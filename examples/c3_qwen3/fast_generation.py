"""One dedicated agent call per source version, with a scoped fixture-only helper."""

import json
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Lock, get_ident

from kernel_optimizer.agents.model_operator_rewriter import (
    ModelOperatorContext, ModelOperatorRewriterAgent, ModelOperatorRewriteResult, OperatorStageBudget, OperatorTaskFacet,
)
from kernel_optimizer.agents.runtime import AgentCallError
from kernel_optimizer.agents.sandbox import Sandbox
from kernel_optimizer.agents.task_rewriter import TaskRewriteInputs
from kernel_optimizer.control.direct_task import TaskSpace
from kernel_optimizer.models.core import ParamSet
from kernel_optimizer.tuning.objective import Objective
from .fast_artifacts import preserve_bundle
from .fast_evaluation import FastEvaluator
from .fast_prepare import FastPrepared
from .fast_records import Artifact, CandidateResult, Framework
from .device_records import LocalReport, LocalRequest
from .model_binding import load_bundle
from .runner_records import FrozenRecord, RunnerError
from .search_helper import HelperService


class QualityFacet(FrozenRecord):
    local_operator_rtol: float
    local_operator_atol: float
    logits_relative_l2_max_per_prompt: float
    paired_mean_nll_delta_max_nat_per_token: float


class PolicyFacet(FrozenRecord):
    quality: QualityFacet


class SafeFacet(FrozenRecord):
    shared_policy: PolicyFacet


@dataclass(frozen=True, slots=True)
class GenerationContext:
    prepared: FastPrepared
    framework: Framework
    goal_text: str


@dataclass(frozen=True, slots=True)
class _LocalCall:
    request: LocalRequest
    result: Future[LocalReport]


class _OwnerCallbacks:
    """Single-invocation bridge: transport waits off-thread; resident work stays on its caller."""

    def __init__(self, callback: Callable[[LocalRequest], LocalReport]) -> None:
        self.callback = callback
        self.owner = get_ident()
        self.queue: Queue[_LocalCall | None] = Queue()
        self.lock = Lock()
        self.accepting = False
        self.used = False

    def request(self, request: LocalRequest) -> LocalReport:
        result: Future[LocalReport] = Future()
        with self.lock:
            if not self.accepting:
                raise RunnerError("agent helper execution is closed")
            self.queue.put(_LocalCall(request, result))
        return result.result()

    def _close(self) -> None:
        with self.lock:
            self.accepting = False
            while True:
                try:
                    call = self.queue.get_nowait()
                except Empty:
                    return
                if call is not None:
                    call.result.set_exception(RunnerError("agent ended before helper execution"))

    def run[T](self, operation: Callable[[], T]) -> T:
        if get_ident() != self.owner or self.used:
            raise RunnerError("owner callback bridge must run once on its creating thread")
        self.used = True
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="c3-agent-transport")
        with self.lock:
            self.accepting = True
        try:
            agent = pool.submit(copy_context().run, operation)
            agent.add_done_callback(lambda _: self.queue.put(None))
            while (call := self.queue.get()) is not None:
                if agent.done():
                    call.result.set_exception(RunnerError("agent ended before helper execution"))
                    continue
                try:
                    result = self.callback(call.request)
                except BaseException as exc:  # noqa: BROAD_EXCEPT_OK - transfer every callback failure to its waiting RPC
                    call.result.set_exception(exc)
                    if not isinstance(exc, Exception):
                        raise
                else:
                    call.result.set_result(result)
            return agent.result()
        finally:
            # Close/reject RPC waiters before joining transport or shutting down the helper socket.
            self._close()
            pool.shutdown(wait=True, cancel_futures=True)


class DraftGenerator:
    def __init__(self, agent: ModelOperatorRewriterAgent, evaluator: FastEvaluator, context: GenerationContext) -> None:
        if agent.execution_profile != "model_operator" or agent.task_kind != "model_project_operator":
            raise RunnerError("fast generation requires the explicitly selected model_operator agent")
        if agent.cfg.max_retries != 0 or agent.cfg.max_transport_retries != 0:
            raise RunnerError("fast generation must not use generic or transport retry draws")
        self.agent, self.evaluator, self.context = agent, evaluator, context

    def invoke(self, version: int, previous: CandidateResult | None = None) -> CandidateResult:
        engine, agent = self.evaluator, self.agent
        engine.check_framework()
        if engine.budget.admit_agent(engine.stage) is None:
            return CandidateResult(version=version, status="censored", error="agent admission denied")
        engine.begin_version(version)
        parent = load_bundle(engine.files.baseline, {})
        goal = next(g for g in self.context.prepared.task.contract.goals if g.id == engine.target.goal)
        safe = SafeFacet.model_validate(self.context.prepared.agent_context(goal.id))
        context = ModelOperatorContext(framework_revision=self.context.framework.revision,
            parent_bundle_sha256=parent.bundle_sha256,
            task_facet=OperatorTaskFacet(contract_sha256=self.context.prepared.task.contract_sha256,
                quality_limits=safe.shared_policy.quality.model_dump()), operator_brief=engine.target.brief,
            stage_budget=OperatorStageBudget(local_evaluations_remaining=engine.budget.remaining("local"),
                model_evaluations_remaining=engine.budget.remaining("model")),
            repair_feedback=json.dumps({"failure": previous.error, "immutable_artifact": str(previous.failed_sandbox),
                "repair_of": previous.artifact.bundle_sha256 if previous.artifact else None,
                "instruction": "Repair the failed artifact under this unchanged framework; do not change the goal or reference."}) if previous else None)
        inputs = TaskRewriteInputs(project_root=engine.files.output, candidate_id=f"C{version}",
            candidate_path=parent.path.parent / parent.document.entry, goal=self.context.goal_text,
            context=context.model_dump(mode="json"), objective=Objective(direction=goal.direction, unit=goal.unit),
            params=ParamSet(values={}), space=TaskSpace.model_validate(parent.document.space),
            bundle_document=parent.document.model_dump(mode="json"),
            bundle_sources={n: data.decode("utf-8") for n, data in parent.sources.items()})
        old_hook = agent.self_test_context
        sandbox: Sandbox | None = None
        before = len(list(agent.store.iter_events()))
        callbacks = _OwnerCallbacks(lambda request: engine.local(request, from_agent=True))
        try:
            with HelperService(fixture_handler=callbacks.request) as helper:
                def seed(sb: Sandbox) -> str:
                    nonlocal sandbox
                    sandbox = sb
                    hint = helper.seed_sandbox(sb)
                    sb.write_input("task/fast-budget.json", json.dumps({"stage": engine.stage.model_dump(mode="json"),
                        "deadline_unix_s": engine.budget.spec.deadline_unix_s, "prompt_timeout_s": 480,
                        "source_versions_max": 2, "local_remaining": engine.budget.remaining("local"),
                        "model_remaining": engine.budget.remaining("model"), "helper_configs": "declared default only"}))
                    sb.write_input("task/source-admission.md", "The helper recomputes SOURCE_ELIGIBLE from the actual bundle/params/declarations. "
                        "Do not load another model. Triton-only/direct output operands. Submit the default first. "
                        "After first GPU admission source bytes are frozen; return failures for one framework-driven artifact repair. "
                        "Declare at most one useful alternative in recommended_configs before its measurement. "
                        "Do not run private GPU tests outside this shared allowance.\n")
                    return hint
                agent.self_test_context = seed
                outcome = callbacks.run(lambda: agent.invoke(inputs))
            output = ModelOperatorRewriteResult.model_validate(outcome.output.model_dump())
            path = (outcome.sandbox.root / output.bundle_file).resolve()
            frozen = preserve_bundle(path, engine.files.output / f"C{version}")
            bundle = load_bundle(frozen, {})
            params = ParamSet.model_validate({"values": bundle.document.params})
            effective = load_bundle(frozen, params.values)
            artifact = Artifact(author="agent", version=version, session_id=outcome.session_id, bundle=frozen,
                bundle_sha256=effective.bundle_sha256, params_sha256=effective.params_sha256,
                source_hashes=dict(effective.source_hashes), params=params, kernels=output.device_kernels,
                alternative=next((p for p in output.recommended_configs if p != params), None),
                repair_of=previous.artifact.bundle_sha256 if previous and previous.artifact else None)
            return CandidateResult(version=version, status="draft_ready", artifact=artifact,
                failed_sandbox=self._preserve(sandbox, version))
        except (AgentCallError, OSError, ValueError, RuntimeError) as exc:
            events = list(agent.store.iter_events())[before:]
            transport = any(e.type == "AGENT_CALL_FAILED" and not e.payload.get("final") for e in events)
            return CandidateResult(version=version, status="transport_failed" if transport else "source_rejected",
                error=f"{type(exc).__name__}: {exc}", failed_sandbox=self._preserve(sandbox, version))
        finally:
            agent.self_test_context = old_hook

    def _preserve(self, sandbox: Sandbox | None, version: int) -> Path | None:
        if sandbox is None:
            return None
        destination = self.evaluator.files.output / f"C{version}-workspace"
        destination.mkdir()
        for path in sandbox.root.rglob("*"):
            if path.is_file() and (path.suffix == ".py" or path.name in ("bundle.json", "operator_declaration.json")):
                target = destination / path.relative_to(sandbox.root)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
        return destination
