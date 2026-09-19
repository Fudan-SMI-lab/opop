"""Two goal-local structural opportunities using the existing TaskRewriter and native TPE."""

import json
from collections import Counter
from collections.abc import Mapping

from pydantic import JsonValue

from kernel_optimizer.agents.runtime import AgentCallError
from kernel_optimizer.agents.task_rewriter import TaskRewriteInputs, TaskRewriterAgent
from kernel_optimizer.config import BudgetConfig
from kernel_optimizer.control.direct_task import TaskSearch, TaskSpace
from kernel_optimizer.evaluation.task_eval import TaskEvaluation
from kernel_optimizer.models.core import ParamSet, sha256_text
from kernel_optimizer.paramspace.guard import ConstraintError
from kernel_optimizer.tuning.objective import Objective
from .model_binding import BundleSpec, load_bundle
from .runner_records import RunnerError
from .search_bundle import anchors, export_selection, validate_child
from .search_helper import HelperService
from .search_records import EvaluationRequest, Opportunity, SearchInputs, SearchResult, Selection
from .search_session import OperatorSession


def profile_context(session: OperatorSession) -> Mapping[str, JsonValue]:
    profile = session.profile
    if profile is None:
        raise RunnerError("missing actual baseline profile")
    source_ids = {path: sha256_text(source) for path, source in profile.data.sources.items()}
    catalog = {source_ids[path]: source for path, source in profile.data.sources.items()}
    counts: Counter[tuple[str, str]] = Counter()
    durations: Counter[tuple[str, str]] = Counter()
    modules: dict[tuple[str, str], set[str]] = {}
    shapes: dict[tuple[str, str], set[str]] = {}
    for row in profile.data.trace:
        path = str(row["module_path"])
        key = (source_ids[path], str(row["phase"]))
        counts[key] += 1
        durations[key] += int(str(row.get("cpu_dispatch_us_inclusive", 0)))
        modules.setdefault(key, set()).add(path)
        shapes.setdefault(key, set()).add(str(row["shape"]))
    return {"goal": profile.data.goal, "source_catalog": catalog,
        "duration_semantics": "inclusive CPU dispatch sums can overlap; not GPU critical path",
        "groups": [{"source_id": key[0], "phase": key[1], "module_paths": sorted(modules[key]),
            "calls": counts[key], "cpu_dispatch_us_inclusive_sum": durations[key],
            "shape_count": len(shapes[key]), "shape_examples": sorted(shapes[key])[:4]} for key in counts]}


def selection(session: OperatorSession, bundle: BundleSpec, result: TaskEvaluation) -> Selection:
    return Selection(bundle=bundle.path, bundle_sha256=bundle.bundle_sha256, source_hashes=dict(bundle.source_hashes),
        params=ParamSet.model_validate({"values": dict(bundle.params)}), params_sha256=bundle.params_sha256,
        site_groups={s.site_id: session.runner.binding.sites[s.site_id] for s in bundle.document.sites},
        evaluation=result, contract_sha256=session.runner.prepared.contract_sha256,
        model_revision=session.runner.prepared.asset_spec.revision, baseline_fallback=not bundle.document.sites,
        goal_id=session.goal.id)


def optimize_goal(session: OperatorSession, agent: TaskRewriterAgent, inputs: SearchInputs) -> SearchResult:
    started = session.now()
    if inputs.goal_id != session.goal.id or inputs.contract != session.runner.prepared.asset_spec.contract_path:
        raise RunnerError("search inputs differ from the resident goal/contract")
    if started < inputs.clock.campaign_started_unix_s:
        raise RunnerError("search launch precedes common S")
    deadline = min(started + 5400, inputs.clock.search_cutoff_unix_s)
    inputs.output.mkdir(parents=True, exist_ok=False)
    session.set_profile(inputs.profile)
    profile = dict(profile_context(session))
    session.begin_opportunity(session.baseline, {}, deadline_unix_s=deadline)
    baseline_result = session.measure_baseline(session.baseline)
    baseline = selection(session, load_bundle(session.baseline, {}), baseline_result)
    incumbent = baseline
    parent_instrumentation = session.instrumentation
    objective = Objective(direction=session.goal.direction, unit=session.goal.unit, label=inputs.goal)
    opportunities: list[Opportunity] = []
    original_hook, original_cfg = agent.self_test_context, agent.cfg
    agent.cfg = agent.cfg.model_copy(update={"max_retries": 0, "max_transport_retries": 0})
    try:
        for number in (1, 2):
            session.begin_opportunity(incumbent.bundle, incumbent.site_groups, deadline_unix_s=deadline)
            session.opportunity = number
            parent_hash = incumbent.bundle_sha256
            errors: list[str] = []
            trials = []
            calls = repairs = 0
            status = "censored"
            if incumbent.evaluation.valid and session.budget.available() and not session.runner.closed:
                for attempt in (0, 1):
                    if not session.budget.available() or session.runner.closed:
                        break
                    parent = load_bundle(incumbent.bundle, incumbent.params.values)
                    context = {"opportunity": number, "parent_bundle_sha256": parent.bundle_sha256,
                        "profile": profile, "profile_artifact": str(inputs.profile),
                        "frozen_contract": json.loads(inputs.contract.read_text()),
                        "operator_interface": (inputs.contract.parent / "interface.md").read_text(),
                        "slots_remaining": session.budget.limit - session.budget.used,
                        "deadline_unix_s": deadline, "fresh_probes": False,
                        "previous_attempts": session.attempts,
                        "repair_feedback": errors[-1] if attempt and errors else None}
                    request = TaskRewriteInputs(project_root=inputs.output, candidate_id=f"{inputs.goal_id}-r{number}",
                        candidate_path=parent.path.parent / parent.document.entry, goal=inputs.goal, context=context,
                        objective=objective, params=incumbent.params, space=TaskSpace.model_validate(parent.document.space),
                        bundle_sources={n: s.decode("utf-8") for n, s in parent.sources.items()},
                        bundle_document=parent.document.model_dump(mode="json"))
                    calls += 1
                    repairs += int(attempt > 0)
                    try:
                        with HelperService(session) as helper:
                            agent.self_test_context = helper.seed_sandbox
                            outcome = agent.invoke(request)
                    except (AgentCallError, OSError, ValueError, RuntimeError) as exc:
                        errors.append(str(exc))
                        status = "failed"
                        break
                    if outcome.output.bundle_file is None:
                        errors.append("C3 requires the complete cumulative bundle_file")
                        status = "failed"
                        break
                    path = (outcome.sandbox.root / outcome.output.bundle_file).resolve()
                    try:
                        bundle = load_bundle(path, {})
                        space = validate_child(bundle, incumbent.bundle)
                        if outcome.output.space is not None and outcome.output.space != space:
                            raise RunnerError("agent output space differs from bundle space")
                        session.groups = {**incumbent.site_groups, **outcome.output.site_groups}
                        with session.lock:
                            session.prepare_bundle(bundle, session.groups)
                        if session.instrumentation != parent_instrumentation:
                            aligned = session.align_parent(incumbent.bundle, incumbent.params)
                            if not aligned.valid:
                                raise RunnerError(f"same-instrumentation parent witness failed: {aligned.detail}")
                            incumbent = incumbent.model_copy(update={"evaluation": aligned})
                            parent_instrumentation = session.instrumentation
                        session.candidate = path
                        search = TaskSearch(session, objective,
                            BudgetConfig(trials_per_space=max(1, session.budget.limit - session.budget.used)), device=session.device)
                        search.probe_budget = 0
                        suggested = anchors(bundle, tuple(outcome.output.recommended_configs), session.device)
                        session._write("anchors.jsonl", {"bundle": str(path), "default": bundle.document.params,
                            "requested": [p.model_dump(mode="json") for p in outcome.output.recommended_configs],
                            "admitted": [p.model_dump(mode="json") for p in suggested]})
                        best = search.evaluate_candidate(path.parent / bundle.document.entry, space, {},
                            startup_trials=2, anchors=suggested, can_evaluate=session.budget.available, stop_on_invalid=True)
                        trials.extend(search.trials)
                        if best is None:
                            raise RunnerError(search.trials[-1].failure_detail if search.trials else "no native trial admitted")
                        measured = best.task_evaluation
                        if measured is not None and measured.valid and measured.score is not None and incumbent.evaluation.score is not None:
                            improved = objective.gain(incumbent.evaluation.score, measured.score) > 0
                            if improved:
                                incumbent = selection(session, load_bundle(path, best.params.values), measured)
                                status = "accepted"
                            else:
                                status = "retained"
                        break
                    except (OSError, ValueError, RuntimeError, SyntaxError, ArithmeticError, ConstraintError) as exc:
                        errors.append(f"{type(exc).__name__}: {exc}")
                        status = "failed"
                        if not trials:
                            session.self_test(EvaluationRequest(bundle=path, params=ParamSet(values={}),
                                                               site_groups=outcome.output.site_groups))
                agent.self_test_context = original_hook
            if status == "failed" and session.budget.clock() >= session.budget.stop:
                status = "censored"
            opportunities.append(Opportunity(number=number, parent_bundle_sha256=parent_hash, status=status,
                slots_used=session.budget.used, model_calls=calls, repair_calls=repairs, trials=trials, errors=errors))
            (inputs.output / f"round-{number}.json").write_text(opportunities[-1].model_dump_json(indent=2), encoding="utf-8")
        incumbent = incumbent.model_copy(update={"clock": inputs.clock})
        frozen = export_selection(incumbent, inputs.output / "selected")
        admitted = [r for r in session.attempts if r.get("admitted")]
        slot_counts = {purpose: sum(r["purpose"] == purpose for r in admitted)
                       for purpose in ("baseline", "native", "self_test", "parent_alignment")}
        result = SearchResult(goal_id=inputs.goal_id, goal=inputs.goal, clock=inputs.clock, started_unix_s=started,
            deadline_unix_s=deadline, finished_unix_s=session.now(),
            drain_s=max(0, session.now() - deadline) if any(r.get("admitted") for r in session.attempts) else 0,
            baseline=baseline, selected=frozen, opportunities=opportunities, slot_counts=slot_counts,
            slots_reconciled=all(o.slots_used == sum(r.get("opportunity") == o.number for r in admitted)
                                 for o in opportunities))
        (inputs.output / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return result
    finally:
        agent.self_test_context, agent.cfg = original_hook, original_cfg
