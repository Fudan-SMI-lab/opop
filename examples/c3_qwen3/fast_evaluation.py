"""One resident, serialized local feedback and separately admitted model evaluation."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from time import time

from kernel_optimizer.evaluation.task_eval import TaskEvaluation
from kernel_optimizer.models.core import ParamSet
from .device_capture import capture_target_fixture
from .device_gate import require_device_proof
from .device_records import CaptureSpec, LocalReport, LocalRequest, LocalRuntime, PureFixture, SourceEligibility
from .fast_artifacts import source_gate, preserve_bundle
from .fast_budget import StageBudget
from .fast_model import FastModelRequest, evaluate_fast_model
from .fast_prepare import FastAccess
from .fast_records import Artifact, TargetSpec
from .local_eval import evaluate_local
from .model_binding import load_bundle
from .runner_records import RunnerError


@dataclass(frozen=True, slots=True)
class EvaluationFiles:
    baseline: Path
    oracle: Path
    output: Path


class FastEvaluator:
    """Own per-version admission state; borrow the existing runner and device adapter."""

    def __init__(self, runtime: LocalRuntime, target: TargetSpec, files: EvaluationFiles) -> None:
        if not isinstance(runtime.admission, StageBudget):
            raise RunnerError("fast workflow requires its durable stage budget")
        self.runtime, self.target, self.files = runtime, target, files
        self.budget = runtime.admission
        self.stage = self.budget.spec.stage
        self.lock = RLock()
        self.fixtures: tuple[PureFixture, ...] = ()
        self.reports: list[LocalReport] = []
        self.version = 0
        self.source_identity: str | None = None
        self.configs: set[str] = set()
        self.check_framework: Callable[[], None] = lambda: None
        files.output.mkdir(parents=True, exist_ok=False)
        runtime.runner.output_dir = files.output / "raw"

    def capture(self) -> None:
        self.check_framework()
        runner = self.runtime.runner
        runner.binding.restore()
        existing = runner.binding.sites.get(self.target.brief.site_id)
        if existing is None:
            runner.binding.register_site(self.target.brief.site_id, self.target.brief.module_paths)
        elif existing != self.target.brief.module_paths:
            raise RunnerError("canonical operator registration differs")
        runner.bind(load_bundle(self.files.baseline, {}))
        fixtures = []
        for path, step in self.target.representatives:
            result = capture_target_fixture(CaptureSpec(stage=self.stage, goal=self.target.goal,
                site_id=self.target.brief.site_id, module_path=path, decode_step=step), self.runtime)
            if not result.valid or result.fixture is None:
                raise RunnerError(result.detail or "required target fixture unavailable")
            fixtures.append(result.fixture)
        self.fixtures = tuple(fixtures)

    def begin_version(self, version: int) -> None:
        self.version, self.source_identity, self.configs, self.reports = version, None, set(), []

    def _rejected(self, request: LocalRequest, detail: str) -> LocalReport:
        admitted = self.budget.admit(self.stage, "local") is not None
        path = self.files.output / f"rejected-{self.version}-{len(self.reports)}.json"
        report = LocalReport(stage=self.stage, valid=False, failure_stage="source_gate", detail=detail,
            bundle_sha256="", params_sha256="", source_hashes={}, declared_kernels=request.kernels,
            fixtures=(), counts={"admitted": int(admitted), "full_model_forwards": 0}, wall_ms=0, raw_path=path)
        path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        self.reports.append(report)
        return report

    def local(self, request: LocalRequest, *, from_agent: bool = False) -> LocalReport:
        with self.lock:
            self.check_framework()
            if request.stage != self.stage:
                raise RunnerError("helper request belongs to another stage/framework")
            gate = source_gate(request, self.files.baseline, self.target)
            destination = self.files.output / f"gate-{self.version}-{len(self.reports)}.json"
            destination.write_text(gate.model_dump_json(indent=2), encoding="utf-8")
            if not gate.eligible:
                return self._rejected(request, gate.detail or "static source rejection")
            bundle = load_bundle(request.bundle_path, request.params)
            identity = gate.bundle_sha256 + json.dumps([k.model_dump(mode="json") for k in request.kernels], sort_keys=True)
            if self.source_identity is not None and identity != self.source_identity:
                return self._rejected(request, "source changed after admission; return for the one bounded artifact repair")
            if from_agent and request.params != bundle.document.params:
                return self._rejected(request, "draft helper admits the declared default; framework admits bounded alternatives")
            if gate.params_sha256 not in self.configs and len(self.configs) >= 4:
                return self._rejected(request, "candidate already used four distinct configurations")
            self.source_identity = identity
            self.configs.add(gate.params_sha256)
            frozen = preserve_bundle(request.bundle_path, self.files.output / f"source-{self.version}-{len(self.reports)}")
            scoped = request.model_copy(update={"bundle_path": frozen, "source_gate": SourceEligibility(
                eligible=True, bundle_sha256=gate.bundle_sha256, params_sha256=gate.params_sha256)})
            started = time()
            report = evaluate_local(scoped, self.fixtures, self.runtime)
            self.reports.append(report)
            with (self.files.output / "local-timeline.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"version": self.version, "origin": "agent_helper" if from_agent else "framework",
                    "started_unix_s": started, "ended_unix_s": time(), "report": str(report.raw_path),
                    "deadline_unix_s": self.budget.spec.deadline_unix_s,
                    "drain_s": max(0, time() - self.budget.spec.deadline_unix_s)}) + "\n")
            return report

    def request(self, artifact: Artifact, params: ParamSet | None = None) -> LocalRequest:
        return LocalRequest(mode="fixture_only", stage=self.stage, bundle_path=artifact.bundle,
            params=(params or artifact.params).values, site_id=self.target.brief.site_id, kernels=artifact.kernels,
            source_gate=SourceEligibility(eligible=False, bundle_sha256="", params_sha256=""))

    def model(self, artifact: Artifact | None) -> TaskEvaluation:
        with self.lock:
            self.check_framework()
            proofs: tuple[LocalReport, ...] = ()
            if artifact is not None:
                if not self.budget.model_with_local_available():
                    return TaskEvaluation(valid=False, detail="no combined local/model admission remaining")
                local = self.local(self.request(artifact))
                if not proven_local(local):
                    return TaskEvaluation(valid=False, detail=local.detail or "fresh local prerequisite failed")
                proofs = (local,)
            goal = next(g for g in self.runtime.runner.prepared.contract.goals if g.id == self.target.goal)
            phase = "formal_search" if self.stage.stage == "formal_search" else "development"
            return evaluate_fast_model(FastModelRequest(stage=self.stage,
                access=FastAccess(profile="c3_fast_device", phase=phase),
                bundle_path=artifact.bundle if artifact else self.files.baseline,
                params=artifact.params.values if artifact else {}, goal=goal.id, prompt_ids=goal.search_prompt_ids,
                oracle_refs=self.files.oracle, local_reports=proofs), self.runtime)


def proven_local(report: LocalReport) -> bool:
    if not report.valid or report.latency_us is None or report.official_model_score is not None or not report.fixtures:
        return False
    try:
        for fixture in report.fixtures:
            if not fixture.quality_passed or len(fixture.reference_us) != 20 or len(fixture.candidate_us) != 20 or not fixture.device_evidence:
                return False
            for evidence in fixture.device_evidence:
                require_device_proof(evidence)
    except RunnerError:
        return False
    return True
