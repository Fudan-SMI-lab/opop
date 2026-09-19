"""Serialized borrowed-runner evaluations with one persistent provided TaskEvaluator."""

import json
from collections.abc import Callable, Mapping
from fnmatch import fnmatchcase
from pathlib import Path
from threading import RLock
from time import monotonic, time
from types import TracebackType
from typing import Literal, Self

from pydantic import JsonValue

from kernel_optimizer.evaluation.task_eval import TaskEvaluation, TaskEvaluator
from kernel_optimizer.models.core import DeviceLimits, ParamSet
from kernel_optimizer.paramspace.guard import check_config
from .manual_data import CALIBRATION_IDS, fingerprint, verify_frozen
from .model_binding import BundleSpec, load_bundle
from .model_runner import ResidentRunner
from .runner_records import FrozenRecord, GoalSpec, OracleManifest, RunnerError
from .search_bundle import baseline_bundle, parameter_space, validate_child
from .search_records import AdmittedSlot, EvaluationRequest, SlotBudget


class ProfileData(FrozenRecord):
    sources: dict[str, str]
    trace: list[dict[str, JsonValue]]
    goal: str


class ProfileDocument(FrozenRecord):
    binding: dict[str, JsonValue]
    data: ProfileData


class OperatorSession:
    """Own evaluator/attempt records; serialize shared resident state without owning its lifetime."""

    def __init__(self, runner: ResidentRunner, goal: GoalSpec, oracle: Path, output: Path,
                 *, eval_file: Path | None = None) -> None:
        manual = Path(__file__).with_name("manual_task.py").resolve()
        if goal not in runner.prepared.contract.goals or eval_file is not None and eval_file.resolve() != manual:
            raise RunnerError("C3 requires its frozen goal and provided manual evaluator")
        self.runner, self.goal, self.oracle = runner, goal, oracle.resolve()
        self.now: Callable[[], float] = time
        self.clock: Callable[[], float] = monotonic
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        runner.output_dir = self.output / "raw"
        self.device = DeviceLimits()
        self.lock = RLock()
        self.budget = SlotBudget(0, time())
        self.parent = baseline_bundle(self.output / "baseline")
        self.baseline = self.parent
        self.groups: dict[str, tuple[str, ...]] = {}
        self.profile: ProfileDocument | None = None
        self.candidate = self.parent
        self.instrumentation = 0
        self.opportunity = 0
        self.attempts: list[dict[str, JsonValue]] = []
        prepared = runner.prepared
        contract = prepared.asset_spec.contract_path
        self.frozen = {p: fingerprint(p) for p in (contract, contract.parent / "interface.md",
            contract.parent / "proposed-prompts.json", prepared.asset_spec.assets_manifest, self.oracle)}
        manifest = OracleManifest.model_validate_json(self.oracle.read_bytes())
        self.frozen.update({self.oracle.parent / p.logits_file: p.logits_sha256 for p in manifest.prompts})
        self.frozen.update({p.resolve(): fingerprint(p) for p in Path(__file__).parent.glob("*.py")})
        self.evaluator = TaskEvaluator(manual, "evaluate")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 traceback: TracebackType | None) -> None:
        self.evaluator.close()

    def set_profile(self, path: Path) -> None:
        profile = ProfileDocument.model_validate_json(path.read_bytes())
        if profile.data.goal != self.goal.id or profile.binding.get("contract_sha256") != self.runner.prepared.contract_sha256:
            raise RunnerError("profile goal or frozen contract mismatch")
        self.profile = profile
        self.frozen[path.resolve()] = fingerprint(path)
        (self.output / "frozen-inputs.json").write_text(json.dumps({str(p): sha for p, sha in self.frozen.items()},
                                                                  indent=2), encoding="utf-8")

    def begin_opportunity(self, parent: Path, groups: Mapping[str, tuple[str, ...]], *, deadline_unix_s: float) -> None:
        self.parent = parent.resolve()
        self.groups = dict(groups)
        self.budget = SlotBudget(8, deadline_unix_s, now=self.now, clock=self.clock)

    def _write(self, name: str, data: Mapping[str, JsonValue]) -> None:
        with (self.output / name).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(data), allow_nan=False) + "\n")

    def prepare_bundle(self, bundle: BundleSpec, groups: Mapping[str, tuple[str, ...]], *, admitted: bool = False) -> None:
        verify_frozen(self.frozen)
        if self.profile is None and bundle.document.sites:
            raise RunnerError("operator discovery requires the actual baseline phase profile")
        traced = {str(row["module_path"]) for row in self.profile.data.trace} if self.profile else set()
        selected: dict[str, tuple[str, ...]] = {}
        occupied = {p for paths in self.runner.binding.sites.values() for p in paths}
        for site in bundle.document.sites:
            patterns = groups.get(site.site_id, self.runner.binding.sites.get(site.site_id, (site.site_id,)))
            if not patterns or any(not any(fnmatchcase(p, pattern) for p in traced) for pattern in patterns):
                raise RunnerError(f"selected site has no trace coverage: {site.site_id}")
            paths = tuple(dict.fromkeys(p for pattern in patterns for p in sorted(traced) if fnmatchcase(p, pattern)))
            existing = self.runner.binding.sites.get(site.site_id)
            if existing is not None:
                if paths != existing:
                    raise RunnerError("cannot redefine an existing site group")
                continue
            if occupied.intersection(paths):
                raise RunnerError("selected groups overlap")
            occupied.update(paths)
            selected[site.site_id] = paths
        if selected:
            if not admitted and not self.budget.available():
                raise RunnerError("cutoff or budget reached before fixture preparation")
            before, started = self.runner.backend.forward_calls, self.now()
            self.runner.binding.restore()
            for site, paths in selected.items():
                self.runner.binding.register_site(site, paths)
            self.instrumentation += 1
            self.runner.bind(load_bundle(self.baseline, {}))
            try:
                fixture_path = self.runner.capture_fixtures(CALIBRATION_IDS)
                self._write("preparation.jsonl", {"kind": "selected_site_fixtures", "sites": list(selected),
                    "opportunity": self.opportunity,
                    "raw": str(fixture_path), "forward_calls": self.runner.backend.forward_calls - before,
                    "started_unix_s": started, "ended_unix_s": self.now()})
            finally:
                self.runner.binding.restore()

    def _evaluate(self, request: EvaluationRequest, purpose: str, *, structural: bool,
                  budget: SlotBudget, split: Literal["search", "heldout"] = "search",
                  prompt_ids: tuple[str, ...] | None = None, oracle: Path | None = None) -> TaskEvaluation:
        with self.lock:
            started, before = self.now(), self.runner.backend.forward_calls
            admitted = budget.admit()
            result = TaskEvaluation(valid=False, detail="budget/deadline denied" if not admitted else "incomplete")
            identity: dict[str, JsonValue] = {}
            preparation_calls = 0
            self._write("starts.jsonl", {"purpose": purpose, "admitted": admitted, "slot": budget.used,
                "opportunity": self.opportunity,
                "bundle": str(request.bundle), "params": request.params.model_dump(mode="json"), "started_unix_s": started})
            try:
                if admitted:
                    bundle = load_bundle(request.bundle, request.params.values)
                    identity = {"bundle_sha256": bundle.bundle_sha256, "source_hashes": dict(bundle.source_hashes),
                                "params_sha256": bundle.params_sha256}
                    if structural:
                        validate_child(bundle, self.parent)
                    rejection = check_config(parameter_space(bundle), request.params, self.device)
                    if rejection:
                        raise RunnerError(f"invalid parameters: {rejection.reason}: {rejection.detail}")
                    preparation_before = self.runner.backend.forward_calls
                    try:
                        self.prepare_bundle(bundle, request.site_groups, admitted=True)
                    finally:
                        preparation_calls = self.runner.backend.forward_calls - preparation_before
                    identity["site_groups"] = {s.site_id: list(self.runner.binding.sites[s.site_id]) for s in bundle.document.sites}
                    context = {"contract": self.runner.prepared.asset_spec.contract_path,
                        "assets_manifest": self.runner.prepared.asset_spec.assets_manifest,
                        "bundle_path": bundle.path, "output_dir": self.output, "goal_id": self.goal.id,
                        "split": split, "prompt_ids": prompt_ids or self.goal.search_prompt_ids,
                        "resident_runner": self.runner, "oracle_refs": oracle or self.oracle,
                        "call_budget": AdmittedSlot(), "frozen_files": self.frozen}
                    result = self.evaluator.evaluate(bundle.path.parent / bundle.document.entry, request.params.values, context)
            except (OSError, ValueError, RuntimeError, SyntaxError, TypeError, ArithmeticError) as exc:
                result = TaskEvaluation(valid=False, detail=f"{type(exc).__name__}: {exc}")
            finally:
                row: dict[str, JsonValue] = {"purpose": purpose, "admitted": admitted, "slot": budget.used,
                    "opportunity": self.opportunity,
                    "bundle": str(request.bundle), "params": request.params.model_dump(mode="json"), **identity,
                    "evaluation": result.model_dump(mode="json"),
                    "forward_calls": self.runner.backend.forward_calls - before - preparation_calls,
                    "preparation_forward_calls": preparation_calls,
                    "total_forward_calls": self.runner.backend.forward_calls - before,
                    "started_unix_s": started, "ended_unix_s": self.now(), "deadline_unix_s": budget.deadline_unix_s,
                    "drain_s": max(0, self.now() - budget.deadline_unix_s) if admitted else 0}
                self.attempts.append(row)
                self._write("attempts.jsonl", row)
            return result

    def evaluate[T](self, candidate_path: Path, params: Mapping[str, JsonValue], context: Mapping[str, T]) -> TaskEvaluation:
        request = EvaluationRequest(bundle=self.candidate, params=ParamSet.model_validate({"values": dict(params)}),
                                    site_groups=self.groups)
        return self._evaluate(request, "native", structural=True, budget=self.budget)

    def self_test(self, request: EvaluationRequest) -> TaskEvaluation:
        return self._evaluate(request, "self_test", structural=True, budget=self.budget)

    def measure_baseline(self, bundle: Path) -> TaskEvaluation:
        return self._evaluate(EvaluationRequest(bundle=bundle, params=ParamSet(values={})), "baseline",
                              structural=False, budget=SlotBudget(1, self.budget.deadline_unix_s, now=self.now, clock=self.clock))

    def align_parent(self, bundle: Path, params: ParamSet) -> TaskEvaluation:
        return self._evaluate(EvaluationRequest(bundle=bundle, params=params, site_groups=self.groups), "parent_alignment",
                              structural=False, budget=self.budget)
