"""Three fresh formal goal lanes under one frozen framework, then the 4x3x3 final matrix.

WHY THIS FILE EXISTS
--------------------
The fast path can prove and optimise ONE device kernel for ONE goal (`fast_cli develop`), and it
already knows how to evaluate a bundle against a `final` prompt block:
`fast_model.evaluate_fast_model` reads `goal.final_prompt_groups` whenever the access phase is
"final". What it never had is the loop that ties those together -- freeze the framework, run the
three goals from the ORIGINAL baseline, then measure all four rows against all three goals.
`fast_development.freeze_framework` and `fast_development.formal_goal` were written for exactly
this and had no caller anywhere in the repository.

This is a driver, not a new engine. Every scientific decision -- device proof, quality limits,
budget admission, per-goal clocks, accept/fallback -- stays in the module that already owns it.

WHY NOT `search_matrix.run_matrix`
---------------------------------
That function produced the earlier model-level matrix, but it consumes
`search_records.Selection`, which has no field for a device-kernel declaration, and its
`readiness`/`prepare` chain pins a DIFFERENT frozen contract (`runner_prepare.CONTRACT_SHA`
9d1ef6a7... against `fast_prepare.CONTRACT_SHA` 2c0e2fc7...), so it rejects this task's bytes
outright. Reading the blocks through `evaluate_fast_model` keeps one contract pin and keeps the
per-row device evidence that makes this table worth publishing.

WHAT MAKES A ROW TRUSTWORTHY
----------------------------
An optimised row is only written after `fast_proof.prove_candidate` accepted it, which requires a
real agent session, a fresh local receipt whose device evidence passed
`device_gate.require_device_proof` (compiled + CUDA-trace-confirmed launch + output
participation), full-model quality inside the contract's limits, and a model receipt whose
bundle/params/source hashes match the artifact. The baseline row is explicitly NOT device-proven
and is recorded as such; it must never be presented as an authored-kernel result.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from .device_gate import launch_route
from .device_records import LocalReport, StageIdentity
from .fast_artifacts import framework_id
from .fast_development import formal_goal, freeze_framework
from .fast_prepare import FastAccess, prepare_fast
from .fast_records import Artifact, FrozenFramework, ProbeResult, TargetSpec
from .fast_runtime import RunInputs, open_evaluator, open_generator
from .manual_data import fingerprint
from .runner_records import FrozenRecord, GoalId, RunnerError
from .search_records import SearchClock

GOALS: tuple[GoalId, ...] = ("ttft", "single", "multi")
ROWS: tuple[str, ...] = ("baseline", *GOALS)


class KernelEvidenceRow(FrozenRecord):
    """A declared kernel paired with what was actually observed for it on the device."""

    backend: str
    entry: str
    source_file: str
    kernel_name: str
    launches: int
    cuda_names: tuple[str, ...]
    output_used: bool
    stored_output_args: tuple[int, ...]
    skip_control_rejected: bool
    # Which independent observation established that this kernel ran: "cuda_trace" (its name was in
    # the profiler's CUDA events) or "causal_chain" (the profiler collected no CUDA activity at all,
    # and the counted launch plus a FAILING skip control established it instead). Recorded per kernel
    # so a reader can see which evidence carried each published row rather than assuming the trace
    # did. See `device_gate.launch_route`.
    launch_route: str | None = None
    n_regs: int | None = None
    shared_bytes: int | None = None


class RowEvidence(FrozenRecord):
    """Everything a reader needs to decide whether a table row is an authored-kernel result."""

    row: str
    device_proven: bool
    fallback_reason: str | None = None
    framework_id: str
    original_baseline_sha256: str
    bundle_sha256: str | None = None
    params_sha256: str | None = None
    source_hashes: dict[str, str] = {}
    params: dict = {}
    session_id: str | None = None
    target_site_id: str | None = None
    kernels: tuple[KernelEvidenceRow, ...] = ()
    local_receipts: tuple[str, ...] = ()
    model_receipt_detail: str | None = None


@dataclass(frozen=True, slots=True)
class FormalInputs:
    """One goal lane. The clock is shared so the three lanes cannot drift apart."""

    inputs: RunInputs
    frozen: Path
    clock: SearchClock


def freeze(development_result: Path, targets: dict[GoalId, TargetSpec], compatibility: Path | None,
           contract: Path, assets: Path, output: Path) -> FrozenFramework:
    """Turn one accepted development epoch into the frozen F* that the three lanes share.

    `compatibility` may be None; see `freeze_framework` for why a receipt file is optional.
    """
    prepared = prepare_fast(contract, assets, FastAccess(profile="c3_fast_device", phase="development"))
    frozen = freeze_framework(development_result, targets, compatibility, prepared=prepared)
    output.write_text(frozen.model_dump_json(indent=2), encoding="utf-8")
    return frozen


def run_formal_goal(spec: FormalInputs) -> ProbeResult:
    """One goal's formal lane: a fresh agent session starting from the ORIGINAL baseline.

    The development winner is deliberately NOT carried in. `DraftGenerator.invoke` loads
    `engine.files.baseline` as the parent on every call, so each goal re-derives its own kernel
    and the three lanes stay independent observations instead of three copies of one result.
    """
    frozen = FrozenFramework.model_validate_json(spec.frozen.read_bytes())
    with open_evaluator(spec.inputs, "formal_search", "formal_search") as engine:
        with open_generator(spec.inputs, engine) as generator:
            result = formal_goal(generator, frozen, spec.clock, spec.inputs.output)
    (spec.inputs.output / "formal-result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result


def row_evidence(row: str, artifact: Artifact | None, reports: tuple[LocalReport, ...],
                 model_detail: str | None, stage: StageIdentity, baseline_sha256: str,
                 target: TargetSpec | None, reason: str | None) -> RowEvidence:
    """Pair each declared kernel with the evidence the gate actually accepted for it.

    `require_device_proof` has already rejected any declaration whose name was missing from the
    CUDA trace by the time a row reaches here. Carrying both sides means a reader can SEE that
    rather than take it on trust -- and an optimised row with an empty `kernels` tuple is a bug
    in this driver, not a passing row, so it is rejected below.
    """
    if artifact is None:
        return RowEvidence(row=row, device_proven=False, fallback_reason=reason or "baseline bytes",
                           framework_id=stage.framework_id, original_baseline_sha256=baseline_sha256)
    matched: list[KernelEvidenceRow] = []
    for declaration in artifact.kernels:
        found = None
        for report in reports:
            for fixture in report.fixtures:
                for evidence in fixture.device_evidence:
                    if (evidence.entry, evidence.source_file) == (declaration.entry, declaration.source_file):
                        found = evidence
                        break
                if found is not None:
                    break
            if found is not None:
                break
        if found is None:
            raise RunnerError(f"row {row} declares {declaration.entry} with no matching device evidence")
        matched.append(KernelEvidenceRow(backend=declaration.backend, entry=declaration.entry,
            source_file=declaration.source_file, kernel_name=found.kernel_name, launches=found.launches,
            cuda_names=found.cuda_names, output_used=found.output_used,
            stored_output_args=found.stored_output_args, skip_control_rejected=found.skip_control_rejected,
            launch_route=launch_route(found), n_regs=found.n_regs, shared_bytes=found.shared_bytes))
    if not matched:
        raise RunnerError(f"row {row} claims an authored kernel but declares none")
    return RowEvidence(row=row, device_proven=True, framework_id=stage.framework_id,
        original_baseline_sha256=baseline_sha256, bundle_sha256=artifact.bundle_sha256,
        params_sha256=artifact.params_sha256, source_hashes=dict(artifact.source_hashes),
        params=dict(artifact.params.values), session_id=artifact.session_id,
        target_site_id=target.brief.site_id if target is not None else None, kernels=tuple(matched),
        local_receipts=tuple(str(r.raw_path) for r in reports), model_receipt_detail=model_detail)


def formal_lane_summary(frozen_path: Path, formal_results: dict[GoalId, Path], output: Path) -> dict:
    """Record which goal lanes accepted a candidate, before any held-out block is measured.

    Selection is finished once this file is written. `fast_matrix` reads the same lane results and
    never revisits the choice, so no row can be re-picked in response to how it scored on the
    sealed data. Writing it out separately makes that ordering auditable rather than implicit.
    """
    frozen = FrozenFramework.model_validate_json(frozen_path.read_bytes())
    fid = framework_id(frozen.framework)
    lanes: dict[str, dict] = {}
    for goal in GOALS:
        path = formal_results.get(goal)
        if path is None or not path.is_file():
            lanes[goal] = {"status": "not_run", "accepted": False}
            continue
        result = ProbeResult.model_validate_json(path.read_bytes())
        accepted = result.status == "accepted" and result.selected is not None
        lanes[goal] = {"status": result.status, "accepted": accepted,
                       "result": str(path.resolve()), "result_sha256": fingerprint(path),
                       "bundle_sha256": result.selected.bundle_sha256 if accepted else None,
                       "params_sha256": result.selected.params_sha256 if accepted else None,
                       "session_id": result.selected.session_id if accepted else None,
                       "framework_id_matches": framework_id(result.framework) == fid}
    record = {"framework_id": fid, "lanes": lanes,
              "accepted_goals": sorted(g for g, v in lanes.items() if v["accepted"])}
    output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record
