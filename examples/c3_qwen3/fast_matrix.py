"""The 4x3x3 final matrix: every row measured against every goal on the sealed held-out blocks.

The rows are frozen before this runs (`fast_formal.collect_rows`), so nothing here can change
which kernel represents a goal -- this module only measures. Each cell goes through
`fast_model.evaluate_fast_model`, which enforces full-model quality against the block's own
reference oracle before it will return a score, so a fast-but-wrong row cannot produce a number.

Why every row is measured against every goal rather than just its own: a kernel tuned for one
objective may cost something on another, and a table that only fills its diagonal cannot show
that. The off-diagonal cells are what distinguish a goal-specific optimisation from a general
speedup that any objective would have found.

WHERE THE DEVICE PROOF COMES FROM
---------------------------------
It is NOT re-measured here, and that is structural rather than a shortcut: `local_eval.
evaluate_local` refuses to run at all once the stage is "final" ("fixture-only feedback is not
available during sealed final evaluation"). So the sealed phase CANNOT produce a fresh local proof
whatever a budget says, and the proof is inherited from the formal lane that selected the row --
the same evidence `fast_proof.prove_candidate` already accepted. `evaluate_fast_model` re-checks it
by identity -- bundle/params hashes, validity and `framework_id` -- so a report from another row,
another framework or another candidate is rejected rather than silently reused.

THE FINAL LANE'S ALLOWANCE IS DERIVED, NOT GUESSED
--------------------------------------------------
This lane is the one place where the allowance is a FACT about the work rather than a research
budget: the sealed corpus contains exactly three held-out blocks per goal, so the table is
`len(ROWS) x sum(blocks per goal)` model evaluations and `sum(blocks per goal)` reference oracles,
and nothing legitimate needs more. `final_allowance` computes both from the loaded corpus, so the
numbers move if the corpus does instead of drifting out of step with it. Everywhere else, caps are
opt-in and unset by default.
"""

from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import time

from .device_records import LocalReport, LocalRuntime, StageIdentity
from .fast_artifacts import framework_id, read_artifact
from .fast_budget import BudgetSpec, StageBudget
from .fast_evaluation import proven_local
from .fast_formal import GOALS, ROWS, RowEvidence, row_evidence
from .fast_model import FastModelRequest, evaluate_fast_model
from .fast_prepare import FastAccess, attach_fast, load_fast, prepare_fast
from .fast_records import Artifact, FrozenFramework, ProbeResult
from .local_device import TorchLocalDevice
from .model_binding import load_bundle
from .runner_records import FrozenRecord, GoalId, RunnerError
from .search_bundle import baseline_bundle


class MatrixCell(FrozenRecord):
    row: str
    goal_id: GoalId
    block: int
    # "valid" only when full-model quality passed AND a score came back; otherwise "invalid" with
    # its reason, or "censored" when the clock/budget refused the measurement. A measurement that
    # did not happen is never written as a zero.
    status: str
    score: float | None = None
    metrics: dict = {}
    detail: str | None = None


class MatrixReport(FrozenRecord):
    started_unix_s: float
    finished_unix_s: float
    framework_id: str
    contract_sha256: str
    corpus_sha256: str
    device_identity: dict
    rows: tuple[RowEvidence, ...]
    cells: tuple[MatrixCell, ...]
    # Per (row, goal) median over the three blocks, plus each row's gain against the baseline of
    # the SAME column. Descriptive only: three blocks do not support a significance claim.
    summary: dict
    complete: bool


@dataclass(frozen=True, slots=True)
class MatrixInputs:
    contract: Path
    assets_manifest: Path
    frozen_framework: Path
    # goal -> that goal's frozen formal result.json. A goal missing here, or one whose lane did
    # not accept a candidate, is measured as a baseline fallback and labelled as such.
    formal_results: dict[GoalId, Path]
    budget: Path
    output: Path
    started_unix_s: float
    deadline_unix_s: float
    device: str = "cuda:0"
    # The same measured equivalence floor the search lanes used. It MUST be the same file: a row
    # accepted under a floored limit during search and then judged against the unfloored limit here
    # would be rejected at final for the amplification the floor exists to account for, leaving the
    # cell invalid for a reason that has nothing to do with the row. See `runner_quality.criteria`.
    floor_evidence: Path | None = None


def cell_plan(final) -> tuple[tuple[GoalId, int, tuple[str, ...]], ...]:
    """Every (goal, block, prompt_ids) the sealed corpus supplies, in a fixed order.

    One function answers both "how many cells are there" and "which prompts is each one", so the
    allowance, the loop and the completeness check cannot disagree about the size of the table.
    """
    return tuple((goal.id, block, prompt_ids)
                 for goal in final.task.contract.goals
                 for block, prompt_ids in enumerate(goal.final_prompt_groups))


def final_allowance(final) -> dict:
    """The final lane's allowances, derived from the sealed corpus rather than declared by hand.

    `model` is one measurement per row per block and `capture` one reference oracle per block, which
    is all the sealed phase can legitimately do. `local` and `agent` are zero: the sealed phase must
    neither take local feedback (`evaluate_local` refuses it outright) nor call an agent, because the
    rows were chosen before this lane opened and may not be revised in response to a held-out score.
    """
    blocks = len(cell_plan(final))
    return {"agent": 0, "local": 0, "model": len(ROWS) * blocks, "capture": blocks, "profile": 0}


def gain_pct(goal: GoalId, baseline: float, row: float) -> float:
    """Positive always means improvement, whichever direction the goal runs in."""
    if goal == "ttft":
        return 100.0 * (baseline - row) / baseline
    return 100.0 * (row / baseline - 1.0)


def summarise(cells: tuple[MatrixCell, ...]) -> dict:
    """Median over each cell's three blocks, then gain against the same column's baseline.

    A (row, goal) pair is summarised only when all three of its blocks are valid; a partial pair
    would let one surviving block stand in for the two that failed.
    """
    per: dict[tuple[str, GoalId], list[float]] = {}
    for cell in cells:
        if cell.status == "valid" and cell.score is not None:
            per.setdefault((cell.row, cell.goal_id), []).append(cell.score)
    medians = {f"{row}|{goal}": median(values) for (row, goal), values in per.items() if len(values) == 3}
    gains: dict[str, float] = {}
    for goal in GOALS:
        base = medians.get(f"baseline|{goal}")
        if base is None or base <= 0:
            continue
        for row in ROWS:
            if row == "baseline":
                continue
            value = medians.get(f"{row}|{goal}")
            if value is not None:
                gains[f"{row}|{goal}"] = gain_pct(goal, base, value)
    return {"median_by_row_goal": medians, "gain_pct_vs_baseline": gains}


def inherited_proof(result_path: Path, artifact: Artifact, fid: str) -> tuple[LocalReport, ...]:
    """The formal lane's own local receipts for exactly this artifact, under this framework.

    Filtered by bundle/params hash and framework so a row cannot borrow another candidate's
    proof. `evaluate_fast_model` applies the same identity check again before it measures; this
    filter exists so a missing proof fails HERE, with a readable message, rather than as an
    opaque rejection thirty-six cells later.
    """
    result = ProbeResult.model_validate_json(result_path.read_bytes())
    reports = tuple(r for candidate in result.candidates for r in candidate.locals
                    if proven_local(r) and r.bundle_sha256 == artifact.bundle_sha256
                    and r.params_sha256 == artifact.params_sha256 and r.stage.framework_id == fid)
    if not reports:
        raise RunnerError(f"selected row has no passing device proof in {result_path}")
    return reports


def register_row_sites(runner, frozen: FrozenFramework,
                       selected: dict[str, tuple[Artifact, tuple[LocalReport, ...]]]) -> tuple[str, ...]:
    """Register every site a row will bind, before the first cell is measured.

    `ModelBinding.bind` refuses a bundle whose site was never registered ("unresolved traced site"),
    because a replacement may only be installed on modules the binding has been told are a group.
    Every other lane registers through `FastEvaluator.capture`; this lane has no capture step -- the
    sealed phase cannot take local feedback at all -- so it must register the sites itself or every
    cell of every optimised row fails on binding while the baseline rows pass, which looks exactly
    like a kernel that cannot be measured rather than like a missing registration.

    The registration comes from the FROZEN targets, so it is the same site the row was searched and
    proven on: a row's bundle declares a `site_id`, and only the frozen target for that goal says
    which modules that ID covers. Registering all of them up front (rather than per row) also means
    a stale or mismatched target fails here, once, instead of thirty-six times.

    `register_site` rejects a duplicate ID and rejects overlapping module sets, and the three goals
    in this campaign legitimately share one site, so an already-correct registration is accepted and
    a conflicting one is an error.
    """
    registered: list[str] = []
    for goal in GOALS:
        if goal not in selected:
            continue  # a baseline fallback row binds no site, so it needs no registration
        target = frozen.targets.get(goal)
        if target is None:
            raise RunnerError(f"row {goal} has an accepted candidate but no frozen target to site it")
        site_id, paths = target.brief.site_id, target.brief.module_paths
        existing = runner.binding.sites.get(site_id)
        if existing is None:
            runner.binding.register_site(site_id, paths)
            registered.append(site_id)
        elif existing != paths:
            raise RunnerError(f"frozen targets disagree about the modules of site {site_id}")
    return tuple(registered)


def run_matrix(spec: MatrixInputs) -> MatrixReport:
    """Measure 4 rows x 3 goals x 3 blocks on the sealed corpus, in one resident model.

    The model is loaded ONCE in the development phase -- whose 16-prompt corpus the baseline
    bundle belongs to -- and then re-attached to the sealed final phase with `attach_fast`, which
    swaps the prepared task while keeping the same 14 GB of resident weights. Loading per row
    would cost minutes each for no scientific gain.
    """
    started = time()
    frozen = FrozenFramework.model_validate_json(spec.frozen_framework.read_bytes())
    fid = framework_id(frozen.framework)
    stage = StageIdentity(profile="c3_fast_device", stage="final", framework_id=fid)
    # Open the sealed corpus first: it is CPU work, and the lane's allowances are derived from it.
    final = prepare_fast(spec.contract, spec.assets_manifest,
                         FastAccess(profile="c3_fast_device", phase="final", formal_complete=True))
    plan = cell_plan(final)
    budget = StageBudget(spec.budget, BudgetSpec(lane="final", stage=stage,
        started_unix_s=spec.started_unix_s, deadline_unix_s=spec.deadline_unix_s,
        limits=final_allowance(final)))
    if not budget.available():
        raise RunnerError("final lane expired before the resident model was loaded")
    spec.output.mkdir(parents=True, exist_ok=True)

    # Resolve every row BEFORE the model is loaded: a missing proof or a changed byte should cost
    # nothing and fail immediately, not after fourteen gigabytes and the first measurements.
    baseline = baseline_bundle(spec.output / "baseline")
    baseline_sha = load_bundle(baseline, {}).bundle_sha256
    selected: dict[str, tuple[Artifact, tuple[LocalReport, ...]]] = {}
    fallbacks: dict[str, str] = {}
    for goal in GOALS:
        path = spec.formal_results.get(goal)
        if path is None or not path.is_file():
            fallbacks[goal] = "formal lane not run; baseline bytes measured in its place"
            continue
        result = ProbeResult.model_validate_json(path.read_bytes())
        if result.status != "accepted" or result.selected is None:
            fallbacks[goal] = f"lane status {result.status}; baseline bytes measured in its place"
            continue
        artifact = read_artifact(result.selected.bundle.parent / "artifact.json")
        selected[goal] = (artifact, inherited_proof(path, artifact, fid))

    development = prepare_fast(spec.contract, spec.assets_manifest,
                               FastAccess(profile="c3_fast_device", phase="development"))
    runner = load_fast(development, spec.device)
    cells: list[MatrixCell] = []
    rows: list[RowEvidence] = []
    try:
        runner.output_dir = spec.output / "raw"
        runtime = LocalRuntime(runner, TorchLocalDevice(spec.device), budget)
        identity = dict(runtime.device.identity())

        attach_fast(runner, final)
        # AFTER `attach_fast`, never before: swapping the prepared task calls
        # `runner.binding.sites.clear()`, so anything registered earlier is discarded silently.
        register_row_sites(runner, frozen, selected)
        # Before any row binds: `capture_fixtures` refuses once a replacement is installed, and the
        # per-operator check needs these originals for every optimised row.
        capture_original_fixtures(runner, selected)
        oracles = final_oracles(runtime, baseline, final, stage)

        for row in ROWS:
            artifact, reports = selected.get(row, (None, ()))
            detail = None
            for goal, block, prompt_ids in plan:
                if not budget.available():
                    cells.append(MatrixCell(row=row, goal_id=goal, block=block, status="censored",
                                            detail="final deadline reached"))
                    continue
                evaluation = evaluate_fast_model(FastModelRequest(stage=stage,
                    access=FastAccess(profile="c3_fast_device", phase="final", formal_complete=True),
                    bundle_path=artifact.bundle if artifact is not None else baseline,
                    params=dict(artifact.params.values) if artifact is not None else {},
                    goal=goal, prompt_ids=prompt_ids, oracle_refs=oracles[(goal, block)],
                    local_reports=reports, floor_evidence=spec.floor_evidence), runtime)
                detail = detail or evaluation.detail
                cells.append(MatrixCell(row=row, goal_id=goal, block=block,
                    status="valid" if evaluation.valid and evaluation.score is not None else "invalid",
                    score=evaluation.score, metrics=dict(evaluation.metrics), detail=evaluation.detail))
                _write(spec.output, cells, rows, started, fid, development, final, identity, len(plan))
            reason = None if artifact is not None else (
                "original baseline bytes" if row == "baseline" else fallbacks.get(row, "no accepted candidate"))
            rows.append(row_evidence(row, artifact, reports, detail, stage, baseline_sha,
                                     frozen.targets.get(row), reason))
            _write(spec.output, cells, rows, started, fid, development, final, identity, len(plan))
    finally:
        runner.close()
    return _write(spec.output, cells, rows, started, fid, development, final, identity, len(plan))


def capture_original_fixtures(runner, selected: dict) -> int:
    """Capture the ORIGINAL operator fixtures the per-operator correctness check compares against.

    `evaluate_quality` calls `ModelBinding.validate_local`, which re-runs each bound replacement on the
    original operator's captured call and checks its output element-wise. Without those fixtures it
    raises "missing original local fixtures" — so every cell of every optimised row fails while the
    baseline rows, which bind nothing and never reach the check, pass.

    Skipping the check for the sealed lane would be the wrong fix: it is the per-operator gate that
    caught the two deliberately wrong kernels in B12 before they ever reached the logits statistic. So
    the fixtures are captured here instead, and the capture is legitimate at this phase for two
    reasons: it comes from the ORIGINAL dispatch (`capture_fixtures` refuses once anything is bound),
    and the frozen 64-token calibration prompts it uses are present in the sealed corpus as well as the
    development one — they are the same two prompts the formal lanes used, not held-out data.

    Nothing is captured when no row binds a site, so a baseline-only matrix does no extra model work.
    A capture that produced nothing raises here rather than letting all nine cells of the row fail one
    by one with a message about fixtures, which is how this defect presented in the first place.
    """
    if not selected:
        return 0
    runner.binding.restore()
    ids = tuple(pid for pid, prompt in runner.prepared.prompts_by_id.items()
                if len(prompt.input_ids) == 64)
    if not ids:
        raise RunnerError("sealed corpus has no 64-token calibration prompt to capture fixtures from")
    runner.capture_fixtures(ids)
    captured = len(runner.binding.fixtures)
    if not captured:
        raise RunnerError("original fixture capture produced nothing; the per-operator check would "
                          "reject every optimised row")
    return captured


def final_oracles(runtime: LocalRuntime, baseline: Path, final, stage: StageIdentity) -> dict:
    """One reference oracle per (goal, block) -- one per block the sealed corpus holds.

    Generated from the ORIGINAL baseline before any row is bound, so every row is compared
    against the same references instead of against itself.
    """
    runner = runtime.runner
    runner.binding.restore()
    runner.bind(load_bundle(baseline, {}))
    oracles: dict = {}
    for goal, block, prompt_ids in cell_plan(final):
        if runtime.admission.admit(stage, "capture") is None:
            raise RunnerError("final reference oracle admission denied")
        oracles[(goal, block)] = runner.create_oracles(prompt_ids)
    return oracles


def _write(output: Path, cells: list[MatrixCell], rows: list[RowEvidence], started: float, fid: str,
           development, final, identity: dict, blocks: int) -> MatrixReport:
    """Persist after every cell: a run that dies at cell 30 must not lose the first 29.

    `blocks` is how many (goal, block) pairs the sealed corpus supplies, so `complete` is measured
    against the table the corpus actually defines rather than a hardcoded 36.
    """
    expected = len(ROWS) * blocks
    report = MatrixReport(started_unix_s=started, finished_unix_s=time(), framework_id=fid,
        contract_sha256=development.task.contract_sha256,
        corpus_sha256=final.task.corpus_sha256 if final is not None else "",
        device_identity=identity, rows=tuple(rows), cells=tuple(cells),
        summary=summarise(tuple(cells)),
        complete=len(cells) == expected and all(c.status == "valid" for c in cells))
    (output / "matrix.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report
