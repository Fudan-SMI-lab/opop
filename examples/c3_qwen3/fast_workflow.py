"""Fixed per-epoch generation/local/model sequence; framework revision decisions stay external."""

import json
import shutil
from dataclasses import replace
from pathlib import Path
from time import time

from kernel_optimizer.config import AppConfig
from kernel_optimizer.tuning.objective import Objective
from .fast_acceptance import DEFAULT_PATIENCE, Incumbent, meaningfully_beats, score_spread_frac
from .fast_artifacts import framework_id, write_artifact
from .fast_generation import DraftGenerator
from .fast_proof import prove_candidate
from .fast_records import CandidateResult, ProbeResult
from .fast_search import tune_local
from .fast_setup import measure_equivalence_floor, measure_reference
from .fast_target import require_target
from .model_binding import load_bundle
from .runner_records import RunnerError
from .search_records import SearchClock


def scoped_config(cfg: AppConfig) -> AppConfig:
    """Scope the shared config to one fast epoch WITHOUT shortening the transport ceiling.

    `request_timeout_s` is deliberately left at whatever the config declares (1500s for every
    C2 and C3 launch config, and the `OpencodeConfig` default). An earlier version of this
    function forced it to 480, which is below the MEASURED rewriter call duration: across the
    two window-2 C2 runs the per-call median is 519.2s and 736.2s with maxima of 902.8s and
    1306.3s, so 480 could not accommodate even the median, let alone a call that also has to
    absorb a local-gate failure and revise. Both fast probes ended in a 480s ReadTimeout, and
    the F1 receipts show why that was structural rather than unlucky: its local gate returned
    at t=422.2s, leaving 57.8s for a rewrite that has never once finished that fast.

    The ceiling is also not a standalone number -- `idle_abort_frac` is a FRACTION of it, so
    480 silently cut the "produced nothing" tolerance to 240s while a working rewriter is
    routinely quiet for longer. Restoring the declared value restores both together, and keeps
    this path consistent with every other agent in the project instead of inventing a second,
    stricter clock for one module.

    Retries stay at zero: a fast epoch draws exactly one artifact, and that is a sampling rule
    (no redraw on transport failure), not a timing rule.
    """
    scoped = cfg.model_copy(deep=True)
    scoped.agents.rewriter.max_retries = 0
    scoped.agents.rewriter.max_transport_retries = 0
    return scoped


def _incumbent_from(candidates: list[CandidateResult], objective: Objective,
                    baseline_score: float | None) -> Incumbent | None:
    """Reconstruct the best confirmed win from a list of candidates, for resume.

    The incumbent is the accepted candidate whose CONFIRMATION (not first measurement) ranks best
    under the objective -- the same rule the live loop applies, read back from disk. `gain` is
    against the reloaded baseline when one is known, else 0.0 (the value is only used for the
    "strictly better than incumbent" comparison, which `objective.rank` drives).
    """
    best: Incumbent | None = None
    for c in candidates:
        if c.status != "valid" or c.confirmation is None or c.confirmation.score is None:
            continue
        if best is None or objective.rank(c.confirmation.score) < objective.rank(best.score):
            gain = objective.gain(baseline_score, c.confirmation.score) if baseline_score is not None else 0.0
            best = Incumbent(candidate=c, score=c.confirmation.score, gain=gain)
    return best


def _load_resumable(output: Path, frame) -> ProbeResult | None:
    """Load a prior lane result from `output` to continue it, or None if there is nothing to resume.

    A resume must be of the SAME framework: continuing a lane with different code would pool draws
    made under different rules, exactly what `framework_id` exists to prevent. A missing result is
    not an error under `resume=True` (the first run of a lane that will later be resumed), so it
    returns None and the caller creates the directory fresh.
    """
    result_path = output / "result.json"
    if not result_path.is_file():
        return None
    prior = ProbeResult.model_validate_json(result_path.read_bytes())
    if framework_id(prior.framework) != framework_id(frame):
        raise RunnerError("cannot resume a lane frozen under a different framework_id")
    return prior


def run_probe(generator: DraftGenerator, output: Path, *, epoch: int = 0,
              clock: SearchClock | None = None, max_versions: int | None = None,
              patience: int = DEFAULT_PATIENCE, resume: bool = False) -> ProbeResult:
    """Draw candidates until the window/budget runs out, keeping the best proven speed win.

    `max_versions` was fixed at two by the original research protocol (`for version in (0, 1)`).
    It is now unbounded by default: a draw that fails for an ordinary reason -- a kernel that does
    not compile, a numerical mismatch, a transport timeout -- says nothing about whether the NEXT
    draw is valid, so capping the count only decided how much sampling was permitted. Pass an
    integer to restore a ceiling.

    ACCEPTANCE AND STOPPING (formal_search). A candidate is a real win only when its repeated
    full-model measurement beats the baseline by more than `NOISE_MULTIPLE` times the goal's own
    measured score spread (`fast_acceptance.meaningfully_beats`); a gain inside that band is
    indistinguishable from measuring the baseline twice. Unlike the original protocol, the FIRST
    such win does not end the lane: the loop keeps drawing, remembers the best win as the incumbent,
    and stops only when the window/budget closes or `patience` consecutive draws produce no new
    incumbent. The published `selected` artifact is the best incumbent, not the first acceptance.

    `resume` continues a lane that stopped with budget or window left instead of starting fresh: it
    reloads the prior `result.json` from `output`, keeps its candidates and best incumbent, and draws
    more against the same budget. Without it, an existing `output` is an error (a fresh lane must not
    silently write into a directory that already holds another lane's work). The patience counter
    restarts on resume -- resuming is a deliberate "keep searching", so it earns a fresh window.

    Development is unchanged: it still stops at the first proven candidate (DEV_GO seals a framework,
    it does not search for an optimum).

    Correctness is never touched here: `prove_candidate`, the logits/NLL floor and the device proof
    gate every candidate before this code runs, so more attempts and a higher speed bar only ever
    change which proven-correct candidate is published, never the standard it had to meet.
    """
    engine = generator.evaluator
    prepared, frame = generator.context.prepared, generator.context.framework
    require_target(engine.target, noise=False)
    if prepared.access.phase not in ("development", "formal_search"):
        raise RunnerError("candidate workflow cannot receive sealed final data")
    if engine.stage.framework_id != framework_id(frame):
        raise RunnerError("generation and admission framework identities differ")
    prior = _load_resumable(output, frame) if resume else None
    if prior is None:
        output.mkdir(parents=True, exist_ok=False)
    started = engine.budget.now()
    candidates: list[CandidateResult] = list(prior.candidates) if prior is not None else []
    baseline = prior.baseline if prior is not None else None
    selected = prior.selected if prior is not None else None
    phase = prepared.access.phase
    status = "PARENT_TRIAGE" if phase == "development" else "baseline"
    goal = next(g for g in prepared.task.contract.goals if g.id == engine.target.goal)
    objective = Objective(direction=goal.direction, unit=goal.unit)
    try:
        if not engine.fixtures:
            engine.capture()
        engine.target = measure_reference(engine)
        require_target(engine.target)
        # Calibrate this site group's equivalence floor before any candidate is judged, unless the lane
        # was handed one already (the setup lane's authored control writes one, and a formal campaign
        # inherits that file so every lane and the matrix enforce the identical limit).
        #
        # WHY HERE: the floor is a property of the SITE GROUP, not of a candidate -- the same
        # equivalence-preserving kernel measured 0.0 at one module and 0.010349 at 73 -- so it has to be
        # measured on the group this lane will search, before the first verdict. The site is registered
        # and the fixtures are captured by this point, and nothing is bound yet, which is exactly what
        # the measurement needs.
        #
        # This raises the logits criterion only, and only to what a ONE-ULP implementation difference
        # actually costs; faithfulness stays with the per-operator check, which measurement shows
        # rejects an identity pass-through, a zeroed output, a dropped weight and a beyond-tolerance
        # scaling before the logits criterion is ever consulted.
        if engine.files.floor_evidence is None and (output / "equivalence-floor.json").is_file():
            # Resume: a floor was calibrated on the first run and written here. Reuse it verbatim
            # rather than re-measuring -- recalibration would spend budget and could land on a
            # slightly different value, so a resumed lane would judge candidates against a floor the
            # earlier draws never saw. The unavailable-marker case is left to fall through to the
            # calibration block, which is a no-op that re-records why there is no floor.
            engine.files = replace(engine.files, floor_evidence=output / "equivalence-floor.json")
        if engine.files.floor_evidence is None:
            # `engine.model` reads `self.files.floor_evidence` afresh on every call, so every verdict
            # from here on -- including the baseline's -- enforces the same calibrated limit.
            #
            # A floor is OPTIONAL: absent one, `runner_quality.criteria` enforces the contract's declared
            # limits exactly as before. So a backend that cannot produce one (no torch, a non-tensor
            # stand-in, an output dtype with no integer view) must leave the lane running under the
            # declared limit rather than abort it -- the floor can only ever RAISE that limit, so failing
            # to measure one is strictly conservative. The reason is recorded so a lane that silently ran
            # under the declared limit can be told apart from one that chose to.
            try:
                floor = measure_equivalence_floor(engine, output / "equivalence-floor.json")
            except (ImportError, RunnerError, TypeError) as exc:
                floor = None
                (output / "equivalence-floor-unavailable.json").write_text(
                    json.dumps({"reason": f"{type(exc).__name__}: {exc}",
                                "consequence": "the contract's declared logits limit stays in force"},
                               indent=2), encoding="utf-8")
            if floor is not None:
                engine.files = replace(engine.files, floor_evidence=floor)
        previous = None
        version = len(candidates) - 1  # resume continues numbering past the reloaded draws
        incumbent = _incumbent_from(candidates, objective,
                                    baseline.score if baseline is not None else None) if candidates else None
        since_best = 0                       # patience restarts on resume: it is a "keep searching"
        while max_versions is None or version + 1 < max_versions:
            version += 1
            if not engine.budget.available() or engine.runtime.runner.closed:
                status = "censored"
                break
            candidate = generator.invoke(version, previous)
            if candidate.status == "censored":
                # Admission itself was refused: more draws cannot help.
                candidates.append(candidate)
                break
            if candidate.status == "transport_failed":
                # A timed-out or dropped agent call produced no artifact to judge. Under the
                # original two-draw protocol this ended the epoch; with sampling unbounded it is
                # just a lost draw, so the next one proceeds. `previous` stays as it was: there is
                # no failed artifact to hand back for repair.
                candidates.append(candidate)
                continue
            if candidate.artifact is None:
                candidates.append(candidate)
                previous = candidate
                continue
            artifact, reports = tune_local(engine, candidate.artifact)
            if artifact is None:
                candidate = candidate.model_copy(update={"status": "local_failed", "locals": reports,
                    "error": reports[-1].detail if reports else "no valid local point"})
                candidates.append(candidate)
                previous = candidate
                continue
            if baseline is None:
                baseline = engine.model(None)
            if not baseline.valid:
                candidates.append(candidate.model_copy(update={"status": "model_failed", "artifact": artifact,
                    "locals": reports, "error": baseline.detail}))
                break
            measured = engine.model(artifact)
            candidate = candidate.model_copy(update={"artifact": artifact, "model": measured,
                "locals": tuple(engine.reports), "status": "valid" if measured.valid else "model_failed", "error": measured.detail})
            if not prove_candidate(candidate, engine.stage, goal):
                candidates.append(candidate)
                previous = candidate
                continue
            if phase == "development":
                status = "DEV_GO"
                candidates.append(candidate)
                break
            # A CORRECT CANDIDATE THAT DOES NOT MEANINGFULLY BEAT THE BASELINE IS NOT THE END.
            #
            # "Meaningfully" is the change from the original `gain > 0`: the gain must clear
            # `NOISE_MULTIPLE` times the goal's own measured score spread, or it is a win inside the
            # measurement noise (the accepted ttft candidate on up_proj improved 0.11% while that
            # site's full-model repeat spread was 0.372%). A candidate that fails this is handed back
            # as `previous` -- the rewriter's feedback path -- exactly like a slower one.
            spread = score_spread_frac(baseline)
            if (baseline.score is None or measured.score is None
                    or not meaningfully_beats(objective, baseline.score, measured.score, spread)):
                gain = None if baseline.score is None or measured.score is None else objective.gain(baseline.score, measured.score)
                candidate = candidate.model_copy(update={"status": "not_improved",
                    "error": (f"proven correct; gain {gain} does not clear {spread:.4%} noise band "
                              f"({objective.direction} {measured.score} vs {baseline.score} {objective.unit or ''})".strip())})
                candidates.append(candidate)
                previous = candidate
                since_best += 1
                if incumbent is not None and since_best >= patience:
                    status = "accepted"  # plateau: keep the incumbent, stop spending the window
                    break
                continue
            # The first measurement clears the band; confirm it with a repeat that must ALSO clear it,
            # so a single lucky read cannot promote a candidate.
            confirmation = engine.model(artifact)
            confirmed = candidate.model_copy(update={"model": confirmation, "locals": tuple(engine.reports)})
            candidate = candidate.model_copy(update={"confirmation": confirmation, "locals": tuple(engine.reports)})
            confirmed_ok = (prove_candidate(confirmed, engine.stage, goal) and confirmation.score is not None
                            and meaningfully_beats(objective, baseline.score, confirmation.score, score_spread_frac(confirmation)))
            if not confirmed_ok:
                # A real outcome for THIS candidate, not a verdict on the next one. Count it against
                # patience and keep drawing.
                candidate = candidate.model_copy(update={"status": "not_confirmed",
                    "error": "first measurement cleared the noise band; the repeat did not confirm it"})
                candidates.append(candidate)
                previous = candidate
                since_best += 1
                if incumbent is not None and since_best >= patience:
                    status = "accepted"
                    break
                continue
            # Confirmed win. `valid` is the established label for "passed every gate AND beat the
            # baseline on a confirmed repeat" (see CandidateResult.status). Keep it if it beats the
            # current incumbent; either way the lane keeps searching -- the FIRST acceptance no longer
            # ends the search, so the published row is the best win the window buys, not the earliest.
            confirmed_gain = objective.gain(baseline.score, confirmation.score)
            candidate = candidate.model_copy(update={"status": "valid"})
            candidates.append(candidate)
            previous = candidate
            if incumbent is None or confirmed_gain > incumbent.gain:
                incumbent = Incumbent(candidate=candidate, score=confirmation.score, gain=confirmed_gain)
                # `preserve_bundle` refuses to write into an existing directory, so a new incumbent
                # replacing an earlier one must clear the old selection first.
                if (output / "selected").exists():
                    shutil.rmtree(output / "selected")
                selected = write_artifact(artifact, output / "selected")
                status = "accepted"
                since_best = 0
            else:
                since_best += 1
                if since_best >= patience:
                    break
            continue
    except (OSError, ValueError, RuntimeError) as exc:
        status = "framework_invalid"
        candidates.append(CandidateResult(version=len(candidates), status="framework_invalid", error=f"{type(exc).__name__}: {exc}"))
    if not engine.budget.available() and status not in ("DEV_GO", "accepted"):
        status = "censored"
    result = ProbeResult(phase=phase, epoch=epoch, framework=frame, target=engine.target,
        original_baseline_sha256=load_bundle(engine.files.baseline, {}).bundle_sha256,
        started_unix_s=started, finished_unix_s=engine.budget.now(), deadline_unix_s=engine.budget.spec.deadline_unix_s,
        drain_s=max(0, engine.budget.now() - engine.budget.spec.deadline_unix_s), status=status, candidates=tuple(candidates),
        baseline=baseline, selected=selected, budget_path=engine.budget.path, clock=clock,
        floor_evidence=engine.files.floor_evidence)
    (output / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result
