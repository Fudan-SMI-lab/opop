# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["kernel-optimizer"]
# ///
# Existing v5 environment: python -m scripts.experiments.c2_closed_loop --help
"""Exactly two G0/G2 opportunities; ready times are observations for offline comparison."""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Literal, assert_never

from kernel_optimizer.conditional.task_response import TaskResponse
from kernel_optimizer.config import AppConfig, load_config
from kernel_optimizer.control.direct_task import TaskSpace
from kernel_optimizer.models.core import Candidate, ParameterSpace, TrialRecord
from kernel_optimizer.paramspace.materializer import MaterializeError, extract_defaults, materialize
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.tuning.objective import rank_record
from scripts.experiments.c2_information import run_information
from scripts.experiments.c2_information_inputs import InformationResult, InformationRun
from scripts.experiments.c2_local_costs import Costs
from scripts.experiments.c2_local_execution import acquire
from scripts.experiments.c2_local_inputs import InputError, Responses, Shared, Strict
from scripts.experiments.c2_local_runner import isolated_config, worker_environment

LoopGroup = Literal["G0", "G2"]


@dataclass(frozen=True, slots=True)
class LoopRun:
    cfg: AppConfig
    group: LoopGroup
    run_dir: Path


class RoundState(Strict):
    number: int
    elapsed_started_s: float
    elapsed_ready_s: float | None
    information_result: str | None
    responses: str | None
    artifact: str | None
    incumbent: TrialRecord | None
    fresh_finals: list[TrialRecord]
    error: str | None


class LoopResult(Strict):
    task: str
    group: LoopGroup
    initial_parent_artifact: str
    initial_parent_finals: list[TrialRecord]
    rounds: list[RoundState]
    elapsed_total_s: float
    terminal_error: str | None


def full_blocks_valid(records: list[TrialRecord]) -> bool:
    return len(records) == 3 and all(t.status == "complete" and t.latency_ms is not None
                                    and t.latency_ms.n_samples == 100 for t in records)


def updated_parent(current: Shared, result: InformationResult, opportunity: Path) -> Shared:
    child = result.child
    if child is None or child.selected is None:
        raise InputError("selected child has no native tuning selection")
    matches = [t for t in child.trials if t.status == "complete" and t.latency_ms is not None
               and t.candidate_id == child.selected.candidate_id and t.params == child.selected.params]
    if not matches:
        raise InputError("selected child has no matching native tuning trial")
    parent = child.selected_trial or min(matches, key=lambda t: rank_record(t, None))
    if parent not in matches:
        raise InputError("selected artifact trial disagrees with native selection")
    events = RunStore.open(opportunity / "retune").iter_events()
    spaces = [ParameterSpace.model_validate(e.payload["space"]) for e in events
              if e.type == "SPACE_PUBLISHED" and e.payload["space"]["space_id"] == parent.space_id
              and e.payload["space"]["candidate_id"] == parent.candidate_id]
    candidates = [Candidate.model_validate(e.payload["candidate"]) for e in events
                  if e.type == "CANDIDATE_REGISTERED" and e.payload["candidate"]["candidate_id"] == parent.candidate_id]
    if len(spaces) != 1 or len(candidates) != 1:
        raise InputError("selected child lacks one consistent native space/candidate")
    if any(t.candidate_id != parent.candidate_id for t in child.trials):
        raise InputError("child tuning history mixes candidates")
    source = (opportunity / result.selected_artifact).read_text(encoding="utf-8")
    if extract_defaults(source) != parent.params.values:
        raise InputError("selected artifact PARAMS differ from native tuning selection")
    reused = {e.payload["trial"]["trial_id"] for e in events
              if e.type == "TRIAL_DONE" and e.payload.get("reused_measurement")}
    trials = [t.model_copy(deep=True, update={
        "failure_detail": "[reused_measurement=true] native validation witness; " + t.failure_detail,
    }) if t.trial_id in reused else t.model_copy(deep=True)
        for t in child.trials if t.space_id == parent.space_id]
    parent = parent.model_copy(deep=True)
    return current.model_copy(deep=True, update={
        "source": source, "parent": parent, "trials": trials, "cutoff_seq": None,
        "space": TaskSpace(params=spaces[0].domains, constraints=spaces[0].constraints),
        "backend": candidates[0].backend, "failed_hypotheses": [],
    })


def retained_feedback(current: Shared, outcome: InformationResult, opportunity: Path) -> Shared:
    proposals = [e.payload for e in RunStore.open(opportunity / "generation").iter_events()
                 if e.type == "LEGACY_PROPOSAL_GENERATED"
                 and e.payload["parent_candidate_id"] == current.parent.candidate_id]
    if not proposals:
        return current
    proposal = proposals[-1]
    child = outcome.child
    result = "parameterization_or_tuning_failed"
    child_ms = None
    if child is not None:
        child_ms = child.selected.latency_ms if child.selected else None
        match child.status:
            case "rejected":
                result = "validation_rejected"
            case "no_best":
                result = "no_valid_tuning_result"
            case "final_failed":
                result = "final_measurement_failed"
            case "artifact_error":
                result = "artifact_unavailable"
            case "complete":
                result = "valid_but_not_faster"
            case unreachable:
                assert_never(unreachable)
    entry = {"id": proposal["hypothesis_id"], "change": proposal["change_summary"], "outcome": result,
             "parent_candidate_id": current.parent.candidate_id, "parent_ms": outcome.parent_baseline_ms,
             "child_best_ms": child_ms, "error": outcome.error}
    return current.model_copy(deep=True, update={"failed_hypotheses": [*current.failed_hypotheses, entry]})


def run_closed_loop(initial: Shared, run: LoopRun) -> LoopResult:
    started = monotonic()
    root = run.run_dir.resolve()
    store = RunStore.create(root.parent, root.name, {
        "task": initial.task, "group": run.group, "planned_rounds": 2,
        "sampler_seed": 0, "evaluation_seed": 0, "initial_parent": initial.parent.model_dump(mode="json"),
    })
    (root / "initial-parent.py").write_text(materialize(initial.source, initial.parent.params), encoding="utf-8")
    current = initial.model_copy(deep=True)
    rounds: list[RoundState] = []
    initial_finals: list[TrialRecord] = []
    terminal_error = None
    for number in (1, 2):
        elapsed_started = monotonic() - started
        directory = root / f"round-{number}"
        directory.mkdir()
        (directory / "shared.json").write_text(current.model_dump_json(indent=2), encoding="utf-8")
        opportunity = directory / "opportunity"
        result_path = None
        responses_path = None
        artifact = None
        elapsed_ready = None
        fresh: list[TrialRecord] = []
        error = None
        try:
            match run.group:
                case "G0":
                    acquisition = Responses(shared_id=current.identity(), probe_calls=0, costs=Costs(), responses=[
                        TaskResponse(candidate_id=current.parent.candidate_id, axis=d.name,
                                     a_params=current.parent.params.model_copy(deep=True), reason="not_acquired")
                        for d in current.space.params])
                case "G2":
                    acquisition_store = RunStore.create(directory, "acquisition", {"round": number})
                    cfg = isolated_config(run.cfg, acquisition_store.run_dir)
                    cfg.run.seed = 0
                    with worker_environment(cfg):
                        acquisition = acquire(current, (cfg, acquisition_store), probe_budget=12)
                case unreachable:
                    assert_never(unreachable)
            (directory / "responses.json").write_text(acquisition.model_dump_json(indent=2), encoding="utf-8")
            responses_path = f"round-{number}/responses.json"
            outcome = run_information(current, acquisition, InformationRun(run.cfg, run.group, opportunity))
            result_path = f"round-{number}/opportunity/result.json"
            error = outcome.error
            if number == 1:
                initial_finals = [t.model_copy(deep=True) for t in outcome.parent_finals]
            if not full_blocks_valid(outcome.parent_finals):
                raise InputError("parent full measurements are invalid or incomplete")
            match outcome.selected:
                case "parent":
                    fresh = outcome.parent_finals
                    current = retained_feedback(current, outcome, opportunity)
                case "child":
                    if outcome.child is None or outcome.child.status != "complete" or not full_blocks_valid(outcome.child.finals):
                        raise InputError("selected child full measurements are invalid or incomplete")
                    current = updated_parent(current, outcome, opportunity)
                    fresh = outcome.child.finals
                case unreachable:
                    assert_never(unreachable)
            artifact = (opportunity / outcome.selected_artifact).relative_to(root).as_posix()
            elapsed_ready = monotonic() - started
        except (MaterializeError, OSError, ValueError, RuntimeError) as exc:
            terminal_error = f"round-{number}: {type(exc).__name__}: {exc}"
            error = terminal_error
        row = RoundState(number=number, elapsed_started_s=elapsed_started, elapsed_ready_s=elapsed_ready,
                         information_result=result_path, responses=responses_path, artifact=artifact,
                         incumbent=current.parent.model_copy(deep=True) if elapsed_ready is not None else None,
                         fresh_finals=[t.model_copy(deep=True) for t in fresh], error=error)
        rounds.append(row)
        store.append("ROUND_READY" if elapsed_ready is not None else "ROUND_INTEGRITY_FAILED", row.model_dump(mode="json"))
        if terminal_error is not None:
            break
    result = LoopResult(task=initial.task, group=run.group, initial_parent_artifact="initial-parent.py",
                        initial_parent_finals=initial_finals, rounds=rounds,
                        elapsed_total_s=monotonic() - started, terminal_error=terminal_error)
    (root / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    store.append("RUN_FINISHED", {"elapsed_total_s": result.elapsed_total_s, "terminal_error": terminal_error})
    return result


class Options(Strict):
    config: Path
    shared: Path
    group: LoopGroup
    run_dir: Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "shared", "run-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--group", choices=("G0", "G2"), required=True)
    options = Options.model_validate(vars(parser.parse_args(argv)))
    try:
        initial = Shared.model_validate_json(options.shared.read_text(encoding="utf-8"))
        result = run_closed_loop(initial, LoopRun(load_config(options.config), options.group, options.run_dir))
        print(f"rounds={len(result.rounds)}; terminal_error={result.terminal_error}")
        return 1 if result.terminal_error else 0
    except (MaterializeError, OSError, ValueError, RuntimeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
