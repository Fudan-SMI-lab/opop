# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["kernel-optimizer"]
# ///
# Existing environment: python -m scripts.experiments.c2_opportunity_program --help
"""One fixed opportunity; the operator launches slots/waves under one absolute deadline."""

import argparse
import sys
from pathlib import Path
from time import time
from typing import Literal

from pydantic import Field

from kernel_optimizer.agents.runtime import AgentCallError
from kernel_optimizer.config import load_config
from kernel_optimizer.gpu.worker_client import to_wsl_path
from kernel_optimizer.models.core import sha256_text
from kernel_optimizer.store.run_store import RunStore
from scripts.experiments.c2_information import run_information
from scripts.experiments.c2_information_inputs import InformationResult, InformationRun
from scripts.experiments.c2_local_costs import costs
from scripts.experiments.c2_local_execution import acquire
from scripts.experiments.c2_local_inputs import InputError, Strict
from scripts.experiments.c2_local_runner import isolated_config, worker_environment
from scripts.experiments.c2_method_admission import admission
from scripts.experiments.c2_method_files import copy_helpers
from scripts.experiments.c2_method_gates import block_values
from scripts.experiments.c2_opportunity_artifacts import export_incumbent
from scripts.experiments.c2_opportunity_inputs import CELLS, CampaignRun, G0Responses, OpportunityInputs, validate_deadline
from scripts.experiments.c2_opportunity_records import OpportunityResult


def outcome_status(info: InformationResult) -> Literal["complete", "failed", "censored"]:
    if info.status == "censored" or not block_values(info.parent_finals):
        return "censored"
    if any(pair.parent is not None and pair.parent.status != "complete" for pair in info.confirmation_pairs):
        return "censored"
    if info.status == "valid":
        return "complete"
    if info.child is None:
        return "censored" if info.studies else "failed"
    if info.child.rejection is None and (not info.studies or any(s.asked != 40 for s in info.studies)):
        return "censored"
    return "failed"


def run_opportunity(inputs: OpportunityInputs, run: CampaignRun) -> OpportunityResult:
    started, deadline = run.start()
    shared = inputs.load_parent(run.cfg)
    cell = CELLS[inputs.wave, inputs.slot]
    root = run.output.resolve()
    store = RunStore.create(root.parent, root.name, {"inputs": inputs.model_dump(mode="json"),
        "deadline_unix_s": run.deadline_unix_s, "campaign_started_unix_s": run.deadline_unix_s - 14400,
        "task": cell.task, "arm": cell.arm, "rep": cell.rep, "sampler_seed": cell.seed})
    local = isolated_config(run.cfg, root)
    local.run.seed = 0
    local.budgets.space_expansions_per_candidate = 1
    local.v3.slope_guide.enabled = False
    local.v4.conditional_scan.mode = "off"
    project = Path(__file__).resolve().parents[2]
    local.wsl.extra_pythonpath = f"{to_wsl_path(root / 'imports')}:{to_wsl_path(project / 'src')}:{to_wsl_path(project)}"
    copy_helpers(inputs.helpers, inputs.reference.parent, root / "imports")
    info, incumbent, response_path, acquisition_calls, error = None, None, None, None, None
    status: Literal["complete", "failed", "censored"] = "censored"
    admitted = False
    try:
        deadline.check()
        admitted = True
        with admission(deadline), worker_environment(local):
            store.append("ACQUISITION_STARTED", {"arm": cell.arm})
            if cell.arm == "C2" and inputs.probe_strategy == "provided":
                acquisition_store = RunStore.create(root, "acquisition", {"probe_budget": 12})
                responses = acquire(shared, (local, acquisition_store), probe_budget=12)
            else:
                responses = G0Responses(shared_id=shared.identity())
            acquisition_calls = responses.probe_calls
            response_path = root / "responses.json"
            response_path.write_text(responses.model_dump_json(indent=2), encoding="utf-8")
            store.append("ACQUISITION_FINISHED", {"probe_calls": acquisition_calls})
            deadline.check()
            store.append("INFORMATION_STARTED", {"sampler_seed": cell.seed, "expansion_cap": 1, "promotion_policy": "full"})
            info = run_information(shared, responses, InformationRun(local, "H" if cell.arm == "C2" else "G0",
                root / "information", sampler_seed=cell.seed, deadline=deadline,
                helpers=inputs.helpers, helper_root=inputs.reference.parent, require_b40=True,
                space_expansions_per_candidate=1, promotion_policy="full", probe_strategy=inputs.probe_strategy))
            targeted_path = root / "information/acquisition/responses.json"
            if targeted_path.is_file():
                response_path = targeted_path
                acquisition_calls = info.acquisition_costs.worker_attempts
                store.append("TARGETED_ACQUISITION_FINISHED", {"probe_calls": acquisition_calls})
            status, error = outcome_status(info), info.error
            store.append("INFORMATION_FINISHED", {"status": status, "asked": info.asked, "expanded_count": info.expanded_count})
        if status != "censored":
            incumbent = export_incumbent(shared, info, root)
    except (AgentCallError, OSError, ValueError, RuntimeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        status = "censored"
        store.append("OPPORTUNITY_CENSORED", {"error": error})
        if (root / "acquisition/events.jsonl").is_file():
            acquisition_calls = sum(e.type == "LOCAL_EVAL_STARTED" for e in RunStore.open(root / "acquisition").iter_events())
    events = []
    attempts = 0
    for name in ("generation", "retune"):
        path = root / "information" / name
        if (path / "events.jsonl").is_file():
            recorded = RunStore.open(path)
            events.extend(recorded.iter_events())
            attempts += costs(recorded).agent_attempts
    starts = [e for e in events if e.type == "AGENT_CALL_STARTED"]
    finishes = [e for e in events if e.type == "AGENT_CALL_FINISHED"]
    known_cost = bool(starts) and len(starts) == len(finishes) and all(e.payload.get("cost", 0) > 0 for e in finishes)
    generation = root / "information/generation"
    proposal = None
    if (generation / "events.jsonl").is_file():
        initial = [e for e in RunStore.open(generation).iter_events()
                   if e.type == "AGENT_CALL_STARTED" and e.payload.get("module") == "parameterizer"]
        if initial:
            path = generation / "sandboxes" / initial[0].payload["call_id"] / "candidate/source.py"
            proposal = path if path.is_file() else None
    native_events = root / "information/retune/events.jsonl"
    result = OpportunityResult(wave=inputs.wave, slot=inputs.slot, task=cell.task, rep=cell.rep, arm=cell.arm,
        sampler_seed=cell.seed, shared_id=shared.identity(), parent_source_sha256=sha256_text(shared.source),
        status=status, information=info, incumbent=incumbent, responses=response_path, acquisition_calls=acquisition_calls,
        generation_events=generation / "events.jsonl" if (generation / "events.jsonl").is_file() else None,
        proposal_source=proposal, native_events=native_events if native_events.is_file() else None,
        model_calls_started=len(starts), model_calls_finished=len(finishes), model_attempts=attempts,
        provider_cost=sum(e.payload["cost"] for e in finishes) if known_cost else None,
        started_unix_s=started, deadline_unix_s=run.deadline_unix_s,
        elapsed_s=run.clock() - deadline.started, late_start_s=max(0, started - run.deadline_unix_s),
        drain_s=max(0, run.now() - run.deadline_unix_s) if admitted else 0, error=error)
    (root / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    store.append("RUN_FINISHED", {"status": status, "error": error, "elapsed_s": result.elapsed_s, "drain_s": result.drain_s})
    return result


class Options(Strict):
    config: Path
    inputs: Path
    output: Path
    deadline_unix_s: float = Field(gt=0, allow_inf_nan=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "inputs", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--deadline-unix-s", type=float, required=True)
    try:
        options = Options.model_validate(vars(parser.parse_args(argv)))
        validate_deadline(options.deadline_unix_s, time())
        inputs = OpportunityInputs.model_validate_json(options.inputs.read_text(encoding="utf-8"))
        result = run_opportunity(inputs, CampaignRun(load_config(options.config), options.output, options.deadline_unix_s))
        return 0 if result.status == "complete" else 1
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
