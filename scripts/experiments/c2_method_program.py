# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["kernel-optimizer"]
# ///
# Existing environment: python -m scripts.experiments.c2_method_program --help
"""Fixed core cell runner and fail-closed optional branch dispatch."""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import chdir
from pathlib import Path
from time import monotonic
from typing import Literal

from kernel_optimizer.config import AppConfig, load_config
from kernel_optimizer.gpu.worker_client import to_wsl_path
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime
from scripts.experiments.c2_local_adapter import GpuAdapter
from scripts.experiments.c2_closed_loop import full_blocks_valid
from scripts.experiments.c2_local_agents import Services
from scripts.experiments.c2_local_execution import acquire
from scripts.experiments.c2_local_costs import costs
from scripts.experiments.c2_local_inputs import InputError, Strict, stage_inputs
from scripts.experiments.c2_local_runner import isolated_config, worker_environment
from scripts.experiments.c2_method_admission import admission
from scripts.experiments.c2_method_gates import summarize_core
from scripts.experiments.c2_method_generation import Generated, GenerationContext, generate_chain
from scripts.experiments.c2_method_native import NativeContext, tune_opportunity
from scripts.experiments.c2_method_protocol import (
    CellResult, Deadline, EventCosts, MethodInputs, Opportunity, Phase, SLOTS, SchedulingExpired, Slot, final_orders,
)


def run_core(inputs: MethodInputs, cfg: AppConfig, output: Path, *, deadline: Deadline | None = None) -> CellResult:
    deadline = deadline or Deadline(monotonic())
    shared = inputs.load_parent()
    if shared.evaluation != cfg.evaluation.model_dump(mode="json") or shared.device != cfg.device:
        raise InputError("config evaluation/device differs from original Shared")
    if cfg.evaluation.perf_trials != 100 or cfg.evaluation.correctness_trials != 5:
        raise InputError("full finals require 100 performance samples and 5 correctness trials")
    root = output.resolve()
    store = RunStore.create(root.parent, root.name, {"slot": inputs.slot, "inputs": inputs.model_dump(mode="json")})
    local = isolated_config(cfg, root)
    local.run.seed = 0
    local.budgets.space_expansions_per_candidate = 0
    local.v3.slope_guide.enabled = False
    local.v4.conditional_scan.mode = "off"
    local.wsl.extra_pythonpath = f"{Path(__file__).resolve().parents[2] / 'src'}:{Path(__file__).resolve().parents[2]}"
    common = root / "common"
    local.wsl.extra_pythonpath = f"{to_wsl_path(common)}:{local.wsl.extra_pythonpath}"
    ordinary = stage_inputs(shared, common, [])
    ordinary = ordinary.model_copy(update={"project_root": common})
    for helper in inputs.helpers:
        target = common / helper.resolve().relative_to(inputs.reference.resolve().parent)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(helper.read_bytes())
    rows = [Opportunity(arm=arm, artifact=common / "parent.py", params=shared.parent.params,
                        selected_space=shared.space, backend=shared.backend)
            for arm in SLOTS[inputs.slot].order]
    generated: dict[str, Generated] = {}
    acquisition_calls: int | None = 0
    acquisition_error = None
    parent_finals = []
    try:
        deadline.check()
        with admission(deadline), worker_environment(local), chdir(common), Runtime(local, root) as runtime:
            with ThreadPoolExecutor(max_workers=3) as pool:
                chains: tuple[Literal["G0", "L", "P"], ...] = ("G0", "L", "P")
                stores = {chain: RunStore.create(root, f"generation-{chain}", {"chain": chain}) for chain in chains}
                g0 = pool.submit(generate_chain, "G0", GenerationContext(shared, Services(local, stores["G0"], runtime), ordinary, deadline))
                futures = {"G0": g0}
                try:
                    deadline.check()
                    acquisition_store = RunStore.create(root, "acquisition", {"budget": 12})
                    acquisition_calls = None
                    response = acquire(shared, (local, acquisition_store), probe_budget=12)
                    acquisition_calls = response.probe_calls
                    if response.probe_calls != 12:
                        raise InputError("fresh endpoint batch did not acquire exactly 12 endpoints")
                    informed = ordinary.model_copy(update={"responses": response.responses})
                    for chain in ("L", "P"):
                        futures[chain] = pool.submit(generate_chain, chain, GenerationContext(
                            shared, Services(local, stores[chain], runtime), informed, deadline))
                except (OSError, ValueError, RuntimeError) as exc:
                    acquisition_error = f"{type(exc).__name__}: {exc}"
                    status = "censored" if isinstance(exc, SchedulingExpired) else "dependency_failed"
                    generated.update({arm: Generated(arm, status=status, error=acquisition_error) for arm in ("L", "P", "H")})
                    recorded = root / "acquisition/events.jsonl"
                    if recorded.is_file():
                        events = RunStore.open(recorded.parent).iter_events()
                        acquisition_calls = sum(e.type == "LOCAL_EVAL_STARTED" for e in events)
                native = NativeContext(shared, local, inputs, root, deadline)
                for index, row in enumerate(rows):
                    if row.arm not in generated:
                        chain = "P" if row.arm == "H" else row.arm
                        for item in futures[chain].result():
                            generated[item.arm] = item
                    rows[index] = tune_opportunity(row, generated[row.arm], native)
                    store.append("METHOD_OPPORTUNITY", rows[index].model_dump(mode="json"))
            finals_store = RunStore.create(root, "finals", {"orders": final_orders(inputs.slot)})
            adapter = GpuAdapter(shared, local, finals_store)
            adapter.full = True
            for block, order in enumerate(final_orders(inputs.slot)):
                for name in order:
                    deadline.check()
                    row = next((r for r in rows if r.arm == name), None)
                    helper_root = row.artifact.parent.parent / "inputs" if row and row.selected == "child" else common
                    adapter.correctness.worker.cfg.extra_pythonpath = f"{to_wsl_path(helper_root)}:{local.wsl.extra_pythonpath}"
                    adapter.backend = row.backend if row else shared.backend
                    adapter.phase = f"final_{name}_{block}"
                    record = adapter.measure(row.artifact if row else common / "parent.py", row.params if row else shared.parent.params)
                    if row:
                        row.finals.append(record)
                    else:
                        parent_finals.append(record)
    except Exception as exc:  # noqa: BROAD_EXCEPT_OK -- job boundary persists every opportunity, including adapter bugs.
        store.append("METHOD_INTERRUPTED", {"error": f"{type(exc).__name__}: {exc}"})
        rows = [r.model_copy(update={"error": r.error or str(exc)}) for r in rows]
    rows = [r.model_copy(update={"final_status": "censored" if len(r.finals) != 3 else
        "complete" if all(t.status == "complete" and t.latency_ms and t.latency_ms.n_samples == 100
                          for t in r.finals) else "failed"}) for r in rows]
    accounting: list[EventCosts] = []
    store_paths = [root / "acquisition", root / "finals", *(root / f"generation-{c}" for c in ("G0", "L", "P")),
                   *(root / r.arm / "retune" for r in rows)]
    for path in store_paths:
        if not (path / "manifest.json").is_file():
            continue
        recorded = RunStore.open(path)
        measured = costs(recorded)
        events = recorded.iter_events()
        ends = [e for e in events if e.type == "AGENT_CALL_FINISHED"]
        worker_accounted = any(e.type == "LOCAL_EVAL_STARTED" for e in events)
        known = bool(ends) and len(ends) == measured.agent_calls
        accounting.append(EventCosts(store=path, worker_attempts=measured.worker_attempts if worker_accounted else None,
            worker_wall_s=measured.worker_wall_s if worker_accounted else None, agent_calls=measured.agent_calls,
            agent_attempts=measured.agent_attempts, agent_wall_s=measured.agent_wall_s,
            provider_cost=measured.agent_cost if known and all(e.payload.get("cost", 0) > 0 for e in ends) else None,
            provider_tokens=measured.agent_tokens if known and all(e.payload.get("tokens") for e in ends) else None))
    elapsed = deadline.clock() - deadline.started
    result = CellResult(slot=inputs.slot, task=shared.task, rep=SLOTS[inputs.slot].rep, shared_id=shared.identity(),
        opportunities=rows, parent_finals=parent_finals, acquisition_calls=acquisition_calls,
        acquisition_error=acquisition_error, elapsed_s=elapsed, drain_s=max(0, elapsed - deadline.seconds), costs=tuple(accounting))
    (root / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    store.append("RUN_FINISHED", {"elapsed_s": elapsed, "drain_s": result.drain_s})
    return result


class Options(Strict):
    phase: Phase
    slot: Slot
    config: Path
    inputs: Path
    output: Path


def main(argv: list[str] | None = None) -> int:
    started = monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("core", "A", "B", "C"), required=True)
    parser.add_argument("--slot", choices=tuple(SLOTS), required=True)
    for name in ("config", "inputs", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    options = Options.model_validate(vars(parser.parse_args(argv)))
    try:
        inputs = MethodInputs.model_validate_json(options.inputs.read_text(encoding="utf-8"))
        if inputs.slot != options.slot:
            raise InputError("CLI slot differs from input wrapper")
        inputs.load_parent()
        if options.phase != "core":
            summary = summarize_core([CellResult.model_validate_json(path.read_text(encoding="utf-8"))
                                      for path in inputs.core_results])
            options.output.mkdir(parents=True, exist_ok=False)
            (options.output / "dispatch.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
            raise InputError(f"not_ready: requested {options.phase}; selected {summary.branch.phase}; {summary.branch.reason}")
        result = run_core(inputs, load_config(options.config), options.output, deadline=Deadline(started))
        complete = full_blocks_valid(result.parent_finals) and all(
            r.status == "complete" and r.final_status == "complete" for r in result.opportunities)
        return 0 if complete else 1
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
