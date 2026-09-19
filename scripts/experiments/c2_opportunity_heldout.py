# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["kernel-optimizer"]
# ///
# Existing environment: python -m scripts.experiments.c2_opportunity_heldout --help
"""Fresh nine-block same-host GPU0 heldout; never selects, tunes, or reuses screen timings."""

import argparse
import sys
from pathlib import Path
from time import time
from typing import Literal, Self

from pydantic import model_validator

from kernel_optimizer.config import load_config
from kernel_optimizer.gpu.worker_client import to_wsl_path
from kernel_optimizer.models.core import TrialRecord
from kernel_optimizer.paramspace.materializer import materialize
from kernel_optimizer.store.run_store import RunStore
from scripts.experiments.c2_local_adapter import GpuAdapter
from scripts.experiments.c2_local_inputs import InputError
from scripts.experiments.c2_local_runner import isolated_config, worker_environment
from scripts.experiments.c2_method_admission import admission
from scripts.experiments.c2_method_files import copy_helpers
from scripts.experiments.c2_method_gates import block_values
from scripts.experiments.c2_opportunity_artifacts import stage_incumbent
from scripts.experiments.c2_opportunity_inputs import CELLS, CampaignRun, OpportunityInputs, validate_deadline
from scripts.experiments.c2_opportunity_program import Options
from scripts.experiments.c2_opportunity_records import HeldoutResult, OpportunityResult


class HeldoutInputs(OpportunityInputs):
    slot: Literal["A0", "B0"]
    g0: Path
    c2: Path

    @model_validator(mode="after")
    def absolute_results(self) -> Self:
        if not self.g0.is_absolute() or not self.c2.is_absolute():
            raise InputError("heldout result paths must be absolute")
        return self


def run_heldout(inputs: HeldoutInputs, run: CampaignRun) -> HeldoutResult:
    started, deadline = run.start(heldout=True)
    shared = inputs.load_parent(run.cfg)
    g0 = OpportunityResult.model_validate_json(inputs.g0.read_text(encoding="utf-8"))
    c2 = OpportunityResult.model_validate_json(inputs.c2.read_text(encoding="utf-8"))
    for result, arm in ((g0, "G0"), (c2, "C2")):
        spec = CELLS.get((result.wave, result.slot))
        if (spec is None or result.wave != inputs.wave or result.slot[0] != inputs.slot[0] or result.arm != arm
                or spec.arm != arm or result.task != shared.task or spec.task != shared.task
                or result.shared_id != shared.identity() or result.deadline_unix_s != run.deadline_unix_s
                or result.pilot_clock != run.pilot_clock):
            raise InputError("heldout pair identity or campaign deadline differs")
    root = run.output.resolve()
    store = RunStore.create(root.parent, root.name, {"inputs": inputs.model_dump(mode="json"),
        "deadline_unix_s": run.deadline_unix_s, "studies": 0, "model_calls": 0,
        "campaign_started_unix_s": run.campaign_started_unix_s,
        "pilot_clock": run.pilot_clock.model_dump(mode="json") if run.pilot_clock else None,
        "admission_deadline_unix_s": run.admission_deadline(heldout=True), "admission_purpose": "heldout"})
    records: dict[str, list[TrialRecord]] = {"parent": [], "G0": [], "C2": []}
    status: Literal["complete", "failed", "censored"] = "censored"
    error = None
    admitted = False
    try:
        deadline.check()
        admitted = True
        if g0.status == "censored" or c2.status == "censored" or g0.incumbent is None or c2.incumbent is None:
            raise InputError("censored or missing incumbent cannot supply complete heldout")
        parent = root / "parent"
        parent.mkdir()
        parent_source = parent / "parent.py"
        parent_source.write_text(materialize(shared.source, shared.parent.params), encoding="utf-8")
        copy_helpers(inputs.helpers, inputs.reference.parent, parent / "imports")
        g0_source = stage_incumbent(g0.incumbent, root / "G0")
        c2_source = stage_incumbent(c2.incumbent, root / "C2")
        local = isolated_config(run.cfg, root)
        local.run.seed = 0
        project = Path(__file__).resolve().parents[2]
        base = f"{to_wsl_path(project / 'src')}:{to_wsl_path(project)}"
        targets = {"parent": (parent_source, shared.parent.params, shared.backend),
                   "G0": (g0_source, g0.incumbent.params, g0.incumbent.backend),
                   "C2": (c2_source, c2.incumbent.params, c2.incumbent.backend)}
        with admission(deadline), worker_environment(local):
            adapter = GpuAdapter(shared, local, store)
            adapter.full = True
            for block, order in enumerate((("parent", "G0", "C2"), ("C2", "G0", "parent"), ("G0", "C2", "parent"))):
                for role in order:
                    deadline.check()
                    path, params, backend = targets[role]
                    adapter.backend, adapter.phase = backend, f"heldout_{block}_{role}"
                    adapter.correctness.worker.cfg.extra_pythonpath = f"{to_wsl_path(root / role / 'imports')}:{base}"
                    records[role].append(adapter.measure(path, params))
        status = "complete" if all(block_values(r) for r in records.values()) else "failed"
    except (OSError, ValueError, RuntimeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        store.append("HELDOUT_CENSORED", {"error": error})
    result = HeldoutResult(wave=inputs.wave, slot=inputs.slot, task=shared.task, shared_id=shared.identity(),
        g0_result=inputs.g0, c2_result=inputs.c2, status=status, parent_finals=records["parent"],
        g0_finals=records["G0"], c2_finals=records["C2"], started_unix_s=started,
        deadline_unix_s=run.deadline_unix_s, elapsed_s=run.clock() - deadline.started,
        transfer_wait_s=max(0, started - max(r.started_unix_s + r.elapsed_s for r in (g0, c2))),
        late_start_s=max(0, started - run.deadline_unix_s),
        drain_s=max(0, run.now() - run.deadline_unix_s) if admitted else 0, error=error,
        pilot_clock=run.pilot_clock, admission_deadline_unix_s=run.admission_deadline(heldout=True))
    (root / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    store.append("RUN_FINISHED", {"status": status, "error": error, "elapsed_s": result.elapsed_s, "drain_s": result.drain_s})
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "inputs", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--deadline-unix-s", type=float, required=True)
    try:
        options = Options.model_validate(vars(parser.parse_args(argv)))
        validate_deadline(options.deadline_unix_s, time())
        inputs = HeldoutInputs.model_validate_json(options.inputs.read_text(encoding="utf-8"))
        result = run_heldout(inputs, CampaignRun(load_config(options.config), options.output, options.deadline_unix_s))
        return 0 if result.status == "complete" else 1
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
