"""Selected bounded exploratory A; raw core fidelity remains failed."""

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from pydantic import Field

from kernel_optimizer.config import AppConfig
from kernel_optimizer.store.run_store import RunStore
from scripts.experiments.c2_closed_loop import LoopResult, LoopRun, run_closed_loop
from scripts.experiments.c2_local_inputs import InputError, Shared, Strict
from scripts.experiments.c2_method_admission import admission
from scripts.experiments.c2_method_files import copy_helpers
from scripts.experiments.c2_method_gates import GateSummary, summarize_core
from scripts.experiments.c2_method_protocol import CellResult, Deadline, MethodInputs, Slot

A_ARMS: Final[dict[Slot, Literal["G0", "H"]]] = {"A0": "G0", "A1": "H", "B0": "G0", "B1": "H"}


class AInputs(MethodInputs):
    core_results: tuple[Path, ...] = Field(min_length=4, max_length=4)
    source_fidelity: Literal["failed"]
    execution_kind: Literal["bounded_exploratory_confirmation"]


@dataclass(frozen=True, slots=True)
class ARun:
    cfg: AppConfig
    output: Path
    deadline: Deadline
    started_unix_s: float


class AResult(Strict):
    phase: Literal["A"] = "A"
    slot: Slot
    arm: Literal["G0", "H"]
    task: str
    original_shared_id: str
    source_fidelity: Literal["failed"] = "failed"
    execution_kind: Literal["bounded_exploratory_confirmation"] = "bounded_exploratory_confirmation"
    clean_core_pass: Literal[False] = False
    provider_cost: float | None = None
    raw_gate: GateSummary
    status: Literal["valid", "failed", "censored"]
    started_unix_s: float
    deadline_unix_s: float
    drain_s: float
    loop: LoopResult
    current: Shared
    exported_source: Path
    exported_source_sha256: str
    exported_helper_root: Path
    exported_helpers: tuple[Path, ...]
    exported_helper_sha256: dict[str, str]


def run_a(inputs: AInputs, run: ARun) -> AResult:
    original = inputs.load_parent()
    core = [CellResult.model_validate_json(p.read_text(encoding="utf-8")) for p in inputs.core_results]
    gate = summarize_core(core)
    if len(core) != 4 or gate.status != "pass" or gate.branch.phase != "A":
        raise InputError("A requires the four-cell raw numeric A decision")
    if any(r.shared_id != original.identity() for r in core if r.task == original.task):
        raise InputError("A must restart the original core P1, not a core winner")
    if run.deadline.seconds != 9000:
        raise InputError("A scheduling cap must be exactly 150 minutes")
    if run.cfg.evaluation.perf_trials != 100 or run.cfg.evaluation.correctness_trials != 5:
        raise InputError("A requires full 100-sample/5-correctness blocks")
    root = run.output.resolve()
    store = RunStore.create(root.parent, root.name, {"phase": "A", "slot": inputs.slot,
        "source_fidelity": "failed", "execution_kind": inputs.execution_kind, "opportunities": 2,
        "inputs": inputs.model_dump(mode="json"), "started_unix_s": run.started_unix_s})
    arm = A_ARMS[inputs.slot]
    cfg = run.cfg.model_copy(deep=True)
    cfg.budgets.space_expansions_per_candidate = 0
    cfg.v4.conditional_scan.mode = "off"
    cfg.v3.slope_guide.enabled = False
    with admission(run.deadline):
        loop = run_closed_loop(original, LoopRun(cfg, arm, root / "trajectory", (1, 2),
            run.deadline, inputs.helpers, inputs.reference.parent, True))
    current = Shared.model_validate_json((root / "trajectory" / loop.terminal_shared).read_text(encoding="utf-8"))
    ready = [r for r in loop.rounds if r.elapsed_ready_s is not None and r.artifact]
    selected = root / "trajectory" / (ready[-1].artifact if ready and ready[-1].artifact else loop.initial_parent_artifact)
    exported = root / "export/selected.py"
    exported.parent.mkdir()
    shutil.copyfile(selected, exported)
    helper_root = exported.parent / "imports"
    helpers = copy_helpers(loop.terminal_helpers, loop.terminal_helper_root, helper_root)
    status = "censored" if any(r.status == "censored" for r in loop.rounds) else "failed" if any(
        r.status == "failed" for r in loop.rounds) else "valid"
    result = AResult(slot=inputs.slot, arm=arm, task=original.task, original_shared_id=original.identity(),
        raw_gate=gate, status=status, started_unix_s=run.started_unix_s, deadline_unix_s=run.started_unix_s + 9000,
        drain_s=max(0, run.deadline.clock() - run.deadline.started - 9000), loop=loop, current=current,
        exported_source=exported, exported_source_sha256=hashlib.sha256(exported.read_bytes()).hexdigest(),
        exported_helper_root=helper_root, exported_helpers=helpers,
        exported_helper_sha256={p.relative_to(helper_root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in helpers})
    (root / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    store.append("RUN_FINISHED", {"status": status, "source_fidelity": "failed", "drain_s": result.drain_s})
    return result
