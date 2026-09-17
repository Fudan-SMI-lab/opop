# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["kernel-optimizer"]
# ///
# Existing environment: python -m scripts.experiments.c2_method_a_terminal --help
"""Terminal-only exploratory A finals on A0/B0; no new model or tuning opportunity."""

import argparse
import hashlib
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, time
from typing import Literal

from kernel_optimizer.config import AppConfig, load_config
from kernel_optimizer.gpu.worker_client import to_wsl_path
from kernel_optimizer.models.core import TrialRecord
from kernel_optimizer.paramspace.materializer import extract_defaults
from kernel_optimizer.store.run_store import RunStore
from scripts.experiments.c2_local_adapter import GpuAdapter
from scripts.experiments.c2_local_inputs import InputError, Strict
from scripts.experiments.c2_local_runner import isolated_config, worker_environment
from scripts.experiments.c2_method_a import AInputs, AResult
from scripts.experiments.c2_method_a_compare import Normalized, ReadyComparison, compare_ready, normalize
from scripts.experiments.c2_method_admission import admission
from scripts.experiments.c2_method_files import copy_helpers
from scripts.experiments.c2_method_gates import Comparison, block_values, compare_blocks
from scripts.experiments.c2_method_protocol import Deadline, SchedulingExpired


class TerminalArm(Strict):
    result: Path
    source: Path
    helper_root: Path
    helpers: tuple[Path, ...] = ()

    def load(self) -> AResult:
        if not all(p.is_absolute() for p in (self.result, self.source, self.helper_root, *self.helpers)):
            raise InputError("terminal input paths must be absolute")
        result = AResult.model_validate_json(self.result.read_text(encoding="utf-8"))
        if result.deadline_unix_s != result.started_unix_s + 9000:
            raise InputError("terminal cannot extend the original 150-minute deadline")
        return result

    def verify_export(self, result: AResult) -> None:
        if hashlib.sha256(self.source.read_bytes()).hexdigest() != result.exported_source_sha256:
            raise InputError("terminal source differs from the native selected export")
        if extract_defaults(self.source.read_text(encoding="utf-8")) != result.current.parent.params.values:
            raise InputError("terminal source parameters differ from the native incumbent")
        helpers = {p.resolve().relative_to(self.helper_root.resolve()).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in self.helpers}
        if helpers != result.exported_helper_sha256:
            raise InputError("terminal helpers differ from the trajectory export")


class TerminalInputs(Strict):
    slot: Literal["A0", "B0"]
    original: Path
    g0: TerminalArm
    h: TerminalArm


@dataclass(frozen=True, slots=True)
class TerminalRun:
    cfg: AppConfig
    output: Path
    now: Callable[[], float] = time
    clock: Callable[[], float] = monotonic


class TerminalResult(Strict):
    slot: Literal["A0", "B0"]
    task: str
    status: Literal["valid", "failed", "censored"]
    source_fidelity: Literal["failed"] = "failed"
    execution_kind: Literal["bounded_exploratory_confirmation"] = "bounded_exploratory_confirmation"
    g0_finals: list[TrialRecord]
    h_finals: list[TrialRecord]
    baseline: list[float]
    normalized_g0: Normalized | None
    normalized_h: Normalized | None
    comparison: Comparison
    ready: ReadyComparison
    started_unix_s: float
    deadline_unix_s: float
    transfer_and_wait_s: float
    terminal_elapsed_s: float
    program_elapsed_s: float
    drain_s: float
    error: str | None


def run_terminal(inputs: TerminalInputs, run: TerminalRun) -> TerminalResult:
    started, started_clock = run.now(), run.clock()
    if not inputs.original.is_absolute():
        raise InputError("terminal original input path must be absolute")
    g0, h = inputs.g0.load(), inputs.h.load()
    original_inputs = AInputs.model_validate_json(inputs.original.read_text(encoding="utf-8"))
    original = original_inputs.load_parent()
    expected_h = "B1" if inputs.slot == "A0" else "A1"
    if (g0.slot != inputs.slot or h.slot != expected_h or original_inputs.slot != inputs.slot
            or any(r.original_shared_id != original.identity() or r.task != original.task for r in (g0, h))):
        raise InputError("terminal pairing must use original P1 and fixed A0/B0 comparison GPU")
    if (original.evaluation != run.cfg.evaluation.model_dump(mode="json") or original.device != run.cfg.device
            or run.cfg.evaluation.perf_trials != 100 or run.cfg.evaluation.correctness_trials != 5):
        raise InputError("terminal config differs from the original full 100/5 contract")
    ready = compare_ready(g0, h)
    end = min(g0.deadline_unix_s, h.deadline_unix_s)
    deadline = Deadline(run.clock(), run.clock, max(0, end - run.now()))
    baseline = block_values(g0.loop.initial_parent_finals)
    root = run.output.resolve()
    store = RunStore.create(root.parent, root.name, {"inputs": inputs.model_dump(mode="json"),
        "source_fidelity": "failed", "deadline_unix_s": end, "studies": 0})
    finals: dict[str, list[TrialRecord]] = {"G0": [], "H": []}
    status, error = "censored", None
    try:
        deadline.check()
        inputs.g0.verify_export(g0)
        inputs.h.verify_export(h)
        deadline.check()
        if not baseline or any(len(r.loop.rounds) != 2 or any(s.elapsed_ready_s is None for s in r.loop.rounds) for r in (g0, h)):
            raise InputError("incomplete trajectories cannot supply complete terminal evidence")
        local = isolated_config(run.cfg, root)
        local.run.seed = 0
        base_pythonpath = local.wsl.extra_pythonpath
        with admission(deadline), worker_environment(local):
            adapter = GpuAdapter(original, local, store)
            adapter.full = True
            for arm, spec in (("G0", inputs.g0), ("H", inputs.h)):
                directory = root / arm
                directory.mkdir()
                shutil.copyfile(spec.source, directory / "selected.py")
                copy_helpers(spec.helpers, spec.helper_root, directory / "imports")
            for block, order in enumerate((("G0", "H"), ("H", "G0"), ("G0", "H"))):
                for arm in order:
                    deadline.check()
                    result = g0 if arm == "G0" else h
                    adapter.backend, adapter.phase = result.current.backend, f"terminal_{arm}_{block}"
                    adapter.correctness.worker.cfg.extra_pythonpath = f"{to_wsl_path(root / arm / 'imports')}:{base_pythonpath}"
                    finals[arm].append(adapter.measure(root / arm / "selected.py", result.current.parent.params))
        status = "censored" if deadline.remaining() <= 0 else "valid" if all(block_values(v) for v in finals.values()) else "failed"
    except SchedulingExpired as exc:
        error = str(exc)
    except (OSError, ValueError, RuntimeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        status = "censored" if any(len(v) != 3 for v in finals.values()) else "failed"
    outcome = TerminalResult(slot=inputs.slot, task=original.task, status=status,
        g0_finals=finals["G0"], h_finals=finals["H"], baseline=baseline,
        normalized_g0=normalize(finals["G0"], baseline), normalized_h=normalize(finals["H"], baseline),
        comparison=compare_blocks(block_values(finals["G0"]), block_values(finals["H"])), ready=ready,
        started_unix_s=started, deadline_unix_s=end,
        transfer_and_wait_s=max(0, started - max(r.started_unix_s + r.loop.elapsed_total_s for r in (g0, h))),
        terminal_elapsed_s=run.clock() - started_clock, program_elapsed_s=run.now() - min(g0.started_unix_s, h.started_unix_s),
        drain_s=max(0, run.now() - end), error=error)
    (root / "result.json").write_text(outcome.model_dump_json(indent=2), encoding="utf-8")
    store.append("RUN_FINISHED", {"status": status, "error": error, "terminal_elapsed_s": outcome.terminal_elapsed_s})
    return outcome


class Options(Strict):
    config: Path
    inputs: Path
    output: Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "inputs", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = Options.model_validate(vars(parser.parse_args(argv)))
    try:
        inputs = TerminalInputs.model_validate_json(args.inputs.read_text(encoding="utf-8"))
        result = run_terminal(inputs, TerminalRun(load_config(args.config), args.output))
        return 0 if result.status == "valid" else 1
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
