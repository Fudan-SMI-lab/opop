# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["kernel-optimizer"]
# ///
# Existing environment: python -m scripts.experiments.c2_helper_pilot --help
"""Two sequential local pairs; subprocess exit notifications dispatch GPU0 heldout immediately."""

import argparse
import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path
from time import time
from typing import Literal, assert_never

from kernel_optimizer.config import load_config
from scripts.experiments.c2_helper_pilot_records import (
    DispatchRecord, HostPilot, HostReceipt, PairDispatch, ProcessExit, WorkerJob,
    terminal_heldout, terminal_opportunity,
)
from scripts.experiments.c2_local_inputs import InputError, Strict
from scripts.experiments.c2_opportunity_heldout import HeldoutInputs, run_heldout
from scripts.experiments.c2_opportunity_inputs import CampaignRun, OpportunityInputs, Slot, Wave
from scripts.experiments.c2_opportunity_program import run_opportunity


def job_environment(job: WorkerJob) -> Mapping[str, str]:
    env = {"CUDA_VISIBLE_DEVICES": str(job.gpu), "PYTHONDONTWRITEBYTECODE": "1"}
    for key, folder in (("XDG_STATE_HOME", "xdg-state"), ("XDG_CONFIG_HOME", "xdg-config"),
                        ("XDG_RUNTIME_DIR", "xdg-runtime"), ("TMP", "tmp"), ("TEMP", "tmp"),
                        ("CUDA_CACHE_PATH", "cuda")):
        path = job.inputs.parent / "runtime" / job.inputs.stem / folder
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    return env


def launch_job(job: WorkerJob) -> subprocess.Popen[bytes]:
    path = job.inputs.with_suffix(".job.json")
    path.write_text(job.model_dump_json(indent=2), encoding="utf-8")
    env = {**os.environ, **job_environment(job)}
    with job.inputs.with_suffix(".log").open("wb") as log:
        return subprocess.Popen([sys.executable, "-m", "scripts.experiments.c2_helper_pilot", "worker", "--inputs", str(path)],
                                stdout=log, stderr=subprocess.STDOUT, env=env)


def run_worker(job: WorkerJob) -> int:
    cfg = load_config(job.config)
    cfg.opencode.server_env = {**cfg.opencode.server_env, **job_environment(job)}
    run = CampaignRun(cfg, job.output, job.clock.final_deadline_unix_s, pilot_clock=job.clock)
    match job.mode:
        case "opportunity":
            inputs = OpportunityInputs.model_validate_json(job.inputs.read_text(encoding="utf-8"))
            result = run_opportunity(inputs, run)
        case "heldout":
            heldout = HeldoutInputs.model_validate_json(job.inputs.read_text(encoding="utf-8"))
            result = run_heldout(heldout, run)
        case unreachable:
            assert_never(unreachable)
    return 0 if result.status == "complete" else 1


def run_host(spec: HostPilot, launcher: Callable[[WorkerJob], subprocess.Popen[bytes]] = launch_job,
             *, now: Callable[[], float] = time) -> HostReceipt:
    if now() < spec.clock.campaign_started_unix_s:
        raise InputError("host launch precedes the explicit common pilot start")
    spec.output.mkdir(parents=True, exist_ok=False)
    requests = spec.output / "requests"
    requests.mkdir()
    records = spec.output / "records"
    records.mkdir()
    slots: tuple[Slot, Slot] = ("A0", "A1") if spec.host == "A" else ("B0", "B1")
    waves: tuple[Wave, Wave] = (1, 2)
    opportunities = [DispatchRecord(wave=w, slot=s) for w in waves for s in slots]
    pairs = [PairDispatch(wave=w, slot=slots[0]) for w in waves]

    def save() -> HostReceipt:
        receipt = HostReceipt(host=spec.host, clock=spec.clock, opportunities=opportunities, pairs=pairs)
        (spec.output / "controller.json").write_text(receipt.model_dump_json(indent=2), encoding="utf-8")
        return receipt

    def wait(process: subprocess.Popen[bytes]) -> ProcessExit:
        code = process.wait()
        return ProcessExit(exit_code=code, ended_unix_s=now())

    save()
    for pair_index, wave in enumerate(waves):
        jobs: list[WorkerJob] = []
        exits: dict[int, ProcessExit] = {}
        for gpu, slot in enumerate(slots):
            path = requests / f"wave{wave}-{slot}.json"
            path.write_text(spec.opportunity(wave, slot).model_dump_json(indent=2), encoding="utf-8")
            jobs.append(WorkerJob(mode="opportunity", config=spec.gpu0_config if gpu == 0 else spec.gpu1_config,
                inputs=path, output=spec.output / f"wave{wave}" / slot, gpu=0 if gpu == 0 else 1, clock=spec.clock))
        with ExitStack() as owned, ThreadPoolExecutor(max_workers=2) as waits:
            pending = {}
            for gpu, job in enumerate(jobs):
                launched = now()
                if launched >= spec.clock.work_cutoff_unix_s:
                    exits[gpu] = ProcessExit(ended_unix_s=launched, error="work_cutoff_before_launch")
                    continue
                try:
                    process = owned.enter_context(launcher(job))
                    pending[waits.submit(wait, process)] = (gpu, launched)
                except OSError as exc:
                    exits[gpu] = ProcessExit(ended_unix_s=now(), error=f"launch: {exc}")
            for finished in as_completed(pending):
                gpu, launched = pending[finished]
                exits[gpu] = finished.result().model_copy(update={"launched_unix_s": launched})
        ready = max(exited.ended_unix_s for exited in exits.values())
        arm_paths: dict[str, Path] = {}
        for gpu, job in enumerate(jobs):
            exited = exits[gpu]
            result = terminal_opportunity(job, exited)
            path = records / f"wave{wave}-{slots[gpu]}.json"
            path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            arm_paths[result.arm] = path
            opportunities[pair_index * 2 + gpu] = DispatchRecord(wave=wave, slot=slots[gpu], status=result.status,
                **exited.model_dump(exclude={"error"}), result=path, error=exited.error or result.error,
                drain_s=max(0, exited.ended_unix_s - spec.clock.work_cutoff_unix_s) if exited.launched_unix_s else 0,
                final_drain_s=max(0, exited.ended_unix_s - spec.clock.final_deadline_unix_s) if exited.launched_unix_s else 0)
        heldout = HeldoutInputs.model_validate({**spec.opportunity(wave, slots[0]).model_dump(),
                                               "g0": arm_paths["G0"], "c2": arm_paths["C2"]})
        path = requests / f"wave{wave}-heldout.json"
        path.write_text(heldout.model_dump_json(indent=2), encoding="utf-8")
        job = WorkerJob(mode="heldout", config=spec.gpu0_config, inputs=path, gpu=0, clock=spec.clock,
                        output=spec.output / "heldout" / f"wave{wave}")
        pairs[pair_index] = PairDispatch(wave=wave, slot=slots[0], pair_ready_unix_s=ready)
        save()
        launched = now()
        exited = ProcessExit(ended_unix_s=launched, error="final_cutoff_before_launch")
        if launched < spec.clock.final_deadline_unix_s:
            try:
                with launcher(job) as process:
                    exited = wait(process).model_copy(update={"launched_unix_s": launched})
            except OSError as exc:
                exited = ProcessExit(ended_unix_s=now(), error=f"heldout launch: {exc}")
        result, observed = terminal_heldout(job, exited)
        path = records / f"wave{wave}-heldout.json"
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        actual_start = result.started_unix_s if observed else None
        pairs[pair_index] = PairDispatch(wave=wave, slot=slots[0], status=result.status,
            **exited.model_dump(exclude={"error"}), result=path, error=exited.error or result.error,
            pair_ready_unix_s=ready, heldout_started_unix_s=actual_start,
            ready_to_heldout_start_s=max(0, actual_start - ready) if actual_start is not None else None,
            drain_s=max(0, exited.ended_unix_s - spec.clock.final_deadline_unix_s) if exited.launched_unix_s else 0,
            final_drain_s=max(0, exited.ended_unix_s - spec.clock.final_deadline_unix_s) if exited.launched_unix_s else 0)
        save()
    return save()


class Options(Strict):
    mode: Literal["host", "worker"]
    inputs: Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for name in ("host", "worker"):
        sub.add_parser(name).add_argument("--inputs", type=Path, required=True)
    try:
        options = Options.model_validate(vars(parser.parse_args(argv)))
        raw = options.inputs.read_text(encoding="utf-8")
        match options.mode:
            case "host":
                receipt = run_host(HostPilot.model_validate_json(raw))
                return 0 if all(row.status == "complete" for row in [*receipt.opportunities, *receipt.pairs]) else 1
            case "worker":
                return run_worker(WorkerJob.model_validate_json(raw))
            case unreachable:
                assert_never(unreachable)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
