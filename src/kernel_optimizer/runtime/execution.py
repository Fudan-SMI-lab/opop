"""Explicit device leases and a process group per rank for project GPU work."""
from __future__ import annotations

import csv
from contextlib import ExitStack
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ExecutionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int = 1
    entry: Path
    function: str = "run"
    parameters: dict[str, Any] = Field(default_factory=dict)
    devices: list[str] = Field(default_factory=list)
    cwd: Path
    output: Path
    lease_dir: Path
    python: str = sys.executable
    distributed_backend: str | None = None
    worker_timeout_s: float | None = None
    collective_timeout_s: float = 120.0
    env: dict[str, str] = Field(default_factory=dict)
    launcher: list[str] = Field(default_factory=list)
    wait_for_devices: bool = False


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def gpu_inventory() -> list[dict[str, str]]:
    fields = ["index", "uuid", "name", "pci.bus_id"]
    result = subprocess.run(["nvidia-smi", "--query-gpu=" + ",".join(fields),
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    return [dict(zip(fields, (cell.strip() for cell in row), strict=True))
            for row in csv.reader(io.StringIO(result.stdout))]


def resolve_devices(selectors: list[str], *, inventory=None, count: int | None = None) -> list[str]:
    """Resolve current-host indices/UUIDs; UUIDs identify jobs, not experiment eligibility.

    ``auto`` selects the first ``count`` devices (all devices when count is absent).
    An empty list or ``cpu`` selects CPU execution. Indices are nvidia-smi indices.
    """
    if not selectors or selectors == ["cpu"] or (selectors == ["auto"] and count == 0):
        return []
    rows = gpu_inventory() if inventory is None else inventory
    if selectors == ["auto"]:
        count = len(rows) if count is None else count
        if count > len(rows):
            raise ValueError(f"task needs {count} GPUs; this host exposes {len(rows)}")
        return [row["uuid"] for row in rows[:count]]
    lookup = {key: row["uuid"] for row in rows
              for key in (row["uuid"], row["index"], "cuda:" + row["index"])}
    unknown = [device for device in selectors if device not in lookup]
    if unknown:
        raise ValueError(f"devices absent from this host: {unknown}; select current indices or auto. "
                         "For a saved project on new hardware, use --reuse-evaluator with --devices auto "
                         "and a new output to refresh its baseline.")
    devices = [lookup[device] for device in selectors]
    if len(set(devices)) != len(devices):
        raise ValueError("device selectors must refer to distinct physical GPUs")
    return devices


class DeviceLease:
    """Cooperative, cross-process exclusive lease keyed by physical UUID."""

    def __init__(self, directory: Path, device: str, *, wait=False):
        self.path = directory / (hashlib.sha256(device.encode()).hexdigest() + ".lock")
        self.stream = None
        self.wait = wait

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                if self.path.stat().st_size == 0:
                    self.stream.write(b"0"); self.stream.flush()
                self.stream.seek(0)
                while True:
                    try:
                        msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError:
                        if not self.wait:
                            raise
                        time.sleep(0.1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | (0 if self.wait else fcntl.LOCK_NB))
        except OSError:
            self.stream.close()
            raise RuntimeError(f"device lease is busy: {self.path.name}") from None
        return self

    def __exit__(self, *exc):
        if self.stream is not None:
            self.stream.close()


def _terminate(process: subprocess.Popen) -> None:
    if os.name != "nt":
        # Even an exited rank can leave children in its process group.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


class LocalExecutor:
    def run(self, request: ExecutionRequest) -> dict:
        selectors = list(request.devices)
        request = request.model_copy(update={
            "entry": request.entry.resolve(), "cwd": request.cwd.resolve(),
            "output": request.output.resolve(), "lease_dir": request.lease_dir.resolve(),
            "devices": resolve_devices(selectors)})
        if not request.entry.is_file():
            raise FileNotFoundError(request.entry)
        if request.distributed_backend and len(request.devices) < 2:
            raise ValueError("distributed execution requires at least two devices")
        request.output.mkdir(parents=True, exist_ok=False)
        spec = request.output / "request.json"
        write_json(spec, request.model_dump(mode="json"))
        world = len(request.devices) if request.distributed_backend else 1
        processes: list[subprocess.Popen] = []
        started = time.monotonic()
        failure = None
        ranks = []
        with ExitStack() as stack:
            for device in sorted(request.devices):
                stack.enter_context(DeviceLease(request.lease_dir, device, wait=request.wait_for_devices))
            lease_wait_s = time.monotonic()-started
            started = time.monotonic()
            try:
                for rank in range(world):
                    env = {**os.environ, **request.env, "CUDA_VISIBLE_DEVICES": ",".join(request.devices),
                           "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "RANK": str(rank),
                           "LOCAL_RANK": str(rank), "WORLD_SIZE": str(world),
                           "PYTHONUNBUFFERED": "1"}
                    # Workers change cwd to the immutable project snapshot. Preserve
                    # the controller's imports even when launched with PYTHONPATH=src:.
                    import_roots = [str(Path(__file__).resolve().parents[2])]
                    import_roots.extend(str(Path(p or ".").resolve()) for p in env.get("PYTHONPATH", "").split(os.pathsep))
                    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(import_roots))
                    log = stack.enter_context((request.output / f"rank-{rank}.log").open("wb"))
                    prefix = [part.replace("{rank}", str(rank)).replace("{output}", str(request.output)) for part in request.launcher]
                    processes.append(subprocess.Popen(
                        [*prefix, request.python, "-B", "-m", "kernel_optimizer.runtime.worker", str(spec), str(rank)],
                        cwd=request.cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=(os.name != "nt")))
                while True:
                    codes = [p.poll() for p in processes]
                    if any(code is not None and code != 0 for code in codes):
                        failure = "rank_failed"
                        break
                    if all(code is not None for code in codes):
                        break
                    if request.worker_timeout_s is not None and time.monotonic() - started > request.worker_timeout_s:
                        failure = "worker_timeout"
                        break
                    time.sleep(0.05)
            except BaseException as exc:
                failure = type(exc).__name__
                raise
            finally:
                for process in processes:
                    _terminate(process)
                for rank, process in enumerate(processes):
                    result_path = request.output / f"rank-{rank}.json"
                    try:
                        result = json.loads(result_path.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        result = {"status": "missing_result"}
                    ranks.append({"rank": rank, "pid": process.pid, "exit_code": process.returncode, **result})
                receipt = {"schema_version": 1, "status": "failed" if failure or len(ranks) != world or
                           any(r.get("status") != "passed" for r in ranks) else "passed",
                           "failure": failure, "wall_s": time.monotonic() - started, "ranks": ranks,
                           "lease_wait_s": lease_wait_s,
                           "device_selectors": selectors,
                           "entry_sha256": hashlib.sha256(request.entry.read_bytes()).hexdigest(),
                           "devices": request.devices, "worker_pids_reaped": all(p.poll() is not None for p in processes)}
                write_json(request.output / "receipt.json", receipt)
        return receipt
