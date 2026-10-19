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


class DeviceLease:
    """Cooperative, cross-process exclusive lease keyed by physical UUID."""

    def __init__(self, directory: Path, device: str):
        self.path = directory / (hashlib.sha256(device.encode()).hexdigest() + ".lock")
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                if self.path.stat().st_size == 0:
                    self.stream.write(b"0"); self.stream.flush()
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
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
        request = request.model_copy(update={
            "entry": request.entry.resolve(), "cwd": request.cwd.resolve(),
            "output": request.output.resolve(), "lease_dir": request.lease_dir.resolve()})
        if not request.entry.is_file():
            raise FileNotFoundError(request.entry)
        if len(set(request.devices)) != len(request.devices):
            raise ValueError("device UUIDs must be unique")
        available = {row["uuid"] for row in gpu_inventory()} if request.devices else set()
        if not set(request.devices) <= available:
            raise ValueError("requested device UUID is unavailable")
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
                stack.enter_context(DeviceLease(request.lease_dir, device))
            try:
                for rank in range(world):
                    env = {**os.environ, **request.env, "CUDA_VISIBLE_DEVICES": ",".join(request.devices),
                           "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "RANK": str(rank),
                           "LOCAL_RANK": str(rank), "WORLD_SIZE": str(world),
                           "PYTHONUNBUFFERED": "1"}
                    log = stack.enter_context((request.output / f"rank-{rank}.log").open("wb"))
                    processes.append(subprocess.Popen(
                        [request.python, "-B", "-m", "kernel_optimizer.runtime.worker", str(spec), str(rank)],
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
                           "entry_sha256": hashlib.sha256(request.entry.read_bytes()).hexdigest(),
                           "devices": request.devices, "worker_pids_reaped": all(p.poll() is not None for p in processes)}
                write_json(request.output / "receipt.json", receipt)
        return receipt
