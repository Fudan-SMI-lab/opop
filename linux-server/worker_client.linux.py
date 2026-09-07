"""GPU worker client (native Linux): argv dispatch, rw-lock, process-group kill."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from kernel_optimizer.config import GpuConcurrencyConfig, WslConfig
from kernel_optimizer.gpu.jobs import failure_result


def to_wsl_path(p: Path | str) -> str:
    """Native-Linux port: the orchestrator and the GPU worker share one filesystem,
    so no path translation happens. Kept as a function (not deleted) because three
    call sites and the job-dict rewrite in run_job() route through it."""
    return str(Path(p).resolve())


class GpuRwLock:
    """In-process reader-writer lock + cross-process exclusive file lock.

    shared: correctness / compile / static-check jobs (bounded concurrency).
    exclusive: all timing jobs (whole GPU).
    Cross-process safety comes from a lock file with pid + heartbeat timestamp;
    within this orchestrator process, the rw semantics are enforced in-memory.
    """

    def __init__(self, lock_path: Path, max_shared: int, stale_after_s: float = 3600.0):
        self.lock_path = lock_path
        self.stale_after_s = stale_after_s
        self._cond = threading.Condition()
        self._readers = 0
        self._writer = False
        self._max_shared = max(1, max_shared)

    # -- cross-process file lock (best effort) --------------------------------

    def _acquire_file(self) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.stale_after_s
        while True:
            try:
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w") as f:
                    f.write(json.dumps({"pid": os.getpid(), "ts": time.time()}))
                return
            except FileExistsError:
                try:
                    holder = json.loads(self.lock_path.read_text())
                    if time.time() - float(holder.get("ts", 0)) > self.stale_after_s:
                        self.lock_path.unlink(missing_ok=True)  # stale takeover
                        continue
                except (OSError, ValueError):
                    self.lock_path.unlink(missing_ok=True)
                    continue
                if time.monotonic() > deadline:
                    raise TimeoutError(f"GPU lock held too long: {self.lock_path}")
                time.sleep(1.0)

    def _release_file(self) -> None:
        self.lock_path.unlink(missing_ok=True)

    # -- rw semantics ------------------------------------------------------------

    def acquire(self, mode: str) -> None:
        with self._cond:
            if mode == "shared":
                while self._writer or self._readers >= self._max_shared:
                    self._cond.wait()
                if self._readers == 0:
                    self._acquire_file()
                self._readers += 1
            else:  # exclusive
                while self._writer or self._readers > 0:
                    self._cond.wait()
                self._writer = True
                self._acquire_file()

    def release(self, mode: str) -> None:
        with self._cond:
            if mode == "shared":
                self._readers -= 1
                if self._readers == 0:
                    self._release_file()
            else:
                self._writer = False
                self._release_file()
            self._cond.notify_all()


class WslGpuWorker:
    """Runs one job per fresh WSL process; JSON file in / JSON file out."""

    def __init__(
        self,
        wsl_cfg: WslConfig,
        conc_cfg: GpuConcurrencyConfig,
        jobs_dir: Path,
        worker_main_path: Path | None = None,
    ):
        self.cfg = wsl_cfg
        self.conc = conc_cfg
        self.jobs_dir = jobs_dir
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        if worker_main_path is None:
            worker_main_path = Path(__file__).with_name("worker_main.py")
        self.worker_main_path = worker_main_path
        max_shared = conc_cfg.max_shared_jobs if conc_cfg.enabled else 1
        self.lock = GpuRwLock(jobs_dir / "gpu.lock", max_shared=max_shared)

    def _build_command(
        self, job_path: Path, out_path: Path
    ) -> tuple[list[str], dict[str, str]]:
        """(argv, env) for one worker job.

        Native Linux: there is no shell in the chain any more, so the three things
        `bash -lc` used to do must be done explicitly here --
          * `VAR=x cmd` env prefixing -> the returned env dict;
          * `~` expansion in venv / triton_cache_dir (the config defaults use `~`)
            -> os.path.expanduser. Python never does this itself, and a literal
            `./~/...` path would fail every job as worker_crash;
          * login-shell PATH lookup -> an absolute interpreter path.
        A list argv also sidesteps quoting: the old string command broke on a run
        directory containing a space.
        """
        venv = os.path.expanduser(self.cfg.venv)
        py = str(Path(venv) / "bin" / "python")
        cache = os.path.expanduser(self.cfg.triton_cache_dir)
        pythonpath = os.path.expanduser(self.cfg.kernelbench_src)
        extra = getattr(self.cfg, "extra_pythonpath", "")
        if extra:
            pythonpath = f"{pythonpath}:{os.path.expanduser(extra)}"
        argv = [
            py,
            str(Path(self.worker_main_path).resolve()),
            "--job", str(Path(job_path).resolve()),
            "--out", str(Path(out_path).resolve()),
        ]
        env = {
            **os.environ,
            "TRITON_CACHE_DIR": cache,
            "PYTHONPATH": pythonpath,
        }
        return argv, env

    def run_job(
        self,
        job: dict[str, Any],
        timeout_s: float,
        tag: str,
        lock_mode: str = "exclusive",
    ) -> dict[str, Any]:
        if not self.conc.enabled:
            lock_mode = "exclusive"
        job_id = f"{tag}-{uuid.uuid4().hex[:8]}"
        job_path = self.jobs_dir / f"{job_id}.json"
        out_path = self.jobs_dir / f"{job_id}.out.json"

        # Translate any host paths in the job to WSL paths.
        job = dict(job)
        for key in ("ref_src_path", "kernel_src_path", "build_dir"):
            if job.get(key):
                job[key] = to_wsl_path(job[key])
        job_path.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")

        argv, env = self._build_command(job_path, out_path)
        self.lock.acquire(lock_mode)
        try:
            if lock_mode == "exclusive" and self.conc.timing_cooldown_s > 0:
                time.sleep(self.conc.timing_cooldown_s)
            # start_new_session gives the job its own process group, so a timeout
            # kills only THIS job's tree. The old path ran `pkill -f worker_main.py`,
            # which killed every worker on the box: with max_shared_jobs: 2 a timeout
            # in one shared-lane job also killed the other, healthy job, which then
            # reported worker_crash ("no result file").
            proc = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                start_new_session=True,
            )
            try:
                _stdout, stderr = proc.communicate(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                self._kill_process_group(proc)
                proc.communicate()
                return failure_result("timeout", f"job {job_id} exceeded {timeout_s}s")
        finally:
            self.lock.release(lock_mode)

        if not out_path.exists():
            err = (stderr or b"").decode("utf-8", errors="replace")
            return failure_result(
                "worker_crash",
                f"no result file; rc={proc.returncode}; stderr tail: {err[-2000:]}",
            )
        try:
            return json.loads(out_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            return failure_result("worker_crash", f"unparseable result: {exc}")

    def _kill_process_group(self, proc: subprocess.Popen) -> None:
        """Kill only this job's process group -- see start_new_session at spawn."""
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except OSError:
                pass
