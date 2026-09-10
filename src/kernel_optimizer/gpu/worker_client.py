"""GPU worker client: worker dispatch, rw-lock, timeout kill.

ONE FILE FOR BOTH TOPOLOGIES. The orchestrator either reaches the GPU worker through WSL (a
Windows host, where the worker runs in a Linux distro and paths must be translated) or runs on the
same OS as the worker (a native-Linux box, where nothing needs translating). Everything else is
shared, because most of what looked platform-specific was not:

  PLATFORM-SPECIFIC (genuinely): path translation, whether a `wsl.exe ... bash -lc` wrapper is
  needed, and how a process tree is killed.

  NOT PLATFORM-SPECIFIC (these were bugs on BOTH, fixed here for both):
    * `pkill -f worker_main.py` killed EVERY worker on the box, so with max_shared_jobs > 1 a
      timeout in one shared-lane job also killed the other, healthy job -- which then reported
      `worker_crash` ("no result file"). A timeout must kill only its own job.
    * `~` in `venv` / `triton_cache_dir` / `kernelbench_src` was never expanded. The config
      defaults use `~`, and a literal `./~/...` path fails every job as `worker_crash`. The old
      string command happened to hide this on Windows only because `bash -lc` expanded it; nothing
      else does, and the expansion belongs here.
    * A string command breaks on a run directory containing a space. An argv list does not.
"""

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


def _wsl_hop_needed() -> bool:
    """Does reaching the GPU worker mean crossing into WSL?

    Decided by the ORCHESTRATOR's own OS, not by config, and not by looking for a WSL install:
    on Windows the worker lives in a distro and every path it receives must be translated; on
    Linux the orchestrator and the worker share one filesystem and one interpreter. A native-Linux
    box that also happened to have `wsl.exe` on PATH must still take the native route, which is
    why this asks about the platform rather than probing for the tool.
    """
    return os.name == "nt"


def to_wsl_path(p: Path | str) -> str:
    r"""Host path -> the path the GPU worker will see.

    On Windows: D:\x\y -> /mnt/d/x/y. On Linux: identity, because the orchestrator and the worker
    share a filesystem. Kept as one function (rather than branching at the three call sites and in
    the job-dict rewrite) so there is a single place where this decision is made.
    """
    resolved = Path(p).resolve()
    if not _wsl_hop_needed():
        return str(resolved)
    drive = resolved.drive.rstrip(":").lower()
    rest = resolved.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{rest}"


def _shq(s: str) -> str:
    """Quote one argv element for the `bash -lc` string on the WSL route.

    `shlex.quote` is the right tool but it is POSIX-quoting a string that is assembled on Windows,
    so it is spelled out here to make clear it quotes for the DISTRO's shell, not for cmd.exe. Only
    the WSL branch needs this: the native branch never builds a shell string, which is why a run
    directory containing a space broke the old code and cannot break the new native path.
    """
    if s and all(c.isalnum() or c in "@%_-+=:,./" for c in s):
        return s
    return "'" + s.replace("'", "'\"'\"'") + "'"


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

        The env dict and the `~` expansion apply on BOTH platforms. They used to be implicit in
        `bash -lc`: the shell expanded `~`, applied `VAR=x cmd` prefixes and did the PATH lookup.
        Building an argv list removes that shell, so those three jobs move here explicitly -- and
        on the WSL route the wrapper is added back at the end, around an already-correct command.
        """
        venv = os.path.expanduser(self.cfg.venv)
        py = f"{venv}/bin/python"
        cache = os.path.expanduser(self.cfg.triton_cache_dir)
        pythonpath = os.path.expanduser(self.cfg.kernelbench_src)
        extra = getattr(self.cfg, "extra_pythonpath", "")
        if extra:
            pythonpath = f"{pythonpath}:{os.path.expanduser(extra)}"
        # G33: the harness's own `src` as well. The worker is documented as stdlib+torch+triton, and
        # that is still the rule for what it may DEPEND on -- but two of its measurements import a
        # harness module: `gpu.tritonmm` (the four Triton-reachable ceilings, G10) and
        # `evaluation.statics` (the SASS counters). When `kernel_optimizer` is not importable those
        # imports raise inside a broad `except`, and the measurement degrades to 0.0.
        #
        # Which is exactly what happened, undetected, on BOTH boxes: box 3's freshly measured
        # calibration had all four *_triton_tflops = 0.0 with `triton_ceiling_error:
        # ModuleNotFoundError: No module named 'kernel_optimizer'` in the raw worker output, and box
        # 1's cached calibration has the same four zeros. So G10 -- whose whole finding is that
        # cuBLAS is the wrong roof, over-reporting headroom by 19% at fp32 and exceeding 100% at
        # fp16 -- has never once been in effect in a real run. It only ever worked in the standalone
        # probe, which runs in the harness's own interpreter.
        #
        # Derived from THIS file's location rather than configured, so it cannot drift from the code
        # being run and needs no per-box setting. Appended last: kernelbench and any pip --target dir
        # keep precedence, so this cannot shadow them.
        harness_src = Path(__file__).resolve().parents[2]
        pythonpath = f"{pythonpath}:{to_wsl_path(harness_src)}"
        worker_argv = [
            py,
            to_wsl_path(self.worker_main_path),
            "--job", to_wsl_path(job_path),
            "--out", to_wsl_path(out_path),
        ]
        if not _wsl_hop_needed():
            # Native: exec the interpreter directly, and pass the two variables in the child's
            # environment rather than as a shell prefix.
            env = {**os.environ, "TRITON_CACHE_DIR": cache, "PYTHONPATH": pythonpath}
            return worker_argv, env
        # WSL: the variables must cross into the distro, and they cannot do that through the
        # Windows process environment, so they stay as a `VAR=x` prefix inside the shell command.
        # Only this branch needs quoting, and only because a shell is unavoidable here.
        inner = " ".join(
            [f"TRITON_CACHE_DIR={cache}", f"PYTHONPATH={pythonpath}", *map(_shq, worker_argv)]
        )
        return ["wsl.exe", "-d", self.cfg.distro, "bash", "-lc", inner], dict(os.environ)

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
            # Own process group / job tree, so a timeout kills only THIS job. `start_new_session`
            # is POSIX-only; on Windows the equivalent is handled by `taskkill /T` at kill time.
            popen_kw: dict[str, Any] = {}
            if not _wsl_hop_needed():
                popen_kw["start_new_session"] = True
            proc = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                **popen_kw,
            )
            try:
                _stdout, stderr = proc.communicate(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                self._kill_job(proc)
                # Reap it, so the descriptors close and the next job does not inherit them.
                try:
                    proc.communicate(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
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

    def _kill_job(self, proc: subprocess.Popen) -> None:
        """Kill ONLY this job's process tree.

        Replaces `pkill -f worker_main.py`, which matched every worker on the box: with
        `max_shared_jobs: 2`, a timeout in one shared-lane job also killed the other, healthy job,
        which then reported `worker_crash` -- a real failure attributed to the wrong candidate. That
        was wrong on both platforms, so both are fixed here.
        """
        if _wsl_hop_needed():
            # /T takes the child tree with it. The tree here is wsl.exe -> the distro's shell ->
            # python, so killing only the direct child would leave the worker running.
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True, timeout=30,
                )
                return
            except (subprocess.TimeoutExpired, OSError):
                pass  # fall through to the direct kill below
        else:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                return
            except (ProcessLookupError, PermissionError, OSError):
                pass
        try:
            proc.kill()
        except OSError:
            pass
