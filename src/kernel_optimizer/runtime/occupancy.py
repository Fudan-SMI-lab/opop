"""Observe unrelated CUDA clients so concurrent ad hoc work cannot earn a score."""
import os
from pathlib import Path
import subprocess
import threading
import time


def belongs_to_job(pid, spec):
    """Include compiler/helper children owned by the same measured worker."""
    seen = set()
    while pid > 1 and pid not in seen:
        seen.add(pid)
        try:
            root = Path(f"/proc/{pid}")
            argv = (root / "cmdline").read_bytes().split(b"\0")
            if b"kernel_optimizer.runtime.worker" in argv and str(spec).encode() in argv:
                return True
            # comm may contain spaces/parentheses, so split after its closing ).
            pid = int((root / "stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            return False
    return False


def clients(devices, spec):
    query = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
    if query.returncode:
        return {"status": "unavailable", "external": []}
    external = []
    for line in query.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2 or parts[1] not in devices or not parts[0].isdigit():
            continue
        pid = int(parts[0])
        if pid == os.getpid():
            continue
        if not belongs_to_job(pid, spec):
            try:
                executable = Path(f"/proc/{pid}/exe").resolve(strict=True).name
                cwd = str(Path(f"/proc/{pid}/cwd").resolve(strict=True))
            except FileNotFoundError:
                continue
            except OSError:
                executable, cwd = "unknown", "unknown"
            external.append({"pid": pid, "uuid": parts[1], "executable": executable, "cwd": cwd})
    return {"status": "observed", "external": external}


class OccupancyMonitor:
    def __init__(self, devices, spec):
        self.devices, self.spec = devices, Path(spec).resolve()
        self.stop_event = threading.Event()
        self.samples = 0
        self.unavailable = 0
        self.interference = []
        self.wait_s = 0

    def _sample(self):
        try:
            value = clients(self.devices, self.spec)
        except (OSError, subprocess.SubprocessError):
            value = {"status": "unavailable", "external": []}
        self.samples += 1
        self.unavailable += value["status"] != "observed"
        return value["external"]

    def start(self):
        started = time.monotonic()
        # The coordinator's ordinary worker timeout still governs a busy device.
        # This never cancels a provider request or kills another user's process.
        while self._sample():
            time.sleep(1)
        self.wait_s = time.monotonic() - started
        def monitor():
            while not self.stop_event.wait(0.5):
                self.interference.extend(self._sample())
        self.thread = threading.Thread(target=monitor, daemon=True)
        self.thread.start()

    def finish(self):
        self.stop_event.set()
        self.thread.join()
        self.interference.extend(self._sample())
        return {"status": "interference_detected" if self.interference else "sampled_clear" if not self.unavailable else "partly_unavailable",
                "samples": self.samples, "unavailable_samples": self.unavailable, "wait_s": self.wait_s,
                "external_clients": self.interference,
                "scope": "CUDA process snapshots every 0.5s outside the workload; sub-sample interference can be missed"}
