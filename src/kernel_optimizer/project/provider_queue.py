"""Shared adaptive admission after an observed provider overload.

No request/token/cost/elapsed-time quota. An actual 429/temporary server error
activates cooldown and additive-increase/multiplicative-decrease admission.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import time

from kernel_optimizer.runtime.execution import DeviceLease, write_json


def retryable_status(error):
    match = re.search(r'"statusCode"\s*:\s*(\d+)', str(error))
    status = int(match.group(1)) if match else None
    return status if status in {408, 429, 500, 502, 503, 504} else None


class ProviderQueue:
    def __init__(self, config):
        provider = config.agents.default_model.split("/", 1)[0]
        self.root = Path(config.opencode.launch_cwd) / ".opop-provider-coordination"
        self.key = hashlib.sha256(provider.encode()).hexdigest()[:24]
        self.state_path = self.root / (self.key + ".json")

    def _read(self):
        return json.loads(self.state_path.read_text()) if self.state_path.exists() else None

    @contextmanager
    def admit(self):
        if self._read() is None:
            yield
            return
        permit = None
        while permit is None:
            state = self._read()
            remaining = state.get("not_before", 0) - time.time()
            if remaining > 0:
                time.sleep(min(remaining, 1))
                continue
            for slot in range(state["capacity"]):
                lease = DeviceLease(self.root, self.key + f"-slot-{slot}")
                try:
                    lease.__enter__()
                    permit = lease
                    break
                except RuntimeError:
                    pass
            if permit is None:
                time.sleep(0.2)
        try:
            yield
        finally:
            permit.__exit__(None, None, None)

    def overloaded(self, status):
        with DeviceLease(self.root, self.key + "-state", wait=True):
            state = self._read() or {"capacity": 1, "healthy": 0, "failures": 0}
            state["capacity"] = max(1, state["capacity"] // 2)
            state["healthy"] = 0
            state["failures"] += 1
            delay = min(60, 5 * 2 ** min(state["failures"]-1, 4))
            state.update(not_before=max(state.get("not_before", 0), time.time()+delay), last_status=status)
            write_json(self.state_path, state)
            return {"status": status, "cooldown_s": delay, "temporary_parallel_capacity": state["capacity"],
                    "policy": "adaptive provider scheduling after actual overload; usage remains unbounded"}

    def succeeded(self):
        if self._read() is None:
            return
        with DeviceLease(self.root, self.key + "-state", wait=True):
            state = self._read()
            state["failures"] = 0
            state["healthy"] += 1
            if state["healthy"] >= 2:
                state["capacity"] += 1
                state["healthy"] = 0
            write_json(self.state_path, state)
