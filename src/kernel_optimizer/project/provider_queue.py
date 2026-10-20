"""Shared adaptive admission after an observed provider overload.

No request/token/cost/elapsed-time quota. Honor declared provider availability
windows; other actual overloads activate adaptive concurrency and cooldown.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
from math import isfinite
from pathlib import Path
import re
import time

from kernel_optimizer.runtime.execution import DeviceLease, write_json


def retryable_status(error):
    match = re.search(r'"statusCode"\s*:\s*(\d+)', str(error))
    status = int(match.group(1)) if match else None
    return status if status in {408, 429, 500, 502, 503, 504} else None


def retry_deadline(failure, *, now=None, local_timezone=None):
    """Honor observed provider recovery advice; this is not a framework usage quota."""
    if not failure:
        return None
    now = time.time() if now is None else now
    header = failure.get("retry_after")
    if header is not None:
        try:
            deadline = now + max(0, float(header))
        except (ValueError, TypeError):
            try:
                parsed = parsedate_to_datetime(str(header))
                deadline = parsed.replace(tzinfo=timezone.utc).timestamp() if parsed.tzinfo is None else parsed.timestamp()
            except (ValueError, TypeError, OverflowError):
                deadline = now
        if isfinite(deadline) and deadline > now:
            return {"not_before": deadline, "basis": "provider Retry-After header"}
    # Some providers report a recovery time only in the error message. Preserve
    # the interpretation instead of treating every 429 as transient concurrency.
    stamp = r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})?"
    message = str(failure.get("message") or "")
    patterns = [rf"(?:after|retry at|available at)\s+({stamp})", rf"({stamp})\s*后(?:可|重试)"]
    match = next((m for pattern in patterns if (m := re.search(pattern, message, flags=re.I))), None)
    if not match:
        return None
    try:
        parsed = datetime.fromisoformat(match.group(1).replace("Z", "+00:00"))
    except ValueError:
        return None
    assumed = parsed.tzinfo is None
    if assumed:
        parsed = parsed.replace(tzinfo=local_timezone or datetime.now().astimezone().tzinfo)
    if parsed.timestamp() <= now:
        return None
    return {"not_before": parsed.timestamp(), "basis": "provider message recovery timestamp",
            "recovery_time": parsed.isoformat(), "timezone_assumption": "host timezone for unqualified provider timestamp" if assumed else None}


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
            if state.get("admission") == "provider_availability":
                # A declared account/service window does not establish a
                # concurrency limit. Resume unrestricted once it has elapsed.
                yield
                return
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

    def overloaded(self, status, failure=None):
        advice = retry_deadline(failure)
        with DeviceLease(self.root, self.key + "-state", wait=True):
            state = self._read() or {"capacity": 1, "healthy": 0, "failures": 0}
            if not advice:
                state["capacity"] = max(1, state["capacity"] // 2)
            state["admission"] = "provider_availability" if advice else "adaptive_overload"
            state["healthy"] = 0
            state["failures"] += 1
            delay = min(60, 5 * 2 ** min(state["failures"]-1, 4))
            state.update(not_before=max(state.get("not_before", 0), time.time()+delay), last_status=status)
            if advice:
                state["not_before"] = max(state["not_before"], advice["not_before"])
                state["external_wait"] = advice
            write_json(self.state_path, state)
            return {"status": status, "cooldown_s": max(0, state["not_before"]-time.time()),
                    "temporary_parallel_capacity": None if advice else state["capacity"],
                    "provider_recovery": advice,
                    "policy": "adaptive provider scheduling after actual overload; usage remains unbounded"}

    def succeeded(self):
        if self._read() is None:
            return
        with DeviceLease(self.root, self.key + "-state", wait=True):
            state = self._read()
            if state.get("admission") == "provider_availability":
                return
            state["failures"] = 0
            state["healthy"] += 1
            if state["healthy"] >= 2:
                state["capacity"] += 1
                state["healthy"] = 0
            write_json(self.state_path, state)
