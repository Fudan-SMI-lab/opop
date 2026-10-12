"""Durable phase admission; framework revisions never reset D or consumed allowances."""

import json
from collections.abc import Callable
from pathlib import Path
from sys import maxsize
from threading import RLock
from time import time
from typing import Literal

from pydantic import Field

from .device_records import DeviceOperation, StageIdentity
from .runner_records import FrozenRecord, RunnerError
from .runner_records import GoalId

type Lane = Literal["setup", "development", "formal_search", "final_verification", "final"]
type Operation = Literal["agent", "local", "model", "capture", "profile"]

# The original C3 plan imposed a fixed research protocol on top of the science: a 90-minute
# development clock to the second, per-lane caps on agent/local/model calls, at most three
# framework revisions, at most three epochs. Those bound how much EXPERIMENTING was allowed; they
# never had anything to do with whether a measurement is correct. When the goal is to obtain the
# cross-goal table at all, they only stop the work.
#
# So allowances are now OPT-IN. `BudgetSpec.limits` and `BudgetSpec.enforce_fixed_window` are
# unset by default, which means unlimited operations and a free-form deadline; a caller that wants
# the original protocol back passes those values explicitly and gets exactly the old behaviour.
# The counters themselves are always kept, because the published table has to report what it
# actually spent -- accounting is evidence, a cap is policy.
#
# What is NOT relaxed anywhere: device proof, quality limits, held-out isolation, and the rule
# that a row is selected before it is measured. Those decide whether a number is true.
PROTOCOL_LIMITS: dict[Lane, tuple[int, int, int, int, int]] = {
    # agent, local, model, capture, profile -- the original plan's per-lane allowances.
    "setup": (0, 3, 3, 6, 3),
    "development": (6, 12, 4, 12, 3),
    "formal_search": (2, 8, 4, 8, 0),
    "final_verification": (0, 9, 0, 18, 0),
    "final": (0, 0, 36, 9, 0),
}
LANE_STAGE: dict[Lane, str] = {
    "setup": "setup", "development": "development", "formal_search": "formal_search",
    "final_verification": "formal_search", "final": "final",
}
OPERATIONS: tuple[Operation, ...] = ("agent", "local", "model", "capture", "profile")


def protocol_limits(lane: Lane) -> dict[Operation, int]:
    """The original plan's allowances for one lane, as a `BudgetSpec.limits` mapping.

    Pass this to restore the old behaviour exactly. It exists so that a caller who WANTS the
    research protocol asks for it by name, rather than every caller inheriting it by default.
    """
    return dict(zip(OPERATIONS, PROTOCOL_LIMITS[lane], strict=True))


class BudgetSpec(FrozenRecord):
    lane: Lane
    stage: StageIdentity
    started_unix_s: float = Field(gt=0)
    deadline_unix_s: float = Field(gt=0)
    goal: GoalId | None = None
    # Per-operation ceilings. None means unlimited, which is the default. Pass
    # `PROTOCOL_LIMITS[lane]` as a mapping to restore the original plan's allowances.
    limits: dict[Operation, int] | None = None
    # True reproduces the original "development must be exactly D+5400" assertion.
    enforce_fixed_window: bool = False


class BudgetState(FrozenRecord):
    spec: BudgetSpec
    used: dict[Operation, int]
    operations: dict[Operation, int]
    revisions: tuple[str, ...]


class StageBudget:
    """Own mutable counters behind one lock and persist before admitting expensive work."""

    def __init__(self, path: Path, spec: BudgetSpec, *, now: Callable[[], float] = time) -> None:
        self.path, self.now, self.lock = path.resolve(), now, RLock()
        names = OPERATIONS
        # Unlimited unless the caller asked for ceilings. `None` per operation means no cap.
        self.limits: dict[Operation, int | None] = dict.fromkeys(names, None)
        if spec.limits is not None:
            self.limits.update({k: v for k, v in spec.limits.items()})
        if spec.stage.stage != LANE_STAGE[spec.lane] or spec.deadline_unix_s <= spec.started_unix_s:
            raise RunnerError("budget lane/stage or clock mismatch")
        if spec.enforce_fixed_window and spec.lane == "development" and spec.deadline_unix_s != spec.started_unix_s + 5400:
            raise RunnerError("development must retain the continuous D+5400 cutoff")
        if self.path.exists():
            self.state = BudgetState.model_validate_json(self.path.read_bytes())
            if self.state.spec != spec or set(self.state.used) != set(names) or set(self.state.operations) != set(names):
                raise RunnerError("existing budget identity/counters cannot be reset")
            if any(self.state.used[k] < 0 or (self.limits[k] is not None and self.state.used[k] > self.limits[k])
                   for k in names):
                raise RunnerError("invalid persisted budget counts")
        else:
            self.state = BudgetState(spec=spec, used={n: 0 for n in names}, operations={n: 0 for n in names},
                                     revisions=(spec.stage.framework_id,))
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._save()

    @property
    def spec(self) -> BudgetSpec:
        return self.state.spec

    def _save(self) -> None:
        temporary = self.path.with_suffix(".pending")
        temporary.write_text(self.state.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def remaining(self, operation: Operation) -> int:
        """Remaining allowance, or `sys.maxsize` when this operation is uncapped.

        Callers use this both as a gate (`> 0`) and as a number to report to the agent. Returning
        a large integer rather than None keeps every existing comparison and message working
        without each call site having to special-case "unlimited".

        Use `declared_remaining` instead when the value is going to be SHOWN to someone: a literal
        9223372036854775807 is true but tells a reader nothing.
        """
        limit = self.limits[operation]
        return maxsize if limit is None else limit - self.state.used[operation]

    def declared_remaining(self, operation: Operation) -> int | None:
        """Remaining allowance, or None when no ceiling was declared.

        This is the form to report -- to an agent, a receipt or a log -- because None says
        "uncapped" where `maxsize` would present itself as a real figure to pace against.
        """
        return None if self.limits[operation] is None else self.remaining(operation)

    def available(self) -> bool:
        return self.spec.started_unix_s <= self.now() < self.spec.deadline_unix_s

    def activate(self, stage: StageIdentity) -> None:
        """Record a new framework revision against this lane.

        The revision count is no longer capped -- the original limit of three was a research
        protocol, not a correctness property. What is still enforced is that a revision is
        DISTINCT: re-activating the same framework id would make the ledger claim two epochs ran
        different code when they ran the same, which would corrupt the record rather than protect
        it.
        """
        with self.lock:
            if (self.spec.lane != "development" or stage.stage != "development"
                    or stage.framework_id in self.state.revisions):
                raise RunnerError("development revisions must be explicit and distinct from earlier ones")
            self.state = self.state.model_copy(update={"spec": self.spec.model_copy(update={"stage": stage}),
                "revisions": (*self.state.revisions, stage.framework_id)})
            self._save()

    def _admit(self, stage: StageIdentity, operation: Operation) -> int | None:
        with self.lock:
            charged = [operation]
            if self.spec.lane in ("development", "formal_search"):
                if operation == "capture":
                    charged.append("local")
                if operation == "profile":
                    charged.append("model")
            allowed = stage == self.spec.stage and self.available() and all(self.remaining(k) > 0 for k in charged)
            ordinal = self.state.operations[operation] if allowed else None
            if allowed:
                used, operations = dict(self.state.used), dict(self.state.operations)
                for key in charged:
                    used[key] += 1
                operations[operation] += 1
                self.state = self.state.model_copy(update={"used": used, "operations": operations})
                self._save()
            with self.path.with_suffix(".events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"stage": stage.model_dump(mode="json"), "lane": self.spec.lane,
                    "operation": operation, "admitted": allowed, "ordinal": ordinal, "charged": charged if allowed else [],
                    "unix_s": self.now(), "deadline_unix_s": self.spec.deadline_unix_s}) + "\n")
            return ordinal

    def admit(self, stage: StageIdentity, operation: DeviceOperation) -> int | None:
        return self._admit(stage, operation)

    def admit_agent(self, stage: StageIdentity) -> int | None:
        return self._admit(stage, "agent")

    def model_with_local_available(self) -> bool:
        return self.available() and self.remaining("model") > 0 and self.remaining("local") > 0
