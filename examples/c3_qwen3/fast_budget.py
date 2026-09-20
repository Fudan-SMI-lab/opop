"""Durable phase admission; framework revisions never reset D or consumed allowances."""

import json
from collections.abc import Callable
from pathlib import Path
from threading import RLock
from time import time
from typing import Literal, assert_never

from pydantic import Field

from .device_records import DeviceOperation, StageIdentity
from .runner_records import FrozenRecord, RunnerError
from .runner_records import GoalId

type Lane = Literal["setup", "development", "formal_search", "final_verification", "final"]
type Operation = Literal["agent", "local", "model", "capture", "profile"]


class BudgetSpec(FrozenRecord):
    lane: Lane
    stage: StageIdentity
    started_unix_s: float = Field(gt=0)
    deadline_unix_s: float = Field(gt=0)
    goal: GoalId | None = None


class BudgetState(FrozenRecord):
    spec: BudgetSpec
    used: dict[Operation, int]
    operations: dict[Operation, int]
    revisions: tuple[str, ...]


class StageBudget:
    """Own mutable counters behind one lock and persist before admitting expensive work."""

    def __init__(self, path: Path, spec: BudgetSpec, *, now: Callable[[], float] = time) -> None:
        self.path, self.now, self.lock = path.resolve(), now, RLock()
        match spec.lane:
            case "setup":
                limits = (0, 3, 3, 6, 3)
                expected = "setup"
            case "development":
                limits = (6, 12, 4, 12, 3)
                expected = "development"
            case "formal_search":
                limits = (2, 8, 4, 8, 0)
                expected = "formal_search"
            case "final_verification":
                limits = (0, 9, 0, 18, 0)
                expected = "formal_search"
            case "final":
                limits = (0, 0, 36, 9, 0)
                expected = "final"
            case unreachable:
                assert_never(unreachable)
        names: tuple[Operation, ...] = ("agent", "local", "model", "capture", "profile")
        self.limits = dict(zip(names, limits, strict=True))
        if spec.stage.stage != expected or spec.deadline_unix_s <= spec.started_unix_s:
            raise RunnerError("budget lane/stage or clock mismatch")
        if spec.lane == "development" and spec.deadline_unix_s != spec.started_unix_s + 5400:
            raise RunnerError("development must retain the continuous D+5400 cutoff")
        if self.path.exists():
            self.state = BudgetState.model_validate_json(self.path.read_bytes())
            if self.state.spec != spec or set(self.state.used) != set(names) or set(self.state.operations) != set(names):
                raise RunnerError("existing budget identity/counters cannot be reset")
            if any(not 0 <= self.state.used[k] <= self.limits[k] for k in names):
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
        return self.limits[operation] - self.state.used[operation]

    def available(self) -> bool:
        return self.spec.started_unix_s <= self.now() < self.spec.deadline_unix_s

    def activate(self, stage: StageIdentity) -> None:
        with self.lock:
            if (self.spec.lane != "development" or stage.stage != "development" or
                    stage.framework_id in self.state.revisions or len(self.state.revisions) >= 3):
                raise RunnerError("only two distinct, explicit development framework revisions are allowed")
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
