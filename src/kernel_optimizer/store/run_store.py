"""Run store: run directory layout, append-only events.jsonl, sha256 artifacts, replay."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Event:
    seq: int
    ts: float
    type: str
    payload: dict[str, Any]


@dataclass
class ReplayState:
    """State rebuilt from the event log; the resume authority."""

    events: list[Event] = field(default_factory=list)
    steps_done: set[str] = field(default_factory=set)
    candidates: dict[str, dict] = field(default_factory=dict)
    families: dict[str, dict] = field(default_factory=dict)
    spaces: dict[str, dict] = field(default_factory=dict)
    trials: dict[str, list[dict]] = field(default_factory=dict)  # space_id -> trial payloads
    baselines: list[dict] = field(default_factory=list)
    finished: bool = False

    def last_of(self, event_type: str) -> Event | None:
        for ev in reversed(self.events):
            if ev.type == event_type:
                return ev
        return None


class RunStore:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.events_path = run_dir / "events.jsonl"
        self.artifacts_dir = run_dir / "artifacts"
        self._seq = self._count_events()
        self._append_lock = threading.Lock()

    # -- lifecycle -----------------------------------------------------------

    @classmethod
    def create(cls, runs_dir: Path, run_id: str, manifest: dict[str, Any]) -> "RunStore":
        run_dir = runs_dir / run_id
        if run_dir.exists():
            raise FileExistsError(f"run dir already exists: {run_dir}")
        for sub in ("artifacts", "candidates", "sandboxes", "jobs", "report"):
            (run_dir / sub).mkdir(parents=True)
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )
        store = cls(run_dir)
        store.append("RUN_CREATED", {"run_id": run_id})
        return store

    @classmethod
    def open(cls, run_dir: Path) -> "RunStore":
        if not (run_dir / "events.jsonl").exists():
            raise FileNotFoundError(f"not a run dir (no events.jsonl): {run_dir}")
        return cls(run_dir)

    def _count_events(self) -> int:
        if not self.events_path.exists():
            return 0
        with self.events_path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    # -- events ---------------------------------------------------------------

    def append(self, event_type: str, payload: dict[str, Any]) -> Event:
        # Serialized: agent calls may run on a background thread (pipelined
        # parameterization), and both `self._seq += 1` and the append-write would
        # otherwise interleave, producing duplicate seqs or torn lines in the log
        # that replay() depends on.
        with self._append_lock:
            ev = Event(seq=self._seq, ts=time.time(), type=event_type, payload=payload)
            line = json.dumps(
                {"seq": ev.seq, "ts": ev.ts, "type": ev.type, "payload": ev.payload},
                ensure_ascii=False,
                default=str,
            )
            with self.events_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            self._seq += 1
            return ev

    def iter_events(self) -> list[Event]:
        events: list[Event] = []
        if not self.events_path.exists():
            return events
        with self.events_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                events.append(
                    Event(seq=raw["seq"], ts=raw["ts"], type=raw["type"], payload=raw["payload"])
                )
        return events

    # -- artifacts -------------------------------------------------------------

    def put_artifact(self, data: bytes | str, name: str = "artifact") -> str:
        """Store content-addressed; returns 'sha256:<hex>' reference."""
        raw = data.encode("utf-8") if isinstance(data, str) else data
        digest = hashlib.sha256(raw).hexdigest()
        self.artifacts_dir.mkdir(exist_ok=True)
        path = self.artifacts_dir / digest
        if not path.exists():
            path.write_bytes(raw)
        # Keep a human-readable name index (best effort).
        index = self.artifacts_dir / "index.jsonl"
        with index.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"sha256": digest, "name": name}) + "\n")
        return f"sha256:{digest}"

    def get_artifact(self, ref: str) -> bytes:
        digest = ref.removeprefix("sha256:")
        return (self.artifacts_dir / digest).read_bytes()

    def candidate_dir(self, candidate_id: str) -> Path:
        d = self.run_dir / "candidates" / candidate_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    # -- replay ------------------------------------------------------------------

    def replay(self) -> ReplayState:
        state = ReplayState()
        state.events = self.iter_events()
        for ev in state.events:
            p = ev.payload
            if ev.type == "STEP_DONE":
                state.steps_done.add(p["step_key"])
            elif ev.type == "CANDIDATE_REGISTERED":
                state.candidates[p["candidate"]["candidate_id"]] = p["candidate"]
            elif ev.type == "FAMILY_UPDATED":
                state.families[p["family"]["family_id"]] = p["family"]
            elif ev.type == "SPACE_PUBLISHED":
                state.spaces[p["space"]["space_id"]] = p["space"]
            elif ev.type == "TRIAL_DONE":
                state.trials.setdefault(p["trial"]["space_id"], []).append(p["trial"])
            elif ev.type == "BASELINE_DONE":
                state.baselines.append(p)
            elif ev.type == "RUN_FINISHED":
                state.finished = True
        return state

    def write_state_snapshot(self, state: dict[str, Any]) -> None:
        (self.run_dir / "state.json").write_text(
            json.dumps(state, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
