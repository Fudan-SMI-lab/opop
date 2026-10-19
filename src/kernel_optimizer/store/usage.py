"""Observed provider usage. Accounting never makes admission or stopping decisions."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sqlite3
import time
from typing import Any


def _number(value: Any) -> float | None:
    return float(value) if (isinstance(value, (int, float)) and not isinstance(value, bool)
                            and math.isfinite(value) and value >= 0) else None


def _tokens(value: Any, prefix: str = "") -> dict[str, float]:
    result = {}
    if not isinstance(value, dict):
        return result
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(_tokens(item, name))
        elif (number := _number(item)) is not None:
            result[name] = number
    return result


class UsageLedger:
    """Upsert message snapshots so retries/resume/repeated reads cannot double count.

    Cost is provider-reported, not billing verification. Token components remain
    separate (including cache and reasoning); total and components are not added.
    Message content, tool arguments and provider credentials are not stored here.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS messages (
                session TEXT, message TEXT, model TEXT, cost REAL, tokens TEXT,
                finished INTEGER, observed_at REAL, PRIMARY KEY(session, message))""")
            db.execute("""CREATE TABLE IF NOT EXISTS observations (
                session TEXT PRIMARY KEY, status TEXT, detail TEXT, observed_at REAL)""")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    def observe(self, session: str, messages: list[dict]) -> dict:
        with self._connect() as db:
            for message in messages:
                info = message.get("info", {})
                ident = info.get("id")
                if info.get("role") != "assistant" or not isinstance(ident, str):
                    continue
                db.execute("""INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session, message) DO UPDATE SET model=excluded.model,
                    cost=COALESCE(excluded.cost, messages.cost), tokens=excluded.tokens,
                    finished=MAX(messages.finished, excluded.finished), observed_at=excluded.observed_at
                    WHERE excluded.finished >= messages.finished""",
                    (session, ident, str(info.get("providerID", "")) + "/" + str(info.get("modelID", "")),
                     _number(info.get("cost")), json.dumps(_tokens(info.get("tokens"))),
                     int(bool(info.get("finish") or info.get("time", {}).get("completed"))), time.time()))
            db.execute("INSERT OR REPLACE INTO observations VALUES (?, 'observed', NULL, ?)",
                       (session, time.time()))
        return self.summary(session)

    def unavailable(self, session: str, detail: str) -> None:
        with self._connect() as db:
            db.execute("INSERT OR REPLACE INTO observations VALUES (?, 'unavailable', ?, ?)",
                       (session, detail, time.time()))

    def summary(self, session: str | None = None) -> dict:
        where, params = (" WHERE session=?", (session,)) if session else ("", ())
        with self._connect() as db:
            rows = db.execute("SELECT cost,tokens,finished FROM messages" + where, params).fetchall()
            observations = db.execute("SELECT session,status,detail FROM observations" + where, params).fetchall()
        tokens: dict[str, float] = {}
        for _, encoded, _ in rows:
            for key, value in json.loads(encoded).items():
                tokens[key] = tokens.get(key, 0) + value
        known = [row[0] for row in rows if row[0] is not None]
        return {"schema_version": 1, "mode": "observe_only", "messages": len(rows),
                "reported_cost": math.fsum(known) if known else None,
                "messages_without_cost": len(rows) - len(known),
                "unfinished_messages": sum(not row[2] for row in rows), "tokens": tokens,
                "sessions": [{"id": s, "status": status, "detail": detail} for s, status, detail in observations],
                "scope": "Observed assistant message snapshots; provider estimates, not invoice"}
