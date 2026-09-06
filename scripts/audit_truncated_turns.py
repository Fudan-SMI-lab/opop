"""How often does a model turn get truncated (`finish == "length"`), across every run?

The glm-5.3 arm died when three generator attempts were each cut off at 32000 output+reasoning
tokens. Two questions that decide whether that was a one-off or a standing hazard:

  1. how many turns in the whole opencode store ended with `finish == "length"`, split by model;
  2. within a single agent call, does a truncation end the call or get retried?

(2) is answered by the harness code -- `AgentModule.invoke` retries `max_retries + 1` times -- but
this counts what actually happened on disk, per session, so the retry behaviour is observed rather
than asserted.

Read-only against opencode's sqlite store; safe while a run is live.
"""
from __future__ import annotations

import json
import os
import sqlite3
from collections import Counter, defaultdict

db = os.path.expanduser("~/.local/share/opencode/opencode.db")
con = sqlite3.connect("file:" + db.replace(os.sep, "/") + "?mode=ro", uri=True, timeout=30)
con.row_factory = sqlite3.Row
cur = con.cursor()

# Only sessions belonging to this project's runs; the store also holds unrelated work.
sessions = {
    r["id"]: (r["title"], r["directory"], r["model"])
    for r in cur.execute(
        "select id, title, directory, model from session "
        "where directory like 'D:/Pyhon_projects/opop%'"
    )
}
print(f"opop sessions in the store: {len(sessions)}")

by_model_finish: dict[str, Counter] = defaultdict(Counter)
trunc_per_session: dict[str, int] = Counter()
turns_per_session: dict[str, int] = Counter()
trunc_tokens: list[int] = []

for r in cur.execute("select session_id, data from message"):
    sid = r["session_id"]
    if sid not in sessions:
        continue
    d = json.loads(r["data"])
    if d.get("role") != "assistant":
        continue
    model = sessions[sid][2] or "{}"
    mid = json.loads(model).get("id", "?") if model.startswith("{") else model
    fin = d.get("finish")
    by_model_finish[mid][fin] += 1
    turns_per_session[sid] += 1
    if fin == "length":
        trunc_per_session[sid] += 1
        tk = d.get("tokens") or {}
        trunc_tokens.append((tk.get("output") or 0) + (tk.get("reasoning") or 0))

print("\n== finish reasons per model (opop runs only) ==")
for mid, c in sorted(by_model_finish.items(), key=lambda x: -sum(x[1].values())):
    total = sum(c.values())
    length = c.get("length", 0)
    print(f"  {mid:16s} turns={total:6d}  length={length:5d} ({length/total*100:5.2f}%)  "
          f"others={dict((k, v) for k, v in c.items() if k != 'length')}")

print("\n== sessions containing at least one truncated turn ==")
if not trunc_per_session:
    print("  none")
for sid, n in sorted(trunc_per_session.items(), key=lambda x: -x[1]):
    title, directory, model = sessions[sid]
    mid = json.loads(model).get("id", "?") if model and model.startswith("{") else model
    print(f"  {n} truncated of {turns_per_session[sid]} turns   {mid:12s} {title:22s}")
    print(f"      {directory[-72:]}")

if trunc_tokens:
    print(f"\ntruncated turns: {len(trunc_tokens)}, "
          f"output+reasoning min={min(trunc_tokens)} max={max(trunc_tokens)}")
con.close()
