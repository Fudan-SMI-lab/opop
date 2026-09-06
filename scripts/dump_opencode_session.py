"""Dump every part of a session's messages so the model's actual words are visible.

Used to diagnose why glm-5.3 produced no parseable JSON in the L3:21 generator call.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys

SID = sys.argv[1] if len(sys.argv) > 1 else "ses_f8bd092d3ffeOPjvtUcytpvhyu"
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 1500

db = os.path.expanduser("~/.local/share/opencode/opencode.db")
uri = "file:" + db.replace(os.sep, "/") + "?mode=ro"
con = sqlite3.connect(uri, uri=True, timeout=15)
con.row_factory = sqlite3.Row
cur = con.cursor()

msgs = list(
    cur.execute(
        "select id, time_created, data from message where session_id = ? order by time_created",
        (SID,),
    )
)
for m in msgs:
    d = json.loads(m["data"])
    role = d.get("role")
    print("=" * 78)
    print(f"{role.upper()}  {m['id']}  t={m['time_created']}")
    if role == "assistant":
        tk = d.get("tokens") or {}
        print(
            f"  cost={d.get('cost')} tokens total={tk.get('total')} in={tk.get('input')} "
            f"out={tk.get('output')} reasoning={tk.get('reasoning')}"
        )
        if d.get("error"):
            print("  ERROR:", json.dumps(d["error"])[:800])
        fin = d.get("finish") or d.get("finishReason")
        if fin:
            print("  finish:", json.dumps(fin)[:300] if not isinstance(fin, str) else fin)
    parts = list(
        cur.execute(
            "select id, data from part where message_id = ? order by time_created",
            (m["id"],),
        )
    )
    for p in parts:
        pd = json.loads(p["data"])
        ptype = pd.get("type")
        if ptype == "text":
            print(f"  [text] {pd.get('text','')[:LIMIT]}")
        elif ptype == "reasoning":
            txt = pd.get("text", "")
            print(f"  [reasoning {len(txt)} chars] {txt[:LIMIT]}")
        elif ptype == "tool":
            st = pd.get("state") or {}
            print(
                f"  [tool {pd.get('tool')}] status={st.get('status')} "
                f"input={json.dumps(st.get('input'))[:300]}"
            )
            if st.get("error"):
                print(f"      error={str(st.get('error'))[:300]}")
        elif ptype == "step-finish":
            print(f"  [step-finish] {json.dumps(pd)[:300]}")
        else:
            print(f"  [{ptype}] {json.dumps(pd)[:300]}")

con.close()
