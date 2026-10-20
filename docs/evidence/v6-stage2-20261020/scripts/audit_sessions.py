from pathlib import Path
import json
import sqlite3
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.config import load_config

roots = (Path("/root/autodl-tmp/dev/v6-work"), Path("/root/opop-v6-stage2"))
config = load_config(roots[0]/"configs/box42-v6.yaml", [])
data_home = Path(config.opencode.server_env.get("XDG_DATA_HOME", "/root/.local/share"))
db = sqlite3.connect("file:"+str(data_home/"opencode/opencode.db")+"?mode=ro", uri=True)
audit = []
for root in roots:
    for cohort in sorted(root.glob("stage2-autonomy-*")):
        if not (cohort/"controllers.json").exists():
            continue
        for control in json.loads((cohort/"controllers.json").read_text()):
            run = Path(control["output"])
            if not run.exists():
                continue
            events = list(RunStore.open(run).iter_events())
            starts = list({e.payload["session"]: e.payload for e in events
                           if e.type == "PROJECT_AGENT_STARTED"}.values())
            roles = []
            for role in starts:
                session = role["session"]
                if not db.execute("select 1 from session where id = ?", (session,)).fetchone():
                    raise ValueError("role session absent from configured provider database")
                sessions = [session]
                for sid in sessions:
                    sessions.extend(row[0] for row in db.execute("select id from session where parent_id = ?", (sid,))
                                    if row[0] not in sessions)
                tools, costs, messages, tokens = [], 0.0, 0, {}
                for sid in sessions:
                    for (raw,) in db.execute("select data from message where session_id = ?", (sid,)):
                        row = json.loads(raw)
                        if row.get("role") == "assistant":
                            messages += 1
                            costs += row.get("cost", 0) or 0
                            for key, value in row.get("tokens", {}).items():
                                if isinstance(value, (int, float)):
                                    tokens[key] = tokens.get(key, 0) + value
                                elif isinstance(value, dict):
                                    for child, count in value.items():
                                        if isinstance(count, (int, float)):
                                            label = key+"."+child
                                            tokens[label] = tokens.get(label, 0) + count
                    for ident, message, raw in db.execute("select id, message_id, data from part where session_id = ?", (sid,)):
                        row = json.loads(raw)
                        if row.get("type") == "tool":
                            state = row.get("state", {})
                            tools.append({"session": sid, "part": ident, "message": message,
                                          "tool": row.get("tool"), "status": state.get("status"),
                                          "input": state.get("input", {})})
                terms = ("observations/", "previous-feedback", "resources.json", "communication-ownership", "binding-contract")
                evidence_reads = [r for r in tools if r["status"] == "completed"
                                  and any(s in json.dumps(r["input"]) for s in terms)]
                historical_paths = [r for r in tools if any(f"/v6-work/p{i}-" in json.dumps(r["input"]) for i in range(7))]
                roles.append({"role": role["role"], "session": session, "sandbox": role["sandbox"],
                              "sessions_including_children": sessions, "messages": messages,
                              "reported_cost": costs, "tokens": tokens, "tool_count": len(tools),
                              "explicit_read_calls": [r for r in evidence_reads if r["tool"] == "read"],
                              "evidence_accesses": evidence_reads, "historical_run_paths": historical_paths})
            audit.append({"cohort": cohort.name, "case": control["name"], "roles": roles})
out = roots[0]/"stage2-session-audit.json"
out.write_text(json.dumps({"source": "Read-only OpenCode database; no provider requests or messages sent",
                           "scope": "Only this stage's role sessions and descendants; costs are provider estimates",
                           "interpretation": "evidence_accesses means a completed tool input mentioned a path; it includes listings and writes, and does not imply reading or comprehension. explicit_read_calls records only the read tool. Shell/Python reads require inspection of the input. Absence of historical path mentions is an observation, not access isolation.",
                           "runs": audit}, indent=2))
print(json.dumps({"runs": len(audit), "roles": sum(len(r["roles"]) for r in audit),
                  "evidence_accesses": sum(len(role["evidence_accesses"]) for r in audit for role in r["roles"]),
                  "historical_path_mentions": sum(len(role["historical_run_paths"]) for r in audit for role in r["roles"]),
                  "output": str(out)}))
