"""Did an attributed wall actually reach the rewriter, and did the rewrite that followed free it?

This is A3's primary outcome, and it is answerable from the log plus the sandboxes:

  1. the wall was ATTRIBUTED for candidate C on knob K
  2. `analysis/resource_walls.md` exists in the rewriter sandbox for C's family (the file the
     pre-launch fix added -- without it the wall reached the rewriter only as the analyst's
     paraphrase, and a null A3 would be unattributable)
  3. the rewrite that followed: did `shared_bytes` fall, and did K's refused value become feasible?

Step 3 is what needs the run to be over; steps 1-2 are checkable in flight, and checking them in
flight is the point -- if the file is missing, the arm is not testing what it was launched to test.
"""
import json
import os
import sys

rd = sys.argv[1]
evs = []
for ln in open(os.path.join(rd, "events.jsonl"), encoding="utf-8"):
    ln = ln.strip()
    if ln:
        try:
            evs.append(json.loads(ln))
        except Exception:
            pass

attributed = []
for e in evs:
    if e.get("type") != "RESOURCE_WALL_ATTRIBUTED":
        continue
    p = e.get("payload") or {}
    for w in (p.get("walls") or []):
        if w.get("verdict") == "attributed":
            attributed.append((e["seq"], e["ts"], p.get("candidate_id"), w))

print(f"ATTRIBUTED walls: {len(attributed)}")
for seq, ts, cid, w in attributed:
    print(f"  seq{seq} {cid} {w.get('param')} refused={w.get('refused_value')} "
          f"shared={w.get('max_shared')}/{w.get('limit')} "
          f"over={w.get('over_ratio')} tail={w.get('tail_gain_pct')}%")

# which family does that candidate belong to, and which rewriter call came after?
fam_of = {}
for e in evs:
    if e.get("type") == "CANDIDATE_REGISTERED":
        p = e.get("payload") or {}
        c = (p.get("candidate") or p)
        if c.get("candidate_id"):
            fam_of[c["candidate_id"]] = c.get("family_id")

print()
print("=== rewriter sandboxes, and whether the measured wall was IN them ===")
sb_root = os.path.join(rd, "sandboxes")
hits = 0
if os.path.isdir(sb_root):
    for name in sorted(os.listdir(sb_root)):
        if "rewriter" not in name:
            continue
        walls_md = os.path.join(sb_root, name, "analysis", "resource_walls.md")
        ledger = os.path.join(sb_root, name, "history", "prediction_ledger.md")
        bn = os.path.join(sb_root, name, "analysis", "bottleneck.json")
        marks = []
        if os.path.isfile(walls_md):
            hits += 1
            body = open(walls_md, encoding="utf-8").read()
            cond = "最优参数点" in body
            marks.append(f"resource_walls.md {os.path.getsize(walls_md)}B "
                         f"conditional={'YES' if cond else 'NO'}")
            for _, _, _cid, w in attributed:
                if str(w.get("param")) in body:
                    marks.append(f"names {w.get('param')}")
        else:
            marks.append("NO resource_walls.md")
        if os.path.isfile(ledger):
            marks.append("ledger")
        if os.path.isfile(bn):
            marks.append(f"bottleneck.json {os.path.getsize(bn)}B")
        print(f"  {name}: {'; '.join(marks)}")
print()
print(f"rewriter sandboxes carrying the measured wall: {hits}")
if attributed and hits == 0:
    print("  *** the arm attributed a wall but NO rewriter sandbox received it -- A3 cannot be "
          "answered from this run, and the in-prompt wiring is not doing what it was launched for")
