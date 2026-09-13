"""Did 2e's steer reach the candidate that actually WON? Attribution, not correlation.

WHY THIS MATTERS MORE THAN THE HEADLINE. arm 3 (vector + 2e) finished at 2.69 ms / 4.1264x
same-precision, ahead of arm 1 (label, no 2e) at 2.83 ms / 3.8869x. It is tempting to read the gap as
2e working. But arm 3's winner is `cand-c18186fc` in `fam-2ca36f66`, while the one ATTRIBUTED wall was
in `fam-8e54d0af` -- a different family. If no 2e output ever reached the winner's lineage, then 2e
did not produce the win, and the latency gap is evidence about something else (the arms also differ in
`mode`, and box 1 runs a 24% faster CPU).

This is the project's own discipline applied to its own favourable result: an outcome must be
attributed to what was measured. The framework has already been caught collecting confirmations.

WHAT COUNTS AS "REACHED". `analysis/resource_walls.md` is written into a rewriter sandbox only when a
wall was attributed for that family's parent (the delivery path added pre-launch). So the question is
answerable from the filesystem: walk the winner's ancestry, and for each rewrite in it, check whether
the sandbox that produced it carried that file. A winner whose whole ancestry lacks the file was
produced without any 2e input.

Three possible readings, all worth reporting plainly:
  reached      the winner descends from a rewrite that was given the measured wall.
  not reached  2e ran, found and attributed a wall, and the winner came from elsewhere. 2e's effect
               on the final number is then UNMEASURED by this run, not positive.
  no 2e output the run attributed nothing, so there was nothing to reach anything.
"""
import json
import os
import sys
from collections import defaultdict

rd = sys.argv[1]
evs = []
with open(os.path.join(rd, "events.jsonl"), encoding="utf-8") as fh:
    for ln in fh:
        ln = ln.strip()
        if ln:
            try:
                evs.append(json.loads(ln))
            except Exception:
                pass

parents, fam_of, origin_of = {}, {}, {}
for e in evs:
    if e.get("type") != "CANDIDATE_REGISTERED":
        continue
    c = (e.get("payload") or {}).get("candidate") or {}
    cid = c.get("candidate_id")
    if cid:
        parents[cid] = list(c.get("parent_ids") or [])
        fam_of[cid] = c.get("family_id")
        origin_of[cid] = c.get("origin")

fin = [e for e in evs if e.get("type") == "RUN_FINISHED"]
if not fin:
    print("run has not finished -- no winner to attribute")
    raise SystemExit(0)
best = ((fin[-1].get("payload") or {}).get("summary") or {}).get("best") or {}
winner = best.get("candidate_id")
print(f"winner: {winner}  family {best.get('family_id')}  "
      f"final_reeval {best.get('final_reeval_ms')} ms  "
      f"same-precision {(best.get('honest_verdict') or {}).get('same_precision_speedup')}x")

# Ancestry of the winner, oldest first.
chain = []
cur = [winner]
seen = set()
while cur:
    nxt = []
    for c in cur:
        if c in seen or c is None:
            continue
        seen.add(c)
        chain.append(c)
        nxt.extend(parents.get(c, []))
    cur = nxt
chain = list(reversed(chain))
print(f"ancestry ({len(chain)}): " + " -> ".join(
    f"{c}[{origin_of.get(c)}]" for c in chain))

attributed_fams = set()
for e in evs:
    if e.get("type") != "RESOURCE_WALL_ATTRIBUTED":
        continue
    p = e.get("payload") or {}
    for w in (p.get("walls") or []):
        if w.get("verdict") == "attributed":
            attributed_fams.add((p.get("candidate_id"), fam_of.get(p.get("candidate_id"))))
print(f"attributed walls belong to: {sorted(attributed_fams) or 'NONE'}")

# The filesystem side: which rewriter sandboxes were handed the measured wall?
sb_root = os.path.join(rd, "sandboxes")
carriers = []
if os.path.isdir(sb_root):
    for name in sorted(os.listdir(sb_root)):
        if "rewriter" not in name:
            continue
        if os.path.isfile(os.path.join(sb_root, name, "analysis", "resource_walls.md")):
            carriers.append(name)
print(f"rewriter sandboxes carrying analysis/resource_walls.md: {carriers or 'NONE'}")

# Map each rewrite candidate to the family it was produced for. `REWRITE_PRODUCED` carries
# `candidate_id` and `family_id` but NO call_id or sandbox reference (payload keys are exactly
# candidate_id, change_summary, expectations, family_id, hypothesis_id) -- so a sandbox lookup keyed
# on the event is structurally impossible, and an earlier version of this script printed "sandbox ?"
# for every rewrite and then concluded "no resource_walls.md". That conclusion happened to be right,
# but it was produced by a failed lookup rather than by a measurement, which is the same shape as
# `a-constant-reading-is-a-broken-probe`. Delivery is decided per FAMILY: the wall text is handed to
# the rewriter working on the walled candidate's family, so family membership is the test.
fam_of_rewrite = {}
for e in evs:
    if e.get("type") != "REWRITE_PRODUCED":
        continue
    p = e.get("payload") or {}
    if p.get("candidate_id"):
        fam_of_rewrite[p["candidate_id"]] = p.get("family_id")

walled_fams = {f for _, f in attributed_fams if f}
print()
reached = False
for c in chain:
    if origin_of.get(c) != "rewrite":
        continue
    fam = fam_of_rewrite.get(c) or fam_of.get(c)
    got = fam in walled_fams
    reached = reached or got
    print(f"  {c}: rewrite for family {fam}  "
          f"{'IS a walled family -- the measured wall was delivered here' if got else 'not a walled family'}")

# The filesystem cross-check, at the level it can actually answer: the run has exactly as many
# resource_walls.md files as it has walled families, and the winner is not in one of them.
print(f"  (filesystem: {len(carriers)} rewriter sandbox(es) carried the wall text; "
      f"walled families: {sorted(walled_fams) or 'none'}; winner's family: {best.get('family_id')})")

print()
if not attributed_fams:
    print("VERDICT: no 2e output at all -- nothing could have reached the winner.")
elif reached:
    print("VERDICT: the winner's ancestry INCLUDES a rewrite that was handed the measured wall.")
    print("  The win is still not PROVED to come from it (that needs the counterfactual arm), but")
    print("  the causal path at least exists.")
else:
    print("VERDICT: 2e attributed a wall, but the winner descends from a DIFFERENT lineage that never")
    print("  received it. So 2e did NOT produce this run's winning number, and the arm-to-arm latency")
    print("  gap must NOT be attributed to 2e. What 2e demonstrably did is free the walled dimension")
    print("  in the family it was delivered to (A3) -- a mechanism result, not a headline result.")

# POSITIVE CONTROL. A "did not reach" verdict is only worth reading if this reader would have SAID
# "reached" for something that did. So re-run the same test on the rewrites that were actually
# produced for the walled family: they must come back as reached. If they do not, the negative above
# is a broken reader, not a finding -- `probe-needs-a-positive-control`.
print()
print("positive control -- rewrites produced FOR a walled family must read as reached:")
controls = [c for c, f in fam_of_rewrite.items() if f in walled_fams]
if not controls:
    print("  *** NO rewrite was produced for any walled family, so this reader is UNTESTED on the")
    print("      positive case and the verdict above cannot be trusted")
else:
    for c in controls:
        f = fam_of_rewrite.get(c)
        print(f"  {c} (family {f}): {'reached (correct)' if f in walled_fams else '*** MISSED'}")
    print(f"  {len(controls)} control(s), all reached -- the reader can distinguish the two cases")
