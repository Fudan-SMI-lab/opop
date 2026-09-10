"""The actual S1b candidate design, tested prospectively: pooled across spaces, retracted on evidence.

audit_s1b_transfer.py showed that 4 of 18 (knob,value) pairs are dead in every space of a run while 14 are
alive somewhere -- judged with full hindsight. A live rule does not have hindsight. So test the rule
that could actually be implemented:

  * evidence pools across the spaces of ONE run, per (knob, value)   [never across runs]
  * fires when: >= N failures pooled AND 0 passes pooled
  * RETRACTS permanently the moment any pass arrives                  [this is what deweight buys]
  * deweight, not removal, so a retracted rule costs sampling share rather than correctness

Walk every trial of a run in global recorded order and score two things:
  SAVED    failing trials that occur while the rule is fired
  MIS-HIT  passing trials that occur while the rule is fired (the damage, per trial)

A pooled rule that retracts is the strongest version available; if this cannot beat the per-space
ceiling of 59 saved trials safely, S1b has no path and should be dropped with S1.
"""
import json, sys, glob
from collections import defaultdict
from pathlib import Path


def run_events(run_dir: Path):
    """Every trial of a run in recorded order, as (space_id, cfg, passed). Order is append-only."""
    ev = run_dir / "events.jsonl"
    if not ev.exists():
        return
    for line in ev.open(encoding="utf-8"):
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") != "TRIAL_DONE":
            continue
        p = e.get("payload") or {}
        tr = p.get("trial") or p
        vals = (tr.get("params") or {}).get("values") or {}
        sid = tr.get("space_id")
        if not sid or not vals:
            continue
        status, kind = tr.get("status"), tr.get("failure_kind")
        if status == "complete":
            yield sid, {k: str(v) for k, v in vals.items()}, True
        elif kind == "correctness_mismatch":
            yield sid, {k: str(v) for k, v in vals.items()}, False


def simulate(runs, floor, pooled):
    """Returns (saved_fails, mis_hit_passes, fired_kv, retracted_kv)."""
    saved = mis = 0
    fired_kv = set()
    retracted = set()
    for r in runs:
        # key: (knob, value) when pooled, else (space_id, knob, value)
        fails = defaultdict(int)
        passes = defaultdict(int)
        dead = set()
        for sid, cfg, ok in run_events(Path(r)):
            # 1. score the trial against rules already fired, BEFORE updating evidence
            for knob, v in cfg.items():
                key = (knob, v) if pooled else (sid, knob, v)
                if key in dead:
                    if ok:
                        mis += 1
                    else:
                        saved += 1
            # 2. update evidence and fire/retract
            for knob, v in cfg.items():
                key = (knob, v) if pooled else (sid, knob, v)
                if ok:
                    passes[key] += 1
                    if key in dead:
                        dead.discard(key)
                        retracted.add((r, key))
                else:
                    fails[key] += 1
                if passes[key] == 0 and fails[key] >= floor:
                    if key not in dead:
                        dead.add(key)
                        fired_kv.add((r, key))
    return saved, mis, fired_kv, retracted


runs = sorted(glob.glob(sys.argv[1] + "/run-l3-*"))
print("%-10s %-9s %-9s %-11s %-11s %s" % (
    "pooling", "floor N", "fired", "SAVED fail", "MIS-HIT pass", "damage rate"))
print("-" * 78)
for pooled in (False, True):
    for floor in (3, 6, 8, 12):
        saved, mis, fired, retracted = simulate(runs, floor, pooled)
        tot = saved + mis
        print("%-10s %-9d %-9d %-11d %-11d %s" % (
            "cross-sp" if pooled else "per-space", floor, len(fired), saved, mis,
            ("%.2f%% (%d retractions)" % (100.0 * mis / tot, len(retracted))) if tot else "n/a"))
print()
print("SAVED = failing trials the rule would have deweighted away. MIS-HIT = PASSING trials it would")
print("have deweighted, i.e. the over-interception the user asked about, counted per trial.")
print()
print("Read the two blocks against each other: cross-space pooling is only worth its extra risk if")
print("SAVED rises by more than MIS-HIT does. Note deweighting keeps nonzero probability, so a")
print("MIS-HIT is a lost sampling opportunity, not a lost configuration -- unlike S1's domain shrink.")
print()
print("WHICH rules fire under cross-space pooling? If they are all one class of knob, the mechanism")
print("is nameable and the threshold is not merely a lucky number.")
for floor in (8, 12):
    _s, _m, fired, _r = simulate(runs, floor, True)
    print("\ncross-space floor N=%d, %d rules:" % (floor, len(fired)))
    for r, (knob, v) in sorted(fired, key=lambda x: (x[1][0], x[0])):
        print("   %-26s  (%s)" % ("%s=%s" % (knob, v), r.rsplit("/", 1)[-1]))
