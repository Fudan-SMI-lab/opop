"""S1b's over-interception risk, measured in the shape S1b actually has: WITHIN-run, PROSPECTIVE.

An earlier version of this script did leave-one-out across runs. That is the wrong test and its
numbers must not be quoted for S1b: the plan's criterion is explicitly per-run and per-space (M = the
MEDIAN sample count of the same knob's other values in that run), and it fires prospectively as
trials arrive. Cross-run leave-one-out measures whether a conclusion TRANSFERS between runs -- a
different question, and one S1b never claims (see the plan's own "conclusions must not carry across").

So: walk each space's trials in recorded order. After each trial, evaluate the three conditions on
the evidence available SO FAR. The moment a (knob, value) qualifies, freeze the decision and then
count how many LATER trials in that same space used that value and PASSED. That is the mis-kill, and
it is the same shape as the box-2 measurement that refused the count-based blacklist (N=6 -> 38.5%).

Trial order comes from events.jsonl, which is append-only, so the order is the real one.
"""
import json, sys, glob
from collections import defaultdict
from pathlib import Path
from statistics import median


def load_ordered(run_dir: Path) -> dict[str, dict]:
    """Per space: domains plus the trial sequence IN ORDER as (cfg, passed) pairs."""
    spaces: dict[str, dict] = {}
    ev = run_dir / "events.jsonl"
    if not ev.exists():
        return spaces
    for line in ev.open(encoding="utf-8"):
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        t, p = e.get("type"), e.get("payload") or {}
        if t == "SPACE_PUBLISHED":
            sp = p.get("space") or {}
            sid, doms = sp.get("space_id"), sp.get("domains") or []
            if sid and doms:
                spaces.setdefault(sid, {"domains": {}, "seq": []})
                spaces[sid]["domains"] = {d["name"]: list(d["choices"]) for d in doms}
        elif t == "TRIAL_DONE":
            tr = p.get("trial") or p
            sid = tr.get("space_id")
            vals = (tr.get("params") or {}).get("values") or {}
            if not sid or not vals:
                continue
            s = spaces.setdefault(sid, {"domains": {}, "seq": []})
            status = tr.get("status")
            kind = tr.get("failure_kind")
            if status == "complete":
                s["seq"].append(({k: str(v) for k, v in vals.items()}, True))
            elif kind == "correctness_mismatch":
                s["seq"].append(({k: str(v) for k, v in vals.items()}, False))
            # other failure kinds (shared mem, runtime) are not S1b's pool and are skipped
    return spaces


def first_fire_index(seq, knob, value, min_ev, require_median_m, require_control):
    """Earliest index i such that the rule fires on evidence from seq[:i+1]. None if it never does."""
    for i in range(len(seq)):
        seen = defaultdict(int)
        passed = defaultdict(int)
        for cfg, ok in seq[: i + 1]:
            if knob not in cfg:
                continue
            seen[cfg[knob]] += 1
            if ok:
                passed[cfg[knob]] += 1
        n = seen.get(value, 0)
        if n == 0 or passed.get(value, 0) != 0:
            continue                                    # cond 1: unconditional (0 passes so far)
        others = [c for vv, c in seen.items() if vv != value]
        need = min_ev
        if require_median_m and others:
            need = max(min_ev, median(others))
        if n < need:
            continue                                    # cond 2: sample sufficiency
        if require_control and not any(passed.get(vv, 0) > 0 for vv in seen if vv != value):
            continue                                    # cond 3: a live control exists
        return i
    return None


def run_variant(runs, min_ev, require_median_m, require_control):
    fired = 0
    later_pass = 0
    saved = 0
    worst = []
    # The register's 4.7% / 38.5% figures count VALUES with at least one later success, not trials.
    # Both units are reported so the two measurements can be compared without appearing to disagree.
    values_with_later_pass = 0
    for r in runs:
        for sid, s in load_ordered(Path(r)).items():
            seq = s["seq"]
            if len(seq) < 5:
                continue
            for knob, choices in s["domains"].items():
                for value in {str(c) for c in choices}:
                    i = first_fire_index(seq, knob, value, min_ev, require_median_m, require_control)
                    if i is None:
                        continue
                    fired += 1
                    tail = [(cfg, ok) for cfg, ok in seq[i + 1:] if cfg.get(knob) == value]
                    p = sum(1 for _, ok in tail if ok)
                    later_pass += p
                    saved += len(tail) - p
                    if p:
                        values_with_later_pass += 1
                        worst.append((p, len(tail), r.rsplit("/", 1)[-1], sid[:12], knob, value))
    return fired, later_pass, saved, worst, values_with_later_pass


runs = sorted(glob.glob(sys.argv[1] + "/run-l3-*"))
print("S1b prospective within-run test over %d runs\n" % len(runs))
print("%-38s %-7s %-11s %-11s %-13s %s" % (
    "variant", "fired", "later PASS", "saved fail", "trial rate", "VALUE rate (register unit)"))
print("-" * 108)
variants = [
    ("count-only N=3 (refused blacklist)", 3, False, False),
    ("count-only N=6 (refused blacklist)", 6, False, False),
    ("count-only N=12 (refused blacklist)", 12, False, False),
    ("plan S1b: median-M + live control", 1, True, True),
    ("plan S1b + floor N=3", 3, True, True),
    ("plan S1b + floor N=6", 6, True, True),
]
detail = {}
for label, m, med, ctl in variants:
    fired, lp, saved, worst, vwp = run_variant(runs, m, med, ctl)
    denom = lp + saved
    print("%-38s %-7d %-11d %-11d %-13s %s" % (
        label, fired, lp, saved,
        ("%.2f%%" % (100.0 * lp / denom)) if denom else "n/a",
        ("%d/%d = %.1f%%" % (vwp, fired, 100.0 * vwp / fired)) if fired else "n/a"))
    detail[label] = worst

print()
print("trial rate = later-passing trials / all later trials using that value.")
print("VALUE rate = fired values with >=1 later success / all fired values. This is the unit the")
print("register's 4.7%% (box 1) and 38.5%% (box 2) figures are in; quote like against like.")
print()
for label in ("plan S1b: median-M + live control", "plan S1b + floor N=3"):
    w = sorted(detail.get(label, []), reverse=True)[:8]
    if not w:
        print("%s -- ZERO values passed after firing." % label)
        continue
    print("%s -- worst mis-kills:" % label)
    for p, tot, run, sid, knob, value in w:
        print("   %s %s %s=%s : %d of %d later trials PASSED" % (run, sid, knob, value, p, tot))
