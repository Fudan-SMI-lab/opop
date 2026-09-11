"""Did the two arms get comparable SEARCH? The load-bearing precondition J2-5 does not state.

`scripts/audit_arm_comparability.py` verified the two arms' CONFIGS match field-by-field. That is
necessary and not sufficient: the budget has two limits, `trials_per_space` and `wall_clock_hours`.
Equal per-space trial counts only settle the first. If one box is slower, it reaches fewer SPACES and
fewer rewrite rounds inside the same 12 h, and a latency difference then has two explanations -- the
switch, or the extra search -- which is the ambiguity the paired cross-box design exists to remove
(see `scripts/compare_calibrations.py`, which settles the *denominator* half of the same question).

WHAT THIS DOES NOT MEASURE, and why it matters more than it sounds
-----------------------------------------------------------------
Inter-event gaps are NOT per-trial costs. `max_shared_jobs: 2` puts `compile-screen` and `prescreen`
on the shared lane while `eval` takes the exclusive lock, so a trial's compile overlaps the previous
trial's timing. A gap therefore measures "time until the next TRIAL_DONE landed", which under
concurrency can be near zero for a trial that did real work -- and a first pass at this comparison
produced per-config medians of 0.0 s and a ratio of 192988x from dividing by one.

So the comparable quantity is the AGGREGATE: trials completed per wall-clock hour, which is
insensitive to how the work is interleaved. Per-trial numbers are printed only as a distribution
with the tail called out, because the useful question about them is "is the total dominated by a few
pathological configs" -- and that question is answerable from the gap sum even when individual gaps
are not attributable.

    python scripts/check_arm_search_parity.py <control_run_dir> <treatment_run_dir>
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

# A rate difference this large would let the faster arm reach a whole extra space inside 12 h.
_RATE_TOL = 1.15


def _events(run_dir: Path) -> list[dict]:
    p = run_dir / "events.jsonl"
    if not p.exists():
        raise SystemExit("not a run dir (no events.jsonl): %s" % run_dir)
    out = []
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if isinstance(e.get("ts"), (int, float)):
                out.append(e)
    if not out:
        raise SystemExit("no timestamped events in %s" % run_dir)
    return out


def read_arm(run_dir: Path) -> dict:
    evs = _events(run_dir)
    span = evs[-1]["ts"] - evs[0]["ts"]

    per_space: collections.Counter = collections.Counter()
    kinds: collections.Counter = collections.Counter()
    gaps: list[tuple[float, str, dict]] = []
    ok = 0
    prev = evs[0]["ts"]
    agent_s = 0.0
    open_calls: dict[str, float] = {}
    rounds = 0
    spaces_expanded = 0
    for e in evs:
        t = e.get("type")
        p = e.get("payload") or {}
        if t == "TRIAL_DONE":
            tr = p.get("trial") or p
            per_space[tr.get("space_id")] += 1
            fk = tr.get("failure_kind") or "ok"
            kinds[fk] += 1
            if fk == "ok" and tr.get("status") == "complete":
                ok += 1
            gaps.append((e["ts"] - prev, fk, (tr.get("params") or {}).get("values") or {}))
        elif t == "AGENT_CALL_STARTED":
            open_calls[str(p.get("call_id") or e.get("seq"))] = e["ts"]
        elif t in ("AGENT_CALL_FINISHED", "AGENT_CALL_FAILED"):
            k = str(p.get("call_id") or "")
            st = open_calls.pop(k, None)
            if st is None and open_calls:
                # `call_id` is present on both sides in v3; the FIFO fallback exists so an older
                # log does not silently contribute 0 h of agent time and read as "all GPU".
                st = open_calls.pop(min(open_calls, key=lambda x: open_calls[x]))
            if st is not None:
                agent_s += e["ts"] - st
        elif t == "FAMILY_ROUND_RECORDED":
            rounds += 1
        elif t == "SPACE_EXPANDED":
            spaces_expanded += 1
        prev = e["ts"]

    n = sum(per_space.values())
    gs = sorted(g for g, _, _ in gaps)
    tail_n = max(1, len(gs) // 10)
    return {
        "span_h": span / 3600.0,
        "trials": n,
        "ok": ok,
        "kinds": dict(kinds),
        "spaces": len(per_space),
        "per_space": dict(per_space),
        "rounds": rounds,
        "expansions": spaces_expanded,
        "trials_per_h": n / (span / 3600.0) if span else 0.0,
        "agent_frac": agent_s / span if span else 0.0,
        "gap_median": statistics.median(gs) if gs else 0.0,
        "gap_sum_h": sum(gs) / 3600.0,
        "tail_share": (sum(sorted(gs, reverse=True)[:tail_n]) / sum(gs)) if sum(gs) else 0.0,
        # Key on the gap alone: a bare `sorted` on the tuples falls through to comparing the params
        # dict whenever two trials share a gap and a failure kind, which raises TypeError.
        "worst": sorted(gaps, key=lambda t: t[0], reverse=True)[:3],
    }


def _mode(counts: list[int]) -> int | None:
    """The per-space budget as the arms actually applied it.

    NOT `max`: a space can exceed the nominal budget (a timeout still counts as a trial), so the max
    reports 41-vs-40 and fires. NOT `min`: mid-run the smallest space is always the one in flight,
    so the min reports 1-vs-40 and fires on every live invocation. The mode is the number most
    spaces agree on, which is what "the budget" means, and it is stable under both.
    """
    if not counts:
        return None
    return collections.Counter(counts).most_common(1)[0][0]


def parity_verdict(a: dict, b: dict, la: str, lb: str) -> tuple[bool, list[str]]:
    """Two independent questions: was the per-space budget applied equally, and did the wall clock
    buy comparable search? A no on either makes a latency difference unattributable."""
    notes: list[str] = []
    okay = True

    ca = sorted(a["per_space"].values())
    cb = sorted(b["per_space"].values())
    ma, mb = _mode(ca), _mode(cb)
    if ma is not None and mb is not None and ma != mb:
        okay = False
        notes.append("PER-SPACE BUDGET DIFFERS: %d vs %d trials per space (modal). The budget "
                     "itself is being applied unequally, which is worse than a rate difference"
                     % (ma, mb))
    else:
        notes.append("per-space budget: %s vs %s (modal count agrees at %s)" % (ca, cb, ma))

    ra, rb = a["trials_per_h"], b["trials_per_h"]
    if ra > 0 and rb > 0:
        ratio = max(ra, rb) / min(ra, rb)
        faster = la if ra > rb else lb
        if ratio > _RATE_TOL:
            # Is the rate gap a TAIL (a few pathological configs, a property of what TPE sampled)
            # or a shifted BODY (every trial slower, a property of the box)? Only the second
            # compounds over a 12 h run. The discriminator is the MEDIAN gap, which is robust to a
            # bounded number of outliers by construction -- no threshold to tune. A first draft used
            # `tail_share > 0.5` and it failed on both sides: the live pair read 68% (correct by
            # luck) while a synthetic single-timeout arm read 48% and was misreported as a confound,
            # because tail_share depends on how many trials the run happens to have.
            gm_a, gm_b = a["gap_median"], b["gap_median"]
            body = max(gm_a, gm_b) / min(gm_a, gm_b) if gm_a > 0 and gm_b > 0 else None
            worse = a if ra < rb else b
            if body is not None and body <= _RATE_TOL:
                notes.append(
                    "rate %.1f vs %.1f trials/h (%.2fx, %s faster) -- but the MEDIAN gap is %.1f vs "
                    "%.1f s (%.2fx, within tolerance), so the body of the distribution agrees and "
                    "this is a TAIL: %s's slowest single trial was %.0f s (%s). A tail does not "
                    "compound over the wall clock"
                    % (ra, rb, ratio, faster, gm_a, gm_b, body,
                       la if ra < rb else lb, worse["worst"][0][0], worse["worst"][0][1]))
            else:
                okay = False
                notes.append(
                    "RATE DIFFERS: %.1f vs %.1f trials/h (%.2fx, %s faster) and the MEDIAN gap "
                    "differs too (%.1f vs %.1f s, %s) => the body of the distribution is shifted, "
                    "not a few bad configs, so the faster arm fits ~%.0f%% more search into the "
                    "same wall clock"
                    % (ra, rb, ratio, faster, gm_a, gm_b,
                       "%.2fx" % body if body is not None else "one side has no gaps",
                       100 * (ratio - 1)))
        else:
            notes.append("rate %.1f vs %.1f trials/h (%.2fx, within %.0f%%): the wall clock buys "
                         "comparable search" % (ra, rb, ratio, 100 * (_RATE_TOL - 1)))
    return okay, notes


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("control", type=Path)
    ap.add_argument("treatment", type=Path)
    args = ap.parse_args(argv)

    rows = [("CONTROL", read_arm(args.control)), ("TREATMENT", read_arm(args.treatment))]
    for lbl, r in rows:
        print("%-10s span %.2f h   %d trials (%d ok) over %d spaces   %.1f trials/h" % (
            lbl, r["span_h"], r["trials"], r["ok"], r["spaces"], r["trials_per_h"]))
        print("%-10s rewrite rounds %d   space expansions %d   agent %.0f%% of wall clock" % (
            "", r["rounds"], r["expansions"], 100 * r["agent_frac"]))
        print("%-10s failure kinds %s" % ("", r["kinds"]))
        print("%-10s inter-event gap: median %.1f s, sum %.2f h, top 10%% hold %.0f%%" % (
            "", r["gap_median"], r["gap_sum_h"], 100 * r["tail_share"]))
        print("%-10s   (gaps are NOT per-trial costs: max_shared_jobs=2 overlaps compile with "
              "timing)" % "")
        for g, fk, vals in r["worst"][:2]:
            keep = {k: vals.get(k) for k in ("COMPUTE_DTYPE", "DOT_MODE") if k in vals}
            print("%-10s   slowest: %.0f s  %-14s %s" % ("", g, fk, keep))
        print()

    print("=" * 78)
    okay, notes = parity_verdict(rows[0][1], rows[1][1], "control", "treatment")
    for n in notes:
        print("  * %s" % n)
    print()
    if okay:
        print("PARITY OK -- a latency difference between these arms is attributable to the switch.")
        return 0
    print("PARITY NOT OK -- record this beside the J2-5 verdict. A difference in the final result "
          "has more than one explanation, and the fix is not a tighter tolerance: it is reporting "
          "the confound alongside the number.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
