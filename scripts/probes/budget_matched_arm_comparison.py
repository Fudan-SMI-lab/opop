"""Does the BUDGET difference explain arm2's 8% lead over arm3? Truncate and see.

WHY. arm2 (vector, no wall steering) finished at 2.49 ms, arm3 (vector + wall steering) at 2.69 ms
-- but arm2 also ran 800 trials / 10 rewrites against arm3's 760 / 8. Two explanations fit that:
the steering hurt, or arm2 simply searched more. They are separable offline, with no GPU time: ask
where arm2 stood at the moment it had spent only arm3's budget.

THE TRUNCATION DIRECTION IS FORCED, and I had it backwards first. arm3 has FEWER trials and FEWER
rewrites, so "truncate arm3 to arm2's count" is not a thing you can do. Only arm2 can be cut down.

WHAT IS COMPARED, and it is not the headline numbers. 2.49 / 2.69 are `final_reeval_ms`, measured
once, at the end, in a fresh process. There is no earlier final_reeval to truncate to, so this probe
compares TUNED best-so-far on both sides -- like for like. Tuned values are systematically optimistic
against a re-evaluation (this project: 1.5-6.7%, and the re-eval gap's SIGN is unstable at +-2-4%),
so a tuned-vs-tuned gap smaller than ~4% carries no ordering information and is reported as such.

TWO BUDGET AXES, REPORTED SEPARATELY. Trials and rewrites are not interchangeable: a rewrite cannot
exist before its parent finished tuning, and this project's champions arrive late (4 of 4 in the
second half, normalised mean 0.78). So "arm2 cut to 760 trials" and "arm2 cut to 8 rewrites" are
different questions and a single number would hide which one bites.

POSITIVE CONTROL. `--selftest` builds one arm whose winner arrives in the LAST 5% of its trials --
truncation must erase that win and say so -- and one whose winner arrives early, where truncation
must change nothing. Without both, a truncation that silently does nothing looks like a finding.
"""
from __future__ import annotations

import json
import os
import sys

_NOISE_PCT = 4.0  # re-evaluation gap on this project, sign unstable


def _read(path: str) -> list[dict]:
    ev = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    ev.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return ev


def _ms(trial: dict) -> float | None:
    """Reproduce LatencyStats.robust_ms: median else mean.

    THE KEYS ARE `median` AND `mean` -- no `_ms` suffix. Writing the suffix makes every trial read
    None, and the probe then reports "no trials" about a run with hundreds.
    """
    lat = trial.get("latency_ms") or {}
    for k in ("median", "mean"):
        v = lat.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def timeline(events: list[dict]) -> dict:
    """best-so-far after each completed trial, plus where each rewrite landed.

    TWO TRIAL COUNTS, and they order the arms OPPOSITELY on the live pair -- which is why both are
    returned and printed. `TRIAL_DONE` fires for failures too (`status: fail`), so:
      * total   = every TRIAL_DONE = the budget the arm SPENT (arm2 800 vs arm3 760)
      * complete= only the ones that produced a latency  = what it got BACK (arm2 617 vs arm3 637)
    arm2 spent more and got less: 183 failures against arm3's 123. Reporting one number would pick a
    winner by choice of denominator, so this reports both and truncates on both.
    """
    n_trials = 0          # complete only -- the curve's x-axis, since only these move best-so-far
    n_all = 0             # every TRIAL_DONE, failures included
    n_rewrites = 0
    best = None
    curve: list[tuple[int, int, int, float]] = []  # (complete_idx, spent_idx, rewrites, best)
    rewrite_at: list[int] = []                 # trial index when the k-th rewrite was produced
    for e in events:
        t = e.get("type")
        if t == "REWRITE_PRODUCED":
            n_rewrites += 1
            rewrite_at.append(n_trials)
            continue
        if t != "TRIAL_DONE":
            continue
        n_all += 1
        tr = (e.get("payload") or {}).get("trial") or {}
        if tr.get("status") != "complete":
            continue
        n_trials += 1
        m = _ms(tr)
        if m is not None and (best is None or m < best):
            best = m
        if best is not None:
            curve.append((n_trials, n_all, n_rewrites, best))
    return {"curve": curve, "rewrite_at": rewrite_at,
            "n_trials": n_trials, "n_all": n_all, "n_rewrites": n_rewrites,
            "final_best": best}


def best_at_trials(tl: dict, limit: int) -> float | None:
    """best-so-far once `limit` COMPLETE trials had been seen."""
    out = None
    for idx, _sp, _rw, best in tl["curve"]:
        if idx <= limit:
            out = best
    return out


def best_at_spent(tl: dict, limit: int) -> float | None:
    """best-so-far once `limit` TRIAL_DONE events had been seen, failures included."""
    out = None
    for _idx, sp, _rw, best in tl["curve"]:
        if sp <= limit:
            out = best
    return out


def best_at_rewrites(tl: dict, limit: int) -> float | None:
    out = None
    for _idx, _sp, rw, best in tl["curve"]:
        if rw <= limit:
            out = best
    return out


def winner_position(tl: dict) -> float | None:
    """Normalised trial index at which the final best was first reached (0=start, 1=end)."""
    if tl["final_best"] is None or not tl["n_trials"]:
        return None
    for idx, _sp, _rw, best in tl["curve"]:
        if best <= tl["final_best"] * (1 + 1e-12):
            return idx / float(tl["n_trials"])
    return None


def _pct(a: float, b: float) -> float:
    """How much better b is than a, in percent."""
    return 100.0 * (a - b) / a


def report(arms: list[tuple[str, dict]]) -> None:
    print("=" * 78)
    print("BUDGET SPENT, and the tuned best each arm reached with it")
    print("  %-8s %7s %8s %6s %8s   %-12s %s"
          % ("arm", "spent", "complete", "fail", "rewrites", "tuned best", "winner at"))
    for label, tl in arms:
        wp = winner_position(tl)
        print("  %-8s %7d %8d %6d %8d   %-12s %s"
              % (label, tl["n_all"], tl["n_trials"], tl["n_all"] - tl["n_trials"],
                 tl["n_rewrites"],
                 "%.4f ms" % tl["final_best"] if tl["final_best"] else "n/a",
                 "%.2f of its trials" % wp if wp is not None else "n/a"))

    if len(arms) != 2:
        return
    (la, ta), (lb, tb) = arms
    if ta["final_best"] is None or tb["final_best"] is None:
        print("  an arm has no complete trial with a latency; nothing to compare.")
        return

    # THE TWO TRIAL COUNTS DISAGREE ABOUT WHO SEARCHED MORE, so say so before truncating: on this
    # pair arm2 spent MORE TRIAL_DONE events but got FEWER usable ones. Picking either silently
    # would decide the question by choice of denominator.
    spent_rich = la if ta["n_all"] >= tb["n_all"] else lb
    comp_rich = la if ta["n_trials"] >= tb["n_trials"] else lb
    print()
    if spent_rich != comp_rich:
        print("!! THE TWO BUDGET AXES ORDER THE ARMS OPPOSITELY:")
        print("   %s spent more trials, but %s got more USABLE ones (failures differ: %d vs %d)."
              % (spent_rich, comp_rich, ta["n_all"] - ta["n_trials"], tb["n_all"] - tb["n_trials"]))
        print("   So \"which arm searched more\" has no single answer, and any truncation below is")
        print("   conditional on which axis you call the budget. Both are reported.")
    else:
        print("Both budget axes agree that %s had more." % spent_rich)

    for axis, key, getter in (("trials SPENT", "n_all", best_at_spent),
                              ("trials COMPLETE", "n_trials", best_at_trials),
                              ("rewrites", "n_rewrites", best_at_rewrites)):
        # Cut whichever arm has MORE on THIS axis down to the other's count.
        if ta[key] >= tb[key]:
            (rl, rich), (ll, lean) = (la, ta), (lb, tb)
        else:
            (rl, rich), (ll, lean) = (lb, tb), (la, ta)
        cut = lean[key]
        cut_best = getter(rich, cut)
        print()
        print("-- axis: %s   (cutting %s down to %s's %d)" % (axis, rl, ll, cut))
        if cut_best is None:
            print("     %s had produced nothing by that point." % rl)
            continue
        d_full = _pct(lean["final_best"], rich["final_best"])
        d_cut = _pct(lean["final_best"], cut_best)
        cost = _pct(rich["final_best"], cut_best)
        print("     %s at the cut: %.4f ms   (its own final %.4f, truncation costs %+.2f%%)"
              % (rl, cut_best, rich["final_best"], -cost))
        print("     %s (untouched): %.4f ms" % (ll, lean["final_best"]))
        print("     %s ahead of %s by %+.2f%% at matched budget  (%+.2f%% unmatched)"
              % (rl, ll, d_cut, d_full))
        if abs(cost) < 1e-9:
            print("     NOTE: truncation changed nothing on this axis -- %s had already reached its"
                  % rl)
            print("           best before the cut, so this axis cannot explain the gap either way.")
        if abs(d_cut) < _NOISE_PCT:
            print("     => within the +-%.0f%% re-eval noise floor: NOT separated on this axis."
                  % _NOISE_PCT)
        elif d_cut > 0:
            print("     => %s still ahead beyond the noise floor after matching." % rl)
        else:
            print("     => %s is ahead beyond the noise floor at matched budget." % ll)

    print()
    print("READ IT WITH THESE LIMITS.")
    print("  * TUNED values on both sides. The headline 2.49/2.69 pair is `final_reeval_ms`, which")
    print("    exists only at the end -- there is no earlier one to truncate to. Tuned is")
    print("    systematically optimistic (1.5-6.7%) and the re-eval SIGN is unstable, so a matched")
    print("    gap under %.0f%% is not an ordering." % _NOISE_PCT)
    print("  * Truncating does NOT re-run the search: the cut arm still had the rewrites and the")
    print("    sampler history it actually had. This answers 'where was it then', not 'what would")
    print("    it have done on a smaller budget'.")
    print("  * n=1 per arm. Nothing here makes the pair significant.")


def _selftest() -> int:
    def _t(ms):
        return {"type": "TRIAL_DONE", "payload": {"trial": {
            "status": "complete", "latency_ms": {"median": ms}}}}

    ok = True
    # Late winner: 20 trials, the win lands at trial 20. Cutting to 19 must erase it.
    late = timeline([_t(10.0)] * 19 + [_t(5.0)])
    if abs(late["final_best"] - 5.0) > 1e-9:
        print("FAIL: late-winner arm did not read its own best (%s)" % late["final_best"])
        ok = False
    elif abs(best_at_trials(late, 19) - 10.0) > 1e-9:
        print("FAIL: truncation did NOT erase a last-trial win -- the cut is inert (%s)"
              % best_at_trials(late, 19))
        ok = False
    elif abs(winner_position(late) - 1.0) > 1e-9:
        print("FAIL: winner position wrong for a last-trial win (%s)" % winner_position(late))
        ok = False
    # Early winner: cutting to 19 must change NOTHING, or the truncation is cutting blindly.
    early = timeline([_t(5.0)] + [_t(10.0)] * 19)
    if abs(best_at_trials(early, 19) - 5.0) > 1e-9:
        print("FAIL: truncation damaged an early win (%s)" % best_at_trials(early, 19))
        ok = False
    elif abs(winner_position(early) - 0.05) > 1e-9:
        print("FAIL: winner position wrong for an early win (%s)" % winner_position(early))
        ok = False
    # Rewrite axis: the win must be attributable to the trial AFTER the 3rd rewrite.
    evs = [_t(10.0), {"type": "REWRITE_PRODUCED", "payload": {}},
           _t(9.0), {"type": "REWRITE_PRODUCED", "payload": {}},
           _t(8.0), {"type": "REWRITE_PRODUCED", "payload": {}}, _t(4.0)]
    tl = timeline(evs)
    if tl["n_rewrites"] != 3:
        print("FAIL: rewrite count wrong (%s)" % tl["n_rewrites"])
        ok = False
    elif abs(best_at_rewrites(tl, 2) - 8.0) > 1e-9:
        print("FAIL: cutting to 2 rewrites did not exclude the post-3rd-rewrite win (%s)"
              % best_at_rewrites(tl, 2))
        ok = False
    # `best_at_spent` must count FAILURES too, or it is a duplicate of best_at_trials and the
    # "two axes disagree" branch below can never fire on real data.
    mixed = timeline([
        {"type": "TRIAL_DONE", "payload": {"trial": {"status": "fail"}}},
        {"type": "TRIAL_DONE", "payload": {"trial": {"status": "fail"}}},
        _t(9.0), _t(4.0)])
    if mixed["n_all"] != 4 or mixed["n_trials"] != 2:
        print("FAIL: spent/complete counts wrong (%s/%s)" % (mixed["n_all"], mixed["n_trials"]))
        ok = False
    elif abs(best_at_spent(mixed, 3) - 9.0) > 1e-9:
        print("FAIL: best_at_spent ignored the two failures (%s)" % best_at_spent(mixed, 3))
        ok = False
    elif abs(best_at_trials(mixed, 1) - 9.0) > 1e-9:
        print("FAIL: best_at_trials mis-indexed complete trials (%s)" % best_at_trials(mixed, 1))
        ok = False
    # And the negative: the two getters must DIFFER at the same numeric limit, else the two axes
    # are the same axis wearing two names.
    if best_at_spent(mixed, 2) is not None:
        print("FAIL: at 2 SPENT trials (both failures) there should be no best yet")
        ok = False
    # The `_ms` suffix trap: a record keyed `median_ms` must read as None, not as a value.
    if _ms({"latency_ms": {"median_ms": 3.0}}) is not None:
        print("FAIL: `median_ms` was accepted; the real key is `median`")
        ok = False
    # An incomplete trial must not enter the curve.
    if timeline([{"type": "TRIAL_DONE", "payload": {"trial": {
            "status": "failed", "latency_ms": {"median": 1.0}}}}])["final_best"] is not None:
        print("FAIL: a failed trial contributed a best")
        ok = False
    print("SELFTEST %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    arms = []
    for a in args:
        path = a if a.endswith(".jsonl") else os.path.join(a, "events.jsonl")
        if not os.path.exists(path):
            print("missing: %s" % path)
            continue
        label = [p for p in path.split(os.sep) if p][-3]
        arms.append((label, timeline(_read(path))))
    if not arms:
        print("no readable arm given")
        raise SystemExit(2)
    report(arms)
