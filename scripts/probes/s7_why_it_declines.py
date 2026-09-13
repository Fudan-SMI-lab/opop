"""Why S7 declines: is the blocker the WALL, or my proposal rule's value-level dedup?

THE MEASUREMENT THAT FORCED THIS PROBE. On the box-4 corpus -- 35 candidates over three 12h L3:43 runs,
every one with a published space and with real `infeasible_shared_memory` refusals -- the shipping
`SlopeGuide` enqueues ONE point in 224 recomputes (2.9% of candidates). 223 recomputes report "no
actionable wall", and the per-knob counter reports 35 skips for "no undrawn value on the walled side".
Yet the feasibility replay finds a probe-worthy hard wall on 8 of those same 35 candidates. Those two
facts can only both be true if the walls ARE found and then discarded while resolving a target.

THE SUSPECT, and it is my own code rather than the criterion. `_toward_wall` requires the proposed CHOICE
to be one no trial has drawn:

    for _, choice in nums:
        if str(choice) not in drawn:      # <-- this
            return choice

The justification I wrote for it was "a drawn value contributes nothing new to `latency_by_value`". That
is wrong in a way worth stating precisely: `latency_by_value` is a MEDIAN over the trials at that value,
so a second measurement at the same value beside DIFFERENT partner knobs does change it -- and the
measurement this mechanism would add is the one taken beside the OPTIMUM's partners, which is exactly the
partner setting the wall's tail slope should be read at. A value measured only beside slow partners
carries a slow median, and that is what makes a wall look worthless.

It also makes the rule structurally self-defeating on this project's domains. A hard wall means the
refused value lies OUTSIDE the measured range, so the tuner has already sampled the whole range; and the
domains hold a median of 4 choices. By the time a wall is detectable, every choice on the walled side has
been drawn -- so the rule can only fire in the narrow window where a wall exists AND a declared choice on
that side happens to be untouched.

WHAT THIS PROBE MEASURES. The same shipping wall criteria and the same anchor and refusal bound, with ONE
difference: the target may be a value that has been drawn before, as long as the resulting POINT (that
value beside the optimum's other knobs) has not been asked. That point-level dedup is the one the tuner
itself uses, and the acceptance run reports ZERO skips from it.

If the relaxed rule fires at a materially higher rate, the blocker is my dedup and not the wall, and the
12h pair is worth running against a fixed mechanism. If it does not, the blocker is the criterion's
applicability and no wiring change rescues it -- which is a result about C2, not about S7.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from kernel_optimizer.evaluation import soft_wall as soft_wall_mod  # noqa: E402
from kernel_optimizer.evaluation import wall_attribution  # noqa: E402
from kernel_optimizer.models.core import ParameterSpace, ParamSet  # noqa: E402
from kernel_optimizer.models.reports import ParamStat, TuningStats  # noqa: E402
from kernel_optimizer.store import read as store_read  # noqa: E402
from kernel_optimizer.tuning import slope_guide as sg  # noqa: E402


class RelaxedGuide(sg.SlopeGuide):
    """The shipping guide with the VALUE-level dedup removed and nothing else changed.

    Subclassed rather than edited so both rules run against the same walls, the same anchor, the same
    refusal bound and the same slope filter in one process -- if the shipping code is wrong about
    something else too, both arms are wrong about it identically and the comparison still isolates the
    dedup.
    """

    def _toward_wall(self, knob, side, drawn, refused_value, incumbent_value):
        domain = next((d for d in self.space.domains if d.name == knob), None)
        if domain is None:
            return None
        here = (wall_attribution._as_num(incumbent_value)
                if incumbent_value is not None else None)
        nums = []
        for c in domain.choices:
            n = wall_attribution._as_num(c)
            if n is None:
                continue
            if refused_value is not None:
                if side == "high" and n >= refused_value:
                    continue
                if side == "low" and n <= refused_value:
                    continue
            if here is not None:
                if side == "high" and n <= here:
                    continue
                if side == "low" and n >= here:
                    continue
            nums.append((n, c))
        if not nums:
            return None
        nums.sort(key=lambda p: p[0], reverse=(side == "high"))
        # The ONLY difference from the shipping rule: the furthest launchable value toward the wall,
        # whether or not some other configuration already used it. The point-level dedup in `suggest`
        # still refuses a point that has actually been asked.
        return nums[0][1]


def _stats_from(trials: list[dict], cid: str) -> TuningStats:
    by_knob: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for t in trials:
        if t.get("status") != "complete":
            continue
        ms = store_read.latency_ms_of(t)
        if ms is None:
            continue
        for knob, raw in (((t.get("params") or {}).get("values")) or {}).items():
            by_knob[str(knob)][str(raw)].append(ms)
    out = []
    for knob, table in sorted(by_knob.items()):
        med = {v: statistics.median(xs) for v, xs in table.items()}
        if not med:
            continue
        best = min(med, key=lambda v: med[v])
        out.append(ParamStat(name=knob, best_value=best, latency_by_value=med))
    return TuningStats(candidate_id=cid, space_id="sp",
                       n_complete=sum(1 for t in trials if t.get("status") == "complete"),
                       n_fail=sum(1 for t in trials if t.get("status") != "complete"),
                       param_stats=out)


def _run_arm(cls, space, trials, every, cap, soft) -> tuple[int, int, float | None, dict]:
    """`(points enqueued, recomputes, first-firing fraction, snapshot)` for one proposal rule."""
    guide = cls(space=space, recompute_every=every, max_enqueued_per_recompute=cap,
                use_soft_wall=soft)
    n = 0
    first: int | None = None
    keys: set[str] = set()
    for i, t in enumerate(trials, start=1):
        vals = ((t.get("params") or {}).get("values")) or {}
        if vals:
            keys.add(ParamSet(values=vals).key())
        if not guide.due(i):
            continue
        prefix = trials[:i]
        got = guide.suggest(_stats_from(prefix, space.candidate_id), prefix, set(keys))
        if got and first is None:
            first = i
        n += len(got)
    frac = (first / len(trials)) if first is not None else None
    return n, guide.n_recomputes, frac, guide.snapshot()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--cap", type=int, default=2)
    ap.add_argument("--soft", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out: list[str] = []

    def say(s: str = "") -> None:
        out.append(s)

    n_cands = 0
    tot = {"strict": 0, "relaxed": 0}
    fired = {"strict": 0, "relaxed": 0}
    fracs: dict[str, list[float]] = {"strict": [], "relaxed": []}
    n_recomputes = 0
    n_walls_found_but_dropped = 0
    rows: list[tuple[str, str, int, int, int]] = []

    for rd in args.runs:
        try:
            events = store_read.read_events(rd)
        except Exception as exc:  # noqa: BLE001
            say(f"!! {rd}: {type(exc).__name__}: {exc}")
            continue
        seq: dict[str, list[dict]] = defaultdict(list)
        published: dict[str, ParameterSpace] = {}
        for ev in events:
            if ev.get("type") == "SPACE_PUBLISHED":
                raw = (ev.get("payload") or ev).get("space") or {}
                try:
                    sp = ParameterSpace.model_validate(raw)
                except Exception:  # noqa: BLE001
                    continue
                published[str(sp.candidate_id)] = sp
                continue
            if ev.get("type") != "TRIAL_DONE":
                continue
            t = store_read.trial_of(ev)
            cid = str(t.get("candidate_id") or "")
            if cid:
                seq[cid].append(t)

        for cid, trials in sorted(seq.items()):
            if len(trials) < args.every or cid not in published:
                continue
            n_cands += 1
            space = published[cid]
            a, nrec, fa, _ = _run_arm(sg.SlopeGuide, space, trials, args.every, args.cap, args.soft)
            b, _, fb, snap_b = _run_arm(RelaxedGuide, space, trials, args.every, args.cap, args.soft)
            n_recomputes += nrec
            tot["strict"] += a
            tot["relaxed"] += b
            if a:
                fired["strict"] += 1
            if b:
                fired["relaxed"] += 1
            if fa is not None:
                fracs["strict"].append(fa)
            if fb is not None:
                fracs["relaxed"].append(fb)
            if b > a:
                n_walls_found_but_dropped += 1
            if a or b:
                rows.append((Path(rd).name, cid, len(trials), a, b))

    say("=" * 100)
    say("WHY S7 DECLINES -- the shipping proposal rule against one with the VALUE-level dedup removed")
    say(f"  recompute_every={args.every}  cap={args.cap}  use_soft_wall={args.soft}")
    say("=" * 100)
    say("Both arms run in the SAME process over the SAME prefixes, with the same wall criteria, the")
    say("same incumbent anchor, the same refusal bound and the same slope filter. The only difference")
    say("is whether the proposed VALUE may be one some other configuration already used. The")
    say("POINT-level dedup (the tuner's own) is enforced in both.")
    say()
    say(f"candidates (with a published space, >= {args.every} trials): {n_cands}")
    say(f"recomputes: {n_recomputes}")
    say()
    say(f"{'':34s}{'SHIPPING':>12s}{'RELAXED':>12s}")
    say(f"{'candidates where it ever fired':34s}{fired['strict']:>12d}{fired['relaxed']:>12d}")
    if n_cands:
        say(f"{'   ...as a share':34s}"
            f"{100.0 * fired['strict'] / n_cands:>11.1f}%{100.0 * fired['relaxed'] / n_cands:>11.1f}%")
    say(f"{'points it would enqueue':34s}{tot['strict']:>12d}{tot['relaxed']:>12d}")
    for arm in ("strict", "relaxed"):
        if fracs[arm]:
            label = "median first firing" if arm == "strict" else ""
            say(f"{label:34s}" +
                (f"{statistics.median(fracs['strict']):>12.2f}" if arm == "strict" else "") +
                (f"{statistics.median(fracs['relaxed']):>12.2f}" if arm == "relaxed" else ""))
    say()
    say(f"candidates where the relaxed rule fires and the shipping one does not: "
        f"{n_walls_found_but_dropped}")
    say()
    if tot["relaxed"] > 2 * max(tot["strict"], 1):
        say("READING: the blocker is the VALUE-LEVEL DEDUP, not the wall. The walls are found and then")
        say("  discarded while resolving a target, because a hard wall means the refused value lies")
        say("  OUTSIDE the measured range -- i.e. the tuner has already sampled the range -- and the")
        say("  domains hold a median of 4 choices. The mechanism is worth fixing before the 12h pair.")
    else:
        say("READING: the blocker is NOT the dedup. Both rules fire at a similar rate, so what is")
        say("  missing is the WALL itself, and no wiring change rescues it. That is a result about C2's")
        say("  criterion applicability, not about S7 -- and it means the 12h pair would measure a")
        say("  treatment arm nearly identical to its control.")
    if rows:
        say()
        say("PER CANDIDATE (any arm fired):")
        say("-" * 100)
        say(f"{'run':30s} {'candidate':15s} {'trials':>7s} {'shipping':>9s} {'relaxed':>8s}")
        say("-" * 100)
        for run, cid, n, a, b in rows[:40]:
            say(f"{run[:30]:30s} {cid[:15]:15s} {n:7d} {a:9d} {b:8d}")
        say("-" * 100)

    text = "\n".join(out) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        sys.stdout.buffer.write(text.encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
