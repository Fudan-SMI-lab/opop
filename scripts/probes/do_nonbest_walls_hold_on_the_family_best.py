"""Item 5: would widening wall delivery beyond the family's best candidate ADD evidence or NOISE?

THE QUESTION, AND WHY IT DECIDES SOMETHING. `orchestrator.py:2932` hands the rewriter
`parent_crun.wall_text`, where the parent is `family.best.candidate_id` -- so a wall found on a
NON-best member of a family is journalled and then never reaches any prompt. Measured on the
three-arm run, only 25% of families ever received wall text. The obvious fix is "deliver the whole
family's walls", and it is exactly the kind of change that looks free and is not: if a non-best
candidate's wall does not exist on the best candidate's source, the text would describe a knob the
rewriter is not editing, at a value it cannot reach, in a kernel it is not looking at.

So the number to get first is: OF THE WALLS FOUND ON A FAMILY'S NON-BEST MEMBERS, HOW MANY STILL
HOLD ON THE BEST MEMBER? Three outcomes, and they are not two:

  TRANSFERS    the best member has the same knob, its measured range is truncated at the same side,
               and the refused value is still outside that range. Delivering it adds a real datum.
  DIMENSION GONE
               the best member has no such knob. Its space was published from a different source --
               a rewrite may have removed the tile loop entirely. Counted apart from both verdicts
               on purpose: as failure it slanders a rewrite that removed the constraint by removing
               the knob; as success it credits one for deleting the evidence. Same three-way rule
               `wall_report.freed_lines` already uses.
  DOES NOT HOLD
               the knob exists and the refused value is INSIDE the best member's measured range --
               that member reached both sides of it, so nothing is truncated there. Delivering it
               would be noise: a wall claimed where the tuner demonstrably ran the value.

ZERO GPU. Everything here is read from events.jsonl through `store/read.py`, which is the project's
single reader for these field paths (`payload.trial`, median-else-mean, no `robust_ms` in the log).

WHAT THIS PROBE CANNOT DO ON THIS HOST, STATED UP FRONT. Every run backed up locally predates the
shared-memory screen, so `failure_kind == "infeasible_shared_memory"` appears ZERO times (verified
per run) and `find_walls`' only input is missing -- an offline replay of the real hard wall yields 0
walls, as `hard_vs_soft_wall_same_corpus.py` already recorded. The transfer question does not depend
on which criterion produced the wall, though: it depends only on whether one candidate's truncated
knob is also truncated on another candidate. So this probe measures the transfer rate on
BOUNDARY-TRUNCATED knobs -- knobs whose own `TuningStats` says the winning value sits at the extreme
choice with a monotone approach (`at_boundary`, which the harness computes and 2e's `find_walls`
requires as well) -- and reports it as such. It is the same geometric question on a signal this
corpus HAS. A run carrying real refusals must re-run this with `--source refusals` before the number
is quoted as being about shared-memory walls.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from kernel_optimizer.evaluation import wall_attribution as wa  # noqa: E402
from kernel_optimizer.store import read as store_read  # noqa: E402


def _num(v):
    return wa._as_num(v)


def _families(events: list[dict]) -> tuple[dict[str, str], dict[str, str]]:
    """`(candidate -> family, family -> best candidate)`.

    The best candidate comes from `RUN_FINISHED.summary.families[*].best_candidate` when present and
    is otherwise DERIVED from the members' own completed trials, because the summary of an older run
    records only `best_ms`. Derived the same way the framework selects: lowest median-else-mean over
    completed trials. A family whose best cannot be established is skipped rather than guessed at --
    guessing it would silently make every wall in that family "transfer" or "not", at random.
    """
    fam_of: dict[str, str] = {}
    for ev in events:
        if ev.get("type") != "CANDIDATE_REGISTERED":
            continue
        c = (ev.get("payload") or {}).get("candidate") or {}
        if c.get("candidate_id") and c.get("family_id"):
            fam_of[str(c["candidate_id"])] = str(c["family_id"])

    best_of: dict[str, str] = {}
    for ev in events:
        if ev.get("type") != "RUN_FINISHED":
            continue
        fams = ((ev.get("payload") or {}).get("summary") or {}).get("families") or {}
        for fid, f in fams.items():
            cid = f.get("best_candidate") or f.get("best", {}).get("candidate_id")
            if cid:
                best_of[str(fid)] = str(cid)
    return fam_of, best_of


def _per_candidate_curves(events: list[dict]) -> dict[str, dict[str, dict[float, float]]]:
    """`candidate -> knob -> {value: median latency}` from the trials themselves.

    Built from trials rather than from `STATS_DONE.param_stats.latency_by_value` on purpose: a
    candidate whose space was EXPANDED emits several STATS_DONE records, and taking any one of them
    would report a range narrower than the candidate actually measured -- which is the exact input
    `find_walls` compares a refused value against, so a stale range manufactures walls.
    """
    buckets: dict[str, dict[str, dict[float, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    for ev in events:
        if ev.get("type") != "TRIAL_DONE":
            continue
        t = store_read.trial_of(ev)
        if t.get("status") != "complete":
            continue
        ms = store_read.latency_ms_of(t)
        cid = t.get("candidate_id")
        if ms is None or not cid:
            continue
        for knob, raw in (((t.get("params") or {}).get("values")) or {}).items():
            v = _num(raw)
            if v is not None:
                buckets[str(cid)][str(knob)][v].append(ms)
    return {cid: {k: {v: statistics.median(xs) for v, xs in vals.items()}
                  for k, vals in knobs.items()}
            for cid, knobs in buckets.items()}


def _next_step(tail: list[float]) -> float | None:
    """The value one step past `tail[-1]`, in the ladder the candidate's own values form.

    GEOMETRIC vs ARITHMETIC matters and the first version of this probe got it wrong in both
    directions, which is why it is its own function with its own reasoning:

      * low side of [16, 32, 64]: an arithmetic step gives 16 - 16 = 0. A tile size of 0 is not a
        value any space offers, and it lies below EVERY range, so every such knob read "transfers".
        That is a constant masquerading as a measurement -- the failure shape recorded as
        `a-constant-reading-is-a-broken-probe`. The geometric step gives 8, a real candidate value.
      * high side of [2, 4, 8, 16]: arithmetic gives 24, which lands INSIDE another candidate's
        [2, 4, 8, 16] and reads "does not hold". Geometric gives 32, which is outside and reads
        "transfers". So the ladder choice flips verdicts both ways and cannot be waved through.

    Tile sizes, warp counts and stage depths in this corpus are overwhelmingly powers of two, so the
    ladder is detected rather than assumed: constant ratio (within 1%) => geometric, else the last
    measured difference. Returns None when the result is not a value a space could hold, so the
    truncation is DROPPED instead of contributing a free transfer. Two such cases, both found by
    reading the probe's own examples rather than by reasoning:

      * `step <= 0` -- the arithmetic-ladder failure described above.
      * a FRACTIONAL step on an all-integer ladder. `NUM_WARPS` measured at [1, 2, 4, 8] gives a
        geometric low-side rung of 0.5, and half a warp is not a configuration; every such knob
        whose low edge is 1 would otherwise hand back a guaranteed "transfers", since 0.5 lies below
        every integer range. The same shape of free positive as the `0` case, one rung further in.
    """
    if len(tail) < 2:
        return None
    a, b = tail[-2], tail[-1]
    ratios = [tail[i + 1] / tail[i] for i in range(len(tail) - 1) if tail[i]]
    geometric = (len(ratios) >= 2
                 and all(r and abs(r / ratios[0] - 1.0) < 0.01 for r in ratios))
    step = b * (b / a) if (geometric and a) else b + (b - a)
    if step <= 0:
        return None
    if all(float(v).is_integer() for v in tail) and not float(step).is_integer():
        return None
    return step


def _truncations(curves: dict[str, dict[float, float]]) -> tuple[list[dict], int]:
    """Every knob of one candidate whose measured optimum sits AT an extreme with a monotone tail.

    The stand-in for a refused value on a corpus with no refusals, and the same geometry `find_walls`
    tests: a knob whose best measured value is the largest (or smallest) one tried, approached
    monotonically, is a knob whose range the tuner would have wanted to extend. The synthetic
    "refused value" is the next rung of the candidate's OWN ladder (see `_next_step`), never a
    constant -- a hardcoded step answers a different question on every space.

    Returns `(truncations, n_dropped)`; the drop count is reported so a reader can see how much the
    ladder rule removed rather than having to trust that it did something.
    """
    out: list[dict] = []
    dropped = 0
    for knob, by_value in curves.items():
        xs = sorted(by_value)
        if len(xs) < wa._MIN_RAN_VALUES:
            continue
        best = min(xs, key=lambda v: by_value[v])
        for side, tail in (("high", xs[-wa._TAIL_LEN:]), ("low", xs[:wa._TAIL_LEN][::-1])):
            edge = tail[-1]
            if best != edge:
                continue
            lats = [by_value[v] for v in tail]
            if not all(lats[i + 1] < lats[i] for i in range(len(lats) - 1)):
                continue
            target = _next_step(tail)
            if target is None:
                dropped += 1
                continue
            gain = (lats[0] - lats[-1]) / lats[0] * 100.0 if lats[0] else 0.0
            out.append({"param": knob, "side": side, "target": target, "ran": xs,
                        "tail_gain_pct": gain})
    return out, dropped


def _fmt(v: float) -> str:
    """A knob value as a space would spell it. `f"{0.5:.0f}"` prints "0", which made a legitimate
    half-step rung look like the impossible target the ladder rule exists to reject."""
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def _holds_on(target: float, side: str, xs: list[float]) -> str:
    if len(xs) < wa._MIN_RAN_VALUES:
        return "too_few_values"
    if side == "high":
        return "transfers" if target > max(xs) else "does_not_hold"
    return "transfers" if target < min(xs) else "does_not_hold"


def _self_check() -> list[str]:
    """Positive and negative controls on the two pieces that decide every verdict.

    Without this the probe could report any transfer rate and there would be no way to tell a real
    measurement from a reader that answers the same thing for every input -- and this probe already
    produced exactly that failure once, when an arithmetic step turned [16, 32, 64]'s low side into
    a target of 0, which lies below every range and read "transfers" unconditionally.
    """
    out = ["SELF-CHECK (must all pass, or every number below is void):"]
    ok = True

    def check(label: str, got, want) -> None:
        nonlocal ok
        good = got == want
        ok = ok and good
        out.append(f"  {'PASS' if good else 'FAIL'}  {label}: got {got!r}, want {want!r}")

    # The ladder. A geometric ladder must step geometrically in BOTH directions.
    check("next rung above [16, 32, 64]", _next_step([16.0, 32.0, 64.0]), 128.0)
    check("next rung below [64, 32, 16]", _next_step([64.0, 32.0, 16.0]), 8.0)
    check("arithmetic ladder [1, 2, 3] steps by 1", _next_step([1.0, 2.0, 3.0]), 4.0)
    check("a step to zero is refused", _next_step([2.0, 1.0]), None)
    check("a fractional rung on an integer ladder is refused",
          _next_step([4.0, 2.0, 1.0]), None)
    check("...but a fractional ladder keeps its own rungs",
          _next_step([2.0, 1.0, 0.5]), 0.25)

    # The transfer verdict, both ways, on ranges taken from the corpus.
    check("128 vs best-ran [16,32,64] high", _holds_on(128.0, "high", [16.0, 32.0, 64.0]),
          "transfers")
    check("32 vs best-ran [16,32,64] high", _holds_on(32.0, "high", [16.0, 32.0, 64.0]),
          "does_not_hold")
    check("8 vs best-ran [16,32,64] low", _holds_on(8.0, "low", [16.0, 32.0, 64.0]), "transfers")
    check("32 vs best-ran [16,32,64] low", _holds_on(32.0, "low", [16.0, 32.0, 64.0]),
          "does_not_hold")

    # The truncation finder: a knob whose optimum is in the MIDDLE is not a truncation, and one at
    # the edge with a monotone tail is. A finder that returned everything, or nothing, would give a
    # plausible-looking rate either way.
    at_edge, _ = _truncations({"K": {16.0: 3.0, 32.0: 2.0, 64.0: 1.0}})
    check("optimum at the top edge is a truncation", [t["param"] for t in at_edge], ["K"])
    check("...and its target is the next rung", [t["target"] for t in at_edge], [128.0])
    middle, _ = _truncations({"K": {16.0: 3.0, 32.0: 1.0, 64.0: 2.0}})
    check("optimum in the middle is NOT a truncation", middle, [])
    flat, _ = _truncations({"K": {16.0: 2.0, 32.0: 2.0, 64.0: 2.0}})
    check("a flat curve is NOT a truncation", flat, [])
    out.append(f"  => {'all controls pass' if ok else 'CONTROLS FAILED -- do not quote the rate'}")
    out.append("")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    lines: list[str] = []

    def say(s: str = "") -> None:
        lines.append(s)

    totals = defaultdict(int)
    per_run: list[tuple[str, dict]] = []
    examples: list[str] = []

    for rd in args.runs:
        try:
            events = store_read.read_events(rd)
        except Exception as exc:  # noqa: BLE001
            say(f"!! {rd}: {type(exc).__name__}: {exc}")
            continue
        refusals = sum(1 for ev in events if ev.get("type") == "TRIAL_DONE"
                       and store_read.trial_of(ev).get("failure_kind")
                       == "infeasible_shared_memory")
        fam_of, best_of = _families(events)
        curves = _per_candidate_curves(events)

        # Derive the family best when the summary did not name it.
        if not best_of:
            fam_best_ms: dict[str, tuple[float, str]] = {}
            for ev in events:
                if ev.get("type") != "TRIAL_DONE":
                    continue
                t = store_read.trial_of(ev)
                if t.get("status") != "complete":
                    continue
                ms = store_read.latency_ms_of(t)
                cid = str(t.get("candidate_id") or "")
                fid = fam_of.get(cid)
                if ms is None or not fid:
                    continue
                cur = fam_best_ms.get(fid)
                if cur is None or ms < cur[0]:
                    fam_best_ms[fid] = (ms, cid)
            best_of = {fid: cid for fid, (_, cid) in fam_best_ms.items()}

        counts = defaultdict(int)
        counts["refusals_in_log"] = refusals
        by_family: dict[str, list[str]] = defaultdict(list)
        for cid, fid in fam_of.items():
            by_family[fid].append(cid)

        for fid, members in sorted(by_family.items()):
            best = best_of.get(fid)
            if not best or best not in curves:
                # A family with no member holding a completed, timed trial. It cannot carry a wall on
                # ANY member, so skipping it cannot bias the transfer rate in either direction -- but
                # the count is reported because a skip whose reason is unstated is indistinguishable
                # from a reader that failed to find data it should have found.
                counts["families_skipped_no_best"] += 1
                if not any(c in curves for c in members):
                    counts["families_skipped_no_trials_at_all"] += 1
                continue
            counts["families"] += 1
            # Does the BEST candidate have a high-side truncation of its own? That decides whether a
            # transferring non-best wall would ADD a family to the delivered set or merely add a row
            # to a family that was already covered -- and the coverage figure the change is meant to
            # improve (25% of families) counts FAMILIES, not rows.
            best_trs, _ = _truncations(curves[best])
            best_has_own = any(t["side"] == "high" and t["tail_gain_pct"] > 0 for t in best_trs)
            counts["families_best_has_own_high_wall"] += 1 if best_has_own else 0
            family_gains = False
            nonbest = [c for c in members if c != best and c in curves]
            counts["nonbest_members_with_trials"] += len(nonbest)
            for cid in nonbest:
                trs, dropped = _truncations(curves[cid])
                counts["dropped_no_next_rung"] += dropped
                for tr in trs:
                    if tr["tail_gain_pct"] <= 0:
                        counts["worthless_skipped"] += 1
                        continue
                    counts["nonbest_walls"] += 1
                    counts[f"side_{tr['side']}"] += 1
                    best_curves = curves[best]
                    if tr["param"] not in best_curves:
                        counts["dimension_gone"] += 1
                        counts[f"dimension_gone_{tr['side']}"] += 1
                        verdict = "dimension_gone"
                    else:
                        verdict = _holds_on(tr["target"], tr["side"],
                                            sorted(best_curves[tr["param"]]))
                        counts[verdict] += 1
                        counts[f"{verdict}_{tr['side']}"] += 1
                        if verdict == "transfers" and tr["side"] == "high":
                            family_gains = True
                    if len(examples) < 12:
                        examples.append(
                            f"  {Path(rd).name[:28]:28s} {fid[:12]} {cid[:13]} "
                            f"{tr['param']:16s} {tr['side']:4s} "
                            f"target {_fmt(tr['target']):>7s}  best-ran "
                            f"{[_fmt(v) for v in sorted(best_curves.get(tr['param'], {}))] or '(no such knob)'}"
                            f"  -> {verdict}")
            if family_gains and not best_has_own:
                counts["families_newly_covered"] += 1
            elif family_gains:
                counts["families_already_covered_gain_rows"] += 1
        per_run.append((Path(rd).name, dict(counts)))
        for k, v in counts.items():
            totals[k] += v

    say("=" * 100)
    say("ITEM 5 -- do a family's NON-BEST candidates' walls still hold on the BEST candidate?")
    say("=" * 100)
    say()
    lines.extend(_self_check())
    say("Signal used: BOUNDARY-TRUNCATED knobs (the candidate's own optimum sits at an extreme")
    say("choice with a monotone tail). Reason, stated because it changes how the number may be")
    say("quoted: every locally backed-up run predates the shared-memory screen, so")
    say(f"`infeasible_shared_memory` appears {totals['refusals_in_log']} times across these runs and")
    say("the real hard wall has NO input to replay. The transfer question is geometric -- is the same")
    say("knob truncated on the other candidate -- so it is answerable on this signal, but the number")
    say("is about boundary truncations, NOT about measured shared-memory refusals.")
    say()
    say("-" * 100)
    say(f"{'run':32s} {'fams':>5s} {'nonbest':>8s} {'walls':>6s} {'transf':>7s} "
        f"{'no-hold':>8s} {'dim-gone':>9s}")
    say("-" * 100)
    for name, c in per_run:
        say(f"{name[:32]:32s} {c.get('families', 0):5d} "
            f"{c.get('nonbest_members_with_trials', 0):8d} {c.get('nonbest_walls', 0):6d} "
            f"{c.get('transfers', 0):7d} {c.get('does_not_hold', 0):8d} "
            f"{c.get('dimension_gone', 0):9d}")
    say("-" * 100)
    say(f"{'TOTAL':32s} {totals['families']:5d} {totals['nonbest_members_with_trials']:8d} "
        f"{totals['nonbest_walls']:6d} {totals['transfers']:7d} "
        f"{totals['does_not_hold']:8d} {totals['dimension_gone']:9d}")
    say()

    n = totals["nonbest_walls"]
    if n:
        t, d, g = totals["transfers"], totals["does_not_hold"], totals["dimension_gone"]
        say(f"TRANSFER RATE: {t}/{n} = {100.0 * t / n:.1f}%   "
            f"(does not hold {100.0 * d / n:.1f}%, dimension gone {100.0 * g / n:.1f}%)")
        say()
        # The HIGH side on its own, because that is the side a shared-memory wall lives on: the
        # compiler refuses a LARGER tile, not a smaller one. A rate pooled over both sides would let
        # the low side -- where "one rung smaller than anything tried" is nearly always outside the
        # other candidate's range too, so transfers are cheap -- carry the headline number.
        # Pooling classes with different base rates has already produced a Simpson reversal in this
        # project (`pooling-register-classes-inverts-correlations`), so the split is not optional.
        for side in ("high", "low"):
            ns = totals[f"side_{side}"]
            if not ns:
                continue
            tr_s, nh_s, dg_s = (totals[f"transfers_{side}"], totals[f"does_not_hold_{side}"],
                                totals[f"dimension_gone_{side}"])
            say(f"  {side:4s} side only: transfers {tr_s}/{ns} = {100.0 * tr_s / ns:.1f}%"
                f"   (does not hold {nh_s}, dimension gone {dg_s})")
            # The DELIVERABLE denominator. `dimension gone` walls cannot be delivered under any
            # policy -- the knob is absent from the source the rewriter edits -- so including them
            # understates the decision at hand, which is only ever about the walls that COULD be
            # delivered. Both denominators are printed because they answer different questions and
            # quoting one as the other is how "44% dimension gone" would silently become evidence
            # against widening delivery.
            deliverable = tr_s + nh_s
            if deliverable:
                say(f"       of the {deliverable} whose knob still EXISTS on the best candidate: "
                    f"{tr_s} transfer = {100.0 * tr_s / deliverable:.1f}%")
        say("  ** the HIGH side is the one that matters for a shared-memory wall: the compiler")
        say("     refuses a LARGER tile. Quote that row, not the pooled one. **")
        say()
        # The number the change is actually FOR. 25% family coverage was the complaint, and coverage
        # counts FAMILIES: a transferring wall in a family whose best candidate already has its own
        # high-side wall adds a row to a prompt that existed, not a prompt that did not.
        fam = totals["families"]
        own = totals["families_best_has_own_high_wall"]
        new = totals["families_newly_covered"]
        extra_rows = totals["families_already_covered_gain_rows"]
        pct = (lambda x: f" ({100.0 * x / fam:.0f}%)") if fam else (lambda x: "")
        say("WHAT IT WOULD DO TO FAMILY COVERAGE (the figure the change is for):")
        say(f"  families examined                                     : {fam}")
        say(f"  ...whose BEST candidate already has its own high wall : {own}{pct(own)}")
        say(f"  ...NEWLY covered by a transferring non-best wall      : {new}{pct(new)}")
        say(f"  ...already covered, would just gain extra rows        : {extra_rows}")
        say()
        say("HOW TO READ IT:")
        say("  A high transfer rate means widening delivery beyond the family's best candidate ADDS")
        say("  evidence -- the wall describes a truncation the rewriter's own source really has.")
        say("  A low one means it adds NOISE: the text would name a knob at a value the best")
        say("  candidate demonstrably ran, and the rewriter would be sent after a non-problem.")
        say("  `dimension gone` is neither: the member's knob does not exist on the best source, so")
        say("  there is nothing to deliver -- counted apart, per `wall_report.freed_lines`' own rule.")
    else:
        say("NO non-best wall found at all. That is itself the answer for this corpus: with no")
        say("transferable wall, widening delivery could not have changed any prompt here.")
    say(f"  (skipped as worthless -- tail not improving: {totals['worthless_skipped']};")
    say(f"   dropped -- no next rung on the candidate's own ladder: "
        f"{totals['dropped_no_next_rung']};")
    say(f"   families with no establishable best: {totals['families_skipped_no_best']}, of which "
        f"{totals['families_skipped_no_trials_at_all']} have no completed trial on ANY member")
    say("   -- such a family cannot carry a wall on any member, so the skip does not bias the rate)")
    if examples:
        say()
        say("EXAMPLES (first 12):")
        lines.extend(examples)

    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        sys.stdout.buffer.write(text.encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
