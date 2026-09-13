"""Read the S7 pair against the predictions declared BEFORE it ran (P1-P5).

WHY A DEDICATED READER, and not just "look at best_ms". The pair's primary outcomes are not latency.
S7 moves the wall search inside the tuning loop, so what it can change first is WHEN a wall becomes
known and HOW MANY are found -- and those are the predictions that were declared in the two config
headers before launch. Latency is secondary and was declared under-powered in advance: a paired 12 h
run on this task has a 2.35% noise floor and a within-arm spread of 4.72% was measured on G9, so a
latency difference smaller than that is not a result in either direction.

WHAT EACH PREDICTION IS READ FROM, and every one is a fact on disk rather than a judgement:

  P1  the first `RESOURCE_WALL_ATTRIBUTED` carrying a probe-worthy wall arrives EARLIER, normalised to
      the candidate's own tuning progress. Normalised, because the arms do not complete the same number
      of trials in 12 h and wall-clock seconds would compare the boxes' speed instead of the mechanism.
  P2  probe-worthy walls per candidate INCREASE (`walls_found` minus `walls_worthless`, per candidate
      that had a space).
  P3  the share of FAMILIES that received wall text rises above 25%. Read from whether the family's
      best candidate produced a deliverable wall at the moment a rewrite was issued -- which is what
      `_rewrite_round` actually hands the rewriter.
  P4  if P1-P3 hold and latency does not improve, the bottleneck is the REWRITE's conversion rate, not
      the timing of the knowledge. Reported as a verdict line, never as a silent omission.
  P5  if P1 holds and P2/P3 fail, the binding constraint is the criterion's APPLICABILITY -- neither
      timing nor dose -- because the preflight measured 181 of 224 recomputes finding no actionable
      wall at all. This one was added BY the preflight and is the reading most likely to be needed.

WHAT THIS DELIBERATELY DOES NOT DO. It does not decide whether S7 "worked". It prints each prediction's
two numbers with the noise floor beside them and states which of P4/P5 the pattern selects. The
distinction matters because the failure mode here is not a wrong number, it is a number read as support
for whichever story is convenient afterwards -- which is exactly what declaring P1-P5 in advance was
meant to prevent.

PARITY IS NOT CHECKED HERE. `scripts/check_arm_search_parity.py` already does it (trials per wall-clock
hour, spaces reached, rewrite rounds) and `scripts/audit_arm_comparability.py` compares the configs
field by field. Run both first: if the arms did not get comparable search, every number below has two
explanations and none of them is the switch.

    python scripts/analyze_s7_pair.py <treatment_run_dir> <control_run_dir> [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# This task's measured re-evaluation noise floor, and the within-arm spread measured on G9. A latency
# difference under either is not evidence in either direction.
NOISE_FLOOR_PCT = 2.35
WITHIN_ARM_SPREAD_PCT = 4.72
# The figure P3 has to beat: the share of families that received wall text on the three-arm run.
P3_BASELINE = 0.25


def _events(run_dir: Path) -> list[dict]:
    p = run_dir / "events.jsonl"
    if not p.exists():
        raise SystemExit(f"not a run dir (no events.jsonl): {run_dir}")
    out = []
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    if not out:
        raise SystemExit(f"no parseable events in {run_dir}")
    return out


def _pl(e: dict) -> dict:
    return e.get("payload") or e


def _trial_of(e: dict) -> dict:
    return _pl(e).get("trial") or {}


def _robust_ms(t: dict) -> float | None:
    """Median else mean, reproduced because `robust_ms` is a @property and never serialized."""
    lat = t.get("latency_ms")
    if not isinstance(lat, dict):
        return None
    for key in ("median", "mean"):
        v = lat.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
            return float(v)
    return None


def _worthy(p: dict) -> int:
    """Probe-worthy walls in one RESOURCE_WALL_ATTRIBUTED payload.

    `walls_found` counts every truncation, including the ones the slope filter drops as worthless
    (measured on box 2: 3 of 6). P2 is about walls worth ACTING on, so the worthless ones are
    subtracted rather than quietly included -- including them would let a run look better by finding
    more walls it then correctly ignored.
    """
    found = p.get("walls_found")
    worthless = p.get("walls_worthless")
    if not isinstance(found, int):
        return 0
    return found - (worthless if isinstance(worthless, int) else 0)


def _arm(run_dir: Path) -> dict:
    ev = _events(run_dir)
    out: dict = {"dir": run_dir.name}

    # --- trial sequence per candidate, in log order, for the P1 normalisation ------------------
    seq: dict[str, list[int]] = defaultdict(list)      # candidate -> event indices of its trials
    for i, e in enumerate(ev):
        if e.get("type") != "TRIAL_DONE":
            continue
        cid = str(_trial_of(e).get("candidate_id") or "")
        if cid:
            seq[cid].append(i)
    out["n_candidates_with_trials"] = len(seq)
    out["n_trials"] = sum(len(v) for v in seq.values())

    # --- P1: first probe-worthy wall, normalised to the candidate's own trial progress ---------
    # Normalised POSITION rather than wall-clock: the arms complete different numbers of trials in
    # 12 h, so seconds would compare the boxes and not the mechanism.
    first_fracs: list[float] = []
    per_cand_worthy: dict[str, int] = {}
    for i, e in enumerate(ev):
        if e.get("type") != "RESOURCE_WALL_ATTRIBUTED":
            continue
        p = _pl(e)
        cid = str(p.get("candidate_id") or "")
        w = _worthy(p)
        per_cand_worthy[cid] = per_cand_worthy.get(cid, 0) + w
        if w <= 0 or cid not in seq or cid in out.get("_p1_seen", ()):
            continue
        idx = seq[cid]
        # How far through this candidate's own trials the attribution landed. 2e runs after tuning
        # ends, so the control's value is ~1.0 by construction; S7's whole claim is that the
        # mechanism KNOWS earlier, which shows up as an earlier recompute inside the loop -- so this
        # is read from SLOPE_GUIDE_STEP for the treatment as well (below), and the 2e figure is kept
        # for both so the comparison is like-for-like.
        done_before = sum(1 for j in idx if j < i)
        first_fracs.append(done_before / len(idx) if idx else 1.0)
        out.setdefault("_p1_seen", set()).add(cid)
    out["p1_2e_first_frac"] = statistics.median(first_fracs) if first_fracs else None
    out["p1_2e_n"] = len(first_fracs)

    # --- P1 (the mechanism's own clock): when did S7 first have a wall to act on? --------------
    sg_first: list[float] = []
    sg_steps = 0
    sg_enq = 0
    sg_ref = 0
    sg_sources: dict[str, int] = defaultdict(int)
    sg_seen: set[str] = set()
    for e in ev:
        if e.get("type") != "SLOPE_GUIDE_STEP":
            continue
        p = _pl(e)
        sg_steps += 1
        enq = p.get("enqueued") or []
        sg_enq += len(enq)
        sg_ref += len(p.get("refused") or [])
        for row in enq:
            sg_sources[str(row.get("source"))] += 1
        cid = str(p.get("candidate_id") or "")
        n_told = p.get("n_told")
        budget = p.get("budget")
        if enq and cid not in sg_seen and isinstance(n_told, int) and isinstance(budget, int) \
                and budget > 0:
            sg_first.append(n_told / budget)
            sg_seen.add(cid)
    out["s7_steps"] = sg_steps
    out["s7_enqueued"] = sg_enq
    out["s7_refused"] = sg_ref
    out["s7_sources"] = dict(sorted(sg_sources.items()))
    out["s7_first_frac"] = statistics.median(sg_first) if sg_first else None
    out["s7_candidates_fired"] = len(sg_seen)
    out["s7_failed"] = sum(1 for e in ev if e.get("type") == "SLOPE_GUIDE_FAILED")

    # --- P2: probe-worthy walls per candidate --------------------------------------------------
    n_spaces = sum(1 for e in ev if e.get("type") == "SPACE_PUBLISHED")
    worthy_vals = list(per_cand_worthy.values())
    out["n_spaces"] = n_spaces
    out["p2_candidates_probed"] = len(per_cand_worthy)
    out["p2_total_worthy"] = sum(worthy_vals)
    out["p2_per_candidate"] = (sum(worthy_vals) / len(worthy_vals)) if worthy_vals else None
    out["p2_candidates_with_any"] = sum(1 for v in worthy_vals if v > 0)
    # WAS THE INSTRUMENT EVEN ON? A run with 2e disabled emits NO RESOURCE_WALL_ATTRIBUTED, and every
    # P1/P2/P3 figure then reads 0 -- identical to "2e ran and found nothing". Those are different
    # facts, and reading the second as the first would make the pair's whole comparison meaningless in
    # the direction that looks like a result. Caught on a real pair of finished runs: two inherited
    # box-2 runs have 0 attributions despite 260 and 142 shared-memory refusals, purely because the
    # switch was off. The refusal count is the discriminator -- refusals present with zero attributions
    # can only mean the instrument was off (or crashed, which has its own event).
    out["n_refusals"] = sum(1 for e in ev if e.get("type") == "TRIAL_DONE"
                            and _trial_of(e).get("failure_kind") == "infeasible_shared_memory")
    out["n_wall_events"] = sum(1 for e in ev if e.get("type") == "RESOURCE_WALL_ATTRIBUTED")
    out["n_wall_failed"] = sum(1 for e in ev
                               if e.get("type") == "RESOURCE_WALL_ATTRIBUTION_FAILED")
    out["instrument_on"] = out["n_wall_events"] > 0

    # --- P3: family coverage of wall text ------------------------------------------------------
    # Read from the rewrite path, because that is where the text is actually delivered: a family is
    # covered when a REWRITE_PRODUCED for it follows an attribution that carried a deliverable wall on
    # the family's own best candidate. Approximated by "the family had at least one candidate with a
    # probe-worthy wall and at least one rewrite", which is the same set on every run inspected and is
    # stated rather than presented as exact.
    fam_of: dict[str, str] = {}
    for e in ev:
        if e.get("type") == "CANDIDATE_REGISTERED":
            p = _pl(e).get("candidate") or _pl(e)
            cid, fid = str(p.get("candidate_id") or ""), str(p.get("family_id") or "")
            if cid and fid:
                fam_of[cid] = fid
    fams_rewritten = {str(_pl(e).get("family_id") or "")
                      for e in ev if e.get("type") == "REWRITE_PRODUCED"}
    fams_rewritten.discard("")
    fams_with_wall = {fam_of.get(cid) for cid, v in per_cand_worthy.items() if v > 0}
    fams_with_wall.discard(None)
    all_fams = set(fam_of.values())
    out["p3_families"] = len(all_fams)
    out["p3_families_rewritten"] = len(fams_rewritten)
    out["p3_families_with_wall"] = len(fams_with_wall)
    out["p3_covered"] = len(fams_with_wall & fams_rewritten)
    out["p3_share"] = (len(fams_with_wall & fams_rewritten) / len(all_fams)) if all_fams else None

    # --- latency (secondary, declared under-powered) -------------------------------------------
    best = None
    for e in ev:
        if e.get("type") != "TRIAL_DONE":
            continue
        ms = _robust_ms(_trial_of(e))
        if ms is not None and (best is None or ms < best):
            best = ms
    out["best_trial_ms"] = best
    fin = [e for e in ev if e.get("type") == "RUN_FINISHED"]
    out["finished"] = bool(fin)
    if fin:
        s = _pl(fin[0]).get("summary") or {}
        for k in ("best_ms", "final_reeval_ms", "speedup", "elapsed_hours", "stop_kind"):
            if k in s:
                out[f"summary_{k}"] = s[k]
    # Event span rather than wall clock: a resumed run's own clock restarts, and the log's span is the
    # only comparable duration.
    ts = [e["ts"] for e in ev if isinstance(e.get("ts"), (int, float))]
    out["span_hours"] = (max(ts) - min(ts)) / 3600.0 if len(ts) >= 2 else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("treatment")
    ap.add_argument("control")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t = _arm(Path(args.treatment))
    c = _arm(Path(args.control))
    L: list[str] = []

    def say(s: str = "") -> None:
        L.append(s)

    def row(label: str, key: str, fmt: str = "{}") -> None:
        tv, cv = t.get(key), c.get(key)
        f = (lambda v: "n/a" if v is None else fmt.format(v))
        say(f"  {label:44s}{f(tv):>16s}{f(cv):>16s}")

    say("=" * 92)
    say("S7 PAIR -- read against the predictions declared before the run")
    say("=" * 92)
    say(f"  treatment: {t['dir']}")
    say(f"  control:   {c['dir']}")
    say(f"  finished:  treatment={t['finished']}  control={c['finished']}")
    if not (t["finished"] and c["finished"]):
        say("  WARNING: at least one arm has not written RUN_FINISHED. Every number below is a")
        say("  snapshot of an in-flight run, and `ended` rows of an in-flight run are not")
        say("  trustworthy (the report tool conflates running with crashed).")
    say()
    say(f"  {'':44s}{'TREATMENT':>16s}{'CONTROL':>16s}")
    say("  " + "-" * 74)
    row("event span (h)", "span_hours", "{:.2f}")
    row("trials completed", "n_trials")
    row("candidates with trials", "n_candidates_with_trials")
    row("spaces published", "n_spaces")
    say()

    # THE INSTRUMENT CHECK COMES FIRST, before any prediction, because every P1/P2/P3 figure is read
    # from 2e's events. With 2e off they are all 0 -- indistinguishable from "2e ran and found nothing"
    # unless the reader says so. Verified against two real finished runs that have 260 and 142 refusals
    # and zero attributions because the switch was off.
    say("  IS THE INSTRUMENT ON? (2e supplies every P1/P2/P3 number)")
    row("shared-memory refusals recorded", "n_refusals")
    row("RESOURCE_WALL_ATTRIBUTED events", "n_wall_events")
    row("...ATTRIBUTION_FAILED events", "n_wall_failed")
    broken = []
    for label, a in (("treatment", t), ("control", c)):
        if not a["instrument_on"]:
            if a["n_refusals"] > 0:
                broken.append(
                    f"    !! {label}: {a['n_refusals']} refusals but ZERO attributions => 2e was OFF "
                    f"(or crashed). Its P1/P2/P3 zeros are the ABSENCE OF THE INSTRUMENT, not a "
                    f"finding, and the pair cannot be read.")
            else:
                broken.append(
                    f"    !! {label}: no refusals at all => the hard wall has no input on this run. "
                    f"Its zeros mean 'nothing to find', which is a third state again.")
    for b in broken:
        say(b)
    if not broken:
        say("    both arms: 2e produced attributions, so the zeros below (if any) are findings.")
    say()

    say("  P1  DOES THE WALL BECOME KNOWN EARLIER? (normalised to tuning progress)")
    row("2e attribution, median first position", "p1_2e_first_frac", "{:.2f}")
    row("   ...candidates it is measured on", "p1_2e_n")
    row("S7 first firing, median n_told/budget", "s7_first_frac", "{:.2f}")
    row("   ...candidates where S7 fired", "s7_candidates_fired")
    say("      2e runs AFTER tuning in BOTH arms, so its position is ~1.0 by construction and is")
    say("      shown only to confirm the arms are alike there. The mechanism's own clock is the S7")
    say("      row: it is the fraction of the trial budget already spent when a wall was first acted")
    say("      on. The control cannot have one -- that is what the pair is testing.")
    say()

    say("  P2  DO PROBE-WORTHY WALLS PER CANDIDATE INCREASE?")
    row("probe-worthy walls (total)", "p2_total_worthy")
    row("   ...per candidate probed", "p2_per_candidate", "{:.2f}")
    row("candidates with >=1 worthy wall", "p2_candidates_with_any")
    row("candidates probed", "p2_candidates_probed")
    say("      `walls_worthless` is SUBTRACTED: a run must not look better for finding walls whose")
    say("      slope filter then correctly discarded them.")
    say()

    say("  P3  DOES FAMILY COVERAGE RISE ABOVE 25%?")
    row("families", "p3_families")
    row("families with a worthy wall", "p3_families_with_wall")
    row("families that were rewritten", "p3_families_rewritten")
    row("covered (wall AND rewrite)", "p3_covered")
    row("share of families covered", "p3_share", "{:.1%}")
    say(f"      baseline to beat: {P3_BASELINE:.0%} (three-arm run). Coverage is approximated as")
    say("      'the family had a candidate with a worthy wall and was rewritten' -- stated rather")
    say("      than presented as exact.")
    say()

    say("  MECHANISM ACTIVITY (treatment only; a zero here makes P1-P3 unreadable)")
    say(f"    SLOPE_GUIDE_STEP events        {t['s7_steps']}")
    say(f"    points enqueued / refused      {t['s7_enqueued']} / {t['s7_refused']}")
    say(f"    by criterion                   {t['s7_sources']}")
    say(f"    SLOPE_GUIDE_FAILED             {t['s7_failed']}")
    if t["n_trials"]:
        say(f"    enqueued as a share of trials  {t['s7_enqueued'] / t['n_trials']:.2%}")
    if c["s7_steps"]:
        say(f"    !! CONTROL HAS {c['s7_steps']} SLOPE_GUIDE_STEP EVENTS -- the arms are not separated")
    say()

    say("  LATENCY (secondary, declared UNDER-POWERED before the run)")
    row("best trial (ms)", "best_trial_ms", "{:.4f}")
    row("summary best_ms", "summary_best_ms", "{:.4f}")
    row("summary final_reeval_ms", "summary_final_reeval_ms", "{:.4f}")
    tb, cb = t.get("best_trial_ms"), c.get("best_trial_ms")
    delta = None
    if isinstance(tb, (int, float)) and isinstance(cb, (int, float)) and cb > 0:
        delta = (cb - tb) / cb * 100.0
        say(f"    treatment is {delta:+.2f}% vs control "
            f"(positive = treatment faster)")
        say(f"    noise floor {NOISE_FLOOR_PCT}%, within-arm spread {WITHIN_ARM_SPREAD_PCT}% "
            f"=> |{delta:.2f}%| is "
            f"{'INSIDE the noise' if abs(delta) < NOISE_FLOOR_PCT else 'outside the noise floor'}"
            + (", but inside the within-arm spread" if NOISE_FLOOR_PCT <= abs(delta)
               < WITHIN_ARM_SPREAD_PCT else ""))
    say()

    # --- which reading the pattern selects ----------------------------------------------------
    say("=" * 92)
    say("WHICH READING THIS PATTERN SELECTS")
    say("=" * 92)
    p1 = t.get("s7_first_frac") is not None and t["s7_enqueued"] > 0
    tp2, cp2 = t.get("p2_per_candidate"), c.get("p2_per_candidate")
    p2 = (isinstance(tp2, float) and isinstance(cp2, float) and tp2 > cp2)
    tp3 = t.get("p3_share")
    p3 = isinstance(tp3, float) and tp3 > P3_BASELINE
    lat_improved = isinstance(delta, float) and delta > WITHIN_ARM_SPREAD_PCT

    say(f"  P1 (wall known earlier, mechanism fired):  {'HOLDS' if p1 else 'FAILS'}")
    say(f"  P2 (more probe-worthy walls/candidate):    {'HOLDS' if p2 else 'FAILS'}")
    say(f"  P3 (family coverage above 25%):            {'HOLDS' if p3 else 'FAILS'}")
    say(f"  latency improved beyond the spread:        {'YES' if lat_improved else 'NO'}")
    say()
    if not (t["instrument_on"] and c["instrument_on"]):
        say("  => UNREADABLE: 2e did not produce attributions on at least one arm, so P1-P3 are")
        say("     measuring the absence of the instrument. See the instrument check above. Nothing")
        say("     below this line applies, and the verdicts printed above are not findings.")
    elif not p1:
        say("  => THE MECHANISM DID NOT FIRE (or fired without enqueueing). P2-P4 are unreadable:")
        say("     'S7 does not help' and 'S7 barely ran' look identical from here. Check")
        say("     SLOPE_GUIDE_STEP's skip counters -- n_skipped_no_wall vs")
        say("     n_skipped_no_value_toward_wall -- before drawing any conclusion.")
    elif p1 and p2 and p3 and not lat_improved:
        say("  => P4. Knowing about the wall earlier is NOT the bottleneck: the mechanism delivered")
        say("     more walls, earlier, to more families, and latency did not move. That locates C2's")
        say("     problem in the conversion rate of the REWRITE itself. This is a result, and it is")
        say("     the one this module was built to be able to state.")
    elif p1 and not (p2 or p3):
        say("  => P5. The binding constraint is the criterion's APPLICABILITY, neither the timing nor")
        say("     the dose: the mechanism fired and delivered points, but the number of walls did not")
        say("     rise. Consistent with the preflight, where 181 of 224 recomputes found no")
        say("     actionable wall at all. Raising coverage means CHANGING A CRITERION -- the same")
        say("     conclusion 2e's 25% coverage and item 5 reached independently.")
    elif p1 and p2 and p3 and lat_improved:
        say("  => P1-P3 hold AND latency improved. The strongest available outcome -- but it is ONE")
        say("     pair, and the open premise (that slope is a good allocation prior; measured")
        say("     Spearman <= 0.24 against remaining gain) is not settled by a single pair. It needs")
        say("     a replicate before it is claimed.")
    else:
        say("  => MIXED. State each prediction's verdict separately above; do not summarise this as")
        say("     'S7 works' or 'S7 does not'. In particular a P2/P3 split means the walls arrived")
        say("     but were not delivered, which is a REPORTING path question, not a sampling one.")
    say()
    say("  RUN THESE BEFORE TRUSTING ANY OF THE ABOVE:")
    say("    python scripts/check_arm_search_parity.py <control> <treatment>")
    say("    python scripts/audit_arm_comparability.py <control> <treatment>")
    say("  If the arms did not get comparable search, every number here has two explanations.")

    text = "\n".join(L) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        sys.stdout.buffer.write(text.encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
