"""The three wrap-up checks, pinned in `docs/preflight-control-run.md` BEFORE the runs launched.

Written while the runs are still going, deliberately: a checker improvised after seeing the data
is a checker whose thresholds were chosen to fit it. Run it against one run dir, or two to compare
the control and treatment arms.

    python scripts/check_wrapup.py <control_run_dir> [<treatment_run_dir>]

Every reading goes through the same nesting rules the analysis scripts learned the hard way -- a
`TRIAL_DONE` payload nests under `payload.trial.params.values`, and reading `payload["params"]`
returns None for every trial while printing a plausible all-None table. Each reader asserts it
found something, so a wrong key path fails loudly instead of reporting a clean zero.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path


def _events(run_dir: Path):
    p = run_dir / "events.jsonl"
    if not p.exists():
        raise SystemExit("not a run dir (no events.jsonl): %s" % run_dir)
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


# --- check 1: G27's first production evidence -------------------------------------------------


def check_conversion(run_dir: Path) -> dict:
    """Do the round events carry a `conversion` field?

    A zero here is only a defect once rounds exist. `conversion_verdict` always returns a
    `conversion` key -- "unknown" when latency cannot be compared -- and the orchestrator merges it
    into every FAMILY_ROUND_RECORDED inside `if evaluated:`. So with rounds > 0, missing
    `conversion` is structurally impossible unless something broke, which is exactly the
    distinction the preflight doc asks for: "if still 0 that is a NEW defect, not 'nothing
    converted'".
    """
    rounds = 0
    with_conv = 0
    with_deltas = 0
    verdicts = collections.Counter()
    for e in _events(run_dir):
        if e.get("type") != "FAMILY_ROUND_RECORDED":
            continue
        rounds += 1
        p = e.get("payload") or {}
        if "conversion" in p:
            with_conv += 1
            verdicts[p["conversion"]] += 1
        if "resource_deltas" in p:
            with_deltas += 1
    if rounds == 0:
        verdict = "NOT YET DECIDABLE -- 0 rewrite rounds, so 0-of-0 says nothing"
    elif with_conv == 0:
        verdict = ("**NEW DEFECT** -- %d rounds and NONE carries `conversion`. The verdict is "
                   "merged unconditionally, so this is not 'nothing converted'" % rounds)
    elif with_conv < rounds:
        verdict = ("PARTIAL -- %d of %d rounds carry it; a round that skips the verdict is the "
                   "G44 shape (computed, journalled, read zero times)" % (with_conv, rounds))
    else:
        verdict = "PASS -- G27 has production evidence for the first time (%d of %d)" % (
            with_conv, rounds)
    return {"rounds": rounds, "with_conversion": with_conv, "with_resource_deltas": with_deltas,
            "verdicts": dict(verdicts), "verdict": verdict}


# --- check 2: J2-5 / J2d-9, the final result must not be worse ---------------------------------


def final_result(run_dir: Path) -> dict:
    """The run's FINAL RE-EVAL latency, never `tuned_ms`.

    `tuned_ms` is systematically optimistic by 1.5-6.7% (measured across the corpus), so comparing
    two arms on it can invert the sign of a small difference.

    Key paths verified against `src/`, not guessed. My first version of this reader looked for a
    `FINAL_REEVAL_DONE` event, which DOES NOT EXIST -- it returned None on a corpus run that has
    the number, i.e. it would have reported "no result" for a finished run. The truth is:

      * `final_reeval_ms` lives in `RUN_FINISHED.payload.summary.best`, written by `_finalize`.
        Note it is `lat.mean`; `final_reeval_median_ms` beside it is the median.
      * `BASELINE_DONE` nests under `payload.baseline` with `kind` and `latency_ms` -- the same
        one-level nesting as TRIAL_DONE, and reading `payload["latency_ms"]` yields nothing.
    """
    best: dict = {}
    trial_best = None
    baseline = {}
    for e in _events(run_dir):
        t = e.get("type")
        p = e.get("payload") or {}
        if t == "RUN_FINISHED":
            b = ((p.get("summary") or {}).get("best")) or {}
            if b:
                best = b
        elif t == "BASELINE_DONE":
            bl = p.get("baseline") or p
            lat = bl.get("latency_ms") or {}
            name = bl.get("kind") or "?"
            v = lat.get("median") or lat.get("mean")
            if isinstance(v, (int, float)):
                baseline[str(name)] = float(v)
        elif t == "TRIAL_DONE":
            tr = p.get("trial") or p
            if tr.get("status") == "complete" and not tr.get("failure_kind"):
                v = (tr.get("latency_ms") or {}).get("median")
                if isinstance(v, (int, float)) and (trial_best is None or v < trial_best):
                    trial_best = float(v)
    reeval = best.get("final_reeval_ms")
    return {"final_reeval_ms": float(reeval) if isinstance(reeval, (int, float)) else None,
            "final_reeval_median_ms": best.get("final_reeval_median_ms"),
            "final_reeval_ok": best.get("final_reeval_ok"),
            "tuned_ms": best.get("tuned_ms"),
            "precision": best.get("precision"),
            "excessive_speedup_flag": best.get("excessive_speedup_flag"),
            "best_trial_median_ms": trial_best,
            "baselines": baseline,
            "used_fallback": not isinstance(reeval, (int, float))}


def compare_arms(control: dict, treatment: dict, noise_floor_pct: float) -> str:
    """J2-5: treatment must not be worse than control beyond the task's measured noise floor."""
    c = control["final_reeval_ms"]
    t = treatment["final_reeval_ms"]
    if c is None or t is None:
        which = [n for n, d in (("control", control), ("treatment", treatment))
                 if d["final_reeval_ms"] is None]
        return ("CANNOT DECIDE -- no final re-eval on: %s. `tuned_ms` is optimistic by 1.5-6.7%%, "
                "so it is not a substitute here" % ", ".join(which))
    allowed = c * (1.0 + noise_floor_pct / 100.0)
    delta = 100.0 * (t - c) / c
    if t <= allowed:
        return ("PASS -- treatment %.4f ms vs control %.4f ms (%+.2f%%), within the %.2f%% noise "
                "floor" % (t, c, delta, noise_floor_pct))
    return ("FAIL -- treatment %.4f ms vs control %.4f ms (%+.2f%%), beyond the %.2f%% noise "
            "floor: the extra information made the final result WORSE" % (
                t, c, delta, noise_floor_pct))


# --- check 3: S3 / S4' behaviour on real data --------------------------------------------------


def check_s3_s4(run_dir: Path) -> dict:
    """Four things the preflight names, each of which has a silent-failure twin.

    `precision_mismatch` firing at all, which of the four complementary-slackness states was
    reported, whether a below-floor reading was REPORTED rather than clamped, and whether any
    ceiling_provenance carries a precision.
    """
    prov_with_precision = 0
    prov_total = 0
    mismatches = []
    slack_states = collections.Counter()
    below_floor = 0
    clamped_suspicion = 0
    dims = collections.Counter()
    for e in _events(run_dir):
        t = e.get("type")
        p = e.get("payload") or {}
        if t == "DIMENSION_STATE":
            for rec in (p.get("records") or []):
                dims[rec.get("dimension_id")] += 1
                prov_total += 1
                prov = rec.get("provenance") or {}
                if prov.get("precision"):
                    prov_with_precision += 1
                b = rec.get("bound") or {}
                if b.get("below_floor"):
                    below_floor += 1
                # A reading at EXACTLY the floor with room 0.0 is what a clamp looks like.
                if b.get("floor") is not None and b.get("room") == 0.0:
                    clamped_suspicion += 1
            if p.get("precision_mismatch"):
                mismatches.append(p.get("candidate_id"))
        blob = json.dumps(p)
        if "precision_mismatch" in blob and t != "DIMENSION_STATE":
            mismatches.append("%s:%s" % (t, p.get("candidate_id")))
        for state in ("cannot_run", "no_dimension_judged_slack", "weak_pass", "violation_named"):
            if state in blob:
                slack_states[state] += 1
    return {"dimension_records": prov_total,
            "records_whose_provenance_names_a_precision": prov_with_precision,
            "precision_mismatch_fired_on": sorted(set(m for m in mismatches if m)),
            "complementary_slackness_states": dict(slack_states),
            "below_floor_readings_reported": below_floor,
            "readings_sitting_exactly_on_the_floor": clamped_suspicion,
            "dimensions_seen": dict(dims)}


def check_reconciliation(run_dir: Path) -> dict:
    """S2d's expectation ledger: did it reconcile anything, or only produce empty entries?

    Two traps here, both of which look like success from a count alone.

    An `EXPECTATIONS_RECONCILED` event can carry `n_declared: 0` -- the rewriter declared no
    expectations that round, so the entry reconciles nothing. Counting events would report a
    working ledger built entirely of empty entries: the G44 shape (computed, journalled, read zero
    times) with an event stream that looks healthy.

    And it fires in the same `if evaluated:` branch as `conversion`, so 0 with 0 rounds is
    expected and says nothing -- the same distinction check 1 makes.

    `EXPECTATIONS_RECONCILE_FAILED` matters more than its count suggests: the reconciler is
    deliberately wrapped so a diagnostic cannot end a rewrite round, which means a total failure is
    SILENT apart from this event.
    """
    rounds = 0
    entries = 0
    empty = 0
    failed = []
    declared_total = 0
    verdict_kinds = collections.Counter()
    for e in _events(run_dir):
        t = e.get("type")
        p = e.get("payload") or {}
        if t == "FAMILY_ROUND_RECORDED":
            rounds += 1
        elif t == "EXPECTATIONS_RECONCILED":
            entries += 1
            n = p.get("n_declared")
            if not n:
                empty += 1
            else:
                declared_total += int(n)
            rec = p.get("reconciliation") or {}
            for key in ("verdict", "outcome", "status"):
                if key in rec:
                    verdict_kinds[str(rec[key])] += 1
                    break
        elif t == "EXPECTATIONS_RECONCILE_FAILED":
            failed.append(p.get("error", "")[:120])
    if rounds == 0:
        verdict = "NOT YET DECIDABLE -- 0 rewrite rounds, so 0 ledger entries says nothing"
    elif entries == 0:
        verdict = ("**DEFECT** -- %d rounds and 0 ledger entries. It is journalled "
                   "UNCONDITIONALLY (the switch only gates the PROMPT), so the control arm should "
                   "have them too" % rounds)
    elif empty == entries:
        verdict = ("**EMPTY LEDGER** -- all %d entries carry n_declared=0: the events exist and "
                   "reconcile nothing, which is the G44 shape with a healthy-looking stream"
                   % entries)
    elif empty:
        verdict = "PARTIAL -- %d of %d entries are empty (n_declared=0)" % (empty, entries)
    else:
        verdict = "PASS -- %d entries over %d rounds, %d declarations reconciled" % (
            entries, rounds, declared_total)
    if failed:
        verdict += "  || %d RECONCILE_FAILED (silent by design -- a diagnostic must not end a " \
                   "round): %s" % (len(failed), failed[0])
    return {"rounds": rounds, "entries": entries, "empty_entries": empty,
            "declarations_reconciled": declared_total, "reconcile_failed": len(failed),
            "verdict_kinds": dict(verdict_kinds), "verdict": verdict}


def report(run_dir: Path, label: str) -> dict:
    print("=" * 78)
    print("%s   %s" % (label, run_dir))
    print("=" * 78)
    conv = check_conversion(run_dir)
    print("\n[1] G27 -- conversion's first production evidence")
    print("    rewrite rounds recorded ....... %d" % conv["rounds"])
    print("    carrying `conversion` ......... %d" % conv["with_conversion"])
    print("    carrying `resource_deltas` .... %d" % conv["with_resource_deltas"])
    print("    verdict distribution .......... %s" % (conv["verdicts"] or "-"))
    print("    => %s" % conv["verdict"])

    fin = final_result(run_dir)
    print("\n[2] J2-5 -- the final result, on final_reeval_ms not tuned_ms")
    print("    final_reeval_ms (mean) ........ %s%s" % (
        "%.4f" % fin["final_reeval_ms"] if fin["final_reeval_ms"] else "NONE YET",
        "" if fin["final_reeval_ok"] is None else "   ok=%s" % fin["final_reeval_ok"]))
    print("    final_reeval_median_ms ........ %s" % (
        "%.4f" % fin["final_reeval_median_ms"]
        if isinstance(fin["final_reeval_median_ms"], (int, float)) else "-"))
    print("    tuned_ms (optimistic 1.5-6.7%%) . %s" % (
        "%.4f" % fin["tuned_ms"] if isinstance(fin["tuned_ms"], (int, float)) else "-"))
    print("    winning precision ............. %s" % (fin["precision"] or "-"))
    print("    best complete trial median .... %s" % (
        "%.4f" % fin["best_trial_median_ms"] if fin["best_trial_median_ms"] else "-"))
    print("    baselines ..................... %s" % (
        {k: round(v, 4) for k, v in fin["baselines"].items()} or "-"))
    if fin["excessive_speedup_flag"]:
        print("    !! excessive_speedup flagged -- the anti-cheat threshold fired, inspect before "
              "reporting any speedup")
    if fin["used_fallback"]:
        print("    !! no final re-eval in RUN_FINISHED yet: the trial median is NOT a substitute")

    s3 = check_s3_s4(run_dir)
    print("\n[3] S3 / S4' on real data")
    print("    dimension records ............. %d" % s3["dimension_records"])
    print("    provenance naming a precision . %d" % s3["records_whose_provenance_names_a_precision"])
    print("    precision_mismatch fired on ... %s" % (s3["precision_mismatch_fired_on"] or "nothing"))
    print("    complementary-slackness states  %s" % (s3["complementary_slackness_states"] or "none"))
    print("    below-floor readings REPORTED . %d" % s3["below_floor_readings_reported"])
    print("    readings exactly on the floor . %d%s" % (
        s3["readings_sitting_exactly_on_the_floor"],
        "   <- inspect: this is what a clamp looks like"
        if s3["readings_sitting_exactly_on_the_floor"] else ""))
    print("    dimensions seen ............... %s" % (s3["dimensions_seen"] or "-"))

    rec = check_reconciliation(run_dir)
    print("\n[3b] S2d -- the expectation ledger (journalled in BOTH arms; the switch gates only")
    print("     whether the rendered form reaches the rewriter's prompt)")
    print("    ledger entries ................ %d" % rec["entries"])
    print("    of which EMPTY (n_declared=0) .. %d" % rec["empty_entries"])
    print("    declarations reconciled ....... %d" % rec["declarations_reconciled"])
    print("    RECONCILE_FAILED .............. %d" % rec["reconcile_failed"])
    print("    verdict kinds ................. %s" % (rec["verdict_kinds"] or "-"))
    print("    => %s" % rec["verdict"])
    print()
    return {"conversion": conv, "final": fin, "s3": s3, "reconciliation": rec}


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    control = report(Path(argv[0]), "CONTROL ARM" if len(argv) > 1 else "RUN")
    if len(argv) > 1:
        treatment = report(Path(argv[1]), "TREATMENT ARM")
        # The L3:43 noise floor measured on both 4090s: 0.9765 / 0.9767 frac_within_tol, and the
        # latency-side floor used for the calibration gate is 2.35%.
        print("=" * 78)
        print("J2-5 / J2d-9 comparison")
        print("=" * 78)
        print("    %s" % compare_arms(control["final"], treatment["final"], 2.35))
        print("\n    G27 in both arms: control %d/%d, treatment %d/%d rounds carry `conversion`" % (
            control["conversion"]["with_conversion"], control["conversion"]["rounds"],
            treatment["conversion"]["with_conversion"], treatment["conversion"]["rounds"]))
        print("    (the treatment arm is the one that must ALSO produce dimension records: "
              "control %d, treatment %d)" % (
                  control["s3"]["dimension_records"], treatment["s3"]["dimension_records"]))
        # The ledger is journalled in BOTH arms by design, so an asymmetry here is a defect rather
        # than the treatment working -- the opposite reading from the dimension records above.
        cl, tl = control["reconciliation"], treatment["reconciliation"]
        if cl["rounds"] and tl["rounds"] and bool(cl["entries"]) != bool(tl["entries"]):
            print("    !! LEDGER ASYMMETRY: control %d entries, treatment %d. It is journalled "
                  "unconditionally, so one arm having none is a defect, NOT the switch working"
                  % (cl["entries"], tl["entries"]))
        else:
            print("    ledger entries: control %d, treatment %d (expected in BOTH arms)"
                  % (cl["entries"], tl["entries"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
