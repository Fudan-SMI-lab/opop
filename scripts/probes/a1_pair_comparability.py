"""A1's pair reader: comparability, mechanism separation, and the end-to-end result.

WHY THESE THREE LIVE IN ONE FILE. They are three questions about the SAME pair and they must
be read in this order, because each one decides whether the next is meaningful:

  --comparability  Did the two arms buy comparable SEARCH? Every finished run in this project
                   ended on the wall clock, so equal trial COUNTS prove nothing -- an arm can
                   run more trials because its trials were cheaper. GPU wall is the budget
                   that was actually equalised, so that is what gets compared.
  --mechanism      Did the independent variable actually differ? Window 1's n1 pair had
                   conditional_scan.mode=off on BOTH arms and therefore could not test C2 at
                   all; the 0 C4 blocks looked like a negative result and were not one.
  --final          What did each arm end at? Descriptive only -- see below.

WHAT THIS DELIBERATELY REFUSES TO DO.

No pooling. Call it once per pair. box4 is a Xeon 8352V and box1 a 8358P with 32% different
compile time, and the m2b pair runs a different task, so a number computed across pairs is
not an effect size.

No tie band, no verdict on the end-to-end difference. As of window 1 there is no qualified
same-code A/A calibration in any run (docs/result-window1-offline-closeout.md section 5: the
13 declaration-matched pairs ran 0/13 identical code, and the three fresh same-artifact
repeats are cross-space re-tunes on three different candidates spanning 40x in latency). So
`--final` prints both arms and their difference and STOPS. It does not say which won.

Final values come from RUN_FINISHED.summary.best.final_reeval_median_ms, never from the
2-decimal final_reeval_ms, and never from the tuning best -- those are different quantities.
speedup_vs_eager is cross-precision and is not printed at all.

`same_precision_speedup` is NOT used as the arm-difference quantity, and the reason is
arithmetic rather than stylistic. It is a mean-over-mean ratio (orchestrator.py:3474) and
KernelBench rounds every mean to THREE SIGNIFICANT FIGURES at the source
(kernelbench/timing.py:614, `float(f"{np.mean(...):.3g}")`). So both sides arrive
pre-quantized: at ~2.5 ms the candidate's ulp is 0.01/2.54 = 0.39% and an ~11 ms baseline's
is 0.1/11.0 = 0.91%, giving the published ratio a resolution floor near 1.3%. Measured
consequence: n1-b and m1-a print the SAME 4.3307 from different kernels (median 2.537472 vs
2.539456, baselines 10.948608 vs 10.948544) -- an identity that is pure quantization. Any
arm difference below ~1.3% can therefore vanish, or inverted, in that field. The median
quantities are full precision on both sides, so the difference is read from
final_reeval_median_ms and `speedups_median` is printed next to the mean ratio.

Offline: reads events.jsonl only. No GPU, no torch, no candidate execution.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from kernel_optimizer.store.read import read_events  # noqa: E402

# Events that exist only when the conditioned-scan mechanism is live. If these are non-zero
# in the control arm, the pair is not a clean off/active contrast and nothing downstream of
# it can be attributed to the mechanism.
MECHANISM_EVENTS = (
    "UW_PROBE_BATCH",
    "SCAN_BLOCK_ADMITTED",
    "SCAN_BLOCK_DONE",
    "SCAN_POINT_DONE",
    "CONDITIONED_BRIEF_DELIVERED",
)


def _load(run: Path) -> list[dict]:
    return read_events(run)


def _arm_name(run: Path) -> str:
    """The arm label. n1-a and n1-b shared ONE run_id, so the identity is the directory, not
    the journal. Production nests as <runs-v4>/<arm>/<run-id>, but a repacked archive can nest
    it deeper (<box>/runs/<arm>/<run-id>), which made every arm print as the literal "runs" on
    the first live dry-run. So walk up past any structural segment to the first real label."""
    generic = {"runs", "runs-v4", "run", "."}
    for part in (run.parent.name, *[p.name for p in run.parents]):
        if part and part not in generic and not part.startswith("run-"):
            return part
    return run.parent.name


def _counts(events: list[dict]) -> Counter:
    return Counter(e.get("type") for e in events)


def _trial_stats(events: list[dict]) -> dict:
    """GPU wall, trial outcomes and reuse. `job_wall_s` covers lock wait + execution, which is
    what the wall-clock budget actually spent; a trial without it is counted but contributes
    no time, and that count is reported so a silent gap cannot masquerade as cheap trials."""
    gpu = 0.0
    n = complete = fail = reused = no_wall = 0
    best = None
    for e in events:
        if e.get("type") != "TRIAL_DONE":
            continue
        p = e.get("payload") or {}
        t = p.get("trial") or {}
        n += 1
        if p.get("reused_measurement"):
            reused += 1
        w = t.get("job_wall_s")
        if w is None:
            no_wall += 1
        else:
            gpu += float(w)
        if t.get("status") == "complete":
            complete += 1
            lat = t.get("latency_ms") or {}
            ms = lat.get("median")          # keys are median/mean, no _ms suffix
            if ms is not None and (best is None or float(ms) < best):
                best = float(ms)
        else:
            fail += 1
    return {"trials": n, "complete": complete, "fail": fail, "reused": reused,
            "gpu_hours": gpu / 3600.0, "no_wall_field": no_wall, "tuned_best_ms": best}


def _families(events: list[dict]) -> dict:
    seeded: set[str] = set()
    rounds: list[dict] = []
    rewrites = 0
    not_evaluated = 0
    for e in events:
        t = e.get("type")
        p = e.get("payload") or {}
        if t == "FAMILY_SEEDED":
            fam = p.get("family") or p
            seeded.add(str(fam.get("family_id")))
        elif t == "REWRITE_PRODUCED":
            rewrites += 1
        elif t == "FAMILY_ROUND_RECORDED":
            rounds.append(p)
        elif t == "FAMILY_ROUND_NOT_EVALUATED":
            not_evaluated += 1
    # Round DEPTH is the question, not round count: one record per family is one point, not a
    # curve, and `round` is the orchestrator's round counter -- NOT lineage depth. A family
    # with rounds [1, 2] has a second generation; four families each with [1] do not.
    by_family: dict[str, list] = defaultdict(list)
    for p in rounds:
        by_family[str(p.get("family_id"))].append(p.get("round"))
    multi = {f: sorted(r) for f, r in by_family.items() if len(r) > 1}
    return {"seeded": len(seeded), "with_a_round": len(by_family), "rounds": rounds,
            "rewrites": rewrites, "not_evaluated": not_evaluated, "multi_round": multi}


def _final(events: list[dict]) -> dict:
    """The run's own final answer. Only RUN_FINISHED carries it.

    Field shapes verified against a real journal (n1-a), because guessing them is how a
    reader prints `None` for a value that is present: the honest same-precision verdict is
    nested under `best.honest_verdict`, NOT flat on `best`, and the winner's knobs are under
    `best.params.values`. `speedups`/`speedup_vs_eager` are read but deliberately dropped --
    eager is a cross-precision comparison (this winner is fp16 against an fp32 eager
    baseline) and only the same-precision figure is reportable.
    """
    for e in reversed(events):
        if e.get("type") != "RUN_FINISHED":
            continue
        s = (e.get("payload") or {}).get("summary") or {}
        b = s.get("best") or {}
        hv = b.get("honest_verdict") or {}
        med = b.get("speedups_median") or {}
        against = hv.get("compared_against")
        return {
            "candidate_id": b.get("candidate_id"),
            "family_id": b.get("family_id"),
            # the unrounded value; `final_reeval_ms` is the 2-decimal one and loses precision
            "final_reeval_median_ms": b.get("final_reeval_median_ms"),
            "same_precision_speedup": hv.get("same_precision_speedup"),
            # the same comparison computed median-over-median: both sides full precision.
            # `speedups`/`honest_verdict` are mean-over-mean and every mean was rounded to
            # 3 significant figures inside KernelBench, so the published ratio cannot
            # resolve an arm difference below ~1.3%.
            "same_precision_speedup_median": med.get(against) if against else None,
            "compared_against": against,
            "beats_same_precision_baseline": hv.get("beats_same_precision_baseline"),
            "final_reeval_ok": b.get("final_reeval_ok"),
            "excessive_speedup_flag": b.get("excessive_speedup_flag"),
            "precision": b.get("precision") or hv.get("candidate_precision"),
            "tuned_ms": b.get("tuned_ms"),
            "elapsed_hours": s.get("elapsed_hours"),
        }
    return {}


def comparability(off: Path, act: Path) -> None:
    print("  %-22s %14s %14s   %s" % ("", _arm_name(off), _arm_name(act), "arm difference"))
    so, sa = _trial_stats(_load(off)), _trial_stats(_load(act))
    fo, fa = _families(_load(off)), _families(_load(act))
    eo, ea = _final(_load(off)), _final(_load(act))

    def row(label: str, a, b, fmt: str = "%s", diff: str | None = None) -> None:
        sa_ = fmt % a if a is not None else "-"
        sb_ = fmt % b if b is not None else "-"
        print("  %-22s %14s %14s   %s" % (label, sa_, sb_, diff if diff is not None else ""))

    dg = sa["gpu_hours"] - so["gpu_hours"]
    pct = (100.0 * dg / so["gpu_hours"]) if so["gpu_hours"] else float("nan")
    row("GPU wall (h)", so["gpu_hours"], sa["gpu_hours"], "%.2f",
        "%+.2f h (%+.1f%%)  <== the budget that was equalised" % (dg, pct))
    row("elapsed (h)", eo.get("elapsed_hours"), ea.get("elapsed_hours"), "%.3f")
    row("trials", so["trials"], sa["trials"], "%d",
        "%+d  (count is NOT the comparability check)" % (sa["trials"] - so["trials"]))
    row("complete", so["complete"], sa["complete"], "%d")
    row("failed", so["fail"], sa["fail"], "%d")
    row("marked reused", so["reused"], sa["reused"], "%d",
        "reused records are not independent measurements")
    if so["no_wall_field"] or sa["no_wall_field"]:
        fo_ = 100.0 * so["no_wall_field"] / so["trials"] if so["trials"] else 0.0
        fa_ = 100.0 * sa["no_wall_field"] / sa["trials"] if sa["trials"] else 0.0
        row("trials w/o job_wall_s", so["no_wall_field"], sa["no_wall_field"], "%d",
            "%.0f%% / %.0f%% of trials -- they contribute 0 h, so the GPU wall above is a"
            " LOWER BOUND" % (fo_, fa_))
    row("families seeded", fo["seeded"], fa["seeded"], "%d")
    row("families w/ a round", fo["with_a_round"], fa["with_a_round"], "%d")
    row("rewrites produced", fo["rewrites"], fa["rewrites"], "%d")
    row("rounds not evaluated", fo["not_evaluated"], fa["not_evaluated"], "%d")
    print()
    print("  rewrite-round DEPTH (a per-round curve needs >=2 rounds in ONE family):")
    for tag, f in ((_arm_name(off), fo), (_arm_name(act), fa)):
        if f["multi_round"]:
            print("    %-12s multi-round families: %s" % (tag, f["multi_round"]))
        else:
            print("    %-12s NO family reached a second round => one generation only, so"
                  % tag)
            print("    %-12s `latency_gain_pct` is a single cumulative gain vs the SEED,"
                  % "")
            print("    %-12s not a per-round increment, and no convergence curve exists."
                  % "")
    print()
    print("  per-round gains (relative to each family's SEED best, so start-point sensitive):")
    for tag, f in ((_arm_name(off), fo), (_arm_name(act), fa)):
        for p in f["rounds"]:
            print("    %-12s fam=%-16s round=%-3s gain=%6.2f%%  %s"
                  % (tag, str(p.get("family_id"))[:16], p.get("round"),
                     p.get("latency_gain_pct") or 0.0, p.get("conversion")))
    print()
    print("  NOTE: gain% correlates with the seed's starting latency (a 7.06 ms seed falling")
    print("        53.83% and a 3.00 ms seed falling 1.76% are not comparable efforts). Any")
    print("        cross-family or cross-arm comparison must normalise first.")


def mechanism(off: Path, act: Path) -> None:
    co, ca = _counts(_load(off)), _counts(_load(act))
    print("  %-30s %14s %14s" % ("mechanism-only event", _arm_name(off), _arm_name(act)))
    leak = 0
    for t in MECHANISM_EVENTS:
        n_off, n_act = co.get(t, 0), ca.get(t, 0)
        flag = ""
        if n_off:
            flag = "  <== NON-ZERO IN CONTROL ARM"
            leak += n_off
        print("  %-30s %14d %14d%s" % (t, n_off, n_act, flag))
    print()
    if leak:
        print("  !! The control arm fired the mechanism %d times. This pair is NOT a clean" % leak)
        print("  !! off/active contrast, and nothing downstream can be attributed to C2.")
    elif not any(ca.get(t) for t in MECHANISM_EVENTS):
        print("  !! NEITHER arm fired the mechanism. Check conditional_scan.mode in both")
        print("  !! resolved configs -- window 1's n1 pair was off/off and its 0 C4 blocks")
        print("  !! were a design consequence, not a negative result.")
    else:
        print("  => SEPARATED: the mechanism fired only in the active arm. The independent")
        print("     variable is real, so downstream contrasts are interpretable.")


def final(off: Path, act: Path) -> None:
    eo, ea = _final(_load(off)), _final(_load(act))
    so, sa = _trial_stats(_load(off)), _trial_stats(_load(act))
    for tag, e, s in ((_arm_name(off), eo, so), (_arm_name(act), ea, sa)):
        print("  [%s]" % tag)
        print("    tuned best (median, from trials) : %s"
              % ("%.4f ms" % s["tuned_best_ms"] if s["tuned_best_ms"] else "-"))
        print("    final_reeval_median_ms           : %s" % e.get("final_reeval_median_ms"))
        print("    final_reeval_ok                  : %s" % e.get("final_reeval_ok"))
        print("    same_precision_speedup [3sf]     : %s  (vs %s)"
              % (e.get("same_precision_speedup"), e.get("compared_against")))
        print("    same_precision_speedup [median]  : %s  <== full precision on both sides"
              % e.get("same_precision_speedup_median"))
        print("    beats_same_precision_baseline    : %s"
              % e.get("beats_same_precision_baseline"))
        print("    excessive_speedup_flag           : %s  (True => treat as suspect)"
              % e.get("excessive_speedup_flag"))
        print("    winner precision / candidate     : %s / %s"
              % (e.get("precision"), e.get("candidate_id")))
    a, b = eo.get("final_reeval_median_ms"), ea.get("final_reeval_median_ms")
    print()
    if a and b:
        d = (float(b) - float(a)) / float(a) * 100.0
        print("  active - off = %+.2f%%   (%.4f vs %.4f ms)" % (d, float(b), float(a)))
    _quantization_note(eo, ea)
    print("  NO VERDICT IS PRINTED HERE, deliberately. There is no qualified same-code A/A")
    print("  calibration in any run to date, so no tie band exists for this comparison, and")
    print("  P3 is a descriptive endpoint. Report the difference WITH that limitation.")


def _quantization_note(eo: dict, ea: dict) -> None:
    """Say WHY the arm difference is read from the median and not from the published speedup.

    Not decoration: the `[3sf]` row can print one identical number for two different kernels.
    Measured on the real journals -- n1-b (median 2.537472, tf32 baseline 10.948608) and m1-a
    (2.539456, 10.948544) both publish 4.3307, because KernelBench stores every mean as
    `float(f"{np.mean(...):.3g}")` (kernelbench/timing.py:614) and the ratio is mean/mean
    (orchestrator.py:3474). Both sides land on 11.0 / 2.54. Anyone reading the [3sf] row as
    an arm contrast would conclude the arms tied when the medians differ.
    """
    so, sa = eo.get("same_precision_speedup"), ea.get("same_precision_speedup")
    mo, ma = (eo.get("same_precision_speedup_median"),
              ea.get("same_precision_speedup_median"))
    if mo and ma:
        dm = (float(ma) - float(mo)) / float(mo) * 100.0
        print("  same-precision speedup, median convention: %+.2f%%  (%.4f vs %.4f)"
              % (dm, float(ma), float(mo)))
    print("  !! The [3sf] speedup is NOT the arm-difference quantity. It is mean-over-mean and")
    print("  !! every mean is rounded to 3 digits of precision at the source, so at ~2.5 ms vs")
    print("  !! ~11 ms the ratio carries a resolution floor near 1.3%: two runs whose medians")
    print("  !! differ can publish the SAME value (measured: n1-b and m1-a both print 4.3307).")
    if so is not None and sa is not None and float(so) == float(sa):
        print("  !! THIS PAIR IS SUCH A CASE: both arms print %s. Read the median rows." % so)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("off_run", help="control (mode=off) arm run directory")
    ap.add_argument("active_run", help="treatment (mode=active) arm run directory")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--mechanism", action="store_true", help="dose-separation table only")
    g.add_argument("--final", action="store_true", help="end-to-end table only")
    a = ap.parse_args()
    off, act = Path(a.off_run.rstrip("/")), Path(a.active_run.rstrip("/"))
    for r in (off, act):
        if not (r / "events.jsonl").is_file():
            print("no events.jsonl in %s" % r, file=sys.stderr)
            return 2
    if a.mechanism:
        mechanism(off, act)
    elif a.final:
        final(off, act)
    else:
        comparability(off, act)
    return 0


if __name__ == "__main__":
    sys.exit(main())
