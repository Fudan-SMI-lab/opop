"""Known-answer controls for `a1_pair_comparability.py`.

WHY THESE EXIST AND WHY THEY ARE SHAPED THIS WAY. A reader that prints a plausible table on
broken input is worse than no reader, because the table gets believed. Each test here fixes
an answer that is checkable by hand, and each one guards a mistake that has ALREADY been made
in this project at least once:

  - `median`/`mean` have no `_ms` suffix. Reading `median_ms` silently yields None for every
    trial, which then prints as "no comparable sibling" -- a fabricated fact.
  - GPU wall, not trial count, is the comparability check: every finished run ended on the
    wall clock, so an arm can run MORE trials precisely because its trials were cheaper.
  - `round` in FAMILY_ROUND_RECORDED is the orchestrator's round counter, NOT lineage depth.
    Four families each recording round 1, plus one recording round 2, is still ONE generation.
  - Mechanism events in the control arm mean the pair is not an off/active contrast at all.
  - Final values live only in RUN_FINISHED.summary.best.final_reeval_median_ms.

The revert check: flipping the reader's arithmetic must break these. Verified by inverting
the GPU-wall difference, the depth predicate and the leak predicate -- each breaks a test here.
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROBE = Path(__file__).resolve().parents[1] / "scripts" / "probes" / "a1_pair_comparability.py"


def _trial(tid, cid, ms, *, status="complete", wall=10.0, reused=False, space="sp-1"):
    """One TRIAL_DONE. `median`/`mean` deliberately carry no `_ms` suffix -- that is the
    real schema, and a reader that expects the suffix reads every latency as missing."""
    t = {"trial_id": tid, "candidate_id": cid, "space_id": space, "status": status,
         "params": {"values": {"BK": 32}}}
    if wall is not None:
        t["job_wall_s"] = wall
    if ms is not None:
        t["latency_ms"] = {"median": ms, "mean": ms * 1.01}
    return {"type": "TRIAL_DONE", "payload": {"trial": t, "reused_measurement": reused}}


def _run(tmp: Path, name: str, events: list[dict]) -> Path:
    """A run directory shaped like the real thing: <runs>/<arm>/<run-id>/events.jsonl, because
    the arm identity is the PARENT directory -- n1-a and n1-b shared one run_id, so a reader
    keyed on run_id cannot tell the arms apart."""
    d = tmp / name / "run-x"
    d.mkdir(parents=True)
    with io.open(d / "events.jsonl", "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    return d


def _finished(**best):
    # Field shapes copied from a real journal: the honest verdict is NESTED under
    # `honest_verdict`, `precision` is top-level, and knobs live under params.values.
    # `speedups` is mean-over-mean and every mean was rounded to 3 significant figures by
    # KernelBench, so `speedups_median` is carried alongside as the full-precision twin.
    b = {"candidate_id": "cand-1", "family_id": "fam-1", "final_reeval_median_ms": 3.0,
         "final_reeval_ok": True, "excessive_speedup_flag": False, "precision": "fp16",
         "tuned_ms": 3.05,
         "honest_verdict": {"candidate_precision": "fp16",
                            "compared_against": "torch_compile_tf32",
                            "same_precision_speedup": 2.0,
                            "beats_same_precision_baseline": True},
         "speedups": {"torch_compile_tf32": 2.0},
         "speedups_median": {"torch_compile_tf32": 2.0},
         "params": {"values": {"COMPUTE_DTYPE": "fp16"}}}
    b.update(best)
    return {"type": "RUN_FINISHED",
            "payload": {"summary": {"best": b, "elapsed_hours": 12.1}}}


def _run_probe(off: Path, act: Path, *flags: str) -> str:
    out = subprocess.run([sys.executable, str(PROBE), str(off), str(act), *flags],
                         capture_output=True)
    txt = out.stdout.decode("utf-8", errors="replace")
    assert out.returncode == 0, txt + out.stderr.decode("utf-8", errors="replace")
    return txt


def test_gpu_wall_is_the_comparability_axis_not_trial_count(tmp_path):
    """The arm with FEWER trials can hold MORE GPU wall. The reader must say the wall is
    equalised and mark the count as not the check -- reading counts would invert the verdict."""
    off = _run(tmp_path, "off", [_trial("t%d" % i, "c", 5.0, wall=10.0) for i in range(10)]
               + [_finished()])                      # 10 trials x 10 s = 100 s
    act = _run(tmp_path, "act", [_trial("t%d" % i, "c", 5.0, wall=50.0) for i in range(4)]
               + [_finished()])                      # 4 trials x 50 s = 200 s
    txt = _run_probe(off, act)
    assert "0.03" in txt and "0.06" in txt          # 100 s and 200 s, in hours
    assert "+0.03 h (+100.0%)" in txt
    assert "count is NOT the comparability check" in txt
    # the trial-count row must show the OPPOSITE sign from the wall row
    assert "-6" in txt


def test_latency_keys_have_no_ms_suffix(tmp_path):
    """If the reader looked for `median_ms`, tuned_best would be missing for every trial and
    the table would print '-' while the data is perfectly present."""
    off = _run(tmp_path, "off", [_trial("t1", "c", 4.25), _finished()])
    act = _run(tmp_path, "act", [_trial("t1", "c", 3.75), _finished()])
    txt = _run_probe(off, act, "--final")
    assert "4.2500 ms" in txt
    assert "3.7500 ms" in txt


def test_control_arm_firing_the_mechanism_is_flagged(tmp_path):
    """Window 1's n1 pair was off/off. The inverse failure -- leakage INTO the control arm --
    invalidates every downstream attribution, so it must be called out, not averaged."""
    off = _run(tmp_path, "off", [_trial("t1", "c", 5.0),
                                 {"type": "SCAN_BLOCK_DONE", "payload": {}},
                                 _finished()])
    act = _run(tmp_path, "act", [_trial("t1", "c", 5.0),
                                 {"type": "SCAN_BLOCK_DONE", "payload": {}},
                                 _finished()])
    txt = _run_probe(off, act, "--mechanism")
    assert "NON-ZERO IN CONTROL ARM" in txt
    assert "NOT a clean" in txt
    assert "SEPARATED" not in txt


def test_both_arms_silent_is_not_a_negative_result(tmp_path):
    """off/off must read as 'check the config', never as 'the mechanism did nothing'."""
    off = _run(tmp_path, "off", [_trial("t1", "c", 5.0), _finished()])
    act = _run(tmp_path, "act", [_trial("t1", "c", 5.0), _finished()])
    txt = _run_probe(off, act, "--mechanism")
    assert "NEITHER arm fired" in txt
    assert "not a negative result" in txt
    assert "SEPARATED" not in txt


def test_clean_separation_is_reported_as_such(tmp_path):
    off = _run(tmp_path, "off", [_trial("t1", "c", 5.0), _finished()])
    act = _run(tmp_path, "act", [_trial("t1", "c", 5.0),
                                 {"type": "SCAN_BLOCK_DONE", "payload": {}},
                                 {"type": "CONDITIONED_BRIEF_DELIVERED", "payload": {}},
                                 _finished()])
    txt = _run_probe(off, act, "--mechanism")
    assert "SEPARATED" in txt
    assert "NON-ZERO IN CONTROL ARM" not in txt


def test_round_number_two_without_a_second_generation(tmp_path):
    """The exact window-1 shape: four families, three recording round 1 and one recording
    round 2, and NO family with two records. That is one generation, and the reader must say
    so -- calling it a second round is how a cumulative gain gets read as a per-round increment."""
    evs = [_trial("t1", "c", 5.0)]
    for i, (fam, rnd, gain) in enumerate([("fam-a", 1, 11.27), ("fam-b", 1, 26.62),
                                          ("fam-c", 1, 36.05), ("fam-d", 2, 53.83)]):
        evs.append({"type": "FAMILY_SEEDED", "payload": {"family": {"family_id": fam}}})
        evs.append({"type": "FAMILY_ROUND_RECORDED",
                    "payload": {"family_id": fam, "round": rnd, "latency_gain_pct": gain,
                                "conversion": "improved"}})
    evs.append(_finished())
    off = _run(tmp_path, "off", evs)
    act = _run(tmp_path, "act", list(evs))
    txt = _run_probe(off, act)
    assert "NO family reached a second round" in txt
    assert "one generation only" in txt
    assert "not a per-round increment" in txt


def test_a_real_second_generation_is_recognised(tmp_path):
    """The positive counterpart: one family with records for round 1 AND round 2 is a curve,
    and must NOT print the one-generation warning."""
    evs = [_trial("t1", "c", 5.0),
           {"type": "FAMILY_SEEDED", "payload": {"family": {"family_id": "fam-a"}}}]
    for rnd, gain in ((1, 10.0), (2, 4.0)):
        evs.append({"type": "FAMILY_ROUND_RECORDED",
                    "payload": {"family_id": "fam-a", "round": rnd,
                                "latency_gain_pct": gain, "conversion": "improved"}})
    evs.append(_finished())
    off = _run(tmp_path, "off", evs)
    act = _run(tmp_path, "act", list(evs))
    txt = _run_probe(off, act)
    assert "multi-round families" in txt
    assert "'fam-a': [1, 2]" in txt
    assert "NO family reached a second round" not in txt


def test_final_comes_from_run_finished_not_from_trials(tmp_path):
    """The tuned best and the re-eval median are DIFFERENT quantities. A reader that reports
    the trial minimum as the run's result reports a number the run itself did not claim."""
    off = _run(tmp_path, "off", [_trial("t1", "c", 2.9999),
                                 _finished(final_reeval_median_ms=3.0126)])
    act = _run(tmp_path, "act", [_trial("t1", "c", 2.5001),
                                 _finished(final_reeval_median_ms=2.5375)])
    txt = _run_probe(off, act, "--final")
    assert "3.0126" in txt and "2.5375" in txt
    assert "2.9999 ms" in txt and "2.5001 ms" in txt      # printed, but as the tuned best
    # the arm difference must use the re-eval values: (2.5375-3.0126)/3.0126 = -15.77%
    assert "-15.77%" in txt


def test_no_verdict_on_the_end_to_end_difference(tmp_path):
    """No tie band is calibrated, so `--final` must refuse to name a winner however large
    the gap looks."""
    off = _run(tmp_path, "off", [_trial("t1", "c", 5.0), _finished(final_reeval_median_ms=9.0)])
    act = _run(tmp_path, "act", [_trial("t1", "c", 5.0), _finished(final_reeval_median_ms=3.0)])
    txt = _run_probe(off, act, "--final")
    assert "NO VERDICT IS PRINTED HERE" in txt
    assert "-66.67%" in txt
    for word in ("wins", "winner is", "better arm", "significant"):
        assert word not in txt.lower()


def test_reused_records_are_counted_separately(tmp_path):
    """A reused record is not an independent measurement; pooling it into the trial count
    without saying so overstates the evidence."""
    off = _run(tmp_path, "off", [_trial("t1", "c", 5.0),
                                 _trial("t2", "c", 5.0, reused=True), _finished()])
    act = _run(tmp_path, "act", [_trial("t1", "c", 5.0), _finished()])
    txt = _run_probe(off, act)
    assert "marked reused" in txt
    assert "not independent measurements" in txt


def test_trials_missing_job_wall_are_disclosed(tmp_path):
    """A missing `job_wall_s` contributes zero hours. Silently absorbing it would make an arm
    look cheap; the count must surface."""
    off = _run(tmp_path, "off", [_trial("t1", "c", 5.0, wall=None), _finished()])
    act = _run(tmp_path, "act", [_trial("t1", "c", 5.0, wall=36.0), _finished()])
    txt = _run_probe(off, act)
    assert "trials w/o job_wall_s" in txt
    assert "contribute 0 h" in txt


def test_arm_identity_comes_from_the_parent_directory(tmp_path):
    """n1-a and n1-b shared one run_id. The label must come from the parent dir, or the two
    arms of the primary contrast are indistinguishable in the output."""
    off = _run(tmp_path, "m1-a", [_trial("t1", "c", 5.0), _finished()])
    act = _run(tmp_path, "m1-b", [_trial("t1", "c", 5.0), _finished()])
    txt = _run_probe(off, act)
    assert "m1-a" in txt and "m1-b" in txt


def test_speedup_vs_eager_is_never_printed(tmp_path):
    """It is a cross-precision ratio (fp16 vs fp32) and is not reportable, so it must not
    appear even when the journal carries it -- and the real journal always does."""
    fin = _finished()
    fin["payload"]["summary"]["best"]["speedup_vs_eager"] = 7.3356
    fin["payload"]["summary"]["best"]["speedups"] = {"eager": 7.3356,
                                                    "torch_compile_tf32": 3.8408}
    off = _run(tmp_path, "off", [_trial("t1", "c", 5.0), fin])
    act = _run(tmp_path, "act", [_trial("t1", "c", 5.0), _finished()])
    txt = _run_probe(off, act, "--final")
    assert "7.33" not in txt
    assert "speedup_vs_eager" not in txt


def test_gain_percent_carries_its_start_point_caveat(tmp_path):
    """gain% is relative to each family's seed, so it is start-point sensitive. Printing it
    without that caveat invites a cross-family comparison that measures seed quality."""
    evs = [_trial("t1", "c", 5.0),
           {"type": "FAMILY_SEEDED", "payload": {"family": {"family_id": "fam-a"}}},
           {"type": "FAMILY_ROUND_RECORDED",
            "payload": {"family_id": "fam-a", "round": 1, "latency_gain_pct": 53.83,
                        "conversion": "improved"}},
           _finished()]
    off = _run(tmp_path, "off", evs)
    act = _run(tmp_path, "act", list(evs))
    txt = _run_probe(off, act)
    assert "must normalise first" in txt
    assert "53.83" in txt


def test_the_honest_verdict_is_read_from_its_nested_home(tmp_path):
    """Guessing this field's location is how a present value prints as None: the real journal
    nests the same-precision verdict under best.honest_verdict, and the first live dry-run of
    this reader printed None for all three because they were read flat off `best`."""
    off = _run(tmp_path, "off", [_trial("t1", "c", 5.0), _finished()])
    act = _run(tmp_path, "act", [_trial("t1", "c", 5.0), _finished()])
    txt = _run_probe(off, act, "--final")
    assert "2.0" in txt
    assert "torch_compile_tf32" in txt
    assert "None  (vs None)" not in txt


def test_an_identical_published_speedup_is_flagged_as_quantization(tmp_path):
    """The real n1-b / m1-a shape. Two different kernels (median 2.537472 vs 2.539456, tf32
    baselines 10.948608 vs 10.948544) both publish 4.3307, because KernelBench stores every
    mean as `float(f"{np.mean(...):.3g}")` and the published ratio is mean-over-mean: both
    sides land on 11.0 / 2.54. A reader that shows only that row invites the conclusion that
    the arms tied, when the medians differ by 0.08%. The flag must fire, and the median
    convention must be printed with its own arm difference."""
    off = _run(tmp_path, "n1-b", [
        _trial("t1", "c", 2.537472),
        _finished(final_reeval_median_ms=2.537472,
                  speedups_median={"torch_compile_tf32": 4.3148})])
    act = _run(tmp_path, "m1-a", [
        _trial("t1", "c", 2.539456),
        _finished(final_reeval_median_ms=2.539456,
                  speedups_median={"torch_compile_tf32": 4.3114})])
    txt = _run_probe(off, act, "--final")
    assert "THIS PAIR IS SUCH A CASE" in txt
    assert "resolution floor near 1.3%" in txt
    # both conventions present, each with its own arm difference
    assert "4.3148" in txt and "4.3114" in txt
    assert "-0.08%" in txt          # median convention: (4.3114-4.3148)/4.3148
    assert "+0.08%" in txt          # latency: (2.539456-2.537472)/2.537472
    # and the tie must never be stated as a result
    assert "NO VERDICT IS PRINTED HERE" in txt


def test_differing_speedups_do_not_trip_the_quantization_flag(tmp_path):
    """The negative control: the warning about the resolution floor is always printed (it is a
    property of the field, not of this pair), but the 'THIS PAIR' line must fire only when the
    two arms actually collide on one value -- otherwise it would cry wolf on every read."""
    off = _run(tmp_path, "off", [_trial("t1", "c", 5.0), _finished()])
    act = _run(tmp_path, "act", [_trial("t1", "c", 5.0),
                                 _finished(honest_verdict={
                                     "candidate_precision": "fp16",
                                     "compared_against": "torch_compile_tf32",
                                     "same_precision_speedup": 2.4,
                                     "beats_same_precision_baseline": True})])
    txt = _run_probe(off, act, "--final")
    assert "resolution floor near 1.3%" in txt
    assert "THIS PAIR IS SUCH A CASE" not in txt


def test_the_median_speedup_is_keyed_by_the_verdicts_own_comparator(tmp_path):
    """`speedups_median` holds every baseline kind; picking the wrong key would compare the
    candidate against a baseline the honest verdict did not use. The key must come from
    `compared_against`, so a journal whose verdict names tf32 must not read the ieee entry."""
    off = _run(tmp_path, "off", [_trial("t1", "c", 5.0),
                                 _finished(speedups_median={"torch_compile": 9.99,
                                                            "torch_compile_tf32": 4.3148})])
    act = _run(tmp_path, "act", [_trial("t1", "c", 5.0),
                                 _finished(speedups_median={"torch_compile": 8.88,
                                                            "torch_compile_tf32": 4.3114})])
    txt = _run_probe(off, act, "--final")
    assert "4.3148" in txt and "4.3114" in txt
    assert "9.99" not in txt and "8.88" not in txt


def test_a_missing_speedups_median_does_not_fabricate_a_difference(tmp_path):
    """Older journals can carry `speedups_median_note` instead of the dict (it is omitted when
    no baseline has a real median). The row must read '-' and NO median arm difference may be
    printed -- inventing one from the mean ratio is exactly the mixed-statistic error the
    production guard exists to prevent."""
    fin_o = _finished(final_reeval_median_ms=3.0)
    fin_a = _finished(final_reeval_median_ms=2.7)
    for f in (fin_o, fin_a):
        del f["payload"]["summary"]["best"]["speedups_median"]
        f["payload"]["summary"]["best"]["speedups_median_note"] = "not computed"
    off = _run(tmp_path, "off", [_trial("t1", "c", 5.0), fin_o])
    act = _run(tmp_path, "act", [_trial("t1", "c", 5.0), fin_a])
    txt = _run_probe(off, act, "--final")
    assert "same_precision_speedup [median]  : None" in txt
    assert "median convention" not in txt
    assert "-10.00%" in txt        # the latency difference is still read, from the medians
