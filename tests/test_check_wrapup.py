"""The wrap-up checker must not report a clean zero when its key path is wrong.

Every reader in `check_wrapup.py` walks nested event payloads, and this project has a recorded
failure mode for exactly that: `TRIAL_DONE.payload.trial.params.values` read as
`payload["params"]` yields None for every trial and prints a plausible all-None table. The
checker's own first draft had two of these -- it looked for a `FINAL_REEVAL_DONE` event that does
not exist, and read `BASELINE_DONE.payload["latency_ms"]` instead of `payload.baseline.latency_ms`,
so it reported "NONE YET" and "-" for a FINISHED run that has both numbers.

A checker that under-reports is worse than no checker at wrap-up: "0 rounds carry `conversion`" is
the exact string the preflight doc says to treat as a NEW DEFECT, so a broken reader manufactures a
defect report. These tests pin each reader to the real event shapes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import check_wrapup  # noqa: E402


def _write_run(tmp_path: Path, events: list[dict]) -> Path:
    d = tmp_path / "run-test"
    d.mkdir(parents=True)
    with (d / "events.jsonl").open("w", encoding="utf-8") as fh:
        for i, e in enumerate(events):
            fh.write(json.dumps({"seq": i, "ts": 1789000000.0 + i, **e}) + "\n")
    return d


# The REAL shapes, copied from what the orchestrator writes (verified against src/, and against
# the box-2 corpus on disk). If the harness changes these, these tests must fail -- that is the
# point of holding them verbatim rather than constructing them from the reader's expectations.
_BASELINE = {"type": "BASELINE_DONE", "payload": {"baseline": {
    "kind": "eager", "latency_ms": {"mean": 21.5, "median": 21.4569, "std": 0.05,
                                    "min": 21.2, "max": 21.6, "n_samples": 100}}}}
_TRIAL_OK = {"type": "TRIAL_DONE", "payload": {"trial": {
    "trial_id": "tr-1", "candidate_id": "c1", "space_id": "sp",
    "params": {"values": {"COMPUTE_DTYPE": "bf16", "DOT_MODE": "plain"}},
    "status": "complete", "failure_kind": None,
    "latency_ms": {"mean": 2.77, "median": 2.7674, "std": 0.01, "min": 2.7,
                   "max": 2.8, "n_samples": 20}}}}
_RUN_FINISHED = {"type": "RUN_FINISHED", "payload": {"summary": {"best": {
    "candidate_id": "c1", "family_id": "f1", "tuned_ms": 2.7674,
    "final_reeval_ok": True, "final_reeval_ms": 2.76, "final_reeval_median_ms": 2.7628,
    "excessive_speedup_flag": False, "precision": "bf16"}}}}
_ROUND_WITH_CONV = {"type": "FAMILY_ROUND_RECORDED", "payload": {
    "family_id": "f1", "best_ms": 2.9, "round": 1, "conversion": "improved",
    "latency_gain_pct": 3.4, "resource_deltas": {"n_regs": {"before": 200, "after": 128,
                                                            "direction": "improved"}}}}
_ROUND_NO_CONV = {"type": "FAMILY_ROUND_RECORDED", "payload": {
    "family_id": "f1", "best_ms": 2.9, "round": 1}}


# --- reader 2: the one that was actually broken ------------------------------------------------


def test_the_final_reeval_is_found_in_run_finished(tmp_path):
    """The regression. `final_reeval_ms` lives in `RUN_FINISHED.payload.summary.best`; there is no
    `FINAL_REEVAL_DONE` event. The first draft looked for that event and reported NONE YET for a
    finished run -- which at wrap-up reads as "the run produced no result"."""
    d = _write_run(tmp_path, [_BASELINE, _TRIAL_OK, _RUN_FINISHED])
    out = check_wrapup.final_result(d)
    assert out["final_reeval_ms"] == pytest.approx(2.76), (
        "the final re-eval was not found where the orchestrator writes it")
    assert out["used_fallback"] is False
    assert out["final_reeval_median_ms"] == pytest.approx(2.7628)
    assert out["precision"] == "bf16"


def test_the_baseline_is_read_through_its_nesting(tmp_path):
    """`BASELINE_DONE` nests under `payload.baseline`. Reading `payload["latency_ms"]` returns
    nothing and prints "-", which looks like a run that measured no baseline."""
    d = _write_run(tmp_path, [_BASELINE, _RUN_FINISHED])
    out = check_wrapup.final_result(d)
    assert out["baselines"] == {"eager": pytest.approx(21.4569)}, (
        "the baseline latency was not read through payload.baseline")


def test_tuned_ms_is_never_substituted_for_the_reeval(tmp_path):
    """The J2-5 criterion names `final_reeval_ms` because `tuned_ms` is optimistic by 1.5-6.7%.
    An unfinished run must report the absence, not quietly fall back."""
    d = _write_run(tmp_path, [_BASELINE, _TRIAL_OK])       # no RUN_FINISHED
    out = check_wrapup.final_result(d)
    assert out["final_reeval_ms"] is None
    assert out["used_fallback"] is True
    assert out["best_trial_median_ms"] == pytest.approx(2.7674), (
        "the trial median should still be REPORTED -- just never as the result")


def test_the_arm_comparison_refuses_to_decide_without_both_reevals(tmp_path):
    """Silently comparing one arm's re-eval against the other's tuned_ms would bias the answer by
    more than the noise floor it is being compared against."""
    have = {"final_reeval_ms": 3.0}
    missing = {"final_reeval_ms": None}
    msg = check_wrapup.compare_arms(have, missing, 2.35)
    assert "CANNOT DECIDE" in msg and "treatment" in msg
    msg2 = check_wrapup.compare_arms(missing, have, 2.35)
    assert "CANNOT DECIDE" in msg2 and "control" in msg2


def test_the_arm_comparison_uses_the_noise_floor_in_both_directions():
    """A treatment arm inside the floor passes; beyond it fails. Both directions asserted, because
    a comparison that always passes is the same as no comparison."""
    ok = check_wrapup.compare_arms({"final_reeval_ms": 3.0}, {"final_reeval_ms": 3.06}, 2.35)
    assert ok.startswith("PASS"), ok
    bad = check_wrapup.compare_arms({"final_reeval_ms": 3.0}, {"final_reeval_ms": 3.20}, 2.35)
    assert bad.startswith("FAIL"), bad
    # And an IMPROVEMENT must pass, not trip a two-sided test: J2-5 is "not worse".
    better = check_wrapup.compare_arms({"final_reeval_ms": 3.0}, {"final_reeval_ms": 2.5}, 2.35)
    assert better.startswith("PASS"), better


# --- reader 1: G27, where a wrong reading manufactures a defect report -------------------------


def test_zero_rounds_is_not_reported_as_a_defect(tmp_path):
    """0-of-0 says nothing. The preflight doc's rule -- "if still 0 that is a NEW defect" -- only
    applies once rounds exist, and a checker that skips that distinction reports a defect for
    every run that never got to a rewrite."""
    d = _write_run(tmp_path, [_BASELINE, _TRIAL_OK])
    out = check_wrapup.check_conversion(d)
    assert out["rounds"] == 0
    assert "NOT YET DECIDABLE" in out["verdict"]
    assert "DEFECT" not in out["verdict"]


def test_rounds_without_conversion_are_reported_as_the_new_defect(tmp_path):
    """The G27 case. `conversion_verdict` always returns a `conversion` key, so a round lacking it
    cannot mean "nothing converted"."""
    d = _write_run(tmp_path, [_ROUND_NO_CONV, _ROUND_NO_CONV])
    out = check_wrapup.check_conversion(d)
    assert out["rounds"] == 2 and out["with_conversion"] == 0
    assert "NEW DEFECT" in out["verdict"]


def test_rounds_with_conversion_are_reported_as_the_first_evidence(tmp_path):
    d = _write_run(tmp_path, [_ROUND_WITH_CONV, _ROUND_WITH_CONV])
    out = check_wrapup.check_conversion(d)
    assert out["rounds"] == 2 and out["with_conversion"] == 2
    assert out["with_resource_deltas"] == 2
    assert out["verdicts"] == {"improved": 2}
    assert out["verdict"].startswith("PASS")


def test_a_partial_conversion_is_not_rounded_up_to_pass(tmp_path):
    """Some rounds carrying the verdict and some not is the G44 shape and must not read as PASS."""
    d = _write_run(tmp_path, [_ROUND_WITH_CONV, _ROUND_NO_CONV])
    out = check_wrapup.check_conversion(d)
    assert out["verdict"].startswith("PARTIAL")


# --- reader 3: S3/S4', where the silent failure is a clamp ------------------------------------


def test_a_below_floor_reading_is_counted_as_reported(tmp_path):
    """A well-fused candidate reads below `compulsory_bytes`. The requirement is that it be
    REPORTED, not clamped, so the checker has to count it."""
    ev = {"type": "DIMENSION_STATE", "payload": {"candidate_id": "c1", "records": [
        {"dimension_id": "dram_bytes", "measured": 1.0e8, "ceiling": 9.0e8,
         "provenance": {"source": "calibration", "precision": "fp16"},
         "bound": {"floor": 1.3e8, "below_floor": True, "room": -3.0e7}},
    ]}}
    d = _write_run(tmp_path, [ev])
    out = check_wrapup.check_s3_s4(d)
    assert out["below_floor_readings_reported"] == 1
    assert out["per_record_provenance_naming_a_precision"] == 1
    assert out["dimensions_seen"] == {"dram_bytes": 1}


def test_a_reading_pinned_exactly_on_the_floor_is_flagged_for_inspection(tmp_path):
    """`room == 0.0` with a floor present is what a clamp looks like from the outside. The checker
    cannot prove it is one, so it flags rather than concludes."""
    ev = {"type": "DIMENSION_STATE", "payload": {"candidate_id": "c1", "records": [
        {"dimension_id": "dram_bytes", "measured": 1.3e8, "ceiling": 9.0e8,
         "provenance": {"source": "calibration", "precision": ""},
         "bound": {"floor": 1.3e8, "below_floor": False, "room": 0.0}},
    ]}}
    d = _write_run(tmp_path, [ev])
    out = check_wrapup.check_s3_s4(d)
    assert out["readings_sitting_exactly_on_the_floor"] == 1
    assert out["per_record_provenance_naming_a_precision"] == 0, (
        "an empty precision string must not count as naming one")


def test_a_non_derivable_floor_is_not_counted_as_below_floor(tmp_path):
    """`floor: null` is the honest answer for occupancy and registers (no closed form, and the
    SIGN is unreliable). It must not be read as a violation, or every run reports many."""
    ev = {"type": "DIMENSION_STATE", "payload": {"candidate_id": "c1", "records": [
        {"dimension_id": "n_regs", "measured": 128.0, "ceiling": 255.0,
         "provenance": {"source": "device_query", "precision": ""},
         "bound": {"floor": None, "source": "none", "below_floor": False,
                   "room": None, "reason": "not derivable"}},
    ]}}
    d = _write_run(tmp_path, [ev])
    out = check_wrapup.check_s3_s4(d)
    assert out["below_floor_readings_reported"] == 0
    assert out["readings_sitting_exactly_on_the_floor"] == 0


def test_the_checker_runs_against_a_real_corpus_run_if_present():
    """The positive control: a real FINISHED run must yield real numbers, not a clean sheet of
    zeros. Skipped when the fetched corpus is absent, since it is not in the repo."""
    corpus = (Path(__file__).resolve().parents[2] / "external_files" / "box2-runs" /
              "runs-l3" / "run-l3-43-20260909-015247")
    if not (corpus / "events.jsonl").exists():
        pytest.skip("fetched corpus not present on this box")
    fin = check_wrapup.final_result(corpus)
    assert fin["final_reeval_ms"] is not None, (
        "a finished run read as having no result -- the failure this file exists for")
    assert len(fin["baselines"]) >= 2, "baselines came back empty on a run that measured four"
    conv = check_wrapup.check_conversion(corpus)
    assert conv["rounds"] > 0, "a run with rewrite rounds read as having none"
    # This corpus PREDATES the conversion fix, so zero-with-rounds is the correct reading here and
    # is what makes it a usable control for the wrap-up check.
    assert conv["with_conversion"] == 0 and "NEW DEFECT" in conv["verdict"]


def test_the_compute_ceiling_precision_is_read_from_the_top_level_field(tmp_path):
    """The reader bug this test exists for, found on live box-2 data.

    `_do_diagnose` writes `compute_ceiling_provenance` BESIDE `records`, not inside them, because
    exactly one dimension has a precision: the compute-pressure denominator, the only one that can
    be the wrong denominator without anything looking wrong (an fp16 kernel against a tf32 ceiling
    read 107.8% of peak). The per-record blocks are `definitional` / `device_query` and their
    `precision` is legitimately EMPTY -- a hardware limit has no precision.

    So counting per-record precisions reports 0 on a run whose S3 field says `precision: "fp16"`
    with a full calibration identity. That is the same shape as looking for a `FINAL_REEVAL_DONE`
    event: a clean zero on data that has the number.
    """
    ev = {"type": "DIMENSION_STATE", "payload": {
        "candidate_id": "c1",
        "prompt_mode": "vector",
        # Verbatim from box 2's live events.jsonl.
        "compute_ceiling_provenance": {
            "source": "measured", "precision": "fp16", "backend": "",
            "measured_at": "2026-09-10T20:43:45+00:00",
            "calibration_identity": "NVIDIA GeForce RTX 4090|8.9|128|2.13.0+cu129|12.9|3.7.1",
            "note": "the compute roof in force for this candidate, labelled 'tensor-core (fp16), "
                    "Triton-measured (above cuBLAS)' by the classifier"},
        "precision_mismatch": "",
        "unreachable_ceilings": [],
        "records": [
            {"dimension_id": "occupancy", "measured": None, "ceiling": 1.0,
             "provenance": {"source": "definitional", "precision": ""},
             "bound": {"floor": None, "below_floor": False, "room": None}},
            {"dimension_id": "n_regs", "measured": 128.0, "ceiling": 255.0,
             "provenance": {"source": "device_query", "precision": ""},
             "bound": {"floor": None, "below_floor": False, "room": None}},
        ]}}
    d = _write_run(tmp_path, [ev])
    out = check_wrapup.check_s3_s4(d)
    assert out["diagnoses"] == 1
    assert out["diagnoses_whose_compute_ceiling_names_a_precision"] == 1, (
        "the S3 precision was not read from the top-level compute_ceiling_provenance field")
    assert out["calibration_identities"] == [
        "NVIDIA GeForce RTX 4090|8.9|128|2.13.0+cu129|12.9|3.7.1"]
    # And the per-record count must stay 0 and be reported SEPARATELY: a zero means opposite things
    # in the two places, so collapsing them into one number loses the distinction either way.
    assert out["per_record_provenance_naming_a_precision"] == 0
    assert out["dimension_records"] == 2
    assert out["prompt_modes"] == {"vector": 1}


def test_a_diagnosis_with_no_compute_ceiling_reports_zero_not_a_crash(tmp_path):
    """An overhead-bound or cannot-run candidate has no compute roof in force, and
    `compute_ceiling_provenance` then carries `source: "none"` with an empty precision. That zero is
    honest and must be distinguishable from the reader failing to look."""
    ev = {"type": "DIMENSION_STATE", "payload": {
        "candidate_id": "c2",
        "compute_ceiling_provenance": {
            "source": "none", "precision": "", "backend": "", "measured_at": "",
            "calibration_identity": "", "note": "no compute ceiling was in force"},
        "precision_mismatch": "",
        "records": [{"dimension_id": "n_spills", "measured": 0.0, "ceiling": 0.0,
                     "provenance": {"source": "definitional", "precision": ""},
                     "bound": {"floor": 0.0, "below_floor": False, "room": 0.0}}]}}
    d = _write_run(tmp_path, [ev])
    out = check_wrapup.check_s3_s4(d)
    assert out["diagnoses"] == 1
    assert out["diagnoses_whose_compute_ceiling_names_a_precision"] == 0
    assert out["calibration_identities"] == []


def test_the_precision_mismatch_is_read_from_the_top_level_field(tmp_path):
    """`precision_mismatch` sits beside the provenance, and it is the 107.8% incident's detector:
    an fp16 kernel scored against a tf32 ceiling. It must be attributed to its candidate."""
    ev = {"type": "DIMENSION_STATE", "payload": {
        "candidate_id": "c3",
        "compute_ceiling_provenance": {"source": "measured", "precision": "tf32",
                                       "calibration_identity": "box|1"},
        "precision_mismatch": "the ceiling was measured at tf32 but the kernel computes in fp16",
        "records": []}}
    d = _write_run(tmp_path, [ev])
    out = check_wrapup.check_s3_s4(d)
    assert out["precision_mismatch_fired_on"] == ["c3"], (
        "a precision mismatch that fired was not attributed to its candidate")


# --- reader 3b: S2d's ledger, where an event stream can look healthy and say nothing -----------

_RECON_REAL = {"type": "EXPECTATIONS_RECONCILED", "payload": {
    "family_id": "f1", "round": 1, "id": "h1", "change": "fuse the projections",
    "reconciliation": {"verdict": "confirmed", "checked": 2},
    "conversion": "improved", "latency_gain_pct": 3.4, "n_declared": 2}}
_RECON_EMPTY = {"type": "EXPECTATIONS_RECONCILED", "payload": {
    "family_id": "f1", "round": 1, "id": "(no hypothesis id)", "change": "",
    "reconciliation": {}, "conversion": "improved", "n_declared": 0}}
_RECON_FAILED = {"type": "EXPECTATIONS_RECONCILE_FAILED", "payload": {
    "family_id": "f1", "round": 1, "error": "KeyError: 'direction'"}}


def test_an_all_empty_ledger_is_not_reported_as_working(tmp_path):
    """THE trap. `n_declared: 0` means the entry reconciled nothing, and a count of events would
    report a healthy ledger built entirely of empty entries -- computed, journalled, saying
    nothing."""
    d = _write_run(tmp_path, [_ROUND_WITH_CONV, _RECON_EMPTY, _RECON_EMPTY])
    out = check_wrapup.check_reconciliation(d)
    assert out["entries"] == 2 and out["empty_entries"] == 2
    assert "EMPTY LEDGER" in out["verdict"]
    assert not out["verdict"].startswith("PASS")


def test_a_real_ledger_entry_is_reported_as_pass(tmp_path):
    d = _write_run(tmp_path, [_ROUND_WITH_CONV, _RECON_REAL])
    out = check_wrapup.check_reconciliation(d)
    assert out["verdict"].startswith("PASS")
    assert out["declarations_reconciled"] == 2
    assert out["verdict_kinds"] == {"confirmed": 1}


def test_zero_ledger_entries_with_zero_rounds_is_not_a_defect(tmp_path):
    """Reconciliation fires in the same `if evaluated:` branch as `conversion`, so before any
    rewrite round a zero says nothing -- the same distinction check 1 makes."""
    d = _write_run(tmp_path, [_BASELINE, _TRIAL_OK])
    out = check_wrapup.check_reconciliation(d)
    assert "NOT YET DECIDABLE" in out["verdict"]
    assert "DEFECT" not in out["verdict"]


def test_zero_ledger_entries_with_rounds_is_a_defect_in_either_arm(tmp_path):
    """The ledger is journalled UNCONDITIONALLY -- the switch gates only whether the rendered form
    reaches the rewriter's prompt. So a control-arm run with rounds and no entries is a defect,
    not the switch being off."""
    d = _write_run(tmp_path, [_ROUND_WITH_CONV, _ROUND_WITH_CONV])
    out = check_wrapup.check_reconciliation(d)
    assert "DEFECT" in out["verdict"]
    assert "journalled" in out["verdict"].lower() or "UNCONDITIONALLY" in out["verdict"]


def test_a_reconcile_failure_is_surfaced_because_it_is_otherwise_silent(tmp_path):
    """`_record_reconciliation` catches everything so a diagnostic cannot end a rewrite round.
    That is the right design and it makes total failure invisible apart from this event."""
    d = _write_run(tmp_path, [_ROUND_WITH_CONV, _RECON_FAILED])
    out = check_wrapup.check_reconciliation(d)
    assert out["reconcile_failed"] == 1
    assert "RECONCILE_FAILED" in out["verdict"]
    assert "KeyError" in out["verdict"], "the error text is dropped, so it is not actionable"


def test_a_partly_empty_ledger_is_not_rounded_up(tmp_path):
    d = _write_run(tmp_path, [_ROUND_WITH_CONV, _RECON_REAL, _RECON_EMPTY])
    out = check_wrapup.check_reconciliation(d)
    assert out["verdict"].startswith("PARTIAL")


# --- the tolerance itself, which was borrowed from a correctness measurement -------------------


def test_the_latency_floor_is_measured_from_the_reeval_not_borrowed(tmp_path):
    """The 2.35% tolerance is `1 - 0.9765`, where 0.9765 is the reference's own frac_within_tol at
    two precisions -- a fraction of ELEMENTS agreeing. J2-5 compares LATENCIES, and there is no
    reason a numerics figure should equal a timing-jitter figure.

    `final_reeval` re-runs theta_best in a fresh process, so |tuned_ms - final_reeval_ms| is a
    same-kernel, same-box latency re-measurement -- the right units.
    """
    d = _write_run(tmp_path, [_BASELINE, _RUN_FINISHED])
    floor, prov = check_wrapup.latency_floor_from_runs(d)
    # _RUN_FINISHED: tuned 2.7674 -> reeval 2.76, i.e. 0.267%
    assert floor == pytest.approx(0.267, abs=0.01), (
        "the latency floor was not computed from the same-kernel re-eval delta")
    assert "n=1" in prov and "measured" in prov


def test_the_latency_floor_takes_the_WIDEST_delta_across_arms(tmp_path):
    """With two arms there are two same-kernel deltas, and a verdict must not be stricter than the
    wider one -- otherwise a difference smaller than the measurement gets called a regression."""
    a = _write_run(tmp_path / "a", [_RUN_FINISHED])
    wide = {"type": "RUN_FINISHED", "payload": {"summary": {"best": {
        "tuned_ms": 4.0, "final_reeval_ms": 4.2, "final_reeval_ok": True}}}}
    b = _write_run(tmp_path / "b", [wide])
    floor, prov = check_wrapup.latency_floor_from_runs(a, b)
    assert floor == pytest.approx(5.0, abs=0.01), "expected the 5%% delta, not the 0.27%% one"
    assert "n=2" in prov


def test_no_reeval_yet_reports_the_absence_rather_than_zero(tmp_path):
    """A floor of 0.0 would make every difference a regression. The absence has to be reported."""
    d = _write_run(tmp_path, [_BASELINE, _TRIAL_OK])
    floor, prov = check_wrapup.latency_floor_from_runs(d)
    assert floor is None
    assert "no arm has re-evaluated" in prov
