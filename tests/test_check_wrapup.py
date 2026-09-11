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


_WALL_BATCH = {"type": "WALL_CLOCK_REACHED", "payload": {
    "elapsed_hours": 13.51, "budget_hours": 12.0, "pipelined": 1, "skipped": 1,
    "detail": "wall clock reached; the remaining candidates in this batch were not tuned."}}
_WALL_ROUND = {"type": "WALL_CLOCK_REACHED", "payload": {
    "elapsed_hours": 13.51, "budget_hours": 12.0, "round": 2,
    "stopped_before_family": "fam-1402fc69",
    "detail": "wall clock reached mid-round; ending this round here."}}


def test_a_budget_stop_with_rounds_does_not_claim_loop_c_never_ran(tmp_path):
    """The bug my first draft had, caught on the corpus run.

    `_pipeline_batch` is called for the SEED batch AND from inside Loop C for rewrite candidates, so
    a `skipped`-shaped stop does NOT mean the seed pipeline ran out of time. The corpus run
    `run-l3-43-20260909-015247` fired exactly that stop at 13.51 h and has 5 FAMILY_ROUND_RECORDED.
    Inferring the loop from the payload shape asserted the opposite of the truth.
    """
    d = _write_run(tmp_path, [_ROUND_WITH_CONV, _WALL_BATCH, _WALL_ROUND])
    out = check_wrapup.check_budget_stop(d)
    assert out["reached_loop_c"] is True
    assert out["rounds"] == 1
    assert "Loop C DID run" in out["verdict"]
    assert "never got there" not in out["verdict"]
    # And it must say the round count is clock-limited, not a convergence result.
    assert "floor" in out["verdict"]


def test_a_budget_stop_with_no_rounds_overrides_not_yet_decidable(tmp_path):
    """With no round AND a budget stop, `NOT YET DECIDABLE` is wrong: it IS decided, the answer is
    that the run never reached Loop C. This is the box-3 risk case."""
    d = _write_run(tmp_path, [_BASELINE, _TRIAL_OK, _WALL_BATCH])
    out = check_wrapup.check_budget_stop(d)
    assert out["reached_loop_c"] is False
    assert "never got there" in out["verdict"]
    assert "1 candidate(s) were tuned and 1 skipped" in out["verdict"]
    assert "stay registered" in out["verdict"], (
        "a reader has to know the skipped candidates are recoverable by a resume")


def test_both_emission_sites_are_recognised(tmp_path):
    """Two sites with different payload keys. A reader keyed on one drops the other silently."""
    d = _write_run(tmp_path, [_WALL_BATCH, _WALL_ROUND])
    out = check_wrapup.check_budget_stop(d)
    assert {s["site"] for s in out["stops"]} == {"candidate batch", "rewrite round"}
    assert len(out["stops"]) == 2


def test_no_budget_stop_is_reported_as_absence(tmp_path):
    d = _write_run(tmp_path, [_BASELINE, _TRIAL_OK])
    out = check_wrapup.check_budget_stop(d)
    assert out["stops"] == []
    assert "no wall-clock stop" in out["verdict"]
    assert "BUDGET STOPPED" not in out["verdict"]


def test_the_overrun_percentage_is_reported(tmp_path):
    """13.51 h against a 12.0 h budget is a 13% overrun, and the recorded finding is that the wall
    clock is always the binding budget -- so the size of the overrun is the number that matters."""
    d = _write_run(tmp_path, [_WALL_BATCH])
    out = check_wrapup.check_budget_stop(d)
    assert "13% over" in out["verdict"], out["verdict"]


def test_the_informative_no_conversion_verdict_is_not_read_as_a_failure(tmp_path):
    """`no_conversion` is G27's whole point, not a failed round.

    Payload built from the REAL contract: `_rewrite_round` does `**conversion` into the event, and
    `conversion_verdict` returns `conversion`, `conversion_note`, `latency_gain_pct`,
    `latency_ms_before/after`, `resource_deltas` and `resources_improved`. A resource improving
    materially while latency does NOT move is evidence that that resource was not the limit for this
    structure -- so a reader that scores only "improved" as success discards exactly the finding the
    module exists to produce.
    """
    ev = {"type": "FAMILY_ROUND_RECORDED", "payload": {
        "family_id": "f1", "best_ms": 3.0, "round": 1,
        "conversion": "no_conversion",
        "conversion_note": ("n_regs, n_spills improved but latency moved only 0.30% (below the "
                            "2.0% floor), so those resources were NOT the limit for this "
                            "structure."),
        "latency_gain_pct": 0.3, "latency_ms_before": 3.009, "latency_ms_after": 3.0,
        "resources_improved": ["n_regs", "n_spills"],
        "resource_deltas": {
            "n_regs": {"before": 255, "after": 168, "delta": -87.0, "rel": 0.3412,
                       "unit": "registers/thread", "direction": "improved"},
            "n_spills": {"before": 428, "after": 0, "delta": -428.0, "rel": 1.0,
                         "unit": "bytes", "direction": "improved"},
            "shared_bytes": {"before": 12288, "after": 12288, "delta": 0.0, "rel": 0.0,
                             "unit": "bytes", "direction": "flat"}}}}
    d = _write_run(tmp_path, [ev])
    out = check_wrapup.check_conversion(d)
    assert out["rounds"] == 1 and out["with_conversion"] == 1
    assert out["verdict"].startswith("PASS"), (
        "a `no_conversion` round still CARRIES the verdict, which is what G27 is checked on")
    assert out["verdicts"] == {"no_conversion": 1}
    assert out["resources_improved"] == {"n_regs": 1, "n_spills": 1}, (
        "which resources moved is the content of the finding, not decoration")
    assert out["latency_gains_pct"] == [0.3]
    assert out["no_conversion_notes"] and "NOT the limit" in out["no_conversion_notes"][0], (
        "the note carries the actual conclusion; a count alone loses it")


def test_the_unknown_conversion_shape_has_no_resource_keys(tmp_path):
    """When latency cannot be compared, `conversion_verdict` returns EARLY with only two keys --
    no `latency_gain_pct`, no `resource_deltas`. A reader that assumes those keys exist would
    KeyError on the one payload shape that is guaranteed to be sparse."""
    ev = {"type": "FAMILY_ROUND_RECORDED", "payload": {
        "family_id": "f1", "best_ms": None, "round": 1,
        "conversion": "unknown",
        "conversion_note": "latency before/after not both available"}}
    d = _write_run(tmp_path, [ev])
    out = check_wrapup.check_conversion(d)
    assert out["with_conversion"] == 1
    assert out["with_resource_deltas"] == 0
    assert out["verdicts"] == {"unknown": 1}
    assert out["latency_gains_pct"] == []
    assert out["resources_improved"] == {}


# --- reader 3b: S2d's ledger, where an event stream can look healthy and say nothing -----------

# The REAL `Reconciliation` shape (evaluation/reconcile.py), not a guess. It has NO
# `verdict`/`outcome`/`status` field -- my first fixture invented `{"verdict": "confirmed",
# "checked": 2}` and the reader was written to match the fixture, so both agreed on keys that do
# not exist and would have printed "verdict kinds: -" on a working ledger.
_RECON_REAL = {"type": "EXPECTATIONS_RECONCILED", "payload": {
    "family_id": "f1", "round": 1, "id": "h1", "change": "fuse the projections",
    "reconciliation": {
        "hypothesis_id": "h1",
        "per_dimension": [
            {"dimension": "n_regs", "expected": "down", "actual": "down", "match": "hit",
             "before": 255.0, "after": 168.0, "delta": -87.0, "rel": 0.3412,
             "unit": "registers/thread", "why": "declared down, measured down"},
            {"dimension": "shared_bytes", "expected": "down", "actual": "flat", "match": "miss",
             "before": 12288.0, "after": 12288.0, "delta": 0.0, "rel": 0.0,
             "unit": "bytes", "why": "declared down, did not move"},
        ],
        "hits": 1, "misses": 1, "vacuous": 0,
        "dimensions_unpredicted": ["n_spills"],
        "dimensions_unmeasured": [],
        "caveat": ""},
    "conversion": "improved", "latency_gain_pct": 3.4, "n_declared": 2}}
_RECON_EMPTY = {"type": "EXPECTATIONS_RECONCILED", "payload": {
    "family_id": "f1", "round": 1, "id": "(no hypothesis id)", "change": "",
    "reconciliation": {}, "conversion": "improved", "n_declared": 0}}
# Declarations exist but every row is vacuous: the ledger ran and JUDGED nothing. `vacuous` is the
# agent declining to predict, which is distinct from a miss (a judgement) and from unmeasured.
_RECON_NO_JUDGEMENT = {"type": "EXPECTATIONS_RECONCILED", "payload": {
    "family_id": "f1", "round": 1, "id": "h2", "change": "retile",
    "reconciliation": {
        "hypothesis_id": "h2",
        "per_dimension": [
            {"dimension": "n_regs", "expected": "unknown", "actual": "down", "match": "vacuous",
             "unit": "registers/thread", "why": "no direction declared"},
            {"dimension": "occupancy", "expected": "up", "actual": "unknown",
             "match": "unmeasured", "unit": "fraction", "why": "no reading after"},
        ],
        "hits": 0, "misses": 0, "vacuous": 1,
        "dimensions_unpredicted": [], "dimensions_unmeasured": ["occupancy"],
        "caveat": "one dimension could not be read on both sides"},
    "conversion": "flat", "latency_gain_pct": 0.1, "n_declared": 2}}
_RECON_FAILED = {"type": "EXPECTATIONS_RECONCILE_FAILED", "payload": {
    "family_id": "f1", "round": 1, "error": "KeyError: 'direction'"}}


def _recon_for(cand: str, hyp: str, hits: int, misses: int) -> dict:
    """A per-candidate entry, the shape the orchestrator emits since per-candidate attribution.

    Copied field-for-field from `_record_reconciliation`'s own `entry` dict -- `candidate_id` beside
    `family_id`, not inside `reconciliation` -- rather than shaped to suit the reader. `_RECON_REAL`
    above deliberately keeps the OLD shape (no candidate_id) so a replayed pre-fix run is covered too.
    """
    return {"type": "EXPECTATIONS_RECONCILED", "payload": {
        "family_id": "f1", "round": 0, "candidate_id": cand, "id": hyp, "change": "c",
        "reconciliation": {
            "hypothesis_id": hyp,
            "per_dimension": [
                {"dimension": "shared_bytes", "expected": "up", "actual": "up", "match": "hit",
                 "before": 17408.0, "after": 32768.0, "delta": 15360.0, "rel": 0.88,
                 "unit": "bytes", "why": ""}] * max(hits, 0) + [
                {"dimension": "n_regs", "expected": "down", "actual": "flat", "match": "miss",
                 "before": 155.0, "after": 155.0, "delta": 0.0, "rel": 0.0,
                 "unit": "registers/thread", "why": ""}] * max(misses, 0),
            "hits": hits, "misses": misses, "vacuous": 0,
            "dimensions_unpredicted": [], "dimensions_unmeasured": [], "caveat": "c"},
        "conversion": "improved", "latency_gain_pct": 9.8, "n_declared": hits + misses}}


def test_two_entries_for_one_round_is_the_expected_shape_not_an_anomaly(tmp_path):
    """One entry per rewrite CANDIDATE, and a round has two of them (9 of 9 measured rounds).

    A reader that expected `entries == rounds` would flag the per-candidate fix AS the defect, so the
    ratio is stated in the verdict. On box 2's real round 0 the two candidates read 4/1 and 2/5, while
    pooling them read 6/6 -- so a summary that hid the split would hide the whole reason for the fix.
    """
    d = _write_run(tmp_path, [_ROUND_WITH_CONV,
                              _recon_for("cand-2d8eaf9a", "H1+H3", 4, 1),
                              _recon_for("cand-3760b4d7", "H2", 2, 5)])
    out = check_wrapup.check_reconciliation(d)
    assert out["rounds"] == 1 and out["entries"] == 2
    assert out["entries_without_candidate_id"] == 0
    assert out["hits"] == 6 and out["misses"] == 6      # the SUM is still 6/6 ...
    assert out["declarations_reconciled"] == 12
    assert out["verdict"].startswith("PASS")
    # ... so the entries-per-round line is what tells a reader the 6/6 is two candidates, not one
    # scrambled pool.
    assert "2.0 entries per round" in out["verdict"]
    assert "one per rewrite CANDIDATE" in out["verdict"]


def test_a_pre_fix_entry_without_a_candidate_id_is_named_as_incomparable(tmp_path):
    """A resumed run replays entries journalled before per-candidate attribution. Their hit/miss
    counts pool two candidates, so silently mixing them with per-candidate entries would produce a
    total that is neither one thing nor the other."""
    d = _write_run(tmp_path, [_ROUND_WITH_CONV, _RECON_REAL,
                              _recon_for("cand-x", "H9", 1, 0)])
    out = check_wrapup.check_reconciliation(d)
    assert out["entries_without_candidate_id"] == 1
    assert "NO candidate_id" in out["verdict"]
    assert "not\ncomparable" in out["verdict"] or "not comparable" in out["verdict"]


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
    """Against the REAL `Reconciliation` fields: hits/misses/vacuous plus the two named tuples."""
    d = _write_run(tmp_path, [_ROUND_WITH_CONV, _RECON_REAL])
    out = check_wrapup.check_reconciliation(d)
    assert out["verdict"].startswith("PASS")
    assert out["declarations_reconciled"] == 2
    assert (out["hits"], out["misses"], out["vacuous"]) == (1, 1, 0)
    assert out["per_dimension_rows"] == 2
    # Named, not merely counted: "you did not think of spills" is a different sentence from
    # "you were wrong about it", which is why reconcile.py keeps them as separate tuples.
    assert out["dimensions_unpredicted"] == {"n_spills": 1}
    assert out["dimensions_unmeasured"] == {}


def test_a_ledger_that_judged_nothing_is_not_reported_as_pass(tmp_path):
    """Declarations exist and every row is vacuous or unmeasured: the ledger RAN and checked
    nothing. From counts alone this looks identical to a working ledger -- n_declared is 2, there is
    an entry, there are per-dimension rows -- but 0 hits and 0 misses means no judgement was made.
    `vacuous` is the agent declining to predict; `unmeasured` is no reading on one side; only a
    hit or a miss is an actual check."""
    d = _write_run(tmp_path, [_ROUND_WITH_CONV, _RECON_NO_JUDGEMENT])
    out = check_wrapup.check_reconciliation(d)
    assert out["declarations_reconciled"] == 2
    assert out["per_dimension_rows"] == 2
    assert (out["hits"], out["misses"]) == (0, 0)
    assert "NO JUDGEMENT" in out["verdict"], out["verdict"]
    assert not out["verdict"].startswith("PASS")
    assert out["dimensions_unmeasured"] == {"occupancy": 1}
    assert out["caveats"] and "both sides" in out["caveats"][0]


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
    # The provenance must name the SAMPLE SIZE -- "inside the noise floor" means nothing without
    # the floor and its n. Asserting on n rather than on a phrase, so rewording the sentence does
    # not fail a test about behaviour.
    # The provenance must name the SAMPLE SIZE and how much of it is an ARM under comparison --
    # "inside the noise floor" means nothing without the floor and its n. Asserting on the numbers
    # rather than on a phrase, so rewording the sentence does not fail a test about behaviour.
    assert "n=1" in prov and "1 is an arm" in prov, prov


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
    assert "re-evaluated" in prov, prov


def test_the_latency_floor_widens_its_sample_with_sibling_runs(tmp_path):
    """n=2 is too thin to bound a verdict, and sibling runs are the same box and same harness.

    Measured on box 1's five finished runs the real spread is 0.29-4.73% (median 2.91%), so a
    sample of only the two arms under comparison -- 0.27% and 2.99% -- straddles the borrowed 2.35%
    and cannot say whether it is too strict. Passing ONE arm must therefore pick up its siblings.
    """
    # NOT _write_run: it names every directory "run-test", so three calls under one root produce
    # three NESTED paths rather than three siblings, and the sibling scan legitimately finds
    # nothing. The point of this test is the directory LAYOUT, so the layout is built explicitly.
    root = tmp_path / "runs"
    root.mkdir()

    def run(name: str, tuned: float, reeval: float) -> Path:
        d = root / name
        d.mkdir()
        (d / "events.jsonl").write_text(json.dumps({
            "seq": 0, "ts": 1789000000.0, "type": "RUN_FINISHED",
            "payload": {"summary": {"best": {
                "tuned_ms": tuned, "final_reeval_ms": reeval, "final_reeval_ok": True}}},
        }) + "\n", encoding="utf-8")
        return d

    a = run("arm-a", 2.7674, 2.76)   # 0.27%
    run("old-1", 1.48, 1.41)         # 4.73%
    run("old-2", 3.605, 3.50)        # 2.91%
    floor, prov = check_wrapup.latency_floor_from_runs(a)
    assert floor == pytest.approx(4.73, abs=0.05), (
        "the sibling runs were not sampled, so the floor stayed at the single arm's 0.27%%: %s"
        % prov)
    assert "n=3" in prov, prov
    # The ARM must be counted separately from its siblings. Without this the direct pass over
    # `run_dirs` is redundant -- the sibling glob finds the arms too -- and a variant that deletes
    # it passes every test, which is how this was found.
    assert "1 is an arm" in prov, prov
