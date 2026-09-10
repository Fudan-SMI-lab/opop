"""G21: an empty read must raise, not return an empty container.

Each test below reconstructs one of the four real failures from a single day, using the event shapes
verified against a real run on box 1 (run-l3-48-20260909-115701: 680 TRIAL_DONE events, 371 complete,
latency keys {max,mean,median,min,n_samples,samples,std}).

The point is not that the field paths are now shared -- it is that a wrong path can no longer look
like an empty run. All four failures were an empty result accepted as an answer.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from kernel_optimizer.store.read import (
    EmptyResult,
    ReadError,
    best_trial,
    candidate_source,
    completed_trials,
    events_of_type,
    kernel_metadata,
    latency_ms_of,
    read_events,
    trials_with_latency,
)


def _run(tmp_path: Path, events: list[dict], name: str = "run") -> Path:
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    with io.open(d / "events.jsonl", "w", encoding="utf-8") as f:
        for i, e in enumerate(events):
            f.write(json.dumps({"seq": i, "ts": 1.0 + i, **e}) + "\n")
    return d


def _trial_done(status="complete", median=3.65, cid="cand-abc123"):
    """A TRIAL_DONE event with the REAL nesting and the REAL latency keys."""
    return {"type": "TRIAL_DONE", "payload": {"trial": {
        "trial_id": "t1", "candidate_id": cid, "status": status,
        "latency_ms": {"mean": median + 0.06, "std": 0.27, "min": median - 0.09,
                       "max": median + 1.2, "n_samples": 20, "median": median,
                       "samples": [median] * 20},
        "profile": {"n_regs": 255},
    }}}


def test_the_680_trial_failure_cannot_recur(tmp_path):
    """A run full of complete trials must never read as having none.

    The original: 680 TRIAL_DONE events, 371 complete, reported as "no complete trial with a latency
    in this run" -- because the reader looked at payload.status while the record is nested under
    payload.trial.
    """
    run = _run(tmp_path, [_trial_done(median=3.0 + i * 0.1) for i in range(5)])
    trials = completed_trials(run)
    assert len(trials) == 5, trials
    assert all(t["status"] == "complete" for t in trials)

    # And the latency must be found without an `_ms` suffix, which is the second failure.
    assert latency_ms_of(trials[0]) == 3.0, latency_ms_of(trials[0])
    lat = trials[0]["latency_ms"]
    assert "median_ms" not in lat and "robust_ms" not in lat, (
        "the fixture no longer matches the real log shape, so this test would pass on a reader that "
        "guesses the suffixed names")

    assert best_trial(run)["latency_ms_value"] == 3.0


def test_a_run_whose_trials_all_failed_must_be_stated_not_inferred(tmp_path):
    """Zero complete trials raises; accepting it requires writing allow_empty=True."""
    run = _run(tmp_path, [_trial_done(status="fail") for _ in range(4)])
    with pytest.raises(EmptyResult) as exc:
        completed_trials(run)
    msg = str(exc.value)
    assert "4 TRIAL_DONE events" in msg, msg
    assert "payload.trial" in msg, (
        "the error does not name the correct field path, so a reader hitting it learns nothing: "
        + msg)
    # Opting in is explicit, and then the empty answer is the caller's stated expectation.
    assert completed_trials(run, allow_empty=True) == []


def test_a_missing_event_type_raises_with_what_was_present(tmp_path):
    """'Loop D never ran' must be a statement, not an empty list nobody looked at."""
    run = _run(tmp_path, [_trial_done()])
    with pytest.raises(EmptyResult) as exc:
        events_of_type(run, "NOVELTY_PRODUCED")
    msg = str(exc.value)
    assert "TRIAL_DONE" in msg, "the error does not list the types that ARE present: " + msg
    assert events_of_type(run, "NOVELTY_PRODUCED", allow_empty=True) == []


def test_trials_without_any_latency_raises(tmp_path):
    """Complete trials that all lack a latency is the signature of a wrong latency key."""
    ev = _trial_done()
    ev["payload"]["trial"]["latency_ms"] = {"n_samples": 0}
    run = _run(tmp_path, [ev, ev])
    with pytest.raises(EmptyResult) as exc:
        trials_with_latency(run)
    assert "median_ms" in str(exc.value), (
        "the error does not warn about the wrong key names that caused this: " + str(exc.value))


def test_the_candidate_source_layout_failure_cannot_recur(tmp_path):
    """`candidates/<id>/source.py`, not `candidates/<id>.py`.

    The original read 10 candidates as "no source on disk" because it globbed `candidates/*.py`.
    """
    run = _run(tmp_path, [_trial_done()])
    (run / "candidates" / "cand-abc123").mkdir(parents=True)
    (run / "candidates" / "cand-abc123" / "source.py").write_text("PARAMS = {}", encoding="utf-8")
    assert candidate_source(run, "cand-abc123") == "PARAMS = {}"

    # A wrong id must say what IS there, not just fail.
    with pytest.raises(ReadError) as exc:
        candidate_source(run, "cand-nope")
    assert "cand-abc123" in str(exc.value), (
        "the error does not list the candidates present, so a typo is indistinguishable from an "
        "empty run: " + str(exc.value))


def test_kernel_metadata_searches_both_objects(tmp_path):
    """The A800 false alarm: fields split across `compiled` and `compiled.metadata`.

    Reading all of them off the kernel reported `shared`/`num_warps` as unavailable and the probe
    called the box blind to Tier-1 signals -- while the harness's own accessor worked. Reporting a
    working machine as broken invents work and discredits a good box.
    """
    class _Meta:
        shared = 0            # a REAL value: this kernel allocates no shared memory
        num_warps = 4
        num_stages = 3

    class _Compiled:
        n_regs = 26
        n_spills = 0
        metadata = _Meta()

    md = kernel_metadata(_Compiled())
    assert md["n_regs"] == 26 and md["n_spills"] == 0
    assert md["shared"] == 0 and md["num_warps"] == 4 and md["num_stages"] == 3
    assert md["_missing"] == [], (
        "a field present on compiled.metadata was reported missing, which is the false 'this box is "
        "blind' alarm: %s" % md)
    # shared == 0 must not be confused with absent.
    assert md["shared"] is not None

    # Only genuinely absent fields are reported, and THAT is what justifies calling a box blind.
    class _Bare:
        n_regs = 26

    bare = kernel_metadata(_Bare())
    assert "shared" in bare["_missing"] and "n_regs" not in bare["_missing"], bare

    # A version that renamed n_spills -> num_spills must still be read.
    class _Renamed:
        n_regs = 26
        num_spills = 7

    assert kernel_metadata(_Renamed())["n_spills"] == 7


def test_a_missing_or_empty_log_is_distinguished_from_an_empty_result(tmp_path):
    """"The file is not there" and "the file holds nothing" are different faults."""
    with pytest.raises(ReadError):
        read_events(tmp_path / "nope")
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "events.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(EmptyResult):
        read_events(empty)


def test_the_reader_returns_no_empty_container_by_default(tmp_path):
    """The structural guarantee: without allow_empty, nothing here can hand back an empty list.

    This is the fix. Wrapping the field paths in functions that still return [] would only move the
    hazard; all four original failures were an empty container accepted as an answer.
    """
    run = _run(tmp_path, [_trial_done(status="fail")])
    for fn in (completed_trials, trials_with_latency):
        with pytest.raises(EmptyResult):
            fn(run)
        assert fn(run, allow_empty=True) == []
