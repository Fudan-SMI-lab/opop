"""CPU-only coverage contracts using synthetic event journals, not reproduced runs."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Final, TypedDict

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.probes import v41_profile_coverage as coverage

SCRIPT: Final = Path(__file__).resolve().parents[1] / "scripts" / "probes" / "v41_profile_coverage.py"
FIELD: Final = "peak_alloc_bytes"
UNMARKED: Final = "unmarked_NOTverifiedfresh"


class Event(TypedDict):
    type: str
    payload: coverage.Json


def write_run(path: Path, events: list[Event]) -> Path:
    path.mkdir(exist_ok=True)
    _ = (path / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    return path


def event(trial: dict[str, coverage.Json], reused: bool | None = None) -> Event:
    payload: dict[str, coverage.Json] = {"trial": trial}
    if reused is not None:
        payload["reused_measurement"] = reused
    return {"type": "TRIAL_DONE", "payload": payload}


def test_subsets_when_status_and_reuse_overlap(tmp_path: Path) -> None:
    # Given: reuse overlaps both failed and complete rows; non-null is not numeric.
    run = write_run(tmp_path, [
        event({"trial_id": "fail", "status": "failed"}, True),
        event({"trial_id": "ok", "status": "complete", "profile": {FIELD: 0}}),
        event({"trial_id": "reuse", "status": "complete", "profile": {FIELD: 12}}, True),
        event({"status": "complete", "profile": {FIELD: None}}),
        event({"status": "complete", "profile": {FIELD: True}}),
    ])
    # When
    summary = coverage.analyse(run)
    # Then: intersection counts, never an overall reuse subtraction.
    counts = summary.metrics[FIELD]
    assert (counts["present"], counts["nonnull"], counts["valid_numeric"]) == (4, 3, 2)
    assert (counts["complete_nonnull"], counts["complete_valid_numeric"],
            counts["complete_valid_numeric_not_marked_reused"]) == (3, 2, 1)
    assert summary.crosstab[("failed", "marked_reused", FIELD, "missing")] == 1
    assert summary.crosstab[("complete", "marked_reused", FIELD, "valid_numeric")] == 1
    assert summary.crosstab[("complete", UNMARKED, FIELD, "valid_numeric")] == 1


@pytest.mark.parametrize(("value", "state"), [
    (None, "null"), (True, "bad_type"), (False, "bad_type"), ("12", "bad_type"),
    ({}, "bad_type"), ([], "bad_type"), (float("nan"), "nonfinite"),
    (float("inf"), "nonfinite"), (-float("inf"), "nonfinite"),
    (-1, "negative"), (0, "valid_numeric"), (0.0, "valid_numeric"),
    (1.5, "valid_numeric"), (10**400, "valid_numeric"),
])
def test_numeric_class_when_value_varies(tmp_path: Path, value: coverage.Json, state: str) -> None:
    # Given
    run = write_run(tmp_path, [event({"status": "complete", "profile": {FIELD: value}})])
    # When
    summary = coverage.analyse(run)
    # Then
    assert summary.metrics[FIELD][state] == 1
    assert summary.crosstab[("complete", UNMARKED, FIELD, state)] == 1


def test_objects_when_profile_has_occupancy_and_sass(tmp_path: Path) -> None:
    # Given
    run = write_run(tmp_path, [event({"profile": {"occupancy": {}, "sass": {"instructions": 0}}})])
    # When
    summary = coverage.analyse(run)
    # Then
    for field in ("occupancy", "sass"):
        assert summary.metrics[field]["nonnull"] == 1
        assert "valid_numeric" not in summary.metrics[field]
    assert set(summary.metrics) == {
        "peak_alloc_bytes", "peak_reserved_bytes", "peak_above_resident_bytes",
        "candidate_aten_bytes", "candidate_aten_ops", "threads_launched", "n_regs",
        "n_spills", "shared_bytes", "occupancy", "sass", "compile_s", "wall_ms",
        "cpu_issue_ms", "overhead_gpu_ms",
    }


def test_ids_and_outer_flags_when_records_repeat(tmp_path: Path) -> None:
    # Given: nested reuse is not the outer event flag.
    run = write_run(tmp_path, [event({"trial_id": "a", "reused_measurement": True}),
                               event({"trial_id": "a"}, True),
                               event({"trial_id": "b"}, False), event({})])
    original = (run / "events.jsonl").read_bytes()
    # When
    summary = coverage.analyse(run)
    # Then
    assert summary.id_counts == {"a": 2, "b": 1}
    assert summary.duplicate_ids == {"a": 2}
    assert summary.missing_id_count == 1
    assert summary.unique_id_unmarked_count == 2
    assert [row.trial_id for row in summary.records] == ["a", "a", "b", None]
    assert [row.reused_flag for row in summary.records] == [None, True, False, None]
    assert [row.reused_flag_present for row in summary.records] == [False, True, True, False]
    assert (run / "events.jsonl").read_bytes() == original


def test_groups_when_precision_is_declared_or_unknown(tmp_path: Path) -> None:
    # Given: preserve both knobs, do not interpret either as observed dtype.
    run = write_run(tmp_path, [
        {"type": "CANDIDATE_REGISTERED", "payload": {"candidate": {"candidate_id": "c", "backend": "cuda"}}},
        event({"candidate_id": "c", "params": {"values": {"COMPUTE_DTYPE": "bf16", "DOT_PRECISION": "ieee"}}, "profile": {FIELD: 0}}),
        event({"params": {"values": {"DOT_PRECISION": "tf32"}}}), event({}),
    ])
    # When
    summary = coverage.analyse(run)
    # Then
    assert summary.precision_counts == {"COMPUTE_DTYPE=bf16; DOT_PRECISION=ieee": 1,
                                        "DOT_PRECISION=tf32": 1, "unknown": 1}
    assert summary.backend_counts == {"cuda": 1, "unknown": 2}
    assert {row.actual_dtype for row in summary.records} == {"unverified"}
    assert summary.precision_metrics[("COMPUTE_DTYPE=bf16; DOT_PRECISION=ieee", FIELD, "valid_numeric")] == 1


def test_bad_lines_when_journal_is_partial(tmp_path: Path) -> None:
    # Given
    run = write_run(tmp_path, [event({}), {"type": "TRIAL_DONE", "payload": None},
                               {"type": "OTHER", "payload": {}}])
    with (run / "events.jsonl").open("a", encoding="utf-8") as journal:
        _ = journal.write('\n{broken\n[]\n{"type":"TRIAL_DONE","payload":{"trial":[]}}\n')
    # When
    summary = coverage.analyse(run)
    # Then
    assert summary.line_counts == {"total": 7, "empty": 1, "malformed": 2,
                                    "invalid_trial": 2, "other_event": 1}
    assert summary.status_counts == {"unknown": 1}
    assert summary.metrics[FIELD]["complete_nonnull"] == 0


def test_empty_journal_when_no_events(tmp_path: Path) -> None:
    # Given
    run = write_run(tmp_path, [])
    # When
    summary = coverage.analyse(run)
    # Then
    assert summary.records == ()
    assert summary.metrics[FIELD]["present"] == 0


def test_cli_when_two_runs_are_supplied(tmp_path: Path) -> None:
    # Given
    first = write_run(tmp_path / "first", [event({"status": "complete", "profile": {FIELD: 0}})])
    second = write_run(tmp_path / "second", [])
    # When
    result = subprocess.run([sys.executable, "-B", str(SCRIPT), str(first), str(second)],
                            capture_output=True, text=True, check=False)
    # Then: human labels, not a prose snapshot.
    assert result.returncode == 0
    assert str(first) in result.stdout and str(second) in result.stdout
    assert result.stdout.count("TRIAL_DONE =") == 2
    for label in ("present", "nonnull", "valid_numeric", "marked_reused", UNMARKED,
                  "complete_valid_numeric_not_marked_reused", "declared precision", "unverified"):
        assert label in result.stdout


@pytest.mark.parametrize("args", [[], ["--bogus"]])
def test_cli_when_arguments_are_invalid(args: list[str]) -> None:
    # Given / When
    result = subprocess.run([sys.executable, "-B", str(SCRIPT), *args],
                            capture_output=True, text=True, check=False)
    # Then
    assert result.returncode == 2
    assert "usage:" in result.stderr


def test_cli_when_help_is_requested() -> None:
    # Given / When
    result = subprocess.run([sys.executable, "-B", str(SCRIPT), "--help"],
                            capture_output=True, text=True, check=False)
    # Then
    assert result.returncode == 0
    assert "run_dir" in result.stdout
