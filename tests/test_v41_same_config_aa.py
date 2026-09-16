"""Synthetic, offline CLI contracts; no historical measurements are reconstructed."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Final, TypedDict

import pytest

type Json = None | bool | int | float | str | list[Json] | dict[str, Json]
class Event(TypedDict):
    type: str
    payload: dict[str, Json]


class SideReport(TypedDict):
    candidate_ids: list[str]
    trial_ids: list[str]
    values_ms: list[float]
    count: int
    median_ms: float
    run_path: str


class PairReport(TypedDict):
    source_sha: str
    backend: str
    typed_params: str
    a: SideReport
    b: SideReport
    signed_percent: float
    absolute_percent: float


class RunReport(TypedDict):
    counts: dict[str, int]
    unmatched_groups: int


class Report(TypedDict):
    analysis: str
    pairs: list[PairReport]
    runs: list[RunReport]
    summary: dict[str, float | bool | None]


PROBE: Final = Path(__file__).parents[1] / "scripts/probes/v41_same_config_aa.py"


def registered(cid: str = "a", /, **fields: Json) -> Event:
    return {"type": "CANDIDATE_REGISTERED", "payload": {"candidate": {
        "candidate_id": cid, "source_sha": "sha", "backend": "triton", **fields}}}


def done(tid: str = "t", /, ms: Json = 10, **fields: Json) -> Event:
    return {"type": "TRIAL_DONE", "payload": {"trial": {
        "trial_id": tid, "candidate_id": "a", "status": "complete",
        "params": {"values": {"precision": "fp16", "BK": 16}},
        "latency_ms": {"median": ms}, **fields}}}


def invoke(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-B", str(PROBE), *args],
                          capture_output=True, text=True, check=False, timeout=15)


def compare(tmp_path: Path, runs: tuple[list[Event], list[Event]]) -> Report:
    paths = [tmp_path / "a events.jsonl", tmp_path / "b events.jsonl"]
    for path, events in zip(paths, runs, strict=True):
        path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    result = invoke([*map(str, paths), "--json"])
    assert result.returncode == 0, result.stderr
    report: Report = json.loads(result.stdout)
    return report


def test_pair_when_source_matches_across_candidate_ids(tmp_path: Path):
    # Given: repeated timings and two A candidate IDs sharing one declaration.
    a = [registered(), registered("other"), done("a1", 1), done("a2", 10),
         done("a3", 100, candidate_id="other")]
    b = [registered("b"), done("b1", 20, candidate_id="b"),
         done("b2", 40, candidate_id="b")]
    # When
    report = compare(tmp_path, (a, b))
    # Then: median, not minimum; all audit identities survive.
    row = report["pairs"][0]
    assert row["source_sha"] == "sha" and row["backend"] == "triton"
    assert row["a"]["candidate_ids"] == ["a", "other"]
    assert row["a"]["trial_ids"] == ["a1", "a2", "a3"]
    assert row["a"]["values_ms"] == [1, 10, 100]
    assert row["a"]["count"] == 3 and row["a"]["median_ms"] == 10
    assert row["b"]["median_ms"] == 30
    assert row["signed_percent"] == row["absolute_percent"] == 200
    assert row["a"]["run_path"] == str(tmp_path / "a events.jsonl")
    assert json.loads(row["typed_params"])["BK"] == ["int", 16]
    assert report["analysis"] == "registered_source_params_comparison"
    assert report["summary"]["single_key"] is True
    assert report["summary"]["p90_absolute_percent"] == 200


@pytest.mark.parametrize("field,value", [("backend", "cuda"), ("source_sha", "different")])
def test_unmatched_when_registered_metadata_differs(tmp_path: Path, field: str, value: str):
    # Given / When
    report = compare(tmp_path, ([registered(), done()], [registered(**{field: value}), done()]))
    # Then
    assert report["summary"]["n"] == 0
    assert [r["unmatched_groups"] for r in report["runs"]] == [1, 1]
    assert report["summary"]["median_absolute_percent"] is None
    assert report["summary"]["p90_absolute_percent"] is None
    assert report["summary"]["observed_max_absolute_percent"] is None


@pytest.mark.parametrize("left,right", [(True, 1), ("16", 16), (16, 16.0)])
def test_unmatched_when_param_types_differ(tmp_path: Path, left: Json, right: Json):
    # Given / When
    report = compare(tmp_path, ([registered(), done(params={"values": {"BK": left}})],
                                [registered(), done(params={"values": {"BK": right}})]))
    # Then
    assert report["summary"]["n"] == 0


def test_full_params_when_precision_or_extra_knob_differs(tmp_path: Path):
    # Given / When
    report = compare(tmp_path, ([registered(), done()], [registered(),
        done(params={"values": {"BK": 16, "precision": "fp32", "extra": 1}})]))
    # Then
    assert report["summary"]["n"] == 0


@pytest.mark.parametrize("value", [None, True, 0, -1, float("nan"), float("inf"), "10"])
def test_excluded_when_median_invalid(tmp_path: Path, value: Json):
    # Given / When
    report = compare(tmp_path, ([registered(), done(ms=value)], [registered(), done()]))
    # Then
    assert report["runs"][0]["counts"]["invalid_median"] == 1
    assert report["summary"]["n"] == 0


@pytest.mark.parametrize("field", ["source_sha", "backend"])
def test_excluded_when_registration_field_missing(tmp_path: Path, field: str):
    # Given / When
    report = compare(tmp_path, ([registered(**{field: None}), done()], [registered(), done()]))
    # Then
    assert report["runs"][0]["counts"][f"missing_{field}"] == 1


@pytest.mark.parametrize("fields,reason", [({"params": None}, "missing_params"),
    ({"trial_id": None}, "missing_trial_id"), ({"candidate_id": "absent"}, "missing_registration"),
    ({"status": "failed"}, "not_complete"), ({"params": {"values": {"x": None}}}, "invalid_params")])
def test_excluded_when_trial_metadata_invalid(tmp_path: Path, fields: dict[str, Json], reason: str):
    # Given / When
    report = compare(tmp_path, ([registered(), done(**fields)], [registered(), done()]))
    # Then
    assert report["runs"][0]["counts"][reason] == 1


def test_deduplication_when_reused_and_conflicting_ids(tmp_path: Path):
    # Given: same ID is run-local, even across candidates; conflicts invalidate all copies.
    reused = done("reuse", 999)
    reused["payload"]["reused_measurement"] = True
    a = [registered(), registered("other"), done("ok"), done("ok"), reused,
         done("conflict", 1), done("conflict", 2, candidate_id="other")]
    # When
    report = compare(tmp_path, (a, [registered(), done("ok", 5)]))
    # Then
    counts = report["runs"][0]["counts"]
    assert counts["duplicate_records"] == 1
    assert counts["conflicting_trial_ids"] == 1 and counts["conflicting_records"] == 2
    assert counts["reused_records"] == 1
    assert report["pairs"][0]["a"]["trial_ids"] == ["ok"]
    assert report["pairs"][0]["signed_percent"] == -50


def test_nearest_rank_when_ten_keys(tmp_path: Path):
    # Given: absolute differences 10,20,...,100%; repeated A values do not add weight.
    a, b = [registered()], [registered()]
    for i in range(1, 11):
        a.extend(done(f"a{i}-{j}", 100, params={"values": {"key": i}}) for j in range(i))
        b.append(done(f"b{i}", 100 + 10 * i, params={"values": {"key": i}}))
    # When
    report = compare(tmp_path, (a, b))
    # Then
    summary = report["summary"]
    assert summary["n"] == 10 and summary["single_key"] is False
    assert summary["median_absolute_percent"] == 55
    assert summary["p90_absolute_percent"] == 90
    assert summary["observed_max_absolute_percent"] == 100


def test_help_when_requested():
    # Given / When
    result = invoke(["--help"])
    # Then
    assert result.returncode == 0 and "--json" in result.stdout


def test_error_when_path_missing(tmp_path: Path):
    # Given / When
    result = invoke([str(tmp_path / "missing"), str(tmp_path)])
    # Then
    assert result.returncode == 2 and "usage:" in result.stderr


def test_human_when_run_directories_supplied(tmp_path: Path):
    # Given
    (tmp_path / "events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in [registered(), done()]), encoding="utf-8")
    # When
    result = invoke([str(tmp_path), str(tmp_path)])
    # Then
    assert result.returncode == 0
    assert "registered_source_params_comparison" in result.stdout
    assert '"n": 1' in result.stdout
