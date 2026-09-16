"""CPU regressions for legacy N1 candidate-space aggregation through real JSONL."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Final

import pytest

PROBE: Final = Path(__file__).resolve().parents[1] / "scripts/probes/v41_n1_noise_floor.py"
SPEC: Final = importlib.util.spec_from_file_location("v41_n1_noise_floor", PROBE)
assert SPEC is not None and SPEC.loader is not None
n1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(n1)


def _write_events(run: Path, events: tuple[str, ...]) -> Path:
    run.mkdir(parents=True, exist_ok=True)
    (run / "events.jsonl").write_text("\n".join(events) + "\n", encoding="utf-8")
    return run


def _tuning(candidate: str, fields: str, space: str = "initial") -> str:
    return ('{"type":"TUNING_DONE","payload":{"candidate_id":'
            + json.dumps(candidate) + ',"space_id":' + json.dumps(space)
            + ("," + fields if fields else "") + "}}")


@pytest.mark.parametrize("values", [(2.0, 5.0), (5.0, 2.0), (2.0, 2.0)])
def test_candidate_best_when_spaces_arrive_in_either_order(
    tmp_path: Path, values: tuple[float, float],
) -> None:
    # Given: two completed spaces for the same candidate.
    run = _write_events(tmp_path, tuple(
        _tuning("a", f'"best_ms":{value}', f"space-{index}")
        for index, value in enumerate(values)
    ))
    # When: the actual arm loader aggregates the log.
    result = n1.arm(run)
    # Then: the candidate retains its best, not its last space.
    assert result["tuned"] == {"a": 2.0}


@pytest.mark.parametrize("fields", ["", '"best_ms":null', '"best":null',
                                   '"best":{"latency_ms":null}'])
def test_best_survives_when_spaces_have_missing_or_null_latency(
    tmp_path: Path, fields: str,
) -> None:
    # Given: missing values precede and follow a valid measurement.
    run = _write_events(tmp_path, (
        _tuning("a", fields), _tuning("a", '"best_ms":2'),
        _tuning("a", fields, "expanded"), _tuning("empty", fields),
    ))
    # When
    result = n1.arm(run)
    # Then
    assert result["tuned"] == {"a": 2.0}


@pytest.mark.parametrize("value", [
    '"bad"', '"1.0"', "true", "false", "[]", "{}",
    "NaN", "Infinity", "-Infinity", "0", "-1",
])
def test_invalid_latency_is_ignored_when_seen_before_or_after_valid_best(
    tmp_path: Path, value: str,
) -> None:
    # Given: an invalid-only candidate, plus invalid spaces around a valid best.
    fields = f'"best_ms":{value}'
    run = _write_events(tmp_path, (
        _tuning("invalid", fields), _tuning("a", fields),
        _tuning("a", '"best_ms":2'), _tuning("a", fields, "expanded"),
    ))
    # When
    result = n1.arm(run)
    # Then: neither exceptions nor invalid minima escape the reader.
    assert result["tuned"] == {"a": 2.0}


@pytest.mark.parametrize(("fields", "expected"), [
    ('"best":{"latency_ms":{"median":3,"mean":1,"min":0.1}}', 3.0),
    ('"best_ms":null,"best":{"latency_ms":{"median":3,"mean":1}}', 3.0),
    ('"best":{"latency_ms":{"mean":4}}', 4.0),
    ('"best":{"latency_ms":{"median":null,"mean":4}}', 4.0),
    ('"best":{"latency_ms":{"median":0,"mean":4}}', 4.0),
    ('"best":{"latency_ms":{"median":-1,"mean":4}}', 4.0),
    ('"best":{"latency_ms":{"median":NaN,"mean":4}}', 4.0),
    ('"best":{"latency_ms":{"median":true,"mean":4}}', 4.0),
    ('"best_ms":6,"best":{"latency_ms":{"median":1,"mean":2}}', 6.0),
])
def test_legacy_latency_fallback_when_best_ms_is_absent_or_null(
    tmp_path: Path, fields: str, expected: float,
) -> None:
    # Given: a legacy nested latency, distinct from any explicit best_ms.
    run = _write_events(tmp_path, (_tuning("a", fields),))
    # When
    result = n1.arm(run)
    # Then: median precedes mean; sample min is never the objective.
    assert result["tuned"] == {"a": expected}


@pytest.mark.parametrize("fields", [
    '"best":{"latency_ms":{"median":null,"mean":true}}',
    '"best":{"latency_ms":{"mean":Infinity}}',
    '"best":{"latency_ms":{"mean":0}}',
    '"best":{"latency_ms":{"mean":"2"}}',
    '"best":{"latency_ms":{"min":1}}',
    '"best_ms":false,"best":{"latency_ms":{"median":1}}',
])
def test_invalid_selected_latency_does_not_create_a_candidate(
    tmp_path: Path, fields: str,
) -> None:
    # Given
    run = _write_events(tmp_path, (_tuning("a", fields),))
    # When
    result = n1.arm(run)
    # Then
    assert result["tuned"] == {}


def test_distinct_candidates_keep_independent_minima_when_spaces_interleave(tmp_path: Path) -> None:
    # Given: shared space IDs must not pool different candidates.
    run = _write_events(tmp_path, (
        _tuning("a", '"best_ms":2'), _tuning("b", '"best_ms":9'),
        _tuning("a", '"best_ms":5', "expanded"),
        _tuning("b", '"best":{"latency_ms":{"mean":7}}', "expanded"),
    ))
    # When
    result = n1.arm(run)
    # Then
    assert result["tuned"] == {"a": 2.0, "b": 7.0}


def test_cli_pairs_source_sha_and_keeps_final_reevaluation_separate(tmp_path: Path) -> None:
    # Given: different run-local IDs share a seed; search and final values differ.
    runs = []
    for label, cid, best, final in (("A", "a", 2, 10), ("B", "b", 3, 11)):
        events = (
            json.dumps({"type": "CANDIDATE_REGISTERED", "payload": {"candidate": {
                "candidate_id": cid, "origin": "seed", "source_sha": "shared-seed",
            }}}),
            _tuning(cid, f'"best_ms":{best}'),
            _tuning(cid, '"best_ms":8', "expanded"),
            json.dumps({"type": "RUN_FINISHED", "payload": {"summary": {"best": {
                "candidate_id": cid, "tuned_ms": best,
                "final_reeval_median_ms": final, "final_reeval_ms": 99,
            }}}}),
        )
        runs.append(_write_events(tmp_path / label / "run", events))
    # When: run the real CLI, not a replacement aggregation function.
    completed = subprocess.run(
        [sys.executable, "-B", "-X", "utf8", str(PROBE), *(str(run) for run in runs)],
        capture_output=True, text=True, encoding="utf-8", check=True, timeout=10,
    )
    # Then: shared SHA pairs distinct IDs, while final difference uses 10 -> 11.
    assert "SHARED: 1" in completed.stdout
    assert "shared-seed  A=2.0000 ms   B=3.0000 ms   B-A = +50.00%" in completed.stdout
    assert "final_reeval_ms difference (B vs A): +10.00%" in completed.stdout
