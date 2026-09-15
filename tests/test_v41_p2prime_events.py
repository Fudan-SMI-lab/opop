"""CPU event-stream controls; expected values come from independent fixed inputs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Unpack

import pytest

from scripts.probes import v41_p2prime as p2
from scripts.probes.v41_p2prime_history import Event, Payload

_PROBE = Path(p2.__file__)


def event(event_type: str, **payload: Unpack[Payload]) -> Event:
    return {"type": event_type, "payload": payload}


def done(identity: str, value: int, ms: float) -> Event:
    return event("TRIAL_DONE", trial={
        "candidate_id": "a", "space_id": "s1", "trial_id": identity,
        "status": "complete", "params": {"values": {"BK": value}},
        "latency_ms": {"median": ms},
    })


def stream(order: tuple[str, ...] = ("N_d", "F_d", "N_v", "F_v")) -> list[Event]:
    history = [done("h0", 32, 10), done("h1", 64, 8),
               done("h2", 32, 10), done("h3", 64, 8)]
    for e in history[2:4]:
        record = e["payload"].get("trial")
        assert record is not None
        record["space_id"] = "s0"
    for i, value in enumerate((32, 64, 32, 64)):
        foreign = done(f"foreign{i}", value, 100)
        record = foreign["payload"].get("trial")
        assert record is not None
        record["candidate_id"] = "b"
        history.append(foreign)
    values = {"F_d": (32, 20), "N_d": (64, 10),
              "F_v": (32, 20), "N_v": (64, 10)}
    for role in order:
        history.append(done(role, *values[role]))
    # Role notifications deliberately differ from measurement arrival order.
    for role in ("F_v", "N_d", "N_v", "F_d"):
        history.append(event("SCAN_POINT_DONE", candidate_id="a", scan_id="scan",
                             trial_id=role, role=role, kind="C4", axis="BK"))
    history.append(event("SCAN_BLOCK_DONE", kind="C4", candidate_id="a",
                         scan_id="scan", axis="BK", full=True, g_d=0.5, y=0.5,
                         direction=None, axis_f_value="32", axis_n_value="64"))
    history.extend([done("future-f", 32, 1000), done("future-n", 64, 1000)])
    return history


def journal(path: Path, events: list[Event]) -> Path:
    (path / "events.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    return path


@pytest.mark.parametrize("legacy", [False, True])
def test_historical_numbers_when_endpoints_direct_or_wall(tmp_path: Path, legacy: bool):
    # Given: own history spans spaces, foreign history is deliberately different.
    events = stream()
    if legacy:
        close = events[-3]["payload"]
        del close["axis_f_value"], close["axis_n_value"]
        events[:0] = [event("UW_PROBE_BATCH", space_id="s1", walls=[
            {"axis": "BK", "f_value": "32", "n_value": "64"}]),
            event("SCAN_BLOCK_ADMITTED", kind="C4", scan_id="scan", axis="BK", space_id="s1")]
    # When
    result = p2.analyse(journal(tmp_path, events))
    # Then: run medians F=20,N=10; candidate F=15,N=9; historical F=10,N=8.
    c = result["contrasts"][0]
    assert c.g_m == pytest.approx({"old": -1.0, "signfix": 0.5,
                                  "scoped": 0.4, "noleak": 0.2, "discovery_cutoff": 0.2})
    assert c.n_pool == {"old": 12, "signfix": 12, "scoped": 8,
                        "noleak": 4, "discovery_cutoff": 6}
    assert c.n_own_leaked == 4
    assert p2.DEFAULT_DEFINITION == "noleak"


@pytest.mark.parametrize("order", [("F_d", "N_d", "F_v", "N_v"),
                                  ("N_d", "F_d", "N_v", "F_v")])
def test_discovery_inclusive_when_later_regular_trials_interleave(tmp_path: Path, order: tuple[str, ...]):
    # Given: one history point per endpoint, so discovery is needed for eligibility.
    events = stream(order)
    del events[2:4]
    events[8:8] = [done("late-f", 32, 100), done("late-n", 64, 100)]
    # When
    result = p2.analyse(journal(tmp_path, events))
    # Then: discovery F=15,N=9; later regular/validation/future records are excluded.
    c = result["contrasts"][0]
    assert c.g_m["discovery_cutoff"] == pytest.approx(0.4)
    assert c.n_pool["discovery_cutoff"] == 4
    meta = c.pool_metadata["discovery_cutoff"]
    assert meta["cutoff_seq"] == 7
    assert meta["cutoff_inclusive"] is True
    assert meta["scope"] == "candidate_all_spaces"
    assert meta["own_trial_policy"] == "include_discovery"


@pytest.mark.parametrize("defect,reason", [
    ("role", "four_role_identity_required"),
    ("duplicate", "four_role_identity_required"),
    ("trial", "trial_completion_identity_required"),
    ("foreign", "trial_completion_identity_required"),
    ("early", "validation_before_discovery_complete"),
])
def test_cutoff_unavailable_when_identity_or_order_invalid(tmp_path: Path, defect: str, reason: str):
    # Given
    events = stream(("F_v", "N_d", "F_d", "N_v") if defect == "early" else
                    ("N_d", "F_d", "N_v", "F_v"))
    if defect == "role":
        del events[12]["payload"]["role"]
    if defect == "duplicate":
        events[12]["payload"]["trial_id"] = "F_d"
    if defect == "trial":
        record = events[8]["payload"].get("trial")
        assert record is not None
        record["trial_id"] = "unknown"
    if defect == "foreign":
        record = events[8]["payload"].get("trial")
        assert record is not None
        record["candidate_id"] = "b"
    # When
    c = p2.analyse(journal(tmp_path, events))["contrasts"][0]
    # Then: decline only the explicit new definition, not historical regeneration.
    assert c.g_m["discovery_cutoff"] is None
    assert c.g_m_skip["discovery_cutoff"] == "information_cutoff_unavailable"
    assert c.pool_metadata["discovery_cutoff"]["cutoff_reason"] == reason
    assert c.g_m["old"] is not None


def test_thin_bucket_when_own_scan_removed_is_missing_denominator(tmp_path: Path):
    # Given: history exists, but only one sample per endpoint remains after exclusion.
    events = stream()
    del events[2:4]
    # When
    result = p2.analyse(journal(tmp_path, events))
    # Then: unresolved direction remains eligible; missing marginal is not a win.
    c = result["contrasts"][0]
    assert c.n_pool["noleak"] == 2
    assert c.g_m_skip["noleak"] == "thin_bucket_at_cutoff"
    summary = result["definitions"]["noleak"]
    assert summary["eligible_contrasts"] == 1
    assert summary["n"] == 0
    assert summary["missing_reasons"] == {"thin_bucket_at_cutoff": 1}
    assert summary["ratio"] is None


def test_reused_records_when_outer_flag_present_keep_numeric_weight(tmp_path: Path):
    # Given: replay repeats an ID and changes its weighting; do not silently dedupe.
    events = stream()
    reused = done("h0", 32, 30)
    reused["payload"]["reused_measurement"] = True
    events[:0] = [reused, reused]
    # When
    result = p2.analyse(journal(tmp_path, events))
    # Then: F median=(10+30)/2=20, N=8; unmarked does not mean verified fresh.
    c = result["contrasts"][0]
    assert c.g_m["noleak"] == pytest.approx(0.6)
    assert c.pool_metadata["noleak"]["coverage"] == {
        "raw_records": 6, "marked_reused_records": 2, "unmarked_records": 4,
        "distinct_trial_ids": 4, "records_without_trial_id": 0,
    }
    assert result["coverage"]["raw_records"] == 16
    assert result["coverage"]["marked_reused_records"] == 2


def test_zero_error_when_unresolved_full_contrast_is_reported(tmp_path: Path):
    # Given
    run = journal(tmp_path, stream())
    # When
    result = p2.analyse(run)
    # Then
    summary = result["definitions"]["noleak"]
    assert summary["n"] == 1
    assert summary["med_c"] == 0
    assert summary["ratio"] is None
    assert summary["missing_reasons"] == {}
    json.dumps(result["definitions"], allow_nan=False)


@pytest.mark.parametrize("args", [("--definition",), ("--definition", "bogus")])
def test_cli_error_when_definition_malformed(args: tuple[str, ...]):
    # Given / When
    result = subprocess.run([sys.executable, "-B", str(_PROBE), *args],
                            capture_output=True, text=True, check=False)
    # Then
    assert result.returncode == 2
    assert "usage:" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_succeeds_when_discovery_definition_explicit(tmp_path: Path):
    # Given
    run = journal(tmp_path, stream())
    # When
    result = subprocess.run([sys.executable, "-B", str(_PROBE), "--definition",
                             "discovery_cutoff", str(run)],
                            capture_output=True, text=True, check=False)
    # Then
    assert result.returncode == 0, result.stderr
    assert "discovery_cutoff" in result.stdout
