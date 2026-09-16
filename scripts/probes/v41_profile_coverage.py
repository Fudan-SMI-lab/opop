"""Read-only, per-run profile coverage; numeric coverage is not comparison eligibility.

Run: python v41_profile_coverage.py RUN_DIR [RUN_DIR ...]
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeAlias

Json: TypeAlias = str | int | float | bool | None | list["Json"] | dict[str, "Json"]
FIELDS: Final = (
    "peak_alloc_bytes", "peak_reserved_bytes", "peak_above_resident_bytes",
    "candidate_aten_bytes", "candidate_aten_ops", "threads_launched", "n_regs",
    "n_spills", "shared_bytes", "occupancy", "sass", "compile_s", "wall_ms",
    "cpu_issue_ms", "overhead_gpu_ms",
)
OBJECT_FIELDS: Final = {"occupancy", "sass"}
UNMARKED: Final = "unmarked_NOTverifiedfresh"
QUALIFICATIONS: Final = (
    "Numeric coverage is not full measurement eligibility or full rescoring eligibility.",
    "Incomplete policy, dtype and environment evidence cannot establish comparison eligibility.",
    "Unmarked reuse and unique trial IDs do not establish freshness or statistical independence.",
    "COMPUTE_DTYPE/DOT_PRECISION are declared knob labels, not verified profile dtype; actual dtype is unverified.",
    "Allocator peaks have process scope, not code-execution-caused or device-wide GPU memory scope.",
    "peak_above_resident_bytes subtracts resident allocation, including already allocated weights and inputs.",
    "Zero is a valid measured value; missing values are never filled with zero.",
    "Offline journal inspection only: no candidate execution, GPU work, or data reproduction.",
)


def mapping(value: Json) -> Mapping[str, Json]:
    return value if isinstance(value, dict) else {}


def label(value: Json) -> str:
    return value if isinstance(value, str) and value else "unknown"


def metric_state(profile: Mapping[str, Json], field: str) -> str:
    if field not in profile:
        return "missing"
    value = profile[field]
    if value is None:
        return "null"
    if field in OBJECT_FIELDS:
        return "nonnull"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "bad_type"
    if isinstance(value, float) and not math.isfinite(value):
        return "nonfinite"
    return "negative" if value < 0 else "valid_numeric"


@dataclass(frozen=True, slots=True)
class Record:
    """Parsed coverage evidence; raw IDs and outer reuse values retain provenance."""

    trial_id: Json
    reused_flag: Json
    reused_flag_present: bool
    status: str
    provenance: str
    declared_precision: str
    backend: str
    metric_states: Mapping[str, str]
    profile_state: str
    actual_dtype: str = "unverified"
    qualifications: tuple[str, ...] = QUALIFICATIONS


@dataclass(frozen=True, slots=True)
class Summary:
    """Counts use accepted trial records as denominator, including failures and duplicates."""

    run_dir: Path
    records: tuple[Record, ...]
    line_counts: Counter[str]
    metrics: Mapping[str, Counter[str]]
    crosstab: Counter[tuple[str, str, str, str]]
    precision_metrics: Counter[tuple[str, str, str]]
    id_counts: Counter[str]
    duplicate_ids: Mapping[str, int]
    missing_id_count: int
    invalid_id_count: int
    unique_id_unmarked_count: int
    status_counts: Counter[str]
    provenance_counts: Counter[str]
    precision_counts: Counter[str]
    backend_counts: Counter[str]
    profile_counts: Counter[str]
    qualifications: tuple[str, ...] = QUALIFICATIONS


def analyse(run_dir: str | Path) -> Summary:
    """Read events.jsonl without modifying it; reject malformed envelopes explicitly."""
    run = Path(run_dir)
    lines: Counter[str] = Counter(total=0, empty=0, malformed=0, invalid_trial=0, other_event=0)
    pending: list[tuple[Mapping[str, Json], Mapping[str, Json]]] = []
    backends: dict[str, str] = {}
    with (run / "events.jsonl").open(encoding="utf-8") as journal:
        for line in journal:
            lines["total"] += 1
            if not line.strip():
                lines["empty"] += 1
                continue
            try:
                raw: Json = json.loads(line)
            except (ValueError, RecursionError):
                lines["malformed"] += 1
                continue
            if not isinstance(raw, dict) or not isinstance(raw.get("type"), str):
                lines["malformed"] += 1
                continue
            payload = mapping(raw.get("payload"))
            if raw.get("type") == "CANDIDATE_REGISTERED":
                candidate = mapping(payload.get("candidate"))
                candidate_id = label(candidate.get("candidate_id"))
                if candidate_id != "unknown":
                    backends[candidate_id] = label(candidate.get("backend"))
            if raw.get("type") != "TRIAL_DONE":
                lines["other_event"] += 1
                continue
            trial = payload.get("trial")
            if not isinstance(trial, dict):
                lines["invalid_trial"] += 1
                continue
            pending.append((payload, trial))
    records: list[Record] = []
    for payload, trial in pending:
        values = mapping(mapping(trial.get("params")).get("values"))
        precision = "; ".join(f"{key}={values[key]}" for key in ("COMPUTE_DTYPE", "DOT_PRECISION")
                              if label(values.get(key)) != "unknown") or "unknown"
        profile_raw = trial.get("profile")
        profile_state = "object" if isinstance(profile_raw, dict) else "bad_type"
        if profile_raw is None:
            profile_state = "null" if "profile" in trial else "missing"
        backend = label(trial.get("backend"))
        records.append(Record(
            trial_id=trial.get("trial_id"), reused_flag=payload.get("reused_measurement"),
            reused_flag_present="reused_measurement" in payload, status=label(trial.get("status")),
            provenance="marked_reused" if payload.get("reused_measurement") is True else UNMARKED,
            declared_precision=precision,
            backend=backend if backend != "unknown" else backends.get(label(trial.get("candidate_id")), "unknown"),
            metric_states={field: metric_state(mapping(profile_raw), field) for field in FIELDS},
            profile_state=profile_state,
        ))
    metrics = {field: Counter(present=0, nonnull=0, missing=0, null=0, complete_nonnull=0) for field in FIELDS}
    for field in FIELDS:
        if field not in OBJECT_FIELDS:
            metrics[field].update(dict.fromkeys(("bad_type", "nonfinite", "negative", "valid_numeric",
                "complete_valid_numeric", "complete_valid_numeric_not_marked_reused"), 0))
    cross: Counter[tuple[str, str, str, str]] = Counter()
    precision_metrics: Counter[tuple[str, str, str]] = Counter()
    ids: Counter[str] = Counter()
    unmarked_ids: set[str] = set()
    missing_ids = invalid_ids = 0
    for row in records:
        if isinstance(row.trial_id, str) and row.trial_id:
            ids[row.trial_id] += 1
            if row.provenance == UNMARKED:
                unmarked_ids.add(row.trial_id)
        elif row.trial_id is None or row.trial_id == "":
            missing_ids += 1
        else:
            invalid_ids += 1
        for field, state in row.metric_states.items():
            counts = metrics[field]
            counts[state] += 1
            counts["present"] += state != "missing"
            nonnull = state not in ("missing", "null")
            if field not in OBJECT_FIELDS:
                counts["nonnull"] += nonnull
            counts["complete_nonnull"] += row.status == "complete" and nonnull
            if field not in OBJECT_FIELDS and row.status == "complete" and state == "valid_numeric":
                counts["complete_valid_numeric"] += 1
                counts["complete_valid_numeric_not_marked_reused"] += row.provenance == UNMARKED
            cross[(row.status, row.provenance, field, state)] += 1
            precision_metrics[(row.declared_precision, field, state)] += 1
    return Summary(
        run_dir=run, records=tuple(records), line_counts=lines, metrics=metrics, crosstab=cross,
        precision_metrics=precision_metrics, id_counts=ids,
        duplicate_ids={key: count for key, count in ids.items() if count > 1},
        missing_id_count=missing_ids, invalid_id_count=invalid_ids,
        unique_id_unmarked_count=len(unmarked_ids), status_counts=Counter(row.status for row in records),
        provenance_counts=Counter(row.provenance for row in records),
        precision_counts=Counter(row.declared_precision for row in records),
        backend_counts=Counter(row.backend for row in records),
        profile_counts=Counter(row.profile_state for row in records),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("run_dir", nargs="+", type=Path, help="run directory containing events.jsonl")
    args = parser.parse_args(argv)
    runs: list[Path] = args.run_dir
    for run in runs:
        try:
            summary = analyse(run)
        except (OSError, UnicodeError) as exc:
            parser.error(f"{run}: {exc}")
        print(f"\n{run}   TRIAL_DONE = {len(summary.records)} (accepted records; not pooled)")
        print(f"  lines: {dict(summary.line_counts)}")
        print(f"  status: {dict(summary.status_counts)}; profiles: {dict(summary.profile_counts)}")
        print(f"  provenance (marked_reused / {UNMARKED}): {dict(summary.provenance_counts)}")
        print(f"  declared precision: {dict(summary.precision_counts)}; actual dtype: unverified")
        print(f"  backend: {dict(summary.backend_counts)}")
        print(f"  duplicate IDs (occurrences): {dict(summary.duplicate_ids)}; missing IDs: {summary.missing_id_count}; invalid IDs: {summary.invalid_id_count}")
        print(f"  unique_id_unmarked_count: {summary.unique_id_unmarked_count}")
        for field, counts in summary.metrics.items():
            print(f"  {field}: " + ", ".join(f"{key}={value}" for key, value in counts.items()))
        print("  status x provenance x metric x validity (nonzero cells):")
        for key, count in sorted(summary.crosstab.items()):
            print(f"    {' | '.join(key)}: {count}")
        print("  declared precision x metric x validity (nonzero cells):")
        for key, count in sorted(summary.precision_metrics.items()):
            print(f"    {' | '.join(key)}: {count}")
        for note in summary.qualifications:
            print(f"  Note: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
