# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
# How to run (existing interpreter, no installation):
# uv run --no-project --python D:/Anaconda/python.exe python -B v41_same_config_aa.py A B --json
"""Offline registered-declaration comparison, NOT historical N1 reconstruction."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Final, NamedTuple

type Json = None | bool | int | float | str | list[Json] | dict[str, Json]
type Fields = dict[str, Json]
QUALIFICATION: Final = {
    "match": "Same declaration only; provisional, not verified same configuration.",
    "artifact": "Registered source_sha is before parameterization, NOT materialized-source "
                "or byte-identical artifact proof. Pending actual artifact verification.",
    "precision": "Declared precision knobs are included in full typed params; actual timed "
                 "and profile dtype/context are unverified.",
    "context": "Task/shape/environment policy unchecked within and across runs. Do not pool "
               "different tasks as scientific evidence; aggregate is provisional until checked.",
    "inference": "Descriptive only: observed maximum is not a population upper bound, timer "
                 "noise floor, tie threshold or MDE. No scientific eligibility certification.",
    "history": "Original 13 pairs are not reproducible locally and remain unverified. "
               "This protocol is not equivalent to the unavailable original generator.",
}


class Key(NamedTuple):
    source_sha: str
    backend: str
    typed_params: str


class Arguments(argparse.Namespace):
    run_a: Path = Path()
    run_b: Path = Path()
    json: bool = False


@dataclass(frozen=True, slots=True)
class Observation:
    trial_id: str
    candidate_id: str
    latency_ms: float


@dataclass(frozen=True, slots=True)
class Run:
    path: str
    counts: Counter[str]
    groups: dict[Key, list[Observation]]


@dataclass(frozen=True, slots=True)
class Side:
    run_path: str
    candidate_ids: list[str]
    trial_ids: list[str]
    values_ms: list[float]
    count: int
    median_ms: float


@dataclass(frozen=True, slots=True)
class Pair:
    key: Key
    source_sha: str
    backend: str
    typed_params: str
    a: Side
    b: Side
    signed_percent: float
    absolute_percent: float


def fields(value: Json) -> Fields:
    """Shape-check JSON objects only at the event boundary; absence never forms a key."""
    return value if isinstance(value, dict) else {}


def text(value: Json) -> str:
    return value if isinstance(value, str) and value.strip() else ""


def load(run_path: Path) -> Run:
    """Read registered metadata and unique trial identities; no artifact/DB access."""
    path = run_path / "events.jsonl" if run_path.is_dir() else run_path
    counts: Counter[str] = Counter()
    candidates: dict[str, set[tuple[str, str]]] = defaultdict(set)
    trials: dict[str, list[Fields]] = defaultdict(list)
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                decoded: Json = json.loads(line)
            except json.JSONDecodeError:
                counts["invalid_json"] += 1
                continue
            event = fields(decoded)
            payload = fields(event.get("payload"))
            if event.get("type") == "CANDIDATE_REGISTERED":
                candidate = fields(payload.get("candidate"))
                cid = text(candidate.get("candidate_id"))
                if cid:
                    candidates[cid].add((text(candidate.get("source_sha")),
                                         text(candidate.get("backend"))))
            if event.get("type") != "TRIAL_DONE":
                continue
            counts["trial_events"] += 1
            counts["reused_records"] += bool(payload.get("reused_measurement"))
            trial = fields(payload.get("trial"))
            tid = text(trial.get("trial_id"))
            if not tid:
                counts["missing_trial_id"] += 1
                continue
            trials[tid].append(payload)
    groups: dict[Key, list[Observation]] = defaultdict(list)
    for tid, copies in trials.items():
        # Compare payloads, not event timestamps/seq; JSON distinguishes true from 1.
        unique = {json.dumps(p, sort_keys=True, separators=(",", ":")) for p in copies}
        counts["duplicate_records"] += len(copies) - len(unique)
        if len(unique) != 1:
            counts["conflicting_trial_ids"] += 1
            counts["conflicting_records"] += len(copies)
            continue
        payload = copies[0]
        if payload.get("reused_measurement"):
            continue
        trial = fields(payload.get("trial"))
        if trial.get("status") != "complete":
            counts["not_complete"] += 1
            continue
        value = fields(trial.get("latency_ms")).get("median")
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or value <= 0):
            counts["invalid_median"] += 1
            continue
        cid = text(trial.get("candidate_id"))
        metadata = candidates.get(cid, set())
        if len(metadata) != 1:
            counts["conflicting_registration" if metadata else "missing_registration"] += 1
            continue
        sha, backend = next(iter(metadata))
        if not sha or not backend:
            counts["missing_source_sha"] += not bool(sha)
            counts["missing_backend"] += not bool(backend)
            continue
        params = fields(trial.get("params")).get("values")
        if not isinstance(params, dict):
            counts["missing_params"] += 1
            continue
        # ParamSet.values is flat/scalar; preserve every name and its JSON scalar type.
        if any(not isinstance(v, (str, int, float, bool))
               or (isinstance(v, float) and not math.isfinite(v)) for v in params.values()):
            counts["invalid_params"] += 1
            continue
        typed = json.dumps({k: [type(v).__name__, v] for k, v in params.items()},
                           sort_keys=True, separators=(",", ":"), allow_nan=False)
        groups[Key(sha, backend, typed)].append(Observation(tid, cid, float(value)))
        counts["accepted_observations"] += 1
    return Run(str(run_path), counts, dict(groups))


def side(run: Run, key: Key) -> Side:
    observations = sorted(run.groups[key], key=lambda item: item.trial_id)
    values = [item.latency_ms for item in observations]
    return Side(run.path, sorted({item.candidate_id for item in observations}),
                [item.trial_id for item in observations], values, len(values), median(values))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("run_a", type=Path, help="Run directory or events.jsonl path")
    _ = parser.add_argument("run_b", type=Path, help="Run directory or events.jsonl path")
    _ = parser.add_argument("--json", action="store_true", help="Machine-readable JSON only")
    args = parser.parse_args(namespace=Arguments())
    try:
        a, b = load(args.run_a), load(args.run_b)
    except (OSError, UnicodeError) as exc:
        parser.error(str(exc))
    shared = sorted(a.groups.keys() & b.groups.keys())
    pairs: list[Pair] = []
    for key in shared:
        left, right = side(a, key), side(b, key)
        delta = (right.median_ms - left.median_ms) / left.median_ms * 100
        if not math.isfinite(delta):
            parser.error("Relative difference exceeds finite numeric range")
        pairs.append(Pair(key, *key, left, right, delta, abs(delta)))
    absolute = sorted(row.absolute_percent for row in pairs)
    n = len(absolute)
    report = {
        "analysis": "registered_source_params_comparison",
        "qualification": QUALIFICATION,
        "protocol": {
            "observations": "Complete TRIAL_DONE latency_ms.median only; positive finite numeric, "
                            "no bool/string coercion, no mean/min fallback; marked reuse excluded.",
            "identity": "Within-run trial_id: identical payloads deduplicated, conflicting "
                        "payloads exclude entire identity. Missing IDs excluded. Counts may overlap.",
            "parameters": "All recorded params.values names, sorted with explicit scalar types. "
                          "No default filling; completeness against a parameter space is unchecked.",
            "side_aggregation": "Median of per-trial medians, not minimum; values align with trial_ids.",
            "weighting": "Each matched key once, regardless of observation count; provisional.",
            "percent": "100*(B_median-A_median)/A_median; positive means B slower.",
            "p90": "Nearest-rank: sorted absolute percentages at ceil(0.9*n), 1-based.",
        },
        "runs": [{"run_path": r.path, "counts": dict(r.counts), "groups": len(r.groups),
                  "unmatched_groups": len(r.groups) - n} for r in (a, b)],
        "pairs": [asdict(row) for row in pairs],
        "summary": {"n": n, "single_key": n == 1,
                    "median_absolute_percent": median(absolute) if n else None,
                    "p90_absolute_percent": absolute[math.ceil(0.9 * n) - 1] if n else None,
                    "observed_max_absolute_percent": absolute[-1] if n else None},
    }
    if not args.json:
        print("registered_source_params_comparison (PROVISIONAL)")
        for label, qualification in QUALIFICATION.items():
            print(f"{label}: {qualification}")
        print("Auditable records and descriptive counts/statistics:")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
