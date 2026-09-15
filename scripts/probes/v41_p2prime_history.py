"""Offline P2' prefix selection; record multiplicity is never silently deduplicated."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Final, TypedDict


class Params(TypedDict, total=False):
    values: dict[str, str | int | float | bool | None]


class Trial(TypedDict, total=False):
    candidate_id: str
    space_id: str
    trial_id: str
    status: str
    params: Params
    latency_ms: dict[str, float | None]


@dataclass(frozen=True, slots=True)
class Record:
    trial: Trial
    seq: int
    reused: bool


class Wall(TypedDict, total=False):
    axis: str
    f_value: str
    n_value: str


class Payload(Trial, total=False):
    trial: Trial
    reused_measurement: bool
    kind: str
    scan_id: str
    axis: str
    role: str
    full: bool
    g_d: float | None
    y: float | None
    direction: str | None
    axis_f_value: str | None
    axis_n_value: str | None
    walls: list[Wall]
    f_lat: list[float]
    n_lat: list[float]
    tau: float | None
    rho_f: float | None
    rho_n: float | None


class Event(TypedDict):
    type: str
    payload: Payload


class Coverage(TypedDict):
    raw_records: int
    marked_reused_records: int
    unmarked_records: int
    distinct_trial_ids: int
    records_without_trial_id: int


class DefinitionInfo(TypedDict):
    status: str
    scope: str
    cutoff_event: str
    own_trial_policy: str
    formula: str


class PoolMetadata(DefinitionInfo):
    cutoff_seq: int | None
    cutoff_inclusive: bool
    sequence_basis: str
    cutoff_reason: str | None
    deduplicated: bool
    coverage: Coverage


class Prefix(TypedDict):
    payload: Payload
    seq: int
    pools: dict[str, list[Record]]
    metadata: dict[str, PoolMetadata]
    n_own_leaked: int


class Summary(DefinitionInfo):
    eligible_contrasts: int
    missing_reasons: dict[str, int]
    cutoff_inclusive: bool
    deduplicated: bool
    ratio: float | None
    n: int
    med_c: float | None
    med_m: float | None
    cond_closer: int
    sign_m: int


# Journal dictionaries remain lossless for the historical arithmetic.
def coverage(records: list[Record]) -> Coverage:
    marked = sum(r.reused for r in records)
    return {"raw_records": len(records), "marked_reused_records": marked,
            "unmarked_records": len(records) - marked,
            "distinct_trial_ids": len({str(r.trial.get("trial_id")) for r in records
                                       if r.trial.get("trial_id")}),
            "records_without_trial_id": sum(not r.trial.get("trial_id") for r in records)}


DEFINITION_INFO: Final[dict[str, DefinitionInfo]] = {
    "old": {"status": "retracted_diagnostic", "scope": "run_all_candidates",
            "cutoff_event": "SCAN_BLOCK_DONE", "own_trial_policy": "include_all",
            "formula": "(n_median-f_median)/n_median"},
    "signfix": {"status": "retracted_diagnostic", "scope": "run_all_candidates",
                "cutoff_event": "SCAN_BLOCK_DONE", "own_trial_policy": "include_all",
                "formula": "1-n_median/f_median"},
    "scoped": {"status": "retracted_diagnostic", "scope": "candidate_all_spaces",
               "cutoff_event": "SCAN_BLOCK_DONE", "own_trial_policy": "include_all",
               "formula": "1-n_median/f_median"},
    "noleak": {"status": "historical_comparison", "scope": "candidate_all_spaces",
               "cutoff_event": "SCAN_BLOCK_DONE", "own_trial_policy": "exclude_all_scan_ids",
               "formula": "1-n_median/f_median"},
    "discovery_cutoff": {"status": "explicit_information_comparison",
                         "scope": "candidate_all_spaces",
                         "cutoff_event": "last_discovery_TRIAL_DONE",
                         "own_trial_policy": "include_discovery",
                         "formula": "1-n_median/f_median"},
}


def discovery_cutoff(points: list[Payload], history: list[Record], block: Payload) -> tuple[int | None, str | None]:
    """Require four distinct role identities and unambiguous candidate completions."""
    roles = {p.get("role", ""): p.get("trial_id") for p in points}
    if (not block.get("candidate_id") or not block.get("scan_id") or len(points) != 4
            or set(roles) != {"F_d", "N_d", "F_v", "N_v"}
            or not all(roles.values()) or len(set(roles.values())) != 4
            or any(p.get("candidate_id") != block.get("candidate_id") for p in points)
            or any(p.get("axis") != block.get("axis") for p in points)):
        return None, "four_role_identity_required"
    times: dict[str, int] = {}
    for role, identity in roles.items():
        matches = [r for r in history if r.trial.get("trial_id") == identity]
        if len(matches) != 1:
            return None, "trial_completion_identity_required"
        times[role] = matches[0].seq
    cutoff = max(times["F_d"], times["N_d"])
    if min(times["F_v"], times["N_v"]) <= cutoff:
        return None, "validation_before_discovery_complete"
    return cutoff, None


def collect_prefixes(events: list[Event]) -> tuple[list[Prefix], Coverage]:
    """Accumulate legacy prefixes and explicit cutoff metadata without changing weights."""
    run_pool: list[Record] = []
    cand_pool: dict[str, list[Record]] = defaultdict(list)
    own_trials: dict[str, set[str]] = defaultdict(set)
    points: dict[str, list[Payload]] = defaultdict(list)
    wall_endpoints: dict[tuple[str | None, str | None], tuple[str | None, str | None]] = {}
    prefixes: list[Prefix] = []
    # Preserve historical all-log scan-ID exclusion, even for late point notifications.
    for ev in events:
        p: Payload = ev.get("payload") or {}
        if ev.get("type") == "SCAN_POINT_DONE" and p.get("trial_id"):
            own_trials[str(p.get("scan_id"))].add(str(p.get("trial_id")))
    for seq, ev in enumerate(events):
        p = ev.get("payload") or {}
        match ev.get("type"):
            case "TRIAL_DONE":
                rec: Trial = p.get("trial") or p
                if rec.get("status") == "complete":
                    record = Record(rec, seq, bool(p.get("reused_measurement")))
                    run_pool.append(record)
                    cand_pool[str(rec.get("candidate_id"))].append(record)
            case "SCAN_POINT_DONE":
                points[str(p.get("scan_id"))].append(p)
            case "UW_PROBE_BATCH":
                for w in p.get("walls", []):
                    if w.get("f_value") is not None and w.get("n_value") is not None:
                        wall_endpoints[(p.get("space_id"), w.get("axis"))] = (
                            w.get("f_value"), w.get("n_value"))
            case "SCAN_BLOCK_ADMITTED" if p.get("kind") == "C4":
                wall_endpoints.setdefault(("scan:" + str(p.get("scan_id")), p.get("axis")),
                                          wall_endpoints.get(
                                              (p.get("space_id"), p.get("axis")), (None, None)))
            case "SCAN_BLOCK_DONE" if p.get("kind") == "C4":
                sid = str(p.get("scan_id"))
                hist = list(cand_pool[str(p.get("candidate_id"))])
                noleak = [r for r in hist if str(r.trial.get("trial_id")) not in own_trials[sid]]
                cutoff, reason = discovery_cutoff(points[sid], hist, p)
                pools = {"old": list(run_pool), "signfix": list(run_pool),
                         "scoped": hist, "noleak": noleak,
                         "discovery_cutoff": [r for r in hist if cutoff is not None
                                              and r.seq <= cutoff]}
                metadata: dict[str, PoolMetadata] = {}
                for definition, pool in pools.items():
                    explicit = definition == "discovery_cutoff"
                    metadata[definition] = {
                        **DEFINITION_INFO[definition], "cutoff_seq": cutoff if explicit else seq,
                        "cutoff_inclusive": True, "sequence_basis": "zero_based_parsed_event_order",
                        "cutoff_reason": reason if explicit else None,
                        "deduplicated": False, "coverage": coverage(pool),
                    }
                payload: Payload = {**p}
                if p.get("axis_f_value") is None or p.get("axis_n_value") is None:
                    payload["axis_f_value"], payload["axis_n_value"] = wall_endpoints.get(
                        ("scan:" + sid, p.get("axis")), (None, None))
                prefixes.append({"payload": payload, "seq": seq, "pools": pools,
                                 "metadata": metadata, "n_own_leaked": len(hist) - len(noleak)})
            case _:
                # This offline projection intentionally ignores other journal event types.
                continue
    return prefixes, coverage(run_pool)
