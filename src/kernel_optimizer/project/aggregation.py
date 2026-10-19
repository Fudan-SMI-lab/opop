"""Unify observed rank-local plans without assuming identical rank control flow."""
from copy import deepcopy
from collections import defaultdict
import json


def merge_ranks(observations):
    calls, groups, dependencies = {}, {}, {}
    resources = defaultdict(dict)
    for observed in observations:
        rank = observed["rank"]
        for original in observed["calls"]:
            call = calls.setdefault(original["id"], {**deepcopy(original), "ranks": [], "rank_specs": {}})
            if any(call[key] != original[key] for key in ("target", "kind", "phase", "semantics")):
                raise ValueError("call identity collides across ranks")
            call["ranks"].append(rank)
            call["rank_specs"][str(rank)] = {key: deepcopy(original[key]) for key in ("attributes", "signatures", "count", "parents")}
            call["signatures"] = list({json.dumps(s, sort_keys=True): s for s in call["signatures"]+original["signatures"]}.values())
            call["parents"] = sorted(set(call["parents"])|set(original["parents"]))
            resources[call["id"]][str(rank)] = observed["call_resources"].get(call["id"], {})
        for group in observed["groups"]:
            groups.setdefault(group["id"], set()).update(group["calls"])
        for edge in observed["dependencies"]:
            key = (edge["producer"], edge["consumer"], edge["kind"])
            entry = dependencies.setdefault(key, {**edge, "evidence": []})
            entry["evidence"].append(observed["trace"])
    return {"calls": list(calls.values()), "groups": [{"id": key, "calls": sorted(value)} for key, value in groups.items()],
        "dependencies": [{**edge, "evidence": tuple(edge["evidence"])} for edge in dependencies.values()], "call_resources_by_rank": dict(resources),
        "unknown_dependencies": list(dict.fromkeys(item for o in observations for item in o["unknown_dependencies"])),
        "traces": [o["trace"] for o in observations], "trace_sha256": [o["trace_sha256"] for o in observations]}


def rank_requirements(bundle, plan, world_size):
    lookup = {c.id: c for c in plan.calls}
    return {str(rank): [ident for ident in bundle.calls if rank in lookup[ident].signature.get("ranks", range(world_size))]
            for rank in range(world_size)}


def index(discovery):
    """An addressable semantic index; keep full calls/groups as the authority."""
    families = {}
    for group in discovery["groups"]:
        key = (group["kind"], group["semantics"], tuple(group["phases"]))
        item = families.setdefault(key, {"kind": key[0], "semantics": key[1], "phases": list(key[2]),
            "groups": [], "call_count": 0, "diagnostic_cuda_us": 0.0})
        item["groups"].append(group["id"]); item["call_count"] += len(group["calls"])
        item["diagnostic_cuda_us"] += group["diagnostic_cuda_us"]
    return {"families": sorted(families.values(), key=lambda row: row["diagnostic_cuda_us"], reverse=True),
        "interpretation": "Navigation index only: families can mix incompatible shapes. Resolve group and call IDs in full artifacts. Inclusive durations overlap, especially across ancestor/child families; rank durations use max per call, not cross-rank summation."}
