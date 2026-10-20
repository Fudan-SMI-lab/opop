"""Navigable full diagnostic evidence; packaging never selects search actions."""
import hashlib
import json
import re


def artifacts(observations, prefix):
    files, manifest = {}, {}
    for name, observation in (observations or {}).items():
        slug = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
        suffix = hashlib.sha256(name.encode()).hexdigest()[:8]
        path = f"{prefix}/{slug}-{suffix}.json"
        raw = json.dumps(observation, indent=2)
        files[path] = raw
        ranks = []
        for rank in observation.get("ranks", []):
            ranks.append({key: rank[key] for key in
                          ("rank", "scope", "workload_window", "profiled_wall_ms", "clock",
                           "gradient_hooks_enabled") if key in rank})
        manifest[name] = {"status": observation.get("status"), "input": path,
                          "sha256": hashlib.sha256(raw.encode()).hexdigest(),
                          "bytes": len(raw.encode()), "provenance": observation.get("provenance"),
                          "ranks": ranks,
                          "access": "Full unfiltered collector output in the referenced input; raw evidence paths retained"}
    return files, manifest


def feedback_artifacts(raw):
    """Keep full observations accessible without embedding megabytes in feedback."""
    if raw is None:
        return {}, raw
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        return {}, raw
    if not isinstance(value, dict):
        return {}, raw
    resources = value.get("resources")
    if not isinstance(resources, dict) or not isinstance(resources.get("extra_collectors"), dict):
        return {}, raw
    files, indexed = {}, {}
    for arm, extra in resources["extra_collectors"].items():
        part, manifest = artifacts(extra, "observations/previous-"+arm)
        files.update(part)
        indexed[arm] = manifest
    resources["extra_collectors"] = indexed
    return files, json.dumps(value, indent=2)


def ownership(discovery):
    """Join observed parameter ownership to groups without dividing shared costs."""
    groups_by_call = {}
    for group in discovery.get("groups", []):
        for call in group["calls"]:
            groups_by_call.setdefault(call, set()).add(group["id"])
    module_calls = {}
    for call in discovery.get("calls", []):
        if call["kind"] == "module":
            module_calls.setdefault(call["target"], set()).add(call["id"])
    buckets = []
    sources = [("torch", rank) for rank in discovery.get("communication", {}).get("ranks", []) if rank]
    for name, extra in discovery.get("extra_collectors", {}).items():
        sources.extend((name, rank) for rank in extra.get("ranks", []) if rank.get("gradient_readiness"))
    for source, rank in sources:
        inventory = (rank.get("bucket_inventory") or rank.get("gradient_readiness") or {}).get("buckets", [])
        for bucket in inventory:
            calls = set(bucket.get("calls", []))
            for parameter in bucket.get("parameters", []):
                calls.update(module_calls.get(parameter.rpartition(".")[0], ()))
            buckets.append({"collector": source, "rank": rank.get("rank"),
                            "module": bucket.get("module"), "bucket_index": bucket.get("bucket_index"),
                            "parameters": bucket.get("parameters", []), "calls": sorted(calls),
                            "groups": sorted({g for c in calls for g in groups_by_call.get(c, ())}),
                            "timed_collective_identity": "unknown",
                            "basis": "Observed parameter ownership; no cross-capture event identity inferred"})
    return {"revision": discovery["revision"], "plan": discovery["plan"], "registry": discovery["registry"],
            "buckets": buckets,
            "interpretation": "Many-to-many ownership for navigating evidence, not per-group communication time, readiness or causal contribution"}
