"""Explicit controls for comparable search experiments, independent of model names."""
import json

from kernel_optimizer.project.generated import Selection


def eligible_groups(discovery, *, leaf_only=False):
    groups = discovery["groups"]
    if not leaf_only:
        return groups
    parents = {p for call in discovery["calls"] for p in call["parents"]}
    return [g for g in groups if g["kind"] != "project" and not set(g["calls"]) & parents]


def check_selection(path, discovery, allowed_scopes, eligible):
    selection = Selection.model_validate_json((path / "selection.json").read_bytes())
    groups = {g["id"]: g for g in eligible}
    calls = {c["id"] for c in discovery["calls"]}
    if selection.scope not in allowed_scopes or not selection.groups or set(selection.groups) - groups.keys():
        raise ValueError("select only eligible groups and allowed scopes in inputs/search.json")
    if selection.scope == "group" and len(selection.groups) != 1:
        raise ValueError("group scope requires exactly one eligible group")
    required = set().union(*(set(groups[g]["calls"]) for g in selection.groups))
    affected = set(selection.affected_calls)
    if not required <= affected or affected - calls:
        raise ValueError("affected_calls must cover selected groups and use observed call IDs")
    if selection.scope == "group":
        closure = set(required)
        while True:
            expanded = closure | {c["id"] for c in discovery["calls"] if set(c["parents"]) & closure}
            if expanded == closure:
                break
            closure = expanded
        if affected - closure:
            raise ValueError("single-group comparison cannot add unrelated groups or enclosing project boundaries")
    return selection


def local_time_selection(groups):
    measured = [g for g in groups if g.get("diagnostic_cuda_us", 0) > 0]
    if not measured:
        raise ValueError("local-time selection needs measured eligible group durations")
    group = max(measured, key=lambda g: (g["diagnostic_cuda_us"], g["id"]))
    return {"scope": "group", "groups": [group["id"]], "affected_calls": group["calls"],
        "rationale": "Controlled baseline: largest observed eligible leaf-group CUDA duration, without objective-specific agent selection.",
        "hypothesis": "Test whether optimizing this local hotspot improves the complete frozen project objective; local duration is not a predicted project gain."}


def feedback_view(raw, view):
    if view == "full" or raw is None:
        return raw
    if view != "project":
        raise ValueError("feedback view must be full or project")
    value = json.loads(raw)
    if not isinstance(value, dict) or "decision" not in value:
        return raw
    # Keep failure semantics so this control does not suppress correctness repair.
    result = {key: value.get(key) for key in ("current_revision", "previous_trial", "decision",
        "project_measurement", "evaluation", "quality_and_coverage", "missing_coverage")}
    search = value.get("parameter_search") or {}
    result["parameter_observations"] = [{"parameters": o["bundle"]["parameters"],
        "score": o["result"].get("score"), "valid": o["result"].get("valid"),
        "constraints": o.get("constraints"), "ranks": [{"rank": r["rank"], "error": r.get("error"),
            "checks": r.get("result", {}).get("checks")} for r in o["run"].get("ranks", [])]}
        for o in search.get("observations", [])]
    result["feedback_view"] = "project metrics and correctness only; per-group resources and resource responses withheld"
    return json.dumps(result, indent=2)
