"""Route measured scope evidence without inventing independent group effects."""
import json


def candidate_artifacts(raw, store):
    """Make prior implementation bytes available, not just opaque blob IDs.

    A rejected candidate is absent from the accepted parent's source snapshot.
    Keep its exact code and original binding metadata separately for repair.
    Unavailable externally supplied references are explicit in the manifest.
    """
    if raw is None:
        return {}
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    if not isinstance(value, dict):
        return {}
    if "previous_result" in value:
        value = value["previous_result"]
    if not isinstance(value, dict):
        return {}
    bundle = value.get("bundle") or value.get("previous_bundle")
    if not isinstance(bundle, dict) or not bundle.get("installer"):
        return {}
    from pathlib import PurePosixPath
    installer = bundle["installer"]
    targets = str(PurePosixPath(installer).with_suffix(".targets.json"))
    names = {installer: "candidate.py", targets: "targets.json"}
    inputs = {"previous-candidate/bundle.json": json.dumps(bundle, indent=2)}
    manifest = {"bundle": bundle["id"], "parent_revision": bundle.get("parent_revision"),
                "purpose": "Historical implementation for inspection/repair; not installed or accepted automatically",
                "files": []}
    for edit in bundle.get("edits", []):
        name = names.get(edit["path"])
        if name is None:
            continue
        row = {"original_path": edit["path"], "content_ref": edit.get("content_ref"), "input": name}
        try:
            if not edit.get("content_ref"):
                raise ValueError("prior edit deletes this file")
            inputs["previous-candidate/"+name] = store.get_artifact(edit["content_ref"])
            row["status"] = "available"
        except (FileNotFoundError, ValueError, KeyError) as exc:
            row.update(status="unavailable", reason=f"{type(exc).__name__}: {exc}")
        manifest["files"].append(row)
    inputs["previous-candidate/manifest.json"] = json.dumps(manifest, indent=2)
    return inputs


def route(raw, *, current_revision, registry=None, selected_groups=None, plan=None):
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        return raw
    if not isinstance(value, dict) or "decision" not in value:
        return raw
    selected = set(selected_groups or ())
    relevant = set(selected)
    selected_calls, neighbors = set(), set()
    if registry:
        for group in registry.groups:
            if not selected or group.id in selected:
                relevant.add(group.id); relevant.update(group.lineage)
                selected_calls.update(group.calls)
        if plan and selected:
            for edge in plan.dependencies:
                if edge.producer in selected_calls:
                    neighbors.add(edge.consumer)
                if edge.consumer in selected_calls:
                    neighbors.add(edge.producer)
            for group in registry.groups:
                if set(group.calls) & neighbors:
                    relevant.add(group.id); relevant.update(group.lineage)
    resources = value.get("resources", {})
    groups = resources.get("groups", [])
    if selected:
        groups = [g for g in groups if g["group"] in relevant]
    result = {"current_revision": current_revision, "previous_trial": value.get("trial"),
        "previous_bundle": value.get("bundle"), "previous_selection": value.get("selection"),
        "previous_proposal": value.get("proposal"), "decision": value["decision"],
        "supplemental_evidence": value.get("supplemental_evidence"),
        "project_measurement": value.get("measurement"), "evaluation": value.get("evaluation"),
        "parameter_search": value.get("parameter_search"),
        "missing_coverage": value.get("missing_coverage"),
        "complete_version": {key: value.get(key) for key in ("bundles", "revision", "parent", "quality",
            "coverage_errors", "source_revision", "source_output", "attribution_status")}
            if value.get("schema_version") == "project.version-evaluation.v1" else None,
        "quality_and_coverage": [{"rank": r["rank"], "status": r.get("status"), "error": r.get("error"),
            "checks": r.get("result", {}).get("checks"), "coverage": r.get("result", {}).get("coverage")}
            for r in value.get("candidate", {}).get("ranks", [])],
        "resources": {**{key: val for key, val in resources.items() if key != "groups"}, "groups": groups},
        "routing": {"selected_groups": sorted(selected), "neighbor_calls": sorted(neighbors),
                    "lineage_group_ids": sorted(relevant), "omitted_group_records": len(resources.get("groups", []))-len(groups)},
        "version_policy": "All previous measurements retain their original parent/candidate identities. Current grouping and evidence come from the new discovery; historical group timings are not silently reused.",
        "attribution_policy": "The project gain belongs to the complete bundle. Missing independent group effects remain unknown."}
    return json.dumps(result, indent=2)
