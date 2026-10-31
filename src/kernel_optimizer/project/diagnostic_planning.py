"""Small measured summaries for optional post-evaluation collection planning."""


def revision_summary(revision, parent):
    """Report installed composition relative to the candidate's actual parent."""
    current = list(revision.bundles)
    previous = list(parent.bundles) if parent is not None else None
    return {"revision": revision.id, "parent": revision.parent_id, "bundles": current,
            "parent_bundles": previous,
            "bundle_changes": None if previous is None else {
                "added": [ident for ident in current if ident not in previous],
                "removed": [ident for ident in previous if ident not in current],
                "retained": [ident for ident in current if ident in previous]}}


def objective_reference(run, objective):
    from kernel_optimizer.project.measurement import metric
    reducer = objective.get("rank_aggregation", "max" if objective["direction"] == "minimize" else "min")
    order = objective.get("aggregation_order", "samples_then_ranks")
    result = {"metric": objective["name"], "unit": objective["unit"], "rank_aggregation": reducer,
              "aggregation_order": order,
              "scope": "Original reference objective reduced using the frozen request's rank and sample order; "
                       "not a paired improvement or candidate score."}
    try:
        result["value"] = metric(run, objective["name"], reducer, order)
    except (KeyError, TypeError, ValueError):
        result.update(value=None, reason="Original reference objective unavailable")
    return result


def endpoint_summary(run):
    from kernel_optimizer.project.measurement import valid_run
    return {"status": run.get("status"), "failure": run.get("failure"),
            "quality_and_coverage_valid": valid_run(run),
            "ranks": [{"rank": row["rank"], "metrics": row.get("result", {}).get("metrics", {}),
                "setup": [{"bundle": setup["bundle"], "binding": {key: setup.get("binding", {}).get(key)
                    for key in ("requested", "effective", "reason", "functional_sites")}}
                    for setup in row.get("result", {}).get("setup", [])],
                "coverage": {ident: {"candidate_invocations": sum(values.get("counts", {}).values()),
                    "framework_fallbacks": sum(values.get("fallbacks", {}).values()),
                    "reference_invocations": sum(values.get("reference_invocations", {}).values()),
                    **({"candidate_declared_routes": values["candidate_declared_routes"]}
                       if values.get("candidate_declared_routes") else {})}
                    for ident, values in row.get("result", {}).get("coverage", {}).items()}}
                for row in run.get("ranks", [])],
            "interpretation": "Reference invocations may implement fallback, verification or composition. "
                              "Route labels are candidate-reported telemetry, not independent native proof; "
                              "corroborate them with traces or actual exception evidence."}
