"""Small measured summaries for optional post-evaluation collection planning."""


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
