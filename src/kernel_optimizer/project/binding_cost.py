"""Measure incremental binding cost without candidate build or computation."""
from uuid import uuid4


class BindingCost:
    def collect(self, search, revision):
        if not revision.parent_id or not revision.bundles:
            return {"status": "not_applicable", "reason": "no candidate bindings on this revision"}
        parent = search.revisions.revision(revision.parent_id)
        added = set(revision.bundles) - set(parent.bundles)
        if not added or set(parent.bundles) - set(revision.bundles):
            return {"status": "not_applicable", "reason":
                    "passthrough requires additive bundles retaining the complete parent; replacement cost is unknown"}
        name = "binding-cost-" + uuid4().hex[:10]
        rows = []
        result = {"status": "unavailable", "parent": parent.id, "candidate": revision.id,
                  "passthrough_bundles": sorted(added), "runs": rows,
                  "interpretation": "AB/BA clean whole-invocation measurements with full original quality checks. "
                  "Passthrough omits added candidate build and computation, retaining its binding sites and "
                  "the installed parent. Differences include binding interactions and measurement noise; "
                  "they are not per-group costs, candidate performance, or amounts to subtract from acceptance."}
        for pair, order in enumerate((("parent", "passthrough"), ("passthrough", "parent"))):
            for arm in order:
                job = f"{name}-{pair}-{arm}"
                run = search.execute(parent if arm == "parent" else revision, job, "binding_ablation",
                                     collector_options={} if arm == "parent" else
                                     {"passthrough_bundles": sorted(added)})
                ranks = [{"rank": r["rank"], **{key: r.get("result", {}).get(key) for key in
                          ("valid", "checks", "coverage", "setup", "metrics", "samples")},
                          "error": r.get("error")} for r in run.get("ranks", [])]
                rows.append({"pair": pair, "arm": arm, "status": run["status"],
                             "failure": run.get("failure"), "ranks": ranks,
                             "receipt": str(search.output/"jobs"/job/"run"/"receipt.json")})
                if run["status"] != "passed" or not ranks or not all(r["valid"] for r in ranks):
                    result["reason"] = "endpoint execution/quality failed; no binding cost inferred"
                    return result
        result["status"] = "observed"
        elapsed = [{"pair": pair} for pair in range(2)]
        for row in rows:
            elapsed[row["pair"]][row["arm"]] = max(
                rank["metrics"]["framework.elapsed_ms"] for rank in row["ranks"])
        result["elapsed_pairs"] = elapsed
        return result
