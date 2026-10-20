"""Replay original candidate bytes with explicit, non-publishing binding controls."""
import argparse
import hashlib
import json
from pathlib import Path

from kernel_optimizer.config import load_config
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source


def compact(run):
    ranks = []
    for rank in run.get("ranks", []):
        result = rank.get("result", {})
        ranks.append({"rank": rank["rank"], "valid": result.get("valid"), "checks": result.get("checks"),
                      "samples": result.get("samples"), "metrics": result.get("metrics"),
                      "setup": result.get("setup"), "coverage": result.get("coverage"),
                      "framework_source": rank.get("framework_source"), "error": rank.get("error")})
    return {"status": run["status"], "failure": run.get("failure"), "ranks": ranks}


def run_protocol(search, revision, parent, output, *, pairs=2, original_controls=True):
    output.mkdir(parents=True, exist_ok=True)
    new = set(revision.bundles)-set(parent.bundles)
    if not new or set(parent.bundles)-set(revision.bundles):
        raise ValueError("binding ablation requires additive child bundles with the parent retained")
    anchor = search.revisions.revision(search.prepared["revision"])
    arms = {"parent": (parent, {}), "passthrough": (revision, {"passthrough_bundles": sorted(new)}),
            "candidate": (revision, {})}
    comparisons = [("parent", "passthrough"), ("parent", "candidate")]
    if original_controls:
        if anchor.bundles:
            raise ValueError("original controls require an unmodified prepared anchor")
        arms.update(direct=(anchor, {"direct": True}), empty=(anchor, {}))
        comparisons.insert(0, ("direct", "empty"))
    protocol = {"source_run": str(search.output), "revision": revision.id, "parent": parent.id,
                "accepted_at_start": search.revisions.current().id,
                "anchor": anchor.id, "objective": search.request.objective, "workload": search.request.workload,
                "request_sha256": hashlib.sha256((search.output/"request.json").read_bytes()).hexdigest(),
                "framework_source": framework_source(), "pairs": pairs, "comparisons": comparisons,
                "arms": {key: {"revision": r.id, "options": opts} for key, (r, opts) in arms.items()},
                "scope": "Developer-launched diagnostic of unchanged Agent candidates, not autonomous optimization; no publication"}
    protocol = json.loads(json.dumps(protocol))
    p = output/"protocol.json"
    if p.exists() and json.loads(p.read_text()) != protocol:
        raise ValueError("cannot resume changed ablation protocol")
    write_json(p, protocol)
    rows = []
    for left, right in comparisons:
        for pair in range(pairs):
            for arm in ((left, right) if pair % 2 == 0 else (right, left)):
                label = f"{left}-{right}-{pair}-{arm}"
                saved = output/f"{label}.json"
                if saved.exists():
                    measured = json.loads(saved.read_text())
                else:
                    r, opts = arms[arm]
                    measured = search.execute(r, output.name+"-"+label, "binding_ablation", collector_options=opts)
                    write_json(saved, measured)
                rows.append({"comparison": [left, right], "pair": pair, "arm": arm, **compact(measured)})
                write_json(output/"results.json", {"protocol": protocol, "runs": rows})
                print(json.dumps({"arm": arm, "pair": pair, "status": measured["status"],
                    "ranks": [{"valid": r.get("result", {}).get("valid"), "metrics": r.get("result", {}).get("metrics")}
                              for r in measured.get("ranks", [])]}), flush=True)
                if measured["status"] != "passed" or not all(r.get("result", {}).get("valid") for r in measured["ranks"]):
                    raise ValueError(f"ablation quality/execution failed: {label}")
    if search.revisions.current().id != protocol["accepted_at_start"]:
        # This driver never commits. Concurrent updates invalidate its parent claim.
        raise ValueError("accepted parent changed during the diagnostic")
    return {"protocol": protocol, "runs": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--revision", required=True)
    p.add_argument("--parent")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--pairs", type=int, default=2)
    p.add_argument("--skip-original-controls", action="store_true")
    args = p.parse_args()
    if args.pairs < 2:
        p.error("at least two alternating pairs are required")
    search = ProjectSearch(load_config(args.config, []), args.run)
    run_protocol(search, search.revisions.revision(args.revision),
                 search.revisions.revision(args.parent) if args.parent else search.revisions.current(),
                 args.output, pairs=args.pairs, original_controls=not args.skip_original_controls)


if __name__ == "__main__":
    main()
