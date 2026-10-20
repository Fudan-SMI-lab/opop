"""Interleave binding backends on identical candidate bytes without publishing."""
import argparse
import hashlib
import json
from pathlib import Path

from kernel_optimizer.config import load_config
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from examples.v6_validation.run_binding_ablation import compact


def compare(search, parent, candidate, output, *, pairs=2):
    output.mkdir(parents=True, exist_ok=False)
    child = set(candidate.bundles) - set(parent.bundles)
    if not child or set(parent.bundles) - set(candidate.bundles):
        raise ValueError("additive candidate required")
    source = framework_source()
    job_prefix = "binding-" + hashlib.sha256(str(output.resolve()).encode()).hexdigest()[:16]
    accepted = search.revisions.current().id
    protocol = {"source": source, "source_run": str(search.output), "parent": parent.id,
                "candidate": candidate.id, "accepted_at_start": accepted, "pairs": pairs,
                "request": search.request.portable(), "job_prefix": job_prefix,
                "scope": "Developer-started unchanged-candidate backend comparison; no publication"}
    write_json(output / "protocol.json", protocol)
    rows = []
    for arm in ("parent", "passthrough", "candidate"):
        revision = parent if arm == "parent" else candidate
        for pair in range(pairs):
            for backend in (("legacy", "scoped") if pair % 2 == 0 else ("scoped", "legacy")):
                assert framework_source() == source
                name = f"{arm}-{pair}-{backend}"
                options = {"binding_backend": backend}
                if arm == "passthrough":
                    options["passthrough_bundles"] = sorted(child)
                run = search.execute(revision, job_prefix+"-"+name, "binding_ablation", collector_options=options)
                write_json(output / (name+".json"), run)
                row = {"arm": arm, "pair": pair, "backend": backend, **compact(run)}
                rows.append(row)
                write_json(output / "results.json", {"protocol": protocol, "runs": rows})
                print(json.dumps({"arm": arm, "backend": backend, "pair": pair, "status": run["status"],
                                  "ranks": [{"valid": r["valid"], "metrics": r["metrics"]} for r in row["ranks"]]}), flush=True)
                if run["status"] != "passed" or not all(r["valid"] for r in row["ranks"]):
                    raise ValueError("quality/execution failed; preserve full receipt")
    assert search.revisions.current().id == accepted
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "run", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--pairs", type=int, default=2)
    args = parser.parse_args()
    if args.pairs < 2:
        parser.error("at least two alternating pairs required")
    search = ProjectSearch(load_config(args.config, []), args.run)
    compare(search, search.revisions.revision(args.parent), search.revisions.revision(args.candidate),
            args.output, pairs=args.pairs)


if __name__ == "__main__":
    main()
