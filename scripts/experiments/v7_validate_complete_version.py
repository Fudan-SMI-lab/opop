"""One independent original comparison of an accepted complete version."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from kernel_optimizer.config import load_config
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.versioning import promote_version
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("config", "source", "formal-result", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output; never restart an independent comparison")
    config = load_config(args.config, [])
    config.usage.mode = "observe_only"
    formal = json.loads(args.formal_result.read_text())
    if not formal["decision"]["accepted"]:
        raise ValueError("Independent closure requires a formally accepted complete version")
    source = ProjectSearch(config, args.source)
    accepted = source.revisions.revision(formal["decision"]["revision"])
    baseline = source.revisions.revision(source.prepared["revision"])
    source_current = source.revisions.current().id
    protected = [args.formal_result] + [args.source / name for name in (
        "request.json", "prepared.json", "project/current.json")]
    before = {str(p): digest(p) for p in protected}
    fork_run(args.source, args.output, revision_id=baseline.id)
    search = ProjectSearch(config, args.output)
    write_json(args.output / "independent-protocol.json", {
        "core": framework_source(), "source": str(args.source), "source_current": source_current,
        "accepted_revision": accepted.id, "baseline": baseline.id,
        "bundles": list(accepted.bundles),
        "parameters": {b: source.revisions.bundle(b).parameters for b in accepted.bundles},
        "shared_controls": search.prepared["controls_path"], "critical_before": before,
        "scope": "One promote_version(publish=False), exact accepted complete version vs original prepared "
            "baseline, fresh original AB then BA. Original full quality/clean samples/noise/acceptance unchanged. "
            "No Agent, search, parameter tuning, collectors, source-model controls rerun, reference copy, "
            "initial discovery or pointer publication. Preserve failure; no retry until pass.",
    })
    try:
        result = promote_version(search, args.source, revision_id=accepted.id, publish=False)
        write_json(args.output / "independent-summary.json", {key: result[key] for key in (
            "trial", "revision", "parent", "source_revision", "quality", "coverage_errors",
            "measurement", "decision", "publication")})
        print(json.dumps({"trial": result["trial"], "decision": result["decision"],
                          "publication": result["publication"]}), flush=True)
    finally:
        same = {name: digest(Path(name)) == value for name, value in before.items()}
        write_json(args.output / "preservation.json", {
            "files": same, "all_unchanged": all(same.values()),
            "source_current": source.revisions.current().id, "expected_source_current": source_current,
            "branch_current": search.revisions.current().id, "expected_branch_current": baseline.id,
        })


if __name__ == "__main__":
    main()
