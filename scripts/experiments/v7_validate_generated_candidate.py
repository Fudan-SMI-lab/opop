"""One original paired validation of reviewed, immutable generated outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from kernel_optimizer.config import load_config
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("config", "source", "generated", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--trial", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new isolated output; never restart a validation")
    config = load_config(args.config, [])
    config.usage.mode = "observe_only"
    original = ProjectSearch(config, args.source)
    discovery = json.loads((args.source / "discovery.json").read_text())
    if discovery["revision"] != args.parent:
        raise ValueError("Saved discovery must belong to the explicitly selected parent")
    selection_path = args.generated / "inputs/selection.json"
    selection = json.loads(selection_path.read_text())
    protected = [args.source / name for name in (
        "request.json", "prepared.json", "discovery.json", "project/current.json")]
    protected += [args.generated / name for name in ("candidate.py", "proposal.json")]
    protected.append(selection_path)
    before = {str(path): digest(path) for path in protected}
    source_current = original.revisions.current().id
    parent = fork_run(args.source, args.output, revision_id=args.parent)
    search = ProjectSearch(config, args.output)
    if search.request != original.request:
        raise ValueError("Fork changed the original request")
    bundle, proposal = search.bundle_from_outputs(
        args.generated, selection, parent, discovery, args.trial)
    candidate = search.revisions.stage(bundle)
    protocol = {
        "kind": "one-original-paired-validation", "core": framework_source(),
        "source": str(args.source), "source_current": source_current,
        "parent": parent.id, "candidate": candidate.id,
        "bundle": bundle.model_dump(mode="json"), "parameters": bundle.parameters,
        "candidate_sha256": digest(args.generated / "candidate.py"),
        "proposal_sha256": digest(args.generated / "proposal.json"),
        "selection_sha256": digest(selection_path),
        "shared_controls": search.prepared["controls_path"],
        "request_equal": search.request == original.request,
        "discovery_equal": (args.output / "discovery.json").read_bytes()
            == (args.source / "discovery.json").read_bytes(),
        "critical_before": before,
        "scope": "One validate_bundle(publish=False), fresh AB then BA under the original "
            "request, full quality, objective, noise and acceptance. Empty default parameters; "
            "no parameter search, fixed-screen gate, source/model control rerun, reference copy, "
            "selector/candidate generation, candidate edits or pointer publication. Normal "
            "resource_feedback and ordinary collection planner choose existing collectors. "
            "Preserve rejection and errors; historical measurements are not pooled.",
    }
    write_json(args.output / "validation-protocol.json", protocol)
    try:
        result = search.validate_bundle(bundle, {"trial": args.trial, "scope": protocol["scope"]},
            discovery=discovery, candidate=candidate, publish=False)
        write_json(args.output / "validation-summary.json", {
            "decision": result["decision"], "publication": result["publication"],
            "measurement": result["measurement"],
            "result": str(args.output / "trials" / args.trial / "result.json"),
        })
        print(json.dumps({"decision": result["decision"], "publication": result["publication"]}), flush=True)
    finally:
        preserved = {name: digest(Path(name)) == value for name, value in before.items()}
        write_json(args.output / "preservation.json", {
            "files": preserved, "all_unchanged": all(preserved.values()),
            "source_current": original.revisions.current().id,
            "branch_current": search.revisions.current().id,
            "expected_source_current": source_current, "expected_branch_current": parent.id,
        })


if __name__ == "__main__":
    main()
