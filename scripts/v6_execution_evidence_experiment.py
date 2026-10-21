"""Replay one frozen candidate generation with an index of existing evidence.

The treatment adds navigation and unmodified observations, not a repair instruction.
Both fresh sessions keep every original input and use normal GPU validation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

from kernel_optimizer.config import load_config
from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.generated import validate_proposal
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.wiring import Runtime


def evidence_index(value):
    """Index measured execution facts without interpreting them as failures."""
    rows = []
    for i, rank in enumerate(value.get("quality_and_coverage", [])):
        for bundle, coverage in (rank.get("coverage") or {}).items():
            rows.append({"source": f"quality_and_coverage[{i}].coverage[{bundle!r}]",
                         "rank": rank.get("rank"), "bundle": bundle,
                         **{k: coverage[k] for k in ("counts", "fallbacks", "reference_invocations",
                                                     "candidate_declared_routes") if k in coverage}})
    diagnostics = []
    for i, observation in enumerate((value.get("parameter_search") or {}).get("observations", [])):
        diagnostic = observation.get("candidate_diagnostics") or {}
        prefix = f"parameter_search.observations[{i}].candidate_diagnostics"
        for j, rank in enumerate(diagnostic.get("run", {}).get("ranks", [])):
            result = rank.get("result", {}).get("candidate_diagnostics", {})
            diagnostics.append({"source": prefix + f".run.ranks[{j}].result.candidate_diagnostics",
                                "rank": rank.get("rank"), "status": diagnostic.get("status"),
                                "events": [{k: event[k] for k in
                                            ("phase", "bundle", "file", "function", "line", "event",
                                             "type", "message", "count") if k in event}
                                           for event in result.get("events", [])]})
    return {"scope": "Navigation to existing observations in this same feedback document; no new measurement or inferred failure category.",
            "execution": rows, "diagnostics": diagnostics,
            "interpretation": "Quality describes returned outputs; wrapper invocation alone does not prove an optimized path. Routes are candidate reports. Exceptions and False returns may be intentionally handled or normal. Inspect the referenced evidence and implementation; no next action is prescribed."}


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("config", "source", "snapshot", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-trial", required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    saved = json.loads((args.source / "trials" / args.source_trial / "generated.json").read_text())
    manifest, _, files = load(args.snapshot)
    original = files["inputs/previous-feedback.txt"].decode()
    assert original.startswith("{")
    index = evidence_index(json.loads(original))
    addition = json.dumps({"execution_evidence_index": index}, indent=2)[1:-1]
    edit = {"path": "inputs/previous-feedback.txt", "old": original,
            "new": "{" + addition + "," + original[1:]}
    protocol = {"snapshot": str(args.snapshot), "source_trial": args.source_trial,
                "original_feedback_sha256": hashlib.sha256(original.encode()).hexdigest(),
                "original_framework": manifest["framework_source"],
                "treatment": "Only prepend a compact index of existing execution facts; original document bytes after its opening brace are retained.",
                "scope": "Developer-started frozen-input comparison; independent real sessions; no candidate edits; isolated normal parameter search and paired evaluation."}
    write_json(args.output / "protocol.json", protocol)
    write_json(args.output / "execution-evidence-index.json", index)

    def run(arm, offset):
        branch = args.output / arm
        parent = fork_run(args.source, branch, revision_id=saved["bundle"]["parent_revision"])
        config = load_config(args.config, [])
        config.opencode.port += offset
        config.usage.mode = "observe_only"
        search = ProjectSearch(config, branch)
        try:
            with Runtime(search.config, log_dir=branch) as runtime:
                directory, receipt = ProjectAgents(runtime, search.store).replay_snapshot(
                    args.snapshot, edit=edit if arm == "treatment" else None, validator=validate_proposal)
            discovery = search.current_discovery()
            trial_id = "index-" + arm
            bundle, proposal = search.bundle_from_outputs(directory, saved["selection"], parent, discovery, trial_id)
            metadata = {"trial": trial_id, "selection": saved["selection"], "proposal": proposal,
                        "candidate_agent": receipt, "experiment": protocol, "arm": arm}
            write_json(branch / "trials" / trial_id / "generated.json", {**metadata, "bundle": bundle.model_dump(mode="json")})
            result = search._evaluate_generated(bundle, metadata, discovery)
            return {"arm": arm, "output": str(branch), "trial": trial_id,
                    "decision": result["decision"], "measurement": result.get("measurement")}
        except Exception as exc:
            import traceback
            return {"arm": arm, "output": str(branch), "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()}

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, arm, i) for i, arm in enumerate(("control", "treatment"))]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            write_json(args.output / "result.json", {"protocol": protocol, "runs": rows})
            print(json.dumps(row), flush=True)
    write_json(args.output / "completed.json", {"completed": True})


if __name__ == "__main__":
    main()
