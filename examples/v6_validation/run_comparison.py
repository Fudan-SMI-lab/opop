"""Run one explicit comparison arm from the same frozen, discovered project.

Each arm uses real configured Agents and real project workers. Rounds/parameter
trials describe the GPU experiment protocol, never a provider usage quota.
"""
import argparse
import json
from pathlib import Path
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source


POLICIES = {
    "whole": {},
    "leaf": {"allowed_scopes": ("group",), "leaf_groups_only": True},
    "score_only": {"feedback_view": "project"},
    "local_time": {"allowed_scopes": ("group",), "leaf_groups_only": True, "selection_strategy": "local_time"},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--source-revision")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--arm", choices=POLICIES, required=True)
    parser.add_argument("--rounds", default=2, type=int)
    parser.add_argument("--port", required=True, type=int)
    args = parser.parse_args()
    if args.rounds < 2:
        raise ValueError("feedback comparison requires at least two explicit experiment rounds")
    start = framework_source()
    parent = fork_run(args.source_run, args.output, revision_id=args.source_revision)
    config = load_config(args.config, []); config.opencode.port = args.port
    search = ProjectSearch(config, args.output)
    if not (args.output / "discovery.json").exists():
        search.discover()
    protocol = {"arm": args.arm, "policy": POLICIES[args.arm], "rounds": args.rounds,
        "framework_source": start, "source_run": str(args.source_run), "parent": parent.id,
        "reference_revision": search.prepared["revision"], "request": search.request.portable(),
        "usage": "observe only; no provider request/token/cost/duration quota",
        "comparison_scope": "same starting version, evaluator, workload, quality, candidate rounds and maximum parameter trials; report actual setup/profile/worker costs and failures"}
    write_json(args.output / "comparison-protocol.json", protocol)
    results = []
    for index in range(args.rounds):
        if framework_source() != start:
            raise RuntimeError("framework changed during comparison; preserve this run and restart the protocol")
        try:
            feedback = args.output / "latest-feedback.json"
            result = search.trial(feedback=feedback.read_text() if feedback.exists() else None, **POLICIES[args.arm])
            summary = {"round": index, "trial": result["trial"], "decision": result["decision"],
                       "measurement": result.get("measurement"), "revision": search.revisions.current().id}
        except Exception as exc:
            summary = {"round": index, "error": str(exc), "traceback": traceback.format_exc(),
                       "revision": search.revisions.current().id}
        results.append(summary)
        write_json(args.output / "comparison-results.json", {"protocol": protocol, "rounds": results,
            "framework_unchanged": framework_source() == start})
        print(json.dumps(summary), flush=True)
    if framework_source() != start:
        raise RuntimeError("framework changed during comparison")


if __name__ == "__main__":
    main()
