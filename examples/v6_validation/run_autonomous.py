"""Frozen fresh-input validation using the ordinary project workflow and Agents."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.project.request import ProjectRequest
from kernel_optimizer.project.workflow import ProjectWorkflow
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "request", "output"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--case", choices=["qwen", "resnet", "ddp"], required=True)
    parser.add_argument("--readiness-reference", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("autonomous validation requires a fresh output directory")
    if args.rounds < 2:
        raise ValueError("at least two feedback rounds required")
    config = load_config(args.config, [])
    config.opencode.port = args.port
    request = ProjectRequest.model_validate_json(args.request.read_bytes())
    source = framework_source()
    protocol = {"source": source, "request_sha256": hashlib.sha256(args.request.read_bytes()).hexdigest(),
                "source_files": {name: hashlib.sha256((request.source_root/name).read_bytes()).hexdigest()
                                 for name in request.files},
                "request": request.portable(), "rounds": args.rounds,
                "oracle": {"case": args.case, "reference": str(args.readiness_reference)},
                "scope": "Fresh evaluator/discovery/selection/candidates; no imported prepared run or candidate; no human messages during this run",
                "usage": "Observe only; rounds specify the experiment, not an API/request/token quota"}
    rounds = []
    stage = "prepare"
    try:
        prepared = ProjectWorkflow(config, request, args.output).prepare()
        write_json(args.output/"autonomy-protocol.json", protocol)
        if prepared["status"] != "prepared":
            raise ValueError("evaluator preparation failed")
        stage = "independent-source-check"
        cmd = [sys.executable, "-m", "examples.v6_validation.source_fidelity",
               "--project", str(args.output), "--case", args.case,
               "--output", str(args.output/"source-fidelity.json")]
        if args.readiness_reference:
            cmd.extend(["--readiness-reference", str(args.readiness_reference)])
        with (args.output/"source-fidelity.log").open("wb") as log:
            check = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        if check.returncode:
            raise ValueError("independent source-fidelity check failed; preserve run without optimizing a changed task")
        stage = "discovery"
        search = ProjectSearch(config, args.output)
        search.discover()
        stage = "optimization"
        for index in range(args.rounds):
            if framework_source() != source:
                raise RuntimeError("framework changed during frozen run")
            try:
                feedback = args.output/"latest-feedback.json"
                result = search.trial(feedback=feedback.read_text() if feedback.exists() else None)
                row = {"round": index, "trial": result["trial"], "decision": result["decision"],
                       "measurement": result.get("measurement"), "revision": search.revisions.current().id}
            except Exception as exc:
                row = {"round": index, "error": str(exc), "traceback": traceback.format_exc()}
            rounds.append(row)
            write_json(args.output/"autonomy-progress.json", {"protocol": protocol, "rounds": rounds})
            print(json.dumps(row), flush=True)
        write_json(args.output/"autonomy-result.json", {"status": "completed", "protocol": protocol,
                   "rounds": rounds, "accepted_revision": search.revisions.current().id,
                   "framework_unchanged": framework_source() == source,
                   "human_interventions_during_run": []})
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output/"autonomy-protocol.json", protocol)
        write_json(args.output/"autonomy-result.json", {"status": "failed", "stage": stage,
                   "error": str(exc), "traceback": traceback.format_exc(), "protocol": protocol,
                   "rounds": rounds, "framework_unchanged": framework_source() == source,
                   "human_interventions_during_run": []})
        raise


if __name__ == "__main__":
    main()
