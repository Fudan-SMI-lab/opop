"""Explicit project entrypoint; old CLI dispatch and defaults are retained."""
from pathlib import Path
import json

from kernel_optimizer.config import load_config
from kernel_optimizer.project.request import ProjectRequest
from kernel_optimizer.project.workflow import ProjectWorkflow


def add_project_parser(parser):
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prepare-only", action="store_true", help="generate and validate the project evaluator")
    parser.add_argument("--feedback", type=Path, help="actual failure evidence from a previous preparation")
    parser.add_argument("--resume", action="store_true", help="resume the identical frozen project request")
    parser.add_argument("--reuse-evaluator", type=Path, help="revalidate a frozen generated evaluator for a goal/workload variant without regenerating code")
    parser.add_argument("--discover-only", action="store_true")
    parser.add_argument("--rounds", type=int, default=2, help="candidate rounds in this experiment, not an API usage quota")
    parser.add_argument("--scope", action="append", choices=["group", "joint", "region", "execution_plan"])


def cmd_optimize_project(args):
    config = load_config(Path(args.config) if args.config else None, args.override)
    request = ProjectRequest.model_validate_json(args.request.read_bytes())
    if args.rounds < 1:
        raise ValueError("experiment rounds must be positive")
    feedback = args.feedback.read_text(encoding="utf-8") if args.feedback else None
    if args.resume:
        previous = ProjectRequest.model_validate_json((args.output/"request.json").read_bytes())
        if previous != request:
            raise ValueError("resume cannot change the frozen project request")
        result = ProjectWorkflow(config, request, args.output).prepare(feedback=feedback, resume=True)
    elif args.reuse_evaluator:
        from kernel_optimizer.project.variants import prepare_variant
        result = prepare_variant(args.reuse_evaluator, request, args.output)
        if result["status"] != "prepared":
            raise ValueError("frozen evaluator did not pass variant controls; see prepared.json")
    else:
        result = ProjectWorkflow(config, request, args.output).prepare(feedback=feedback)
    if not args.prepare_only:
        from kernel_optimizer.project.search import ProjectSearch
        search = ProjectSearch(config, args.output)
        if args.resume:
            search.revisions.recover()
        discovery_path = args.output/"discovery.json"
        if not discovery_path.exists() or json.loads(discovery_path.read_text())["revision"] != search.revisions.current().id:
            search.discover()
        if not args.discover_only:
            for _ in range(args.rounds):
                previous = args.output/"latest-feedback.json"
                search.trial(allowed_scopes=tuple(args.scope or ("group", "joint", "region", "execution_plan")),
                             feedback=previous.read_text() if previous.exists() else feedback)
        result = {"status": "completed_experiment", "revision": search.revisions.current().id}
    print(json.dumps({"status": result["status"], "output": str(args.output), "revision": result["revision"]}))
    return 0
