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


def cmd_optimize_project(args):
    config = load_config(Path(args.config) if args.config else None, args.override)
    request = ProjectRequest.model_validate_json(args.request.read_bytes())
    if not args.prepare_only:
        raise ValueError("project search is not installed yet; --prepare-only validates the generated evaluator")
    result = ProjectWorkflow(config, request, args.output).prepare(
        feedback=args.feedback.read_text(encoding="utf-8") if args.feedback else None)
    print(json.dumps({"status": result["status"], "output": str(args.output), "revision": result["revision"]}))
    return 0
