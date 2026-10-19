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
    parser.add_argument("--resume-trial", help="resume one saved generated trial without another generation request; implies --resume")
    parser.add_argument("--replay-from", type=Path, help="source run for frozen-candidate cross-goal/held-out evaluation")
    parser.add_argument("--replay-bundle", help="exact stored bundle ID to replay without code/parameter changes")
    parser.add_argument("--publish-replay", action="store_true", help="publish a replay only if its independent paired evaluation passes")
    parser.add_argument("--branch-from", type=Path, help="create an isolated experiment from a prepared run")
    parser.add_argument("--branch-revision", help="explicit quality-passing branch parent, including a non-improving candidate")
    parser.add_argument("--leaf-groups-only", action="store_true", help="restrict eligible groups to observed leaves; use with --scope group for operator-only comparison")
    parser.add_argument("--selection-strategy", choices=["agent", "local_time"], default="agent")
    parser.add_argument("--feedback-view", choices=["full", "project"], default="full")
    parser.add_argument("--discover-only", action="store_true")
    parser.add_argument("--rounds", type=int, default=2, help="candidate rounds in this experiment, not an API usage quota")
    parser.add_argument("--scope", action="append", choices=["group", "joint", "region", "execution_plan"])


def cmd_optimize_project(args):
    config = load_config(Path(args.config) if args.config else None, args.override)
    request = ProjectRequest.model_validate_json(args.request.read_bytes())
    if args.rounds < 1:
        raise ValueError("experiment rounds must be positive")
    feedback = args.feedback.read_text(encoding="utf-8") if args.feedback else None
    replay = getattr(args, "replay_bundle", None)
    if bool(getattr(args, "replay_from", None)) != bool(replay):
        raise ValueError("replay-from and replay-bundle must be supplied together")
    if replay and (args.prepare_only or args.discover_only or args.resume_trial):
        raise ValueError("replay cannot be combined with prepare-only/discover-only/resume-trial")
    branch = getattr(args, "branch_from", None)
    if getattr(args, "branch_revision", None) and not branch:
        raise ValueError("branch-revision requires branch-from")
    if branch and (args.resume or args.resume_trial or args.reuse_evaluator):
        raise ValueError("branch creation cannot be combined with resume/reuse-evaluator")
    if args.resume_trial:
        if args.prepare_only or args.discover_only:
            raise ValueError("resume-trial cannot be combined with prepare-only/discover-only")
        args.resume = True
    if branch:
        from kernel_optimizer.project.branches import fork_run
        if ProjectRequest.model_validate_json((branch/"request.json").read_bytes()) != request:
            raise ValueError("experiment branch cannot change the frozen request; use a validated workload variant")
        fork_run(branch, args.output, revision_id=getattr(args, "branch_revision", None))
        result = json.loads((args.output/"prepared.json").read_text())
    elif args.resume:
        previous = ProjectRequest.model_validate_json((args.output/"request.json").read_bytes())
        if previous != request:
            raise ValueError("resume cannot change the frozen project request")
        result = ProjectWorkflow(config, request, args.output).prepare(feedback=feedback, resume=True)
    elif args.reuse_evaluator:
        from kernel_optimizer.project.variants import prepare_variant
        result = prepare_variant(args.reuse_evaluator, request, args.output, config=config)
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
        if not args.resume_trial and (not discovery_path.exists() or json.loads(discovery_path.read_text())["revision"] != search.revisions.current().id):
            search.discover()
        if not args.discover_only:
            if replay:
                from kernel_optimizer.project.replay import transfer_bundle
                bundle, metadata = transfer_bundle(args.replay_from, replay, search)
                search.validate_bundle(bundle, metadata, publish=getattr(args, "publish_replay", False))
            elif args.resume_trial:
                search.resume_trial(args.resume_trial)
            else:
                for _ in range(args.rounds):
                    previous = args.output/"latest-feedback.json"
                    search.trial(allowed_scopes=tuple(args.scope or ("group", "joint", "region", "execution_plan")),
                                 feedback=previous.read_text() if previous.exists() else feedback,
                                 leaf_groups_only=getattr(args, "leaf_groups_only", False),
                                 selection_strategy=getattr(args, "selection_strategy", "agent"),
                                 feedback_view=getattr(args, "feedback_view", "full"))
        result = {"status": "completed_experiment", "revision": search.revisions.current().id}
    print(json.dumps({"status": result["status"], "output": str(args.output), "revision": result["revision"]}))
    return 0
