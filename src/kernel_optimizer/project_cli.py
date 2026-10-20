"""Explicit project entrypoint; old CLI dispatch and defaults are retained."""
from pathlib import Path
import json

from kernel_optimizer.config import load_config
from kernel_optimizer.project.request import ProjectRequest
from kernel_optimizer.project.workflow import ProjectWorkflow


def add_project_parser(parser):
    entry = parser.add_mutually_exclusive_group(required=True)
    entry.add_argument("--request", type=Path)
    entry.add_argument("--project", help="local project directory or Git repository URL")
    parser.add_argument("--goal", help="natural-language goal for --project")
    parser.add_argument("--command", help="existing project launch command, when known")
    parser.add_argument("--context", action="append", default=[], help="data paths or necessary task constraints")
    parser.add_argument("--project-ref", help="branch/tag when cloning a repository")
    parser.add_argument("--intake-only", action="store_true", help="resolve project input without running optimization")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prepare-only", action="store_true", help="generate and validate the project evaluator")
    parser.add_argument("--feedback", type=Path, help="actual failure evidence from a previous preparation")
    parser.add_argument("--resume", action="store_true", help="resume the identical frozen project request")
    parser.add_argument("--reuse-evaluator", type=Path, help="revalidate a frozen generated evaluator for a goal/workload variant without regenerating code")
    parser.add_argument("--resume-trial", help="resume one saved generated trial without another generation request; implies --resume")
    parser.add_argument("--resume-generation", type=Path, help="continue an interrupted selection/candidate sandbox in its original provider session; implies --resume")
    parser.add_argument("--replay-from", type=Path, help="source run for frozen-candidate cross-goal/held-out evaluation")
    parser.add_argument("--replay-bundle", help="exact stored bundle ID to replay without code/parameter changes")
    parser.add_argument("--publish-replay", action="store_true", help="publish a replay only if its independent paired evaluation passes")
    parser.add_argument("--branch-from", type=Path, help="create an isolated experiment from a prepared run")
    parser.add_argument("--branch-revision", help="explicit quality-passing branch parent, including a non-improving candidate")
    parser.add_argument("--leaf-groups-only", action="store_true", help="restrict eligible groups to observed leaves; use with --scope group for operator-only comparison")
    parser.add_argument("--selection-strategy", choices=["agent", "local_time"], default="agent")
    parser.add_argument("--feedback-view", choices=["full", "project"], default="full")
    parser.add_argument("--search-strategy", choices=["sequential", "bounded_joint"], default="sequential")
    parser.add_argument("--frontier-width", type=int, default=2, help="number of retained feasible exploratory branches")
    parser.add_argument("--experiment-id", help="persisted bounded_joint experiment identity to create or resume")
    parser.add_argument("--promote-from", type=Path, help="independently compare and publish a complete branch version against current")
    parser.add_argument("--promote-revision", help="explicit complete branch revision, otherwise its current version")
    parser.add_argument("--discover-only", action="store_true")
    parser.add_argument("--rounds", type=int, default=2, help="candidate rounds in this experiment, not an API usage quota")
    parser.add_argument("--scope", action="append", choices=["group", "joint", "region", "execution_plan"])


def cmd_optimize_project(args):
    config = load_config(Path(args.config) if args.config else None, args.override)
    if getattr(args, "project", None):
        if not args.goal:
            raise ValueError("--project requires a natural-language --goal")
        from kernel_optimizer.project.intake import prepare_request
        resolved = prepare_request(config, project=args.project, goal=args.goal, command=args.command,
            context=args.context, output=args.output, project_ref=args.project_ref)
        if resolved["status"] != "ready" or args.intake_only:
            print(json.dumps(resolved))
            return 0 if resolved["status"] == "ready" else 2
        args.request = Path(resolved["request"])
    elif getattr(args, "intake_only", False):
        raise ValueError("--intake-only requires --project")
    request = ProjectRequest.model_validate_json(args.request.read_bytes())
    if args.rounds < 1:
        raise ValueError("experiment rounds must be positive")
    feedback = args.feedback.read_text(encoding="utf-8") if args.feedback else None
    replay = getattr(args, "replay_bundle", None)
    promote = getattr(args, "promote_from", None)
    if getattr(args, "promote_revision", None) and not promote:
        raise ValueError("promote-revision requires promote-from")
    if promote:
        if args.prepare_only or args.discover_only or args.resume_trial or replay or getattr(args, "resume_generation", None):
            raise ValueError("complete version promotion cannot be combined with other execution modes")
        args.resume = True
    resume_generation = getattr(args, "resume_generation", None)
    if resume_generation:
        if args.prepare_only or args.discover_only or args.resume_trial or replay:
            raise ValueError("resume-generation cannot be combined with other preparation/replay modes")
        args.resume = True
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
        if not args.resume_trial and not resume_generation and (not discovery_path.exists() or json.loads(discovery_path.read_text())["revision"] != search.revisions.current().id):
            search.discover()
        if not args.discover_only:
            if promote:
                from kernel_optimizer.project.versioning import promote_version
                promote_version(search, promote, getattr(args, "promote_revision", None))
            elif resume_generation:
                search.resume_generation(resume_generation)
            elif replay:
                from kernel_optimizer.project.replay import transfer_bundle
                bundle, metadata = transfer_bundle(args.replay_from, replay, search)
                search.validate_bundle(bundle, metadata, publish=getattr(args, "publish_replay", False))
            elif args.resume_trial:
                search.resume_trial(args.resume_trial)
            elif getattr(args, "search_strategy", "sequential") == "bounded_joint":
                from kernel_optimizer.project.experiment import bounded_experiment
                bounded_experiment(search, rounds=args.rounds, frontier_width=args.frontier_width,
                    experiment_id=args.experiment_id,
                    trial_options={"allowed_scopes": tuple(args.scope or ("group", "joint", "region", "execution_plan")),
                        "leaf_groups_only": args.leaf_groups_only, "selection_strategy": args.selection_strategy,
                        "feedback_view": args.feedback_view})
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
