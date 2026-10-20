"""Run independent control/treatment generations from one saved project role."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from kernel_optimizer.config import load_config
from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.project.runtime_config import project_agent_config
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--edit", type=Path, help="JSON: path, old, new, optional occurrences")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1, help="independent sessions in flight; not a usage quota")
    parser.add_argument("--allow-source-change", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or args.workers < 1:
        parser.error("repeats and workers must be positive")
    config = project_agent_config(load_config(args.config, []))
    config.usage.mode = "observe_only"
    edit = json.loads(args.edit.read_text()) if args.edit else None
    store = RunStore.create(args.output.parent, args.output.name, {
        "kind": "generation-comparison", "snapshot": str(args.snapshot), "edit": edit,
        "repeats": args.repeats, "workers": args.workers,
        "scope": "Fresh role generations; GPU validation is a separate recorded step"})
    rows = []
    with Runtime(config, log_dir=args.output) as runtime:
        agents = ProjectAgents(runtime, store)
        def generate(repeat, arm, change):
            try:
                directory, receipt = agents.replay_snapshot(args.snapshot, edit=change,
                    allow_source_change=args.allow_source_change)
                return {"repeat": repeat, "arm": arm, "sandbox": str(directory), "agent": receipt}
            except Exception as exc:
                return {"repeat": repeat, "arm": arm, "error": f"{type(exc).__name__}: {exc}"}

        tasks = []
        for repeat in range(args.repeats):
            arms = [("control", None), ("treatment", edit)] if edit else [("control", None)]
            if repeat % 2:
                arms.reverse()
            for arm, change in arms:
                tasks.append((repeat, arm, change))
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(generate, *task) for task in tasks]):
                row = future.result()
                rows.append(row)
                write_json(args.output/"result.json", {"runs": rows, "scope": "Generation only; not optimization success"})
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
