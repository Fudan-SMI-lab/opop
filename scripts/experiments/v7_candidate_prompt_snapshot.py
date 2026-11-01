"""One candidate generation from a retained snapshot with an explicit prompt version."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from kernel_optimizer.config import load_config
from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.project.generated import validate_proposal
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.prompt_versions import load_prompt
from kernel_optimizer.project.runtime_config import project_agent_config
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--from-version", default="v6")
    parser.add_argument("--to-version", required=True)
    args = parser.parse_args()
    config = project_agent_config(load_config(args.config, []))
    config.usage.mode = "observe_only"
    saved, original, files = load(args.snapshot)
    if saved["role"] != "candidate" or not saved["fresh_session"]:
        raise ValueError("requires an initial candidate snapshot")
    config.project_defaults.prompt_versions["candidate"] = args.from_version
    previous = load_prompt(config, "candidate")
    config.project_defaults.prompt_versions["candidate"] = args.to_version
    selected = load_prompt(config, "candidate")
    if previous == selected or original.count(previous) != 1:
        raise ValueError("must replace exactly one complete prompt with a different version")
    edit = {"path": "prompt", "old": previous, "new": selected}
    protocol = {
        "at": datetime.now(timezone.utc).isoformat(), "kind": "one-candidate-prompt-comparison",
        "snapshot": str(args.snapshot.resolve()), "original_session": saved["session"],
        "from_version": args.from_version, "to_version": args.to_version,
        "previous_prompt_sha256": hashlib.sha256(previous.encode()).hexdigest(),
        "selected_prompt_sha256": hashlib.sha256(selected.encode()).hexdigest(),
        "full_prompt_sha256": hashlib.sha256(original.replace(previous, selected).encode()).hexdigest(),
        "inputs": len(files), "input_bytes": sum(len(raw) for raw in files.values()),
        "original_framework": saved["framework_source"], "framework": framework_source(),
        "scope": "One fresh candidate session; same original snapshot inputs/model/tools; normal contract repairs. "
                 "No controller selection, candidate hand edits, parameter search, GPU work or acceptance stage; "
                 "actual role tool use remains subject to review. "
                 "Historical control is not a replicated causal comparison. External filesystem is not hermetic.",
    }
    store = RunStore.create(args.output.parent, args.output.name, protocol)
    write_json(args.output / "protocol.json", protocol)
    try:
        with Runtime(config, log_dir=args.output) as runtime:
            directory, receipt = ProjectAgents(runtime, store).replay_snapshot(
                args.snapshot, edit=edit, validator=validate_proposal, allow_source_change=True)
        result = {"status": "generated", "sandbox": str(directory), "agent": receipt}
        write_json(args.output / "result.json", result)
        print(json.dumps({"status": result["status"], "sandbox": str(directory), "session": receipt["session"]}), flush=True)
    except Exception as exc:
        write_json(args.output / "result.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main()
