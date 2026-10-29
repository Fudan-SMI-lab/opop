"""Resume the one interrupted selector session after a recorded container shutdown."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import subprocess
import sys
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.versioning import promote_version
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation


def now():
    return datetime.now(timezone.utc).isoformat()


def preflight(args, config):
    audit = read(args.output / "recovery-audit.json")
    old = Path(audit["old_root"])
    protocol = read(old / "protocol.json")
    assert framework_source() == audit["core"] == protocol["core"]
    assert not (old / "followup-completed.json").exists()
    assert all(sha(path) == value for path, value in audit["preserved_files"].items())
    assert all(sha(path) == value for path, value in audit["input_hashes"].items())
    search = ProjectSearch(config, Path(audit["branch"]))
    assert search.revisions.current().id == audit["current"]
    assert list(search.revisions.current().bundles) == audit["bundles"]
    assert sha(search.output / "latest-feedback.json") == audit["feedback_before_sha256"]
    assert sha(search.output / "discovery.json") == audit["discovery_sha256"]
    assert len(list((search.output / "trials").glob("*/result.json"))) == 1
    assert not (Path(audit["sandbox"]) / "selection.json").exists()
    original = ProjectSearch(config, Path(protocol["source"]))
    assert original.revisions.current().id == original.prepared["revision"] == audit["baseline"]
    return audit, search, original


def main(args):
    os.umask(0o077)
    config = load_config(args.config, [])
    config.usage.mode = "observe_only"
    audit, search, original = preflight(args, config)
    if args.preflight:
        print(json.dumps({"status": "passed", "core": framework_source(),
            "current": audit["current"], "session": audit["interrupted"]["session"],
            "preserved_files": len(audit["preserved_files"]), "input_files": len(audit["input_hashes"]),
            "completed_trials_retained": 1, "unfinished_trials_to_resume": 1,
            "baseline": audit["baseline"], "controls": search.prepared["controls_path"]}))
        return
    assert not (args.output / "protocol.json").exists()
    argv = [sys.executable, "-B", "-m", "kernel_optimizer.cli", "--config", str(args.config),
        "optimize-project", "--request", str(search.output / "request.json"), "--output", str(search.output),
        "--resume-generation", audit["sandbox"], "--rounds", "1"]
    write_json(args.output / "protocol.json", {"at": now(), "core": framework_source(),
        "recovery_audit": str(args.output / "recovery-audit.json"),
        "recovery_audit_sha256": sha(args.output / "recovery-audit.json"), "argv": argv,
        "session": audit["interrupted"]["session"], "current": audit["current"],
        "original_accepted_revision": audit["original_accepted_revision"], "baseline": audit["baseline"],
        "scope": "Continue only the interrupted second selector in its actual persisted session through existing resume_generation. "
            "One completed rejected trial remains frozen. No new preparation, references, initial discovery, manual diagnosis or candidate. "
            "Existing generic recovery wording says temporary service error; the actual interruption was user-confirmed unpaid-balance shutdown. "
            "Conditional new accepted active bundle set gets one independent full-version AB/BA versus original baseline; publish=False. "
            "Measurement periods straddle container restart; prior accepted result and original constraint anchor remain frozen. API observe-only.",
        "disk": disk_observation(Path("/root/autodl-tmp"))})
    results = {}
    try:
        write_json(args.output / "state.json", {"at": now(), "phase": "running", "stage": "resume_second_generation"})
        with (args.output / "search-recovery.log").open("wb") as log:
            process = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
        search = ProjectSearch(config, Path(audit["branch"]))
        current = search.revisions.current()
        trials = []
        for path in sorted((search.output / "trials").glob("*/result.json")):
            if str(path) == audit["completed_trial"]["path"]:
                continue
            value = read(path)
            trials.append({"path": str(path), "sha256": sha(path), "decision": value.get("decision"),
                "generation_recovery": value.get("generation_recovery")})
        changed = list(current.bundles) != audit["bundles"]
        results["recovered-trial"] = {"status": "completed" if process.returncode == 0 else "error",
            "returncode": process.returncode, "current": current.id, "bundles": list(current.bundles),
            "changed_bundles": changed, "new_trials": trials, "prior_trial_retained": audit["completed_trial"]}
        if process.returncode == 0:
            assert len(trials) == 1
            assert trials[0]["generation_recovery"]["session"] == audit["interrupted"]["session"]
        write_json(args.output / "followup-results.json", results)
        if process.returncode != 0 or not changed:
            results["independent"] = {"status": "not_run", "reason": "No newly accepted active bundle set, or recovery execution failed. No old-candidate retest."}
        else:
            write_json(args.output / "state.json", {"at": now(), "phase": "running", "stage": "independent"})
            target = args.output / "qwen-independent"
            fork_run(original.output, target, revision_id=original.prepared["revision"])
            independent = ProjectSearch(config, target)
            result = promote_version(independent, search.output, revision_id=current.id, publish=False)
            assert independent.revisions.current().id == original.prepared["revision"]
            path = target / "version-trials" / result["trial"] / "result.json"
            results["independent"] = {"status": "completed", "path": str(path), "sha256": sha(path),
                **{k: result[k] for k in ["trial", "revision", "quality", "coverage_errors", "measurement", "decision", "publication"]}}
    except Exception:
        results["controller"] = {"status": "error", "traceback": traceback.format_exc()}
    try:
        assert all(sha(path) == value for path, value in audit["preserved_files"].items())
        assert original.revisions.current().id == audit["baseline"]
        after = {path: sha(path) for path in audit["input_hashes"]}
        results["preservation"] = {"status": "completed", "files": len(audit["preserved_files"]),
            "source_current": original.revisions.current().id,
            "recovery_input_changes": {path: {"before": audit["input_hashes"][path], "after": value}
                for path, value in after.items() if value != audit["input_hashes"][path]},
            "scope": "Initial snapshot and session history preserved. Existing recovery may rewrite worker-environment/binding input before capturing a continuation snapshot; report exact differences."}
    except Exception:
        results["preservation"] = {"status": "error", "traceback": traceback.format_exc()}
    terminal = {"phase": "completed" if all(v["status"] in {"completed", "not_run"} for v in results.values()) else "completed_with_errors",
        "finished_at": now(), "results": results, "disk": disk_observation(Path("/root/autodl-tmp")),
        "scope": "Interrupted original two-round experiment completed only its remaining trial. Execution, quality, autonomous acceptance and independent acceptance are separate."}
    write_json(args.output / "followup-results.json", results)
    write_json(args.output / "state.json", terminal)
    write_json(args.output / "followup-completed.json", terminal)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    main(parser.parse_args())
