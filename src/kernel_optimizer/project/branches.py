"""Isolated experiment branches sharing immutable evidence and reference tensors."""
import json
import os
from pathlib import Path
import shutil

from kernel_optimizer.project.transactions import RevisionStore
from kernel_optimizer.models.project import ProjectEvaluation
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.store.run_store import RunStore


def fork_run(source_output, output, *, revision_id=None):
    source_output, output = Path(source_output).resolve(), Path(output).resolve()
    source = RunStore.open(source_output)
    revisions = RevisionStore(source)
    parent = revisions.revision(revision_id) if revision_id else revisions.current()
    prepared = json.loads((source_output / "prepared.json").read_text())
    if prepared["status"] != "prepared":
        raise ValueError("experiment branch requires a validated evaluator")
    if parent.bundles and parent.id != revisions.current().id:
        state_path = revisions.root / "transactions" / (parent.id + ".json")
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        if not state.get("report_ref"):
            raise ValueError("exploratory candidate has no project quality evaluation")
        if state.get("kind") == "complete_version":
            report = json.loads(source.get_artifact(state["report_ref"]))
            if (report["quality"] != "pass" or not report["measurement"]["valid"] or report["coverage_errors"]
                    or any(not c["passed"] for c in report["measurement"].get("constraints", []))):
                raise ValueError("exploration requires a quality-passing feasible complete project")
        else:
            report = revisions.load(state["report_ref"], ProjectEvaluation)
            if report.quality != "pass" or not report.feasible or "original_baseline_constraint_failed" in report.reasons:
                raise ValueError("exploration requires a quality-passing feasible complete project")
    # A branch may begin at a quality-passing non-improving candidate. This does
    # not publish that candidate in the source run or claim a global improvement.
    store = RunStore.create(output.parent, output.name, {"kind": "project_experiment_branch",
        "source_output": str(source_output), "source_revision": parent.id,
        "source_current": revisions.current().id, "publication": "isolated experiment"})
    target = RevisionStore(store)
    for path in source.artifacts_dir.iterdir():
        if path.is_file():
            destination = store.artifacts_dir / path.name
            if len(path.name) != 64 or any(c not in "0123456789abcdef" for c in path.name):
                shutil.copyfile(path, destination)
                continue
            try:
                os.link(path, destination)
            except OSError:
                shutil.copyfile(path, destination)
    for name in ("revisions", "bundles"):
        for path in (revisions.root / name).glob("*.json"):
            shutil.copyfile(path, target.root / name / path.name)
    for name in ("request.json",):
        if (source_output / name).exists():
            shutil.copyfile(source_output / name, output / name)
    if parent.bundles and parent.id != revisions.current().id:
        feedback = (source_output / "version-trials" / report["id"] / "result.json" if isinstance(report, dict)
                    else source_output / "trials" / report.id / "result.json")
        if feedback.exists():
            shutil.copyfile(feedback, output / "latest-feedback.json")
    # References are read-only in evaluate/profile, avoiding multi-GB duplication.
    prepared["controls_path"] = prepared.get("controls_path", str(source_output / "controls"))
    write_json(output / "prepared.json", prepared)
    shutil.copyfile(revisions.root / "anchor.json", target.root / "anchor.json")
    write_json(target.pointer, {"revision": parent.id})
    discovery = source_output / "discovery.json"
    if discovery.exists() and json.loads(discovery.read_text())["revision"] == parent.id:
        shutil.copyfile(discovery, output / "discovery.json")
    store.append("PROJECT_BRANCH_CREATED", {"source_output": str(source_output), "revision": parent.id,
        "source_current_unchanged": revisions.current().id, "reference_revision": prepared["revision"],
        "publication": "isolated experiment; compare against the original source before promotion"})
    return parent
