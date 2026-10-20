"""Explicit finite experiments with a feasible, non-greedy branch frontier.

Rounds bound proposed GPU experiments, not provider requests, repairs or spending.
"""
import json
from pathlib import Path
from uuid import uuid4

from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.versioning import promote_version
from kernel_optimizer.runtime.execution import write_json


def explorable(result):
    report = result.get("evaluation", {})
    return (report.get("quality") == "pass" and report.get("feasible") is True
            and result.get("measurement", {}).get("framework_source") != "mixed"
            and not result.get("missing_coverage")
            and set(result["decision"].get("reasons", [])) <= {
                "paired_gain_not_above_noise", "objective_not_improved"})


def bounded_experiment(search, *, rounds, frontier_width=2, trial_options=None, experiment_id=None):
    if rounds < 1 or frontier_width < 1:
        raise ValueError("experiment rounds and frontier width must be positive")
    options = trial_options or {}
    experiment_id = experiment_id or "experiment-"+uuid4().hex[:12]
    root = search.output/"experiments"/experiment_id
    state_path = root/"experiment.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["rounds"] != rounds or state["frontier_width"] != frontier_width or state["trial_options"] != json.loads(json.dumps(options)):
            raise ValueError("resuming an experiment cannot change its protocol")
    else:
        state = {"id": experiment_id, "strategy": "bounded_joint", "rounds": rounds,
                 "frontier_width": frontier_width, "trial_options": options, "frontier": [], "steps": [],
                 "main_initial": search.revisions.current().id,
                 "policy": "alternate main and least-visited feasible branch; retain best objective branches; independently compare complete versions before publication"}
    def save():
        write_json(state_path, state)
    save()
    while len(state["steps"]) < rounds:
        index = len(state["steps"])
        inflight = state.get("inflight")
        branch_entry = (next((row for row in state["frontier"] if row["output"] == inflight["output"]), None)
                        if inflight else min(state["frontier"], key=lambda row: (row["visits"], row["priority"]))
                        if index % 2 and state["frontier"] else None)
        active = ProjectSearch(search.config, Path(branch_entry["output"])) if branch_entry else search
        if branch_entry and not inflight:
            branch_entry["visits"] += 1
        previous = active.output/"latest-feedback.json"
        feedback = previous.read_text() if previous.exists() else None
        if branch_entry:
            feedback = json.dumps({"experiment_context": "This is an isolated feasible exploratory branch. Its first candidate may be locally slower. Seek complementary changes and explicitly retain or replace installed bundles. The complete resulting project will be independently compared against main before publication.",
                                   "previous_result": json.loads(feedback) if feedback else None})
        if inflight:
            result = resume_round(active, inflight)
        else:
            state["inflight"] = {"round": index, "output": str(active.output), "parent": active.revisions.current().id,
                "known_trials": [p.name for p in (active.output/"trials").glob("*")],
                "known_sandboxes": [p.name for p in (active.output/"sandboxes").glob("*")]}
            save()
            result = active.trial(feedback=feedback, **options)
        state["inflight"]["trial"] = result["trial"]
        save()
        step = {"round": index, "output": str(active.output), "trial": result["trial"],
                "decision": result["decision"], "measurement": result.get("measurement"),
                "explorable": explorable(result)}
        if step["explorable"]:
            revision_id = result["evaluation"]["revision_id"]
            if branch_entry:
                saved_promotion = state["inflight"].get("promotion")
                promotion = (json.loads((search.output/"version-trials"/saved_promotion/"result.json").read_text())
                             if saved_promotion else promote_version(search, active.output, revision_id))
                state["inflight"]["promotion"] = promotion["trial"]
                save()
                step["promotion"] = {"trial": promotion["trial"], "decision": promotion["decision"], "measurement": promotion["measurement"]}
            # Accepted main work is already on the normal sequential path.
            if branch_entry or not result["decision"]["accepted"]:
                destination = root/("branch-"+uuid4().hex[:10])
                fork_run(active.output, destination, revision_id=revision_id)
                score = result["evaluation"]["objective"]
                priority = score if search.request.objective["direction"] == "minimize" else -score
                state["frontier"].append({"output": str(destination), "revision": revision_id,
                    "priority": priority, "visits": 0, "source_trial": result["trial"]})
                state["frontier"] = sorted(state["frontier"], key=lambda row: (row["priority"], row["visits"]))[:frontier_width]
        state["steps"].append(step)
        state["inflight"] = None
        state["main_current"] = search.revisions.current().id
        save()
    state["status"] = "completed_experiment"
    save()
    return state


def resume_round(search, inflight):
    """Recover persisted work rather than silently spend another candidate round."""
    if inflight.get("trial"):
        return search.resume_trial(inflight["trial"])
    trials = [p for p in (search.output/"trials").glob("*")
              if p.name not in inflight["known_trials"] and (p/"generated.json").exists()]
    if len(trials) == 1:
        return search.resume_trial(trials[0].name)
    sandboxes = sorted((p for p in (search.output/"sandboxes").glob("*")
                        if p.name not in inflight["known_sandboxes"] and p.name.startswith(("candidate-", "selection-"))),
                       key=lambda p: p.stat().st_mtime)
    if not trials and sandboxes:
        return search.resume_generation(sandboxes[-1])
    raise ValueError("interrupted experiment has ambiguous or absent generation evidence; inspect the saved inflight record")
