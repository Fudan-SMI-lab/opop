"""Continue an interrupted project role using its actual persisted session."""
import json
from pathlib import Path
from uuid import uuid4

from kernel_optimizer.agents.sandbox import Sandbox
from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.project.experiment_policy import check_selection, eligible_groups
from kernel_optimizer.project.generated import validate_proposal
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.wiring import Runtime


def committed_trial(search, trial_id):
    """Reconcile a crash after pointer publication but before the trial summary."""
    from kernel_optimizer.models.project import ProjectEvaluation
    cursor = search.revisions.current()
    while True:
        path = search.revisions.root / "transactions" / (cursor.id + ".json")
        state = json.loads(path.read_text()) if path.exists() else {}
        if state.get("accepted") and state.get("report_ref") and state.get("kind") != "complete_version":
            report = search.revisions.load(state["report_ref"], ProjectEvaluation)
            if report.id == trial_id:
                bundle = search.revisions.bundle(report.bundle_id)
                search.store.append("PROJECT_TRIAL_SUMMARY_RECOVERED", {"trial": trial_id, "revision": cursor.id})
                return {"trial": trial_id, "bundle": bundle.model_dump(mode="json"),
                    "publication": "committed", "decision": {**state, "state": "committed"},
                    "evaluation": report.model_dump(mode="json"), "measurement": report.metrics,
                    "recovery": "accepted ancestor and immutable evaluation reconciled; no GPU/API work repeated",
                    "resources": {"status": "unavailable", "reason": "interruption before trial summary; inspect retained profile jobs"}}
        if cursor.parent_id is None:
            return None
        cursor = search.revisions.revision(cursor.parent_id)


def resume_generation(search, directory):
    directory = Path(directory).resolve()
    if directory.parent != search.output / "sandboxes":
        raise ValueError("generation recovery requires a sandbox from this run")
    events = search.store.iter_events()
    starts = [e for e in events if e.type == "PROJECT_AGENT_STARTED" and e.payload["sandbox"] == str(directory)]
    if not starts:
        raise ValueError("no recorded provider session for this sandbox")
    event = starts[-1]; role, session = event.payload["role"], event.payload["session"]
    if role not in {"selection", "candidate"}:
        raise ValueError("this recovery entry supports selection/candidate roles")
    inputs = directory / "inputs"
    project = json.loads((inputs / "project.json").read_text())
    parent = search.revisions.current()
    if project["current_revision"] != parent.id:
        raise ValueError("interrupted generation belongs to another parent; explicit rebase/reselection required")
    discovery = json.loads((search.output / "discovery.json").read_text())
    if discovery["revision"] != parent.id:
        raise ValueError("interrupted generation discovery is no longer current")
    policy = json.loads((inputs / "search.json").read_text())
    trial = "trial-recovered-" + uuid4().hex[:10]
    root = search.output / "trials" / trial; root.mkdir(parents=True)
    prompt = ("Continue the original task from this session and the supplied input files. The previous provider request "
              "ended with a temporary service error. Preserve completed work and frozen semantics. Finish the requested "
              "output files using their JSON schemas. Do not launch ad hoc GPU tests; the framework will validate them.")
    selection_receipt = None
    with Runtime(search.config, log_dir=root) as runtime:
        agents = ProjectAgents(runtime, search.store)
        if role == "selection":
            eligible = eligible_groups(discovery, leaf_only=policy.get("leaf_groups_only", False))
            selected, selection_receipt = agents._complete(role, Sandbox(directory), session, prompt,
                ("selection.json",), lambda p: check_selection(p, discovery, policy["allowed_scopes"], eligible), True)
            selection = json.loads((selected / "selection.json").read_text())
            affected = set(selection["affected_calls"])
            while True:
                closure = affected | {c["id"] for c in discovery["calls"] if set(c["parents"]) & affected}
                if closure == affected:
                    break
                affected = closure
            selection["affected_calls"] = sorted(affected)
            common = {p.relative_to(inputs).as_posix(): p.read_bytes() for p in inputs.rglob("*") if p.is_file()}
            previous = common.pop("previous-feedback.txt", None)
            common.update({"selection.json": json.dumps(selection, indent=2),
                "targets.json": json.dumps([c for c in discovery["calls"] if c["id"] in affected], indent=2),
                "target-resources.json": json.dumps({c: discovery.get("call_resources", {}).get(c) for c in affected}, indent=2)})
            generated, candidate_receipt = agents.generate("candidate", common,
                Path(__file__).with_name("candidate_prompt.md").read_text(), ("candidate.py", "proposal.json"),
                previous.decode() if previous else None, validator=validate_proposal)
        else:
            selection = json.loads((inputs / "selection.json").read_text())
            prior = []
            for e in events:
                if e.type != "PROJECT_AGENT_FINISHED" or e.payload["role"] != "selection" or e.seq >= starts[0].seq:
                    continue
                artifact = Path(e.payload["sandbox"]) / "selection.json"
                if artifact.exists():
                    value = json.loads(artifact.read_text())
                    if all(value.get(k) == selection.get(k) for k in ("scope", "groups", "rationale", "hypothesis")):
                        prior.append(e.payload)
            selection_receipt = prior[-1] if prior else {"status": "retained selection artifact; receipt unavailable"}
            generated, candidate_receipt = agents._complete(role, Sandbox(directory), session, prompt,
                ("candidate.py", "proposal.json"), validate_proposal, True)
    bundle, proposal = search.bundle_from_outputs(generated, selection, parent, discovery, trial)
    metadata = {"trial": trial, "selection": selection, "proposal": proposal,
        "selection_agent": selection_receipt, "candidate_agent": candidate_receipt,
        "experiment_policy": {k: policy.get(k) for k in ("allowed_scopes", "leaf_groups_only", "selection_strategy", "feedback_view")},
        "generation_recovery": {"sandbox": str(directory), "session": session, "role": role, "same_parent": parent.id}}
    write_json(root / "generated.json", {**metadata, "bundle": bundle.model_dump(mode="json")})
    return search._evaluate_generated(bundle, metadata, discovery)
