"""Measured project trials over discovered sites; no model-name dispatch."""
from __future__ import annotations

import json
from pathlib import Path
import traceback
from uuid import uuid4

from kernel_optimizer.models.project import (
    CallSite, CandidateBundle, Dependency, ExecutionPlan, FileEdit, GroupRegistry, GroupSpec,
    ProjectContract, ProjectEvaluation,
)
from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.project.request import ProjectRequest
from kernel_optimizer.project.measurement import paired_decision, valid_run
from kernel_optimizer.project.transactions import RevisionStore, identity
from kernel_optimizer.runtime.execution import ExecutionRequest, LocalExecutor, write_json
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime


class ProjectSearch:
    def __init__(self, config, output: Path):
        self.config, self.output = config, output.resolve()
        self.request = ProjectRequest.model_validate_json((output/"request.json").read_bytes())
        self.store = RunStore.open(output)
        self.revisions = RevisionStore(self.store)
        prepared = json.loads((output/"prepared.json").read_text())
        if prepared["status"] != "prepared":
            raise ValueError("project evaluator has not passed its controls")

    def execute(self, revision, name, action):
        job_root = self.output/"jobs"/name
        source = self.revisions.materialize(revision, job_root/"source")
        bundles = []
        for ident in revision.bundles:
            bundle = self.revisions.bundle(ident)
            if bundle.installer is None:
                raise ValueError("candidate has no executable installer")
            bundles.append({"id": ident, "file": str(source/bundle.installer), "parameters": bundle.parameters,
                            "specs": json.loads((source/Path(bundle.installer).with_suffix(".targets.json")).read_text()),
                            "expected_calls": list(bundle.calls)})
        request = self.request
        return LocalExecutor().run(ExecutionRequest(entry=source/"opop_evaluate.py", cwd=source,
            devices=request.devices, distributed_backend=request.distributed_backend, output=job_root/"run",
            lease_dir=request.lease_dir, worker_timeout_s=request.worker_timeout_s,
            parameters={"request": request.portable(), "adapter": str(source/"opop_adapter.py"), "action": action,
                        "reference_dir": str(self.output/"controls"), "bundles": bundles}))

    def discover(self):
        parent = self.revisions.current()
        run = self.execute(parent, "discover-"+uuid4().hex[:10], "profile")
        if run["status"] != "passed":
            write_json(self.output/"discovery-failure.json", run)
            raise ValueError("project profiling failed")
        ranks = [json.loads(Path(rank["result"]["discovery"]).read_text()) for rank in run["ranks"]]
        primary = ranks[0]
        calls = tuple(CallSite(id=c["id"], phase=c["phase"], kind=c["kind"], target=c["target"],
                               signature={"signatures": c["signatures"], "attributes": c["attributes"]},
                               evidence=(primary["trace"],)) for c in primary["calls"])
        plan = ExecutionPlan(id=identity({"parent": parent.id, "trace": primary["trace_sha256"]}), calls=calls,
                             dependencies=tuple(Dependency(**edge) for edge in primary["dependencies"]),
                             unknown_dependencies=tuple(primary["unknown_dependencies"]))
        previous = self.revisions.load(parent.registry_ref, GroupRegistry)
        call_lookup = {c["id"]: c for c in primary["calls"]}
        groups = tuple(GroupSpec(id=g["id"], calls=tuple(g["calls"]),
                                semantics=call_lookup[g["calls"][0]]["semantics"],
                                resource_refs=(primary["trace"],),
                                lineage=tuple(old.id for old in previous.groups if set(old.calls)&set(g["calls"]))) for g in primary["groups"])
        registry = GroupRegistry(id=identity({"plan": plan.id, "groups": groups}), plan_id=plan.id, groups=groups,
            parent_id=previous.id, retired=tuple(g.id for g in previous.groups if g.id not in {item.id for item in groups}),
            invalidated_evidence=tuple(sorted({ref for g in previous.groups for ref in g.resource_refs})))
        revision = self.revisions.attach_discovery(parent, plan, registry)
        summary = []
        for group in groups:
            members = [call_lookup[ident] for ident in group.calls]
            summary.append({"id": group.id, "calls": list(group.calls), "semantics": group.semantics,
                            "kind": members[0]["kind"], "invocations": sum(c["count"] for c in members),
                            "diagnostic_cuda_us": sum(primary["call_resources"].get(c["id"], {}).get("cuda_us", 0) for c in members),
                            "targets": [c["target"] for c in members], "phases": sorted({c["phase"] for c in members})})
        result = {"revision": revision.id, "registry": registry.id, "plan": plan.id,
                  "groups": sorted(summary, key=lambda g: g["diagnostic_cuda_us"], reverse=True),
                  "calls": primary["calls"], "ranks": [r["trace"] for r in ranks],
                  "coverage": [r["coverage"] for r in ranks], "unknown_dependencies": primary["unknown_dependencies"],
                  "raw_discovery": [rank["result"]["discovery"] for rank in run["ranks"]]}
        write_json(self.output/"discovery.json", result)
        return result

    def trial(self, *, allowed_scopes=("group", "joint", "region", "execution_plan"), feedback=None):
        try:
            return self._trial(allowed_scopes=allowed_scopes, feedback=feedback)
        except Exception as exc:
            failure = {"status": "failed", "parent": self.revisions.current().id,
                       "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
            write_json(self.output/"latest-feedback.json", failure)
            self.store.append("PROJECT_TRIAL_FAILED", failure)
            raise

    def _trial(self, *, allowed_scopes, feedback):
        discovery = json.loads((self.output/"discovery.json").read_text())
        parent = self.revisions.current()
        if discovery["revision"] != parent.id:
            discovery = self.discover(); parent = self.revisions.current()
        contract = self.revisions.load(parent.contract_ref, ProjectContract)
        call_lookup = {c["id"]: c for c in discovery["calls"]}
        group_lookup = {g["id"]: g for g in discovery["groups"]}
        sources = {f.path: self.store.get_artifact(f.content_ref) for f in parent.files}
        common = {"request.json": json.dumps(self.request.portable(), indent=2),
                  "groups.json": json.dumps(discovery["groups"], indent=2), "calls.json": json.dumps(discovery["calls"], indent=2),
                  "search.json": json.dumps({"allowed_scopes": allowed_scopes, "unknown_dependencies": discovery["unknown_dependencies"],
                      "installed_bundles": [self.revisions.bundle(ident).model_dump(mode="json") for ident in parent.bundles]}),
                  "adapter.py": sources["opop_adapter.py"]}
        common.update({"source/"+name: data for name, data in sources.items()})
        prompt_root = Path(__file__).parent
        trial_id = "trial-"+uuid4().hex[:10]
        trial_root = self.output/"trials"/trial_id; trial_root.mkdir(parents=True)
        with Runtime(self.config, log_dir=trial_root) as runtime:
            agents = ProjectAgents(runtime, self.store)
            selected, selection_receipt = agents.generate("selection", common,
                (prompt_root/"selection_prompt.md").read_text(), ("selection.json",), feedback)
            selection = json.loads((selected/"selection.json").read_text())
            if selection["scope"] not in allowed_scopes or not selection["groups"] or set(selection["groups"]) - group_lookup.keys():
                raise ValueError("selection refers to unsupported scope or unknown groups")
            required = set().union(*(set(group_lookup[g]["calls"]) for g in selection["groups"]))
            affected = set(selection["affected_calls"])
            if not required <= affected or affected - call_lookup.keys():
                raise ValueError("selection omitted group calls or refers to unknown calls")
            # A parent replacement can eliminate child sites. Include the observed
            # containment closure, without claiming unknown alias/state edges are safe.
            while True:
                closure = affected | {c["id"] for c in discovery["calls"] if set(c["parents"]) & affected}
                if closure == affected:
                    break
                affected = closure
            selection["affected_calls"] = sorted(affected)
            targets = [call_lookup[c] for c in sorted(affected)]
            candidate_inputs = {**common, "selection.json": json.dumps(selection, indent=2),
                                "targets.json": json.dumps(targets, indent=2)}
            generated, candidate_receipt = agents.generate("candidate", candidate_inputs,
                (prompt_root/"candidate_prompt.md").read_text(), ("candidate.py", "proposal.json"), feedback)
        proposal = json.loads((generated/"proposal.json").read_text())
        replacement_calls = tuple(proposal["replacement_calls"])
        if not replacement_calls or set(replacement_calls) - affected:
            raise ValueError("candidate returned undeclared replacement sites")
        code = (generated/"candidate.py").read_bytes()
        compile(code, "candidate.py", "exec")
        installer = f"opop_candidates/{trial_id}.py"
        target_file = str(Path(installer).with_suffix(".targets.json")).replace("\\", "/")
        edits = (FileEdit(path=installer, before_sha256=None, content_ref=self.store.put_artifact(code)),
                 FileEdit(path=target_file, before_sha256=None, content_ref=self.store.put_artifact(json.dumps(targets))))
        bundle = CandidateBundle(id=trial_id, parent_revision=parent.id, contract_ref=parent.contract_ref,
            scope=selection["scope"], implementation=proposal["implementation"], groups=tuple(selection["groups"]),
            calls=replacement_calls, affected_calls=tuple(sorted(affected)), edits=edits,
            read_set=tuple("call:"+c for c in sorted(affected)),
            write_set=("file:"+installer, "file:"+target_file, *("call:"+c for c in replacement_calls)),
            requires=tuple(proposal.get("requires", ())), replaces=tuple(proposal.get("replaces", ())),
            external_invariants=contract.invariants, installer=installer)
        candidate = self.revisions.stage(bundle)
        pairs = []
        pair_count = int(self.request.objective.get("paired_repeats", 2))
        if pair_count < 2:
            raise ValueError("project acceptance requires at least AB and BA pairs")
        for index in range(pair_count):
            pair = {}
            order = (("parent", parent), ("candidate", candidate))
            for role, revision in order if index % 2 == 0 else reversed(order):
                pair[role] = self.execute(revision, f"{trial_id}-pair-{index}-{role}", "evaluate")
                if not valid_run(pair[role]):
                    break
            pairs.append(pair)
            if len(pair) != 2 or not all(valid_run(run) for run in pair.values()):
                break
        baseline = pairs[0].get("parent", {"status": "not_run", "ranks": []})
        measured = pairs[0].get("candidate", {"status": "not_run", "ranks": []})
        valid = valid_run(measured)
        complete = len(pairs) == pair_count and all(len(pair) == 2 for pair in pairs)
        anchor = json.loads((self.output/"prepared.json").read_text())["evaluation"]
        measurement = paired_decision(pairs, self.request.objective, anchor) if complete else {
            "valid": False, "objective": None, "reference": None, "reasons": ["paired_endpoint_failed"]}
        executed = set.intersection(*(set(r["result"].get("coverage", {}).get(bundle.id, {}).get("counts", {}))
                                     for r in measured["ranks"])) if valid else set()
        report = ProjectEvaluation(id=trial_id, revision_id=candidate.id, parent_id=parent.id,
            contract_ref=parent.contract_ref, bundle_id=bundle.id, quality="pass" if valid else "fail",
            feasible=valid and measurement["valid"], objective=measurement["objective"], reference_objective=measurement["reference"],
            metrics=measurement, reasons=tuple(measurement["reasons"]),
            required_ranks=tuple(range(contract.world_size)), rank_bundle_hashes={str(r["rank"]):
                r.get("result", {}).get("bundle_hashes", {}).get(bundle.id, "missing") for r in measured["ranks"]},
            required_calls=bundle.calls, executed_calls=tuple(sorted(executed)),
            evidence=tuple(str(self.output/"jobs"/f"{trial_id}-pair-{i}-{role}"/"run"/"receipt.json")
                           for i, pair in enumerate(pairs) for role in pair))
        decision = self.revisions.evaluate(report)
        resource_feedback = self.resource_feedback(candidate, bundle, discovery) if valid else {
            "status": "unavailable", "reason": "candidate failed quality or actual-call coverage"}
        if decision["accepted"]:
            self.revisions.commit(candidate.id)
        else:
            self.revisions.abort(candidate.id, "; ".join(decision["reasons"]))
        result = {"trial": trial_id, "selection": selection, "proposal": proposal, "bundle": bundle.model_dump(mode="json"),
                  "selection_agent": selection_receipt, "candidate_agent": candidate_receipt,
                  "baseline": baseline, "candidate": measured, "pairs": pairs, "measurement": measurement,
                  "evaluation": report.model_dump(mode="json"),
                  "resources": resource_feedback, "decision": decision}
        write_json(trial_root/"result.json", result)
        write_json(self.output/"latest-feedback.json", result)
        return result

    def resource_feedback(self, candidate, bundle, discovery):
        run = self.execute(candidate, bundle.id+"-profile", "profile")
        if run["status"] != "passed":
            return {"status": "unavailable", "reason": "diagnostic execution failed", "run": run}
        before = [json.loads(Path(path).read_text()) for path in discovery["raw_discovery"]]
        after = [json.loads(Path(rank["result"]["discovery"]).read_text()) for rank in run["ranks"]]
        groups = []
        for group in discovery["groups"]:
            if set(group["calls"]) & set(bundle.affected_calls or bundle.calls):
                groups.append({"group": group["id"], "bundle": bundle.id, "attribution_status": "joint_only",
                    "calls": [{"id": call, "ranks": [{"rank": a["rank"],
                        "before": b["call_resources"].get(call), "after": a["call_resources"].get(call),
                        "after_status": "observed" if call in a["call_resources"] else "absent_or_unassociated"}
                        for b, a in zip(before, after, strict=True)]} for call in group["calls"]]})
        return {"status": "observed", "groups": groups, "coverage": [a["coverage"] for a in after],
                "evidence": [a["trace"] for a in after], "raw_discovery": [r["result"]["discovery"] for r in run["ranks"]],
                "interpretation": "Nested diagnostic durations overlap; project gain belongs to the complete bundle. Unmeasured independent group effects are unknown."}
