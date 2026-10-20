"""Measured project trials over discovered sites; no model-name dispatch."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import traceback
from uuid import uuid4

from kernel_optimizer.models.project import (
    CallSite, CandidateBundle, Dependency, ExecutionPlan, FileEdit, GroupRegistry, GroupSpec,
    ProjectContract, ProjectEvaluation,
)
from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.project.generated import CollectionPlan, Proposal, Selection, validate_proposal
from kernel_optimizer.project.request import ProjectRequest
from kernel_optimizer.project.measurement import paired_decision, valid_run
from kernel_optimizer.project.transactions import RevisionStore, identity
from kernel_optimizer.runtime.execution import ExecutionRequest, LocalExecutor, write_json
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime


class ProjectSearch:
    def __init__(self, config, output: Path):
        from kernel_optimizer.project.runtime_config import project_agent_config
        self.config, self.output = project_agent_config(config), output.resolve()
        self.request = ProjectRequest.model_validate_json((output/"request.json").read_bytes())
        self.store = RunStore.open(output)
        self.revisions = RevisionStore(self.store)
        prepared = self.prepared = json.loads((output/"prepared.json").read_text())
        if prepared["status"] != "prepared":
            raise ValueError("project evaluator has not passed its controls")
        if prepared.get("request_sha256") and prepared["request_sha256"] != hashlib.sha256((output/"request.json").read_bytes()).hexdigest():
            raise ValueError("frozen project request changed after evaluator validation")

    def execute(self, revision, name, action, *, collector_options=None):
        job_root = self.output/"jobs"/name
        source = self.revisions.materialize(revision, job_root/"source")
        bundles = []
        from kernel_optimizer.project.aggregation import rank_requirements
        contract = self.revisions.load(revision.contract_ref, ProjectContract)
        for ident in revision.bundles:
            bundle = self.revisions.bundle(ident)
            if bundle.installer is None:
                raise ValueError("candidate has no executable installer")
            bundles.append({"id": ident, "file": str(source/bundle.installer), "parameters": bundle.parameters,
                            "specs": json.loads((source/Path(bundle.installer).with_suffix(".targets.json")).read_text()),
                            "expected_calls": list(bundle.calls), "expected_calls_by_rank": rank_requirements(
                                bundle, self.revisions.load(self.revisions.revision(bundle.parent_revision).plan_ref, ExecutionPlan), contract.world_size)})
        request = self.request
        env = {}
        launcher = []
        if action == "nsys_profile":
            from kernel_optimizer.project.nsys import command_prefix
            import shutil
            tool = request.tools.get("nsys") or shutil.which("nsys")
            if tool is None:
                return {"status": "unavailable", "reason": "Nsys executable unavailable", "ranks": []}
            options = collector_options or {}
            launcher = command_prefix(tool, counters=options.get("gpu_counters", request.profiling.get("gpu_counters", False)))
        if action in {"profile", "nsys_profile"} and request.communication != "off" and len(request.devices) > 1:
            env = {"NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,GRAPH,COLL",
                   "NCCL_DEBUG_FILE": str(job_root/"run"/"nccl.%h.%p.log")}
        return LocalExecutor().run(ExecutionRequest(entry=source/"opop_evaluate.py", cwd=source,
            devices=request.devices, distributed_backend=request.distributed_backend, output=job_root/"run",
            lease_dir=request.lease_dir, worker_timeout_s=request.worker_timeout_s,
            env=env, launcher=launcher, wait_for_devices=True,
            parameters={"request": request.portable(), "adapter": str(source/"opop_adapter.py"), "action": action,
                        "collector_options": collector_options or {},
                        "reference_dir": self.prepared.get("controls_path", str(self.output/"controls")), "bundles": bundles}))

    def discover(self):
        parent = self.revisions.current()
        run = self.execute(parent, "discover-"+uuid4().hex[:10], "profile")
        if run["status"] != "passed":
            write_json(self.output/"discovery-failure.json", run)
            raise ValueError("project profiling failed")
        ranks = [json.loads(Path(rank["result"]["discovery"]).read_text()) for rank in run["ranks"]]
        from kernel_optimizer.project.aggregation import merge_ranks
        primary = merge_ranks(ranks)
        from kernel_optimizer.project.communication import match_ranks
        communication = match_ranks([r["communication"] for r in ranks if "collectives" in r.get("communication", {})])
        if "cross_rank_collectives" in self.request.required_observations and (
                not communication["matched"] or communication["incomplete"]):
            raise ValueError("required cross-rank collective matching is incomplete")
        calls = tuple(CallSite(id=c["id"], phase=c["phase"], kind=c["kind"], target=c["target"],
                               signature={"signatures": c["signatures"], "attributes": c["attributes"], "ranks": c["ranks"]},
                               evidence=tuple(primary["traces"])) for c in primary["calls"])
        plan = ExecutionPlan(id=identity({"parent": parent.id, "trace": primary["trace_sha256"]}), calls=calls,
                             dependencies=tuple(Dependency(**edge) for edge in primary["dependencies"]),
                             unknown_dependencies=tuple(primary["unknown_dependencies"]))
        previous = self.revisions.load(parent.registry_ref, GroupRegistry)
        previous_plan = self.revisions.load(parent.plan_ref, ExecutionPlan)
        call_lookup = {c["id"]: c for c in primary["calls"]}
        from kernel_optimizer.project.aggregation import lineage_resolver
        lineage = lineage_resolver(previous, call_lookup,
            [self.revisions.bundle(ident) for ident in parent.bundles])
        groups = tuple(GroupSpec(id=g["id"], calls=tuple(g["calls"]),
                                semantics=call_lookup[g["calls"][0]]["semantics"],
                                resource_refs=tuple(primary["traces"]),
                                lineage=lineage(g["calls"])) for g in primary["groups"])
        group_ids = {item.id for item in groups}
        registry = GroupRegistry(id=identity({"plan": plan.id, "groups": groups}), plan_id=plan.id, groups=groups,
            parent_id=previous.id, retired=tuple(g.id for g in previous.groups if g.id not in group_ids),
            invalidated_evidence=tuple(sorted({ref for g in previous.groups for ref in g.resource_refs} |
                                              {ref for c in previous_plan.calls for ref in c.evidence})))
        revision = self.revisions.attach_discovery(parent, plan, registry)
        summary = []
        for group in groups:
            members = [call_lookup[ident] for ident in group.calls]
            summary.append({"id": group.id, "calls": list(group.calls), "semantics": group.semantics,
                            "kind": members[0]["kind"], "invocations": sum(c["count"] for c in members),
                            "diagnostic_cuda_us": sum(max((value.get("cuda_us", 0) for value in primary["call_resources_by_rank"][c["id"]].values()), default=0) for c in members),
                            "ranks": sorted({rank for c in members for rank in c["ranks"]}),
                            "targets": [c["target"] for c in members], "phases": sorted({c["phase"] for c in members})})
        result = {"revision": revision.id, "registry": registry.id, "plan": plan.id,
                  "groups": sorted(summary, key=lambda g: g["diagnostic_cuda_us"], reverse=True),
                  "calls": primary["calls"], "ranks": [r["trace"] for r in ranks],
                  "call_resources": primary["call_resources_by_rank"],
                  "coverage": [r["coverage"] for r in ranks], "unknown_dependencies": primary["unknown_dependencies"],
                  "profiled_wall_ms": [r.get("profiled_wall_ms") for r in ranks],
                  "communication": {"cross_rank": communication, "ranks": [r.get("communication") for r in ranks],
                                    "capabilities": [r.get("capabilities") for r in ranks]},
                  "plan_change": {"retired_calls": sorted({c.id for c in previous_plan.calls} - call_lookup.keys()),
                      "new_calls": sorted(call_lookup.keys() - {c.id for c in previous_plan.calls}),
                      "retired_groups": list(registry.retired), "invalidated_evidence": list(registry.invalidated_evidence),
                      "lineage_basis": "Surviving call identity or containment inside an installed joint bundle; not independent causal attribution"},
                  "raw_discovery": [rank["result"]["discovery"] for rank in run["ranks"]]}
        write_json(self.output/"discovery.json", result)
        if (self.request.communication != "off" and len(self.request.devices) > 1) or self.request.profiling.get("agent_plan", False):
            result["collection_plan"] = self.plan_collectors(result)
            result["extra_collectors"] = self.collect_extra(revision, result["collection_plan"]["plan"]["collectors"])
            write_json(self.output/"discovery.json", result)
        return result

    def plan_collectors(self, discovery):
        from kernel_optimizer.project.collectors import diagnostic_names
        names = diagnostic_names(self.request.profiling)
        inputs = {"request.json": json.dumps(self.request.portable(), indent=2),
                  "available-collectors.json": json.dumps(names),
                  "observations.json": json.dumps(discovery["communication"], indent=2),
                  "coverage.json": json.dumps(discovery["coverage"], indent=2),
                  "collection.schema.json": json.dumps(CollectionPlan.model_json_schema(), indent=2)}
        prompt = ("Choose useful diagnostic collectors for the frozen task objective from actual capability and trace evidence. "
            "Write collection.json following collection.schema.json and use names from available-collectors.json. The torch collector already ran. Nsys is optional and runs separately; "
            "select it when available and useful to resolve overlap, graph tracing or attribution questions. "
            "device_telemetry samples available NVML PCIe rates, utilization and memory separately; these are device-wide readings with native windows, not per-collective bytes. "
            "collective_calibration runs a separate synthetic full-world SUM allreduce at actually observed sizes/dtypes, with correctness checks; it is a reference experiment, not a performance lower bound or model critical-path estimate. "
            "execution_timeline records a separate plain Torch trace, CPU-to-CUDA submit associations, synchronization, per-device intervals, and optional passive CPU gradient post-accumulate callbacks. It reports clock alignment bounds for matching CPU submissions only; GPU readiness and consumer completion may remain unknown. It never replaces a communication hook or determines subsequent search actions. "
            "Never invent PCIe/NVLink traffic from logical payload or memcpy sizes. Nsys timelines alone do not separate NCCL waiting from wire transfer or prove bucket identity. "
            "Exact bucket links require native correlation and observed gradient-storage membership; otherwise retain ambiguous candidates. List unavailable dimensions with observed reasons. "
            "Do not request every counter indiscriminately, change the task, or run tools yourself. "
            "The framework verifies availability and retains missing fields as unknown.")
        directory = self.output/"collector-plans"/uuid4().hex[:10]; directory.mkdir(parents=True)
        def validate(directory):
            plan = CollectionPlan.model_validate_json((directory/"collection.json").read_bytes())
            if set(plan.collectors)-set(names):
                raise ValueError("collector plan refers to an unregistered collector")
        with Runtime(self.config, log_dir=directory) as runtime:
            generated, receipt = ProjectAgents(runtime, self.store).generate("collection-planner", inputs, prompt,
                ("collection.json",), validator=validate)
        plan = CollectionPlan.model_validate_json((generated/"collection.json").read_bytes()).model_dump(mode="json")
        result = {"plan": plan, "agent": receipt}
        write_json(directory/"result.json", result)
        return result

    def collect_extra(self, revision, collectors):
        from kernel_optimizer.project.collectors import collect
        return collect(self, revision, collectors)

    def current_discovery(self):
        """A feasible branch can begin at a version that has never been profiled."""
        path = self.output / "discovery.json"
        discovery = json.loads(path.read_text()) if path.exists() else None
        if discovery is None or discovery["revision"] != self.revisions.current().id:
            return self.discover()
        return discovery

    def trial(self, *, allowed_scopes=("group", "joint", "region", "execution_plan"), feedback=None,
              leaf_groups_only=False, selection_strategy="agent", feedback_view="full"):
        try:
            return self._trial(allowed_scopes=allowed_scopes, feedback=feedback,
                leaf_groups_only=leaf_groups_only, selection_strategy=selection_strategy, feedback_view=feedback_view)
        except Exception as exc:
            failure = {"status": "failed", "parent": self.revisions.current().id,
                       "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
            write_json(self.output/"latest-feedback.json", failure)
            self.store.append("PROJECT_TRIAL_FAILED", failure)
            raise

    def _trial(self, *, allowed_scopes, feedback, leaf_groups_only=False, selection_strategy="agent", feedback_view="full"):
        discovery = self.current_discovery()
        parent = self.revisions.current()
        contract = self.revisions.load(parent.contract_ref, ProjectContract)
        call_lookup = {c["id"]: c for c in discovery["calls"]}
        group_lookup = {g["id"]: g for g in discovery["groups"]}
        from kernel_optimizer.project.experiment_policy import eligible_groups, check_selection, local_time_selection
        from kernel_optimizer.project.experiment_policy import feedback_view as view_feedback
        leaf_groups_only = leaf_groups_only or selection_strategy == "local_time"
        eligible = eligible_groups(discovery, leaf_only=leaf_groups_only)
        if not eligible:
            raise ValueError("no eligible groups for this experiment policy")
        if selection_strategy not in {"agent", "local_time"} or feedback_view not in {"full", "project"}:
            raise ValueError("unknown experiment selection or feedback policy")
        if selection_strategy == "local_time" and "group" not in allowed_scopes:
            raise ValueError("local_time comparison requires group scope")
        policy = {"allowed_scopes": allowed_scopes, "leaf_groups_only": leaf_groups_only,
                  "selection_strategy": selection_strategy, "feedback_view": feedback_view}
        sources = {f.path: self.store.get_artifact(f.content_ref) for f in parent.files}
        common = {"request.json": json.dumps(self.request.portable(), indent=2),
                  "project.json": json.dumps({"current_revision": parent.id, "contract": contract.model_dump(mode="json"),
                      "original_reference_revision": self.prepared["revision"],
                      "original_baseline": [{"rank": r["rank"], "metrics": r["result"]["metrics"],
                                             "diagnostics": r["result"].get("diagnostics")} for r in self.prepared["evaluation"]["ranks"]],
                      "profiled_wall_ms": discovery.get("profiled_wall_ms"),
                      "measurement_policy": "Original baseline belongs to its immutable reference. Discovery uses intrusive hooks and is separate from clean timing and plain diagnostic objective samples. Profiled rank skew and kernel time can be heavily perturbed; they are not project J or wire time."}, indent=2),
                  "selection.schema.json": json.dumps(Selection.model_json_schema(), indent=2),
                  "proposal.schema.json": json.dumps(Proposal.model_json_schema(), indent=2),
                  "resources.json": json.dumps({"communication": discovery.get("communication"), "coverage": discovery["coverage"],
                      "collection_plan": discovery.get("collection_plan"), "extra_collectors": discovery.get("extra_collectors")}, indent=2),
                  "groups.json": json.dumps(discovery["groups"], indent=2), "calls.json": json.dumps(discovery["calls"], indent=2),
                  "call-resources.json": json.dumps(discovery.get("call_resources", {}), indent=2),
                  "search.json": json.dumps({**policy, "eligible_groups": [g["id"] for g in eligible], "unknown_dependencies": discovery["unknown_dependencies"],
                      "installed_bundles": [self.revisions.bundle(ident).model_dump(mode="json") for ident in parent.bundles]}),
                  "adapter.py": sources["opop_adapter.py"]}
        from kernel_optimizer.control.direct_task import TaskSpace
        common["parameter-space.schema.json"] = json.dumps(TaskSpace.model_json_schema(), indent=2)
        from kernel_optimizer.project.aggregation import index
        common["index.json"] = json.dumps(index(discovery), indent=2)
        common.update({"source/"+name: data for name, data in sources.items()})
        from kernel_optimizer.project.feedback import candidate_artifacts
        common.update(candidate_artifacts(feedback, self.store))
        prompt_root = Path(__file__).parent
        trial_id = "trial-"+uuid4().hex[:10]
        trial_root = self.output/"trials"/trial_id; trial_root.mkdir(parents=True)
        with Runtime(self.config, log_dir=trial_root) as runtime:
            agents = ProjectAgents(runtime, self.store)
            from kernel_optimizer.project.feedback import route
            registry = self.revisions.load(parent.registry_ref, GroupRegistry)
            plan = self.revisions.load(parent.plan_ref, ExecutionPlan)
            if selection_strategy == "agent":
                selected, selection_receipt = agents.generate("selection", common,
                    (prompt_root/"selection_prompt.md").read_text(), ("selection.json",),
                    view_feedback(route(feedback, current_revision=parent.id, registry=registry), feedback_view),
                    validator=lambda path: check_selection(path, discovery, allowed_scopes, eligible))
                selection = json.loads((selected/"selection.json").read_text())
            else:
                selection = local_time_selection(eligible)
                selection_receipt = {"role": "selection", "method": "deterministic local-time experiment control", "new_generation": False}
                write_json(trial_root/"selection.json", selection)
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
                                "targets.json": json.dumps(targets, indent=2),
                                "target-resources.json": json.dumps({ident: discovery.get("call_resources", {}).get(ident)
                                                                     for ident in affected}, indent=2)}
            generated, candidate_receipt = agents.generate("candidate", candidate_inputs,
                (prompt_root/"candidate_prompt.md").read_text(), ("candidate.py", "proposal.json"),
                view_feedback(route(feedback, current_revision=parent.id, registry=registry, selected_groups=selection["groups"], plan=plan), feedback_view),
                validator=validate_proposal)
        bundle, proposal = self.bundle_from_outputs(generated, selection, parent, discovery, trial_id)
        metadata = {"trial": trial_id, "selection": selection, "proposal": proposal,
                    "selection_agent": selection_receipt, "candidate_agent": candidate_receipt, "experiment_policy": policy}
        write_json(trial_root/"generated.json", {**metadata, "bundle": bundle.model_dump(mode="json")})
        return self._evaluate_generated(bundle, metadata, discovery)

    def resume_trial(self, trial_id):
        root = self.output/"trials"/trial_id
        if (root/"result.json").exists():
            return json.loads((root/"result.json").read_text())
        from kernel_optimizer.project.recovery import committed_trial
        recovered = committed_trial(self, trial_id)
        if recovered is not None:
            write_json(root/"result.json", recovered)
            return recovered
        saved = json.loads((root/"generated.json").read_text())
        bundle = CandidateBundle.model_validate_json(json.dumps(saved.pop("bundle")))
        if bundle.parent_revision != self.revisions.current().id:
            raise ValueError("saved trial parent changed; explicit rebase and re-evaluation are required")
        discovery = json.loads((self.output/"discovery.json").read_text())
        self.store.append("PROJECT_TRIAL_RESUMED", {"trial": trial_id, "parent": bundle.parent_revision,
                                                   "generation_reused": True})
        return self._evaluate_generated(bundle, saved, discovery)

    def resume_generation(self, directory):
        from kernel_optimizer.project.recovery import resume_generation
        return resume_generation(self, directory)

    def _evaluate_generated(self, bundle, metadata, discovery):
        trial_id = metadata["trial"]
        trial_root = self.output/"trials"/trial_id
        previous_search = trial_root/"parameter-search.json"
        if previous_search.exists():
            archive = trial_root/"prior-attempts"/uuid4().hex
            archive.mkdir(parents=True)
            (archive/previous_search.name).write_bytes(previous_search.read_bytes())
        proposal = metadata["proposal"]
        code = next(edit for edit in bundle.edits if edit.path == bundle.installer)
        candidate_path = trial_root/"submitted-candidate.py"
        candidate_path.write_bytes(self.store.get_artifact(code.content_ref))
        from kernel_optimizer.project.inner_loop import tune
        selected, tuning = tune(self, bundle, candidate_path, proposal["parameter_space"],
                                int(self.request.objective.get("parameter_trials", 3)))
        write_json(trial_root/"parameter-search.json", tuning)
        metadata["parameter_search"] = tuning
        if selected is None:
            result = {**metadata, "bundle": bundle.model_dump(mode="json"), "decision": {
                "accepted": False, "reasons": ["no_valid_parameter_candidate"]}}
            write_json(trial_root/"result.json", result)
            write_json(self.output/"latest-feedback.json", result)
            return result
        return self.validate_bundle(selected["bundle"], metadata, discovery=discovery, candidate=selected["revision"])

    def bundle_from_outputs(self, generated, selection, parent, discovery, trial_id):
        """Resolve saved agent outputs against their exact parent for replay/repair."""
        if discovery["revision"] != parent.id:
            raise ValueError("generated outputs belong to another discovery revision")
        contract = self.revisions.load(parent.contract_ref, ProjectContract)
        call_lookup = {c["id"]: c for c in discovery["calls"]}
        group_lookup = {g["id"]: g for g in discovery["groups"]}
        if set(selection["groups"]) - group_lookup.keys():
            raise ValueError("unknown selected groups")
        affected = set(selection["affected_calls"])
        required = set().union(*(set(group_lookup[g]["calls"]) for g in selection["groups"]))
        if not required <= affected or affected - call_lookup.keys():
            raise ValueError("selected groups are not covered by the effect closure")
        while True:
            closure = affected | {c["id"] for c in discovery["calls"] if set(c["parents"]) & affected}
            if closure == affected:
                break
            affected = closure
        targets = [call_lookup[c] for c in sorted(affected)]
        proposal = Proposal.model_validate_json((generated/"proposal.json").read_bytes()).model_dump(mode="json")
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
        return bundle, proposal

    def validate_bundle(self, bundle, metadata=None, *, discovery=None, candidate=None, publish=True):
        """Replay an immutable generated candidate without another provider request."""
        metadata = metadata or {"trial": bundle.id}
        trial_id = metadata["trial"]
        trial_root = self.output/"trials"/trial_id
        trial_root.mkdir(parents=True, exist_ok=True)
        if (trial_root/"result.json").exists():
            raise ValueError("completed trial is immutable; use a new trial identity for an independent repeat")
        job_tag = trial_id+"-evaluation-"+uuid4().hex[:8]
        parent = self.revisions.revision(bundle.parent_revision)
        contract = self.revisions.load(parent.contract_ref, ProjectContract)
        discovery = discovery or json.loads((self.output/"discovery.json").read_text())
        if discovery["revision"] != parent.id:
            raise ValueError("candidate validation requires its own parent's discovery evidence")
        candidate = candidate or self.revisions.stage(bundle)
        pairs = []
        pair_count = int(self.request.objective.get("paired_repeats", 2))
        if pair_count < 2:
            raise ValueError("project acceptance requires at least AB and BA pairs")
        for index in range(pair_count):
            pair = {}
            order = (("parent", parent), ("candidate", candidate))
            for role, revision in order if index % 2 == 0 else reversed(order):
                pair[role] = self.execute(revision, f"{job_tag}-pair-{index}-{role}", "evaluate")
                if not valid_run(pair[role]):
                    break
            pairs.append(pair)
            if len(pair) != 2 or not all(valid_run(run) for run in pair.values()):
                break
        baseline = pairs[0].get("parent", {"status": "not_run", "ranks": []})
        measured = pairs[0].get("candidate", {"status": "not_run", "ranks": []})
        valid = valid_run(measured) and all(valid_run(pair.get("candidate", {"status": "not_run"})) for pair in pairs)
        complete = len(pairs) == pair_count and all(len(pair) == 2 for pair in pairs)
        anchor = json.loads((self.output/"prepared.json").read_text())["evaluation"]
        measurement = paired_decision(pairs, self.request.objective, anchor) if complete else {
            "valid": False, "objective": None, "reference": None, "reasons": ["paired_endpoint_failed"]}
        from kernel_optimizer.project.quality_report import quality_and_coverage
        quality, missing_coverage = quality_and_coverage([p["candidate"] for p in pairs if "candidate" in p])
        rank_executed = {str(r["rank"]): tuple(c for c, n in r.get("result", {}).get("coverage", {}).get(bundle.id, {}).get("counts", {}).items() if n > 0)
                        for r in measured["ranks"]}
        executed = set().union(*map(set, rank_executed.values()))
        report = ProjectEvaluation(id=trial_id, revision_id=candidate.id, parent_id=parent.id,
            contract_ref=parent.contract_ref, bundle_id=bundle.id, quality=quality,
            feasible=valid and measurement["valid"] and all(c["passed"] for c in measurement.get("constraints", [])),
            objective=measurement["objective"], reference_objective=measurement["reference"],
            metrics=measurement, reasons=tuple(measurement["reasons"]) + tuple(sorted({
                "candidate_coverage_missing" if row["bundle"] == bundle.id else "inherited_candidate_coverage_missing" for row in missing_coverage})),
            required_ranks=tuple(range(contract.world_size)), rank_bundle_hashes={str(r["rank"]):
                r.get("result", {}).get("bundle_hashes", {}).get(bundle.id, "missing") for r in measured["ranks"]},
            required_calls=bundle.calls, executed_calls=tuple(sorted(executed)),
            rank_executed_calls=rank_executed,
            evidence=tuple(str(self.output/"jobs"/f"{job_tag}-pair-{i}-{role}"/"run"/"receipt.json")
                           for i, pair in enumerate(pairs) for role in pair))
        decision = self.revisions.evaluate(report)
        resource_feedback = self.resource_feedback(candidate, bundle, discovery, job_tag=job_tag) if valid else {
            "status": "unavailable", "reason": "candidate failed quality or actual-call coverage"}
        if decision["accepted"] and publish:
            self.revisions.commit(candidate.id)
        elif not decision["accepted"]:
            self.revisions.abort(candidate.id, "; ".join(decision["reasons"]))
        result = {**metadata, "bundle": bundle.model_dump(mode="json"),
                  "evaluation_attempt": job_tag,
                  "missing_coverage": missing_coverage,
                  "publication": "committed" if decision["accepted"] and publish else "exploration" if decision["accepted"] else "rejected",
                  "baseline": baseline, "candidate": measured, "pairs": pairs, "measurement": measurement,
                  "evaluation": report.model_dump(mode="json"),
                  "resources": resource_feedback, "decision": decision}
        write_json(trial_root/"result.json", result)
        write_json(self.output/"latest-feedback.json", result)
        return result

    def resource_feedback(self, candidate, bundle, discovery, *, job_tag=None):
        run = self.execute(candidate, (job_tag or bundle.id)+"-profile", "profile")
        if run["status"] != "passed":
            return {"status": "unavailable", "reason": "diagnostic execution failed", "run": run}
        before = [json.loads(Path(path).read_text()) for path in discovery["raw_discovery"]]
        after = [json.loads(Path(rank["result"]["discovery"]).read_text()) for rank in run["ranks"]]
        groups = []
        affected = set(bundle.affected_calls or bundle.calls)
        for group in discovery["groups"]:
            if set(group["calls"]) & affected:
                groups.append({"group": group["id"], "bundle": bundle.id, "attribution_status": "joint_only",
                    "calls": [{"id": call, "ranks": [{"rank": a["rank"],
                        "before": b["call_resources"].get(call), "after": a["call_resources"].get(call),
                        "replacement_range": a.get("bundle_resources", {}).get(bundle.id, {}).get(call),
                        "after_status": "observed" if call in a["call_resources"] else "absent_or_unassociated"}
                        for b, a in zip(before, after, strict=True)]} for call in group["calls"]]})
        from kernel_optimizer.project.communication import match_ranks
        return {"status": "observed", "groups": groups, "coverage": [a["coverage"] for a in after],
                "communication": {"before": [b.get("communication") for b in before],
                                  "after": [a.get("communication") for a in after],
                                  "cross_rank_after": match_ranks([a["communication"] for a in after if "collectives" in a.get("communication", {})])},
                "evidence": [a["trace"] for a in after], "raw_discovery": [r["result"]["discovery"] for r in run["ranks"]],
                "interpretation": "Nested diagnostic durations overlap; project gain belongs to the complete bundle. Unmeasured independent group effects are unknown."}
