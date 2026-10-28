"""Reuse the existing TaskSearch/TPE loop through a project evaluator adapter."""
from __future__ import annotations

from pathlib import Path
import json
from uuid import uuid4

from kernel_optimizer.control.direct_task import TaskSearch, TaskSpace
from kernel_optimizer.conditional.task_response import TaskResponse, complete_response
from kernel_optimizer.evaluation.task_eval import TaskEvaluation
from kernel_optimizer.models.core import ParamSet
from kernel_optimizer.project.measurement import metric, valid_run
from kernel_optimizer.project.constraints import constraints_for_runs
from kernel_optimizer.tuning.objective import Objective


class ProjectTaskEvaluator:
    def __init__(self, coordinator, template, *, collect_resources=False):
        self.coordinator, self.template = coordinator, template
        self.collect_resources = collect_resources
        self.observations = []

    def evaluate(self, candidate_path, params, context):
        bundle = self.template.model_copy(update={"id": self.template.id+"-param-"+uuid4().hex[:8], "parameters": dict(params)})
        revision = self.coordinator.revisions.stage(bundle)
        run = self.coordinator.execute(revision, bundle.id+"-screen", "evaluate")
        request = self.coordinator.request
        objective = request.objective
        valid = valid_run(run)
        reducer = objective.get("rank_aggregation", "max" if objective["direction"] == "minimize" else "min")
        metrics = {name: metric(run, name, "max" if name.startswith("framework.") and name.endswith("_bytes") else reducer)
                   for name in run["ranks"][0]["result"]["metrics"]} if valid else {}
        if valid:
            metrics[objective["name"]] = metric(run, objective["name"], reducer,
                objective.get("aggregation_order", "samples_then_ranks"))
        if valid:
            for name, spec in getattr(request, "diagnostic_metrics", {}).items():
                metrics[name] = metric(run, name, spec.rank_aggregation, spec.aggregation_order)
        constraints = constraints_for_runs([run], objective, self.coordinator.prepared["evaluation"]) if valid else []
        feasible = valid and all(row["passed"] for row in constraints)
        from kernel_optimizer.project.candidate_diagnostics import reference_sites, should_trace
        candidate_diagnostics = None
        if should_trace(run, bundle.id, request.profiling):
            candidate_diagnostics = self.coordinator.collect_extra(revision, ["candidate_diagnostics"])["candidate_diagnostics"]
            candidate_diagnostics["trigger"] = {
                "policy": request.profiling.get("candidate_diagnostics", "on_reference"),
                "reference_sites": reference_sites(run, bundle.id),
                "interpretation": "Any fully referenced site can expose a hidden fallback in a joint bundle. "
                                  "References can also implement legitimate composition or verification; "
                                  "this trigger does not change quality, feasibility or acceptance.",
            }
        diagnostic = None
        if feasible and self.collect_resources:
            diagnostic = self.coordinator.execute(revision, bundle.id+"-resources", "profile")
            resources = []
            if diagnostic["status"] == "passed":
                for rank in diagnostic["ranks"]:
                    raw = json.loads(Path(rank["result"]["discovery"]).read_text())
                    kernels = [k for k in raw["kernels"] if any(r["bundle"] == bundle.id for r in k.get("replacement_sites", []))]
                    if not kernels:
                        resources.append({})
                        continue
                    values = {"replacement.kernel_launches": float(len(kernels)),
                              "replacement.diagnostic_kernel_sum_us": sum(k["duration_us"] for k in kernels)}
                    for field, metric_name in (("registers per thread", "replacement.max_registers_per_thread"),
                                               ("shared memory", "replacement.max_shared_memory_bytes")):
                        if all(field in k["launch_resources"] for k in kernels):
                            values[metric_name] = float(max(k["launch_resources"][field] for k in kernels))
                    resources.append(values)
                if resources:
                    for name in set.intersection(*(set(r) for r in resources)):
                        metrics[name] = max(r[name] for r in resources)
        result = TaskEvaluation(valid=feasible, score=metrics.get(objective["name"]), metrics=metrics,
                                detail=None if feasible else "project quality/coverage/worker or original-baseline constraint failure; see raw observation")
        self.observations.append({"bundle": bundle, "revision": revision, "run": run, "result": result,
                                  "constraints": constraints, "diagnostic": diagnostic,
                                  "candidate_diagnostics": candidate_diagnostics})
        return result


def tune(coordinator, template, path: Path, space: dict, trials: int):
    collect_resources = bool(space.get("params")) and coordinator.request.profiling.get("parameter_resources", True)
    evaluator = ProjectTaskEvaluator(coordinator, template, collect_resources=collect_resources)
    objective = Objective(direction=coordinator.request.objective["direction"],
                          label=coordinator.request.objective["name"], unit=coordinator.request.objective["unit"])
    # This is an explicit GPU parameter experiment. It makes no provider calls and
    # adds no wall-clock or API quota. Legacy callers retain their original budgets.
    budgets = coordinator.config.budgets.model_copy(update={"trials_per_space": trials, "wall_clock_hours": float("inf")})
    search = TaskSearch(evaluator, objective, budgets, device=coordinator.config.device)
    winner = search.evaluate_candidate(path, TaskSpace.model_validate(space), {}, seed=coordinator.config.run.seed)
    selected = None
    if winner is not None:
        selected = next(item for item in evaluator.observations if item["bundle"].parameters == winner.params.values
                        and item["result"].valid and item["result"].score == winner.task_evaluation.score)
    responses = []
    resource_fields = {"framework."+name: {"unit": "bytes", "kind": "gauge", "scope": "project allocator"}
        for name in ("allocated_bytes", "reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes",
                     "lifecycle_peak_allocated_bytes", "lifecycle_peak_reserved_bytes")}
    resource_fields.update(coordinator.request.profiling.get("resource_metrics", {}))
    if collect_resources:
        resource_fields.update({"replacement."+name: {"unit": unit, "kind": kind,
            "scope": "native launches inside this bundle's observed replacement ranges", "rank_aggregation": "max",
            "interpretation": "separate intrusive diagnostic run; nested work and fallback included; launch metadata is not dynamic utilization"}
            for name, unit, kind in (("kernel_launches", "count", "count"), ("diagnostic_kernel_sum_us", "us", "duration"),
                                    ("max_registers_per_thread", "count", "launch_metadata"), ("max_shared_memory_bytes", "bytes", "launch_metadata"))})
    for name, spec in resource_fields.items():
        if not isinstance(spec, dict) or not all(spec.get(key) for key in ("unit", "kind", "scope")):
            raise ValueError(f"resource metric {name} needs explicit unit, kind and scope")
    resource_fields.pop(coordinator.request.objective["name"], None)
    for left, right in zip(evaluator.observations, evaluator.observations[1:]):
        a, b = left["bundle"].parameters, right["bundle"].parameters
        axes = [name for name in a if a[name] != b.get(name)]
        if len(axes) == 1:
            axis = axes[0]
            response = complete_response(TaskResponse(candidate_id=template.id, axis=axis,
                a_params=ParamSet(values=a), b_params=ParamSet(values=b), a=left["result"], b=right["result"],
                unknown_resources=sorted(resource_fields)), objective,
                isinstance(a[axis], (float, int)) and isinstance(b[axis], (float, int)))
            responses.append(response.model_dump(mode="json"))
    for item in evaluator.observations:
        if item is not selected:
            coordinator.revisions.abort(item["revision"].id, "parameter screening only; not the selected validation candidate")
    receipt = {"engine": "existing TaskSearch + OptunaTPETuner", "measurement_scope": "complete project",
        "resource_fields": resource_fields,
        "response_interpretation": "Finite differences of observed screening endpoints, not independent causal sensitivities or acceptance evidence. Objective/throughput/workload constants are not automatically labelled resources. Zero resource change is observed, not missing. Kernel counters require separately available diagnostic evidence.",
        "parameter_trials": trials, "space": space, "selected": selected["bundle"].id if selected else None,
        "stats": [s.model_dump(mode="json") for s in search.stats], "responses": responses,
        "observations": [{"bundle": x["bundle"].model_dump(mode="json"), "revision": x["revision"].id,
                          "result": x["result"].model_dump(mode="json"), "constraints": x["constraints"], "run": x["run"],
                          "diagnostic": x["diagnostic"], "candidate_diagnostics": x["candidate_diagnostics"]}
                         for x in evaluator.observations]}
    return selected, receipt
