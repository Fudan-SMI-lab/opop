"""Re-evaluate a frozen generated adapter under an explicit workload/goal variant."""
import hashlib
import json
from pathlib import Path

from kernel_optimizer.models.objective import AggregationSpec, ObjectiveSpec, RequiredMetric
from kernel_optimizer.models.project import ExecutionPlan, GroupRegistry, ProjectContract
from kernel_optimizer.project.measurement import valid_run
from kernel_optimizer.project.request import ProjectRequest
from kernel_optimizer.project.transactions import RevisionStore
from kernel_optimizer.runtime.execution import ExecutionRequest, LocalExecutor, write_json
from kernel_optimizer.store.run_store import RunStore


def prepare_variant(parent_output: Path, request: ProjectRequest, output: Path, *, config=None):
    """No new code generation: useful for goal contrasts and held-out workloads.

    Failure stays a failure; this entry never repairs code to fit the held-out case.
    Each variant gets its own controls, reference, contract, and measured anchor.
    """
    parent_output, output = parent_output.resolve(), output.resolve()
    old_request = ProjectRequest.model_validate_json((parent_output/"request.json").read_bytes())
    allowed = {"goal", "objective", "workload", "warmup", "repeats", "worker_timeout_s", "tools", "profiling",
               "communication", "required_observations", "diagnostic_metrics", "notes", "devices", "lease_dir"}
    if old_request.model_dump(exclude=allowed) != request.model_dump(exclude=allowed):
        raise ValueError("variant may change goal/workload/collection protocol/device placement, not source, quality or assets")
    if len(old_request.devices) != len(request.devices):
        raise ValueError("device placement may change, but this frozen evaluator variant keeps the task's device count")
    prepared = json.loads((parent_output/"prepared.json").read_text())
    if prepared["status"] != "prepared":
        raise ValueError("variant requires a validated original evaluator")
    original_store = RunStore.open(parent_output); original_revisions = RevisionStore(original_store)
    original = original_revisions.revision(prepared["revision"])
    contract = original_revisions.load(original.contract_ref, ProjectContract)
    # Prose generated for the original workload can contain literal batch/shape
    # values. Copying it into a different workload creates a contradictory task.
    changed_contract = (contract.workload != request.workload or contract.goal != request.goal
                        or old_request.objective != request.objective)
    if changed_contract and config is None:
        raise ValueError("goal/workload variants need config for a real contract-only adaptation; evaluator code stays frozen")
    store = RunStore.create(output.parent, output.name, {"kind": "project_variant", "parent_output": str(parent_output), "request": request.portable()})
    adaptation = None
    if changed_contract:
        from pydantic import BaseModel, ConfigDict, Field
        from kernel_optimizer.project.agents import ProjectAgents
        from kernel_optimizer.wiring import Runtime
        class Invariants(BaseModel):
            model_config = ConfigDict(extra="forbid")
            invariants: list[str] = Field(min_length=1)
        inputs = {"original-contract.json": contract.model_dump_json(indent=2),
                  "request.json": json.dumps(request.portable(), indent=2),
                  "invariants.schema.json": json.dumps(Invariants.model_json_schema(), indent=2),
                  "adapter.py": original_store.get_artifact(next(f.content_ref for f in original.files if f.path == "opop_adapter.py"))}
        log_dir = output / "contract-adaptation"
        log_dir.mkdir(parents=True, exist_ok=True)
        from kernel_optimizer.project.runtime_config import project_agent_config
        with Runtime(project_agent_config(config), log_dir=log_dir) as runtime:
            generated, adaptation = ProjectAgents(runtime, store).generate("variant_contract", inputs,
                "Write invariants.json following invariants.schema.json. This is a contract-only goal/workload variant. "
                "Keep every source precision, asset, numerical, seed, state, quality and measurement policy. "
                "Update only references to old workload dimensions/amounts, goal, metric scope and goal constraints, using request.json. "
                "The evaluator is frozen and will be executed unchanged; inspect adapter.py to explain its support. "
                "Do not generate or edit adapter code, loosen accuracy, or invent measurements. "
                "Use the new concrete workload values consistently; do not retain contradictory old batch/shape statements.",
                ("invariants.json",), validator=lambda path: Invariants.model_validate_json((path/"invariants.json").read_bytes()))
        invariants = Invariants.model_validate_json((generated / "invariants.json").read_bytes())
        contract = contract.model_copy(update={"invariants": tuple(invariants.invariants)})
    objective = request.objective
    contract = contract.model_copy(update={"goal": request.goal, "workload": request.workload,
        "objective": ObjectiveSpec(name=objective["name"], unit=objective["unit"], direction=objective["direction"],
            required_metrics=(RequiredMetric(name=objective["name"], unit=objective["unit"]),),
            aggregation=AggregationSpec(kind="sum"), improvement_threshold=float(objective.get("improvement_threshold", 0)))})
    write_json(output/"request.json", request.portable())
    source = output/"preparation"/"source"
    original_revisions.materialize(original, source)
    revisions = RevisionStore(store)
    revision = revisions.initialize(source, [file.path for file in original.files], contract,
        ExecutionPlan(id="undiscovered"), GroupRegistry(id="undiscovered", plan_id="undiscovered"), publish=False)
    materialized = revisions.materialize(revision, output/"preparation"/"reference-source")
    controls = output/"preparation"/"controls"
    evaluation = LocalExecutor().run(ExecutionRequest(entry=materialized/"opop_evaluate.py", cwd=materialized,
        devices=request.devices, distributed_backend=request.distributed_backend, output=controls,
        lease_dir=request.lease_dir, worker_timeout_s=request.worker_timeout_s, wait_for_devices=True,
        parameters={"adapter": str(materialized/"opop_adapter.py"), "request": request.portable(), "action": "controls"}))
    valid = valid_run(evaluation)
    result = {"schema_version": "project-prepared.v2", "status": "prepared" if valid else "invalid_evaluator",
        "revision": revision.id, "contract_ref": revision.contract_ref, "evaluation": evaluation,
        "adapter_sha256": prepared["adapter_sha256"], "request_sha256": hashlib.sha256((output/"request.json").read_bytes()).hexdigest(),
        "controls_path": str(controls), "reuse": {"parent_output": str(parent_output), "revision": original.id,
            "placement": {"previous_devices": old_request.devices, "devices": request.devices,
                          "baseline": "remeasured in this variant's controls; no historical timing reused"},
            "origin_agent": prepared.get("agent", prepared.get("reuse")), "new_generation": False,
            "contract_adaptation": adaptation}}
    write_json(output/"prepared.json", result)
    store.append("PROJECT_PREPARED" if valid else "PROJECT_EVALUATOR_REJECTED", result)
    if valid:
        revisions.publish_initial(revision)
    return result
