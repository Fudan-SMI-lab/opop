"""Project coordinator. Frozen task semantics and execution are separate from agents."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

from kernel_optimizer.config import AppConfig
from kernel_optimizer.models.objective import AggregationSpec, ObjectiveSpec, RequiredMetric
from kernel_optimizer.models.project import ExecutionPlan, GroupRegistry, ProjectContract
from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.project.request import ProjectRequest
from kernel_optimizer.project.transactions import RevisionStore
from kernel_optimizer.runtime.execution import ExecutionRequest, LocalExecutor, write_json
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime


class ProjectWorkflow:
    def __init__(self, config: AppConfig, request: ProjectRequest, output: Path):
        self.config, self.request, self.output = config, request, output.resolve()
        if config.usage.mode == "disabled":
            # The new project path defaults to statistics; legacy configuration remains unchanged.
            self.config = config.model_copy(update={"usage": config.usage.model_copy(update={"mode": "observe_only"})})

    def prepare(self, *, feedback: str | None = None) -> dict:
        request = self.request
        store = RunStore.create(self.output.parent, self.output.name, {"kind": "project", "request": request.portable()})
        write_json(self.output / "request.json", request.portable())
        sources = {name: (request.source_root/name).read_bytes() for name in request.files}
        inputs = {"request.json": json.dumps(request.portable(), indent=2)}
        inputs.update({"source/"+name: content for name, content in sources.items()})
        prompt = Path(__file__).with_name("adapter_prompt.md").read_text(encoding="utf-8")
        with Runtime(self.config, log_dir=self.output) as runtime:
            agents = ProjectAgents(runtime, store)
            sandbox, agent_receipt = agents.generate("project-evaluator", inputs, prompt,
                                                     ("adapter.py", "understanding.json"), feedback)
        understanding = json.loads((sandbox/"understanding.json").read_text())
        for field in ("objective", "workload", "quality"):
            if understanding[field] != getattr(request, field):
                raise ValueError(f"Agent changed frozen {field}")
        if understanding["missing_information"]:
            write_json(self.output/"needs-input.json", understanding)
            raise ValueError(f"unresolved project semantics: {understanding['missing_information']}")
        objective = ObjectiveSpec(name=request.objective["name"], unit=request.objective["unit"],
            direction=request.objective["direction"], required_metrics=(RequiredMetric(name=request.objective["name"], unit=request.objective["unit"]),),
            aggregation=AggregationSpec(kind="sum"), improvement_threshold=float(request.objective.get("improvement_threshold", 0)))
        contract = ProjectContract(id="task", goal=request.goal, objective=objective,
            workload=request.workload, quality=request.quality, completion=understanding["completion"],
            invariants=tuple(understanding["invariants"]), state_protocol=understanding["state_protocol"],
            frozen_files=("opop_adapter.py", "opop_evaluate.py"),
            world_size=len(request.devices) if request.distributed_backend else 1)
        source = self.output / "source"; source.mkdir()
        for name, content in sources.items():
            path = source/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(content)
        adapter = source / "opop_adapter.py"
        adapter.write_bytes((sandbox/"adapter.py").read_bytes())
        compile(adapter.read_text(), str(adapter), "exec")
        (source/"opop_evaluate.py").write_text("from kernel_optimizer.project.evaluation import worker_entry as run\n")
        revisions = RevisionStore(store)
        revision = revisions.initialize(source, [*sources, "opop_adapter.py", "opop_evaluate.py"], contract,
                                       ExecutionPlan(id="undiscovered"), GroupRegistry(id="undiscovered", plan_id="undiscovered"))
        materialized = revisions.materialize(revision, self.output / "reference-source")
        evaluation = LocalExecutor().run(ExecutionRequest(entry=materialized/"opop_evaluate.py", cwd=materialized,
            devices=request.devices, distributed_backend=request.distributed_backend, output=self.output/"controls",
            lease_dir=request.lease_dir, worker_timeout_s=request.worker_timeout_s,
            parameters={"adapter": str(materialized/"opop_adapter.py"), "request": request.portable(), "action": "controls"}))
        valid = evaluation["status"] == "passed" and all(r["result"]["valid"] for r in evaluation["ranks"])
        prepared = {"schema_version": "project-prepared.v1", "status": "prepared" if valid else "invalid_evaluator",
                    "revision": revision.id, "contract_ref": revision.contract_ref, "agent": agent_receipt,
                    "adapter_sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(), "evaluation": evaluation}
        write_json(self.output/"prepared.json", prepared)
        store.append("PROJECT_PREPARED" if valid else "PROJECT_EVALUATOR_REJECTED", prepared)
        if not valid:
            raise ValueError(f"generated evaluator failed controls; inspect {self.output/'prepared.json'}")
        return prepared
