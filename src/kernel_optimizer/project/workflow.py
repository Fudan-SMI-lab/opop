"""Project preparation: real generation, independent controls, then publication."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import uuid4

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
        from kernel_optimizer.project.runtime_config import project_agent_config
        config = project_agent_config(config)
        self.config, self.request, self.output = config, request, output.resolve()
        if config.usage.mode == "disabled":
            self.config = config.model_copy(update={"usage": config.usage.model_copy(update={"mode": "observe_only"})})

    def prepare(self, *, feedback: str | None = None, resume: bool = False) -> dict:
        request = self.request
        if resume:
            store = RunStore.open(self.output)
            if ProjectRequest.model_validate_json((self.output/"request.json").read_bytes()) != request:
                raise ValueError("preparation resume cannot change the frozen request")
            previous = self.output/"prepared.json"
            if previous.exists():
                prior = json.loads(previous.read_text())
                if prior["status"] == "prepared":
                    revisions = RevisionStore(store)
                    if not revisions.pointer.exists():
                        revisions.publish_initial(revisions.revision(prior["revision"]))
                    return prior
                feedback = previous.read_text()
        else:
            store = RunStore.create(self.output.parent, self.output.name, {"kind": "project", "request": request.portable()})
            write_json(self.output/"request.json", request.portable())
        input_manifest = self.output/"source-inputs.json"
        if input_manifest.exists():
            sources = {name: store.get_artifact(ref) for name, ref in json.loads(input_manifest.read_text()).items()}
        else:
            sources = {name: (request.source_root/name).read_bytes() for name in request.files}
            write_json(input_manifest, {name: store.put_artifact(content) for name, content in sources.items()})
        inputs = {"request.json": json.dumps(request.portable(), indent=2)}
        inputs.update({"source/"+name: content for name, content in sources.items()})
        if resume:
            generated = [event.payload for event in store.iter_events() if event.type == "PROJECT_AGENT_FINISHED"
                         and event.payload.get("role") == "project-evaluator"]
            if generated:
                prior = Path(generated[-1]["sandbox"])/"adapter.py"
                if prior.exists():
                    inputs["prior_adapter.py"] = prior.read_bytes()
        prompt = Path(__file__).with_name("adapter_prompt.md").read_text(encoding="utf-8")
        revisions = RevisionStore(store)

        def validate_outputs(directory):
            understanding = json.loads((directory/"understanding.json").read_text())
            for field in ("objective", "workload", "quality"):
                if understanding.get(field) != getattr(request, field):
                    raise ValueError(f"Agent changed frozen {field}")
            for field in ("completion", "state_protocol", "invariants", "missing_information"):
                if field not in understanding:
                    raise ValueError(f"understanding.json omitted {field}")
            compile((directory/"adapter.py").read_bytes(), "adapter.py", "exec")

        with Runtime(self.config, log_dir=self.output) as runtime:
            agents = ProjectAgents(runtime, store)
            while True:
                sandbox, agent_receipt = agents.generate("project-evaluator", inputs, prompt,
                    ("adapter.py", "understanding.json"), feedback, validator=validate_outputs)
                understanding = json.loads((sandbox/"understanding.json").read_text())
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
                attempt = self.output/"preparation"/uuid4().hex[:12]
                source = attempt/"source"; source.mkdir(parents=True)
                for name, content in sources.items():
                    path = source/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(content)
                adapter = source/"opop_adapter.py"
                adapter.write_bytes((sandbox/"adapter.py").read_bytes())
                (source/"opop_evaluate.py").write_text("from kernel_optimizer.project.evaluation import worker_entry as run\n")
                revision = revisions.initialize(source, [*sources, "opop_adapter.py", "opop_evaluate.py"], contract,
                    ExecutionPlan(id="undiscovered"), GroupRegistry(id="undiscovered", plan_id="undiscovered"), publish=False)
                materialized = revisions.materialize(revision, attempt/"reference-source")
                evaluation = LocalExecutor().run(ExecutionRequest(entry=materialized/"opop_evaluate.py", cwd=materialized,
                    devices=request.devices, distributed_backend=request.distributed_backend, output=attempt/"controls",
                    lease_dir=request.lease_dir, worker_timeout_s=request.worker_timeout_s,
                    wait_for_devices=True,
                    parameters={"adapter": str(materialized/"opop_adapter.py"), "request": request.portable(), "action": "controls"}))
                valid = evaluation["status"] == "passed" and all(r["result"]["valid"] for r in evaluation["ranks"])
                prepared = {"schema_version": "project-prepared.v2", "status": "prepared" if valid else "invalid_evaluator",
                    "revision": revision.id, "contract_ref": revision.contract_ref, "agent": agent_receipt,
                    "adapter_sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(), "evaluation": evaluation,
                    "request_sha256": hashlib.sha256((self.output/"request.json").read_bytes()).hexdigest(),
                    "controls_path": str(attempt/"controls"), "attempt": str(attempt)}
                write_json(attempt/"prepared.json", prepared)
                write_json(self.output/"prepared.json", prepared)
                store.append("PROJECT_PREPARED" if valid else "PROJECT_EVALUATOR_REJECTED", prepared)
                if valid:
                    revisions.publish_initial(revision)
                    return prepared
                feedback = json.dumps(prepared, indent=2)
                inputs["prior_adapter.py"] = adapter.read_bytes()
