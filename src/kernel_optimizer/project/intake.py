"""Resolve a repository, launch command and goal into the existing project ABI."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.project.request import ProjectRequest
from kernel_optimizer.project.runtime_config import project_agent_config
from kernel_optimizer.runtime.execution import gpu_inventory, write_json
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime


class ConstraintDescription(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metric: str
    op: Literal["le", "lt", "ge", "gt", "eq"]
    baseline_ratio: float | None = None
    threshold: float | None = None
    rank_aggregation: Literal["max", "min", "sum", "mean"] | None = None

    @model_validator(mode="after")
    def limit(self):
        if (self.baseline_ratio is None) == (self.threshold is None):
            raise ValueError("constraint requires exactly one of baseline_ratio and threshold")
        return self


class ObjectiveDescription(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str = Field(description="Exact evaluator metric key, not a descriptive task title")
    unit: str
    direction: Literal["minimize", "maximize"]
    constraints: list[ConstraintDescription] = Field(default_factory=list)
    source: dict[str, Any] | None = Field(default=None, description="Optional diagnostic collector and field")
    rank_aggregation: Literal["max", "min", "sum", "mean"] | None = None

    @model_validator(mode="after")
    def metric_name(self):
        if self.model_extra.get("metric", self.name) != self.name:
            raise ValueError("objective.name is the metric key; do not create a separate descriptive name")
        return self


class Description(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entry: str
    files: list[str] = Field(min_length=1)
    objective: ObjectiveDescription
    workload: dict[str, Any]
    quality: dict[str, Any]
    assets: dict[str, str] = Field(default_factory=dict)
    device_count: int = Field(ge=0)
    distributed_backend: str | None = None
    communication: Literal["auto", "off", "required"] = "auto"
    required_observations: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class IntakeProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["ready", "needs_input"]
    description: Description | None
    evidence: dict[str, list[str]]
    assumptions: list[str]
    questions: list[str]


def repository_files(root):
    result = subprocess.run(["git", "-C", str(root), "ls-files", "-co", "--exclude-standard", "-z", "--", "."],
                            capture_output=True)
    if result.returncode == 0:
        names = sorted(set(result.stdout.decode("utf-8").split("\0")) - {""})
    else:
        names = []
        for current, directories, files in os.walk(root):
            directories[:] = [d for d in directories if d not in {".git", "__pycache__", ".venv", "node_modules"}]
            names.extend((Path(current)/name).relative_to(root).as_posix() for name in files)
    return [{"path": name, "bytes": (root/name).stat().st_size}
            for name in sorted(names) if (root/name).is_file()]


def deployment(config):
    defaults = config.project_defaults
    try:
        inventory = gpu_inventory()
        probe_error = None
    except (OSError, subprocess.SubprocessError) as exc:
        inventory, probe_error = [], f"{type(exc).__name__}: {exc}"
    available = {row["uuid"]: row for row in inventory}
    devices = defaults.devices or list(available)
    if set(devices) - available.keys():
        raise ValueError("configured project device UUID is absent from the actual worker inventory")
    tools = {name: path for name in ("nsys", "ncu") if (path := shutil.which(name))}
    tools.update(defaults.tools)
    return {"devices": devices, "inventory": inventory, "probe_error": probe_error,
            "tools": tools, "lease_dir": str(defaults.lease_dir.expanduser().resolve()),
            "profiling": defaults.profiling, "worker_timeout_s": defaults.worker_timeout_s}


def resolve_request(proposal, source, goal, environment):
    if proposal.status == "needs_input":
        if not proposal.questions:
            raise ValueError("needs_input must identify unresolved task semantics")
        return None
    if proposal.description is None or proposal.questions:
        raise ValueError("ready requires a complete description and no unresolved questions")
    d = proposal.description
    if d.entry.partition(":")[0] not in d.files:
        raise ValueError("entry must be included in the selected source files")
    for name in d.files:
        file = (source/name).resolve()
        if not file.is_relative_to(source.resolve()) or not file.is_file():
            raise ValueError(f"selected source file is not available inside the project: {name}")
    if d.device_count > len(environment["devices"]):
        raise ValueError("declared workload needs more GPUs than the available deployment")
    if d.distributed_backend and d.device_count < 2:
        raise ValueError("this distributed executor requires at least two bound GPUs")
    return ProjectRequest(source_root=source, entry=d.entry, files=list(dict.fromkeys(d.files)), goal=goal,
        objective=d.objective.model_dump(exclude_none=True, exclude_unset=True),
        workload=d.workload, quality=d.quality, assets=d.assets,
        devices=environment["devices"][:d.device_count], distributed_backend=d.distributed_backend,
        communication=d.communication, required_observations=d.required_observations, notes=d.notes,
        lease_dir=Path(environment["lease_dir"]), tools=environment["tools"],
        profiling=environment["profiling"], worker_timeout_s=environment["worker_timeout_s"])


def prepare_request(config, *, project: str, goal: str, command: str | None, context: list[str],
                    output: Path, project_ref: str | None = None):
    """Save a resolved request next to the ordinary run, not inside its lifecycle."""
    root = output.resolve().with_name(output.name + ".intake")
    intent = {"project": project, "goal": goal, "command": command, "context": context, "project_ref": project_ref}
    if root.exists():
        store = RunStore.open(root)
        previous = json.loads((root/"intent.json").read_text())
        if (root/"resolved-request.json").exists():
            if previous != intent:
                raise ValueError("resolved intake is frozen; use a new output for a different task")
            return {"status": "ready", "request": str(root/"resolved-request.json"), "intake": str(root)}
        if any(previous[k] != intent[k] for k in ("project", "goal", "command", "project_ref")):
            raise ValueError("an unresolved intake may only add task context; other changes need a new output")
    else:
        store = RunStore.create(root.parent, root.name, {"kind": "project-intake", "intent": intent})
    write_json(root/"intent.json", intent)
    source = Path(project).expanduser()
    if source.is_dir():
        source = source.resolve()
    elif project.startswith(("https://", "ssh://", "git@")):
        source = root/"repository"
        if not source.exists():
            cmd = ["git", "clone", "--depth", "1"]
            if project_ref:
                cmd += ["--branch", project_ref]
            cmd += [project, str(source)]
            with (root/"clone.log").open("wb") as stream:
                subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, check=True)
    else:
        raise ValueError("project must be a local directory or a Git repository URL")
    environment = deployment(config)
    inventory = repository_files(source)
    write_json(root/"repository.json", {"root": str(source), "files": inventory,
        "scope": "Repository navigation, not an allowlist or a preselected set of optimization targets."})
    inputs = {"intent.json": json.dumps(intent, indent=2),
              "repository.json": (root/"repository.json").read_bytes(),
              "deployment.json": json.dumps(environment, indent=2),
              "intake.schema.json": json.dumps(IntakeProposal.model_json_schema(), indent=2)}
    config = project_agent_config(config)
    config.usage.mode = "observe_only"
    def validate(directory):
        proposed = IntakeProposal.model_validate_json((directory/"intake.json").read_bytes())
        resolve_request(proposed, source, goal, environment)
    prompt = Path(__file__).with_name("intake_prompt.md").read_text(encoding="utf-8")
    with Runtime(config, log_dir=root) as runtime:
        generated, receipt = ProjectAgents(runtime, store).generate("project-intake", inputs, prompt,
            ("intake.json",), validator=validate)
    proposal = IntakeProposal.model_validate_json((generated/"intake.json").read_bytes())
    request = resolve_request(proposal, source, goal, environment)
    result = {"status": proposal.status, "intake": str(root), "proposal": proposal.model_dump(mode="json"),
              "agent": receipt}
    if request is not None:
        snapshot = root/"source"
        snapshot.mkdir(exist_ok=False)
        manifest = []
        for name in request.files:
            original = source/name
            raw = original.read_bytes()
            target = snapshot/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            manifest.append({"path": name, "source": str(original), "sha256": hashlib.sha256(raw).hexdigest()})
        request.source_root = snapshot
        path = root/"resolved-request.json"
        write_json(path, request.portable())
        write_json(root/"source-manifest.json", {"files": manifest,
            "scope": "Selected project files frozen after intake; original repository and external assets were inspected by the Agent and are not claimed to have been snapshotted before intake."})
        result["request"] = str(path)
    write_json(root/"result.json", result)
    store.append("PROJECT_INTAKE_FINISHED", result)
    return result
