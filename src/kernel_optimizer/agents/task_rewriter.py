"""Task-owned structural rewriting through the existing bounded AgentModule loop."""

import json
from pathlib import Path
from typing import ClassVar, override

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from kernel_optimizer.agents.base import AgentModule
from kernel_optimizer.agents.sandbox import Sandbox
from kernel_optimizer.conditional.task_response import TaskResponse
from kernel_optimizer.control.direct_task import TaskSpace
from kernel_optimizer.control.families import structural_signature
from kernel_optimizer.models.core import ParamSet
from kernel_optimizer.tuning.objective import Objective


class TaskRewriteInputs(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    project_root: Path
    source_paths: tuple[Path, ...] = ()
    candidate_id: str
    candidate_path: Path
    goal: str
    context: dict[str, JsonValue]
    objective: Objective
    params: ParamSet
    space: TaskSpace
    resource_metrics: tuple[str, ...] = ()
    resources: dict[str, float] = Field(default_factory=dict)
    responses: list[TaskResponse] = Field(default_factory=list)
    bundle_sources: dict[str, str] = Field(default_factory=dict)
    bundle_document: dict[str, JsonValue] | None = None


class TaskRewriteResult(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")
    candidate_file: str = Field(min_length=1)
    space: TaskSpace | None = None
    bundle_file: str | None = None
    site_groups: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    recommended_configs: list[ParamSet] = Field(default_factory=list, max_length=2)


class BundleFiles(BaseModel):
    entry: str
    files: tuple[str, ...]
    helpers: tuple[str, ...] = ()


class TaskRewriterAgent(AgentModule[TaskRewriteInputs, TaskRewriteResult]):
    name: str = "task_rewriter"
    output_model: type[BaseModel] = TaskRewriteResult

    @override
    def seed_sandbox(self, inputs: TaskRewriteInputs, sb: Sandbox) -> None:
        _ = sb.write_input("candidate/current.py", inputs.candidate_path.read_text(encoding="utf-8"))
        _ = sb.write_input("analysis/task_response.json", inputs.model_dump_json(indent=2))
        if inputs.bundle_document is not None:
            _ = sb.write_input("candidate/bundle/bundle.json", json.dumps(inputs.bundle_document, indent=2))
            for name, source in inputs.bundle_sources.items():
                _ = sb.write_input(f"candidate/bundle/{name}", source.encode("utf-8"))
            _ = sb.write_input("candidate/bundle-sources.json", json.dumps(inputs.bundle_sources))
        root = inputs.project_root.resolve()
        for source in inputs.source_paths:
            path = (root / source).resolve()
            _ = sb.write_input(f"project/{path.relative_to(root).as_posix()}", path.read_text(encoding="utf-8"))

    @override
    def render_prompt(self, inputs: TaskRewriteInputs, sb: Sandbox) -> str:
        bundle_guidance = ("\nFor this bundle task read candidate/bundle/bundle.json and candidate/bundle/*, plus the "
            "task-local instructions/helper in context. Return bundle_file pointing to a complete standalone "
            "cumulative bundle, candidate_file matching its entry, explicit site_groups for newly selected trace "
            "module paths, and at most two recommended_configs. Set parent_bundle to context.parent_bundle_sha256. "
            "Preserve all accepted sites/helpers. "
            "Helper-only computation changes count; PARAMS-only changes do not.\n") if inputs.bundle_document is not None else ""
        return """Read candidate/current.py and analysis/task_response.json, including the
goal, workload context, native objective direction/unit, current parameters, declared
space and measured responses. Read the provided project reference sources too.
Write a structurally different runnable candidate, not just new values for existing
parameters. Preserve the entire task's outputs and callable interface. The same
unchanged evaluator will execute your child with params and check its full outputs.
Do not modify inputs, evaluator, goal, workload, or correctness requirements.

Use actual selected resource observations to guide algorithm, dataflow, allocation,
fusion or scheduling changes. Resource names/units mean only what the task declares.
Responses remeasure both endpoints with all other parameters fixed. delta_j and
slopes retain native signs and units; positive gain means improvement in the declared
direction. Nominal axes have contrasts, not numeric slopes. These are local empirical
responses, not hardware causality or proof of a limiting wall. Missing/invalid data
are unknown, not zero. With no useful response, propose an ordinary source-guided
structural improvement rather than inventing observations. Probes are diagnostic,
not incumbent trials. Your child's actual evaluation and tuning decide its value.

Return candidate_file relative to this workspace and optionally a revised space
(params/constraints); otherwise the parent's space is reused. Write any required
helpers alongside the child. No ModelNew, PARAMS block, GPU backend or task registry
is required. Do not execute GPU code on the host for syntax checking.
""" + bundle_guidance

    @override
    def check_output(self, output: TaskRewriteResult, sb: Sandbox) -> str | None:
        try:
            source = sb.read_output(output.candidate_file)
            _ = compile(source, output.candidate_file, "exec")
            parent = sb.read_output("candidate/current.py")
            if output.bundle_file is not None:
                bundle = BundleFiles.model_validate_json(sb.read_output(output.bundle_file))
                folder = Path(output.bundle_file).parent
                if (folder / bundle.entry).as_posix() != Path(output.candidate_file).as_posix():
                    return "candidate_file must match bundle entry"
                sources = {name: sb.read_output((folder / name).as_posix()) for name in (*bundle.files, *bundle.helpers)}
                for name, text in sources.items():
                    _ = compile(text, name, "exec")
                old = json.loads(sb.read_output("candidate/bundle-sources.json"))
                if {n: structural_signature(s) for n, s in sources.items()} == {n: structural_signature(s) for n, s in old.items()}:
                    return "Bundle must change computation, not just PARAMS."
                return None
        except (OSError, ValueError, SyntaxError) as exc:
            return f"Candidate file check failed: {exc}"
        if structural_signature(source) == structural_signature(parent):
            return "Candidate must change structure, not just comments or PARAMS values."
        return None
