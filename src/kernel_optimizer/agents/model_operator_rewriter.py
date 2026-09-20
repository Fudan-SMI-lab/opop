"""Opt-in one-operator generation; declarations are not device execution evidence."""

import json
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import ClassVar, Final, Literal, override

from pydantic import BaseModel, ConfigDict, Field

from kernel_optimizer.agents.base import AgentOutcome
from kernel_optimizer.agents.sandbox import Sandbox
from kernel_optimizer.agents.task_rewriter import BundleFiles, TaskRewriteInputs, TaskRewriteResult, TaskRewriterAgent
from kernel_optimizer.models.core import sha256_text
from kernel_optimizer.models.device_operator import DeviceKernelDeclaration
from kernel_optimizer.tuning.objective import Objective


class TensorArgument(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    device: str


class RepresentativeCall(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")
    phase: str
    tensors: tuple[TensorArgument, ...] = Field(min_length=1)


class OperatorBrief(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")
    site_id: str = Field(min_length=1)
    module_paths: tuple[str, ...] = Field(min_length=1)
    reference_source: str = Field(min_length=1)
    callable_interface: str = Field(min_length=1)
    numerical_semantics: tuple[str, ...] = Field(min_length=1)
    representative_calls: tuple[RepresentativeCall, ...] = Field(min_length=1)


class OperatorTaskFacet(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore", allow_inf_nan=False)
    contract_sha256: str = Field(min_length=1)
    quality_limits: dict[str, float] = Field(default_factory=dict)


class OperatorStageBudget(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")
    local_evaluations_remaining: int = Field(ge=0)
    model_evaluations_remaining: int = Field(ge=0)


class ModelOperatorContext(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")
    framework_revision: str = Field(min_length=1)
    parent_bundle_sha256: str = Field(min_length=1)
    task_facet: OperatorTaskFacet
    operator_brief: OperatorBrief
    stage_budget: OperatorStageBudget
    repair_feedback: str | None = None


class ModelOperatorRewriteResult(TaskRewriteResult):
    bundle_file: str = Field(min_length=1)
    site_groups: dict[str, tuple[str, ...]] = Field(min_length=1, max_length=1)
    device_kernels: tuple[DeviceKernelDeclaration, ...] = Field(min_length=1, max_length=4)


@dataclass(frozen=True, slots=True)
class _OperatorInvocation:
    agent_id: int
    context: ModelOperatorContext
    goal: str
    objective: Objective


_OPERATOR: Final[ContextVar[_OperatorInvocation | None]] = ContextVar("model_operator_request", default=None)


class ModelOperatorRewriterAgent(TaskRewriterAgent):
    execution_profile: ClassVar[Literal["model_operator"]] = "model_operator"
    task_kind: ClassVar[Literal["model_project_operator"]] = "model_project_operator"
    output_model: type[BaseModel] = ModelOperatorRewriteResult

    @override
    def invoke(self, inputs: TaskRewriteInputs) -> AgentOutcome[TaskRewriteResult]:
        context = ModelOperatorContext.model_validate(inputs.context)
        scoped = inputs.model_copy(update={"context": context.model_dump(mode="json"), "source_paths": ()})
        token = _OPERATOR.set(_OperatorInvocation(id(self), context, inputs.goal, inputs.objective))
        try:
            return super().invoke(scoped)
        finally:
            _OPERATOR.reset(token)

    @override
    def seed_sandbox(self, inputs: TaskRewriteInputs, sb: Sandbox) -> None:
        super().seed_sandbox(inputs, sb)
        state = _OPERATOR.get()
        if state is None or state.agent_id != id(self):
            raise OperatorContextError
        context = state.context
        sb.write_input("task/operator_reference.py", context.operator_brief.reference_source)
        sb.write_input("task/execution_profile.json", json.dumps({
            "execution_profile": self.execution_profile, "task_kind": self.task_kind,
            "framework_revision": context.framework_revision, "goal": inputs.goal,
            "objective": inputs.objective.model_dump(mode="json"),
            "parent_bundle_sha256": context.parent_bundle_sha256,
            "reference_sha256": sha256_text(context.operator_brief.reference_source),
            "site_id": context.operator_brief.site_id, "device_proof_status": "not_run"}, indent=2))

    @override
    def render_prompt(self, inputs: TaskRewriteInputs, sb: Sandbox) -> str:
        return """Read task/operator_reference.py, analysis/task_response.json and
task/execution_profile.json. The framework has already selected ONE traced pure-tensor
operator in context.operator_brief. Implement authored Triton or CUDA computation for
that operator only, using its exact supplied callable interface and canonical module paths.
The goal and native objective are dynamic inputs; do not substitute a predetermined
operator or strategy based on the goal name. Do not expand to the model or decoder stack.

Read the supplied original implementation, representative phase/shape/stride/dtype/device
calls and numerical semantics. Preserve FP32 reductions, intermediate BF16 rounding,
cast-before-weight order, epsilon, weights, aliasing and required side effects exactly
as specified; mathematical equivalence alone is insufficient. Missing reference details
are a framework-context error: report the location rather than inventing semantics.

Write a minimal complete draft first. The physical helper call cap is eight minutes;
observe the supplied remaining stage budgets. Do not execute GPU code before the
framework source gate admits the candidate. Python compile checks are not device proof.
After admission use only the framework's permitted local fixture helper/feedback path.
Do not run ad hoc model evaluation or use sealed heldout prompts, references or results.

Return a complete cumulative standalone bundle_file and its matching candidate_file.
Preserve accepted helpers/sites and the parent's immutable identity; set parent_bundle
to context.parent_bundle_sha256. Sandbox parent/reference files are editable mirrors,
not authority to redefine the baseline. Return exactly the selected site group, an
optional revised space, and at most two recommended_configs. No PARAMS-only rewrite.

Declare one to four device_kernels with backend triton/cuda, bundle-relative source_file,
named or dotted entry, and output_arg_indices (empty when no output arguments apply).
Each declaration must name authored computation in a declared bundle source, not a host
wrapper, import, dummy/unreached launch, framework dispatch, or old-operator fallback.
These declarations are static metadata only. The framework must prove actual compiled
kernel launch and participation in the returned output, then local and model correctness.
Neither your claim, host-only speedup, nor a successful compile constitutes research success.
Keep compile/local/schema errors with their exact locations for bounded framework feedback;
do not relax checks, rewrite the goal/evaluator, or claim a performance result without measurement.
"""

    @override
    def check_output(self, output: TaskRewriteResult, sb: Sandbox) -> str | None:
        error = super().check_output(output, sb)
        if error is not None:
            return error
        if not isinstance(output, ModelOperatorRewriteResult):
            return "model_operator requires its dedicated result schema"
        state = _OPERATOR.get()
        if state is None or state.agent_id != id(self):
            return "model_operator request context unavailable"
        brief = state.context.operator_brief
        if output.site_groups != {brief.site_id: brief.module_paths}:
            return "site_groups must match the single framework-selected operator group"
        bundle = BundleFiles.model_validate_json(sb.read_output(output.bundle_file))
        folder = PurePosixPath(output.bundle_file).parent
        declared = set((*bundle.files, *bundle.helpers))
        sources: dict[str, str] = {}
        for kernel in output.device_kernels:
            path = PurePosixPath(kernel.source_file)
            if path.is_absolute() or ".." in path.parts or "\\" in kernel.source_file or kernel.source_file not in declared:
                return "device_kernels.source_file must be a bundle-relative declared source"
            sources[kernel.source_file] = sha256_text(sb.read_output((folder / path).as_posix()))
        sb.write_input("analysis/operator_declaration.json", json.dumps({
            "execution_profile": self.execution_profile, "task_kind": self.task_kind,
            "framework_revision": state.context.framework_revision, "site_id": brief.site_id,
            "goal": state.goal, "objective": state.objective.model_dump(mode="json"),
            "device_kernels": [kernel.model_dump(mode="json") for kernel in output.device_kernels],
            "source_sha256": sources, "device_proof_status": "not_run"}, indent=2))
        return None


class OperatorContextError(ValueError):
    def __str__(self) -> str:
        return "model_operator must seed through its scoped invoke lifecycle"
