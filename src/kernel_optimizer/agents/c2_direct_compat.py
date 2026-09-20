"""CDA model-facing contract with the current immutable-parent validation lifecycle."""

from dataclasses import replace
from typing import ClassVar, Literal, override

from pydantic import BaseModel, ConfigDict, Field

from kernel_optimizer.agents.base import AgentOutcome
from kernel_optimizer.agents.sandbox import Sandbox
from kernel_optimizer.agents.task_rewriter import TaskRewriteInputs, TaskRewriteResult, TaskRewriterAgent
from kernel_optimizer.control.direct_task import TaskSpace as CurrentTaskSpace
from kernel_optimizer.models.core import ParamDomain


class TaskSpace(CurrentTaskSpace):
    params: list[ParamDomain] = Field(min_length=1)


class C2DirectRewriteResult(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", title="TaskRewriteResult")
    candidate_file: str = Field(min_length=1)
    space: TaskSpace | None = None


class C2DirectTaskRewriterAgent(TaskRewriterAgent):
    execution_profile: ClassVar[Literal["c2_direct_compat"]] = "c2_direct_compat"
    task_kind: ClassVar[Literal["kernelbench"]] = "kernelbench"
    output_model: type[BaseModel] = C2DirectRewriteResult

    @override
    def invoke(self, inputs: TaskRewriteInputs) -> AgentOutcome[TaskRewriteResult]:
        outcome = super().invoke(inputs)
        # Consumers retain the common result type; only the two-field model is provider-visible/parsed.
        return replace(outcome, output=TaskRewriteResult(
            candidate_file=outcome.output.candidate_file, space=outcome.output.space))

    @override
    def seed_sandbox(self, inputs: TaskRewriteInputs, sb: Sandbox) -> None:
        sb.write_input("candidate/current.py", inputs.candidate_path.read_text(encoding="utf-8"))
        sb.write_input("analysis/task_response.json", inputs.model_dump_json(
            indent=2, exclude={"bundle_sources", "bundle_document"}))
        root = inputs.project_root.resolve()
        for source in inputs.source_paths:
            path = (root / source).resolve()
            sb.write_input(f"project/{path.relative_to(root).as_posix()}", path.read_text(encoding="utf-8"))

    @override
    def render_prompt(self, inputs: TaskRewriteInputs, sb: Sandbox) -> str:
        return super().render_prompt(inputs.model_copy(update={"bundle_document": None}), sb)

    @override
    def check_output(self, output: TaskRewriteResult | C2DirectRewriteResult, sb: Sandbox) -> str | None:
        return super().check_output(TaskRewriteResult(candidate_file=output.candidate_file), sb)
