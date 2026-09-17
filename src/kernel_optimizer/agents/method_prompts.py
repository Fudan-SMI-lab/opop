"""Shared C2 reasoning scope and evidence delivery for analyst and rewriter."""

from typing import Final, Literal, assert_never

from kernel_optimizer.agents.sandbox import Sandbox


WHOLE_TASK_GUIDANCE: Final = """Reason about the complete task's dataflow, not only a
resource-heavy kernel or a blocked parameter direction. Consider fusion boundaries,
materialization, layout and scheduling where supported by the source and run semantics.
Choose a coherent structural scope: explain the mechanism, the new retuning opportunity,
its companion conditions, costs and tradeoffs, and what remains uncertain. A resource
peak need not identify the latency bottleneck; minimizing resource use is not the goal.
Use the existing analyst summary or rewrite change_summary for this concise explanation.
The unchanged full-task correctness checks and performance after retuning decide value;
winning at the parent's old defaults is not required. No numeric proof or named region
is required: a dataflow goal and an explicit unknown are valid intent.
"""

LEGACY_LOCAL_GUIDANCE: Final = """This is the explicit legacy_local experimental control.
Focus on local resource mechanisms and parameter directions with remaining headroom:
where measured trends improve toward a boundary, investigate what prevents further
movement and propose structural relief or a resource trade. A declared domain endpoint
alone is not a hardware limit. Explain the local mechanism and its uncertainty in the
existing summary or change_summary. Evaluate the resulting complete task after retuning,
not merely at the parent's old defaults.
"""

ANALYST_RESOURCE_GUIDANCE: Final = """CRITICAL — do not confuse "a resource is saturated" with "that resource is the
performance limiter." Reason about the WHOLE resource balance before proposing a
change:
1. Resource balance: compare each resource at the best config against its device
   limit. High register use (even at the 255/thread max) is often the SIGNATURE of
   the fast configuration (large accumulator tiles live in registers), NOT a
   pathology to relieve — relieving it by spilling to shared memory or recomputing
   usually makes latency WORSE. Only call a saturated resource "blocking" if the
   trial data shows latency still wants to move toward a value that resource
   forbids AND a lower-usage config is not already just as fast.
2. Idle resources: if a resource is far below its limit (e.g. shared memory at 24%
   while registers are maxed), ask whether the kernel could trade the saturated
   resource for the idle one to raise arithmetic throughput — but only if the trial
   data suggests throughput (not that resource) is the wall.
3. Precision / tensor-core path: check how the kernel does its core math. If it
   uses full-IEEE fp32 matmul (e.g. tl.dot(..., input_precision="ieee")) or scalar
   FMA loops, it is NOT using the tensor cores, and a tf32/fp16-accumulate tensor-
   core path can be materially faster on matmul/conv-bound ops -- by the ratio between
   this box's MEASURED ceilings, not a fixed factor (this is how torch.compile
   wins). If the flat latency floor across many configs looks like an arithmetic-
   throughput wall rather than a memory/occupancy wall, say so and propose switching
   the dot path to tf32 (input_precision="tf32") or fp16 inputs with fp32
   accumulation — the harness's dual-precision correctness gate accepts a tf32-
   matching result, so this is allowed. This is frequently the single highest-impact
   change and must be considered explicitly, not omitted.
"""


def seed_method_context(
    reference_source: str | None, conditional_response_text: str | None, sb: Sandbox,
) -> None:
    if reference_source:
        _ = sb.write_input("task/ref.py", reference_source)
    if conditional_response_text:
        _ = sb.write_input("analysis/conditional_responses.md", conditional_response_text)


def method_guidance(
    mode: Literal["whole_task", "legacy_local"],
    reference_source: str | None,
    conditional_response_text: str | None,
) -> str:
    match mode:
        case "whole_task":
            scope = WHOLE_TASK_GUIDANCE
        case "legacy_local":
            scope = LEGACY_LOCAL_GUIDANCE
        case unreachable:
            assert_never(unreachable)
    reference = (
        "Read `task/ref.py` for the complete reference computation and callable semantics.\n"
        if reference_source else
        "Reference source was not supplied; reason from the candidate and supplied task facts, "
        "and state uncertainty rather than inventing missing reference behavior.\n"
    )
    responses = (
        "Read `analysis/conditional_responses.md` as local empirical response evidence.\n"
        if conditional_response_text else ""
    )
    return scope + reference + responses + """
For any response evidence, preserve the native objective direction, signs and units.
Either direction of a contrast is useful; improvement depends on the declared objective,
not on a universally preferred sign. Interpret endpoints with their fixed companion
parameters. Nominal axes have contrasts, not numeric slopes. Responses do not establish
hardware causality or a compile wall. Keep actual compiler refusals distinct from
soft-wall observations such as spills, and both distinct from ordinary response trends.
Missing, failed and uncovered measurements are unknown, not zero; a small endpoint
budget is not a survey of the full domain. With no useful responses, ordinary source-
guided structural reasoning remains valid. Do not invent measurements or expected gains.

"""
