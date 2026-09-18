"""Shared C2 reasoning scope and evidence delivery for analyst and rewriter."""

from typing import Final, Literal, assert_never

from kernel_optimizer.agents.sandbox import Sandbox
from kernel_optimizer.models.core import ParamSet


WHOLE_TASK_GUIDANCE: Final = """Reason from the source, run semantics, ordinary trial
history and measured resource context about the complete task's performance.
Compare plausible mechanisms and their dataflow costs, not just resource peaks.
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

ANALYST_RESOURCE_GUIDANCE: Final = """Use ordinary resource measurements as context:
compare registers, shared memory, spills, occupancy, launches and arithmetic paths
with the measured task cost and device ceilings. High usage can accompany a fast
configuration; spare capacity is not by itself an optimization opportunity. Distinguish
measurements from inferred limiters, and compare tradeoffs against actual latency.
Aggregate profiles do not prove hardware causality or rule out structural hypotheses.
For arithmetic-heavy work, consider the precision/tensor-core paths allowed by the
existing contract and correctness gate, using this box's measured ceilings. A throughput
label or a flat profile alone does not identify a uniquely best implementation path.
"""

C2_OPPORTUNITY_GUIDANCE: Final = """Read `analysis/conditional_responses.md`. For this
response-supplied call, let the measured opportunity lead mechanism choice; the general
whole-task guidance is the scale for checking its net cost, not a replacement menu
of unrelated optimizations. Work through this chain in the existing analyst summary
and hypotheses.change/expected_effect/risk, or the rewriter change_summary:
1. Relate the responses to the supplied source, reference/run semantics and selected
   parameter context. Identify the native objective direction and units, endpoint
   configurations, resource observations and fixed partners. If their source/config
   correspondence is unclear, state that limitation instead of assuming a match.
2. Identify a valuable parameter direction or nominal contrast from valid measurements
   under those partners. Preserve native J/delta signs and units; improvement depends
   on the declared objective, not a preferred sign. Nominal contrasts are not numeric
   slopes. Finite differences are neither global gradients nor hardware-causality proof.
3. Use the source to explain the constraint, cost growth or partner coupling that may
   limit that direction's value. Distinguish measured facts from inferred mechanisms.
   A hard wall is not required: continuous costs, spills, occupancy tradeoffs or coupling
   can motivate a bounded hypothesis. A domain endpoint is not a compiler refusal.
4. Choose a coherent structural action and necessary companion changes that could
   alter the opportunity's benefit or cost. Explain the connection, rather than listing
   generic fusion/layout/scheduling ideas or chasing the largest resource counter.
5. Describe the intended useful joint region and partner relationships. Name axes and
   values when supported; otherwise give the dataflow target and unknowns. Resource
   increases and loss of the parent's old best point are allowed tradeoffs, not rejection
   conditions. Do not demand resource minimization or preservation of that point.
6. Account for added full-task work, traffic, synchronization, correctness risk and
   failure conditions along this opportunity. The child's own retuning and, when native
   eligibility permits, expansion must verify the full-task net effect. Keep claims,
   implementation and measurements distinct; intent alone is not realized opportunity.
Keep actual compiler walls separate from responses, soft costs and analyst judgments.
Missing, failed or uncovered observations are unknown, not zero. If the supplied brief
has no valid contrast, state insufficient response evidence and use ordinary source-
guided reasoning without inventing an opportunity. No numeric gain prediction, new proof
schema, evidence citation/ID gate or extra analysis call is required.
"""

RAW_SOURCE_GUIDANCE: Final = """The supplied source is not declared materialized at
the selected operating point; its PARAMS defaults may differ from selected values.
Use explicit selected context when available, not the filename or defaults as evidence.
"""

MATERIALIZED_SOURCE_GUIDANCE: Final = """The caller marks the supplied source as
materialized. Check its PARAMS against the supplied selected context before treating
it as that operating point; the filename alone does not establish correspondence.
"""

PARAMETERIZER_BODY_GUIDANCE: Final = """Parameter wiring or necessary repairs may
change the computational body while preserving task semantics and existing quality
requirements. If you change indexing, masks or reductions, briefly explain the concrete
relation to wiring or repair and its risk in existing parameter descriptions or constraint
rationales. This is advisory, not a body/dtype ban or an equivalence-proof gate.
"""

PARAMETERIZER_INTENT_GUIDANCE: Final = (
    "Read `analysis/rewrite_intent.md`, the existing rewrite summary, as advisory intent. "
    "Reconcile its structural goal, intended region and companion conditions with "
    "domains, defaults and constraints. In existing parameter descriptions or constraint "
    "rationales, briefly explain what is expressed, omitted or still unknown. "
    "No particular axis/value is mandatory, and matching the intent is not an acceptance "
    "gate. Preserve all existing correctness and parameterization rules.\n\n"
)


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
    responses = C2_OPPORTUNITY_GUIDANCE if conditional_response_text else ""
    return scope + reference + responses + "\n"


def selected_context_guidance(selected_params: ParamSet | None, source_materialized: bool = False) -> str:
    source = MATERIALIZED_SOURCE_GUIDANCE if source_materialized else RAW_SOURCE_GUIDANCE
    selected = (
        "Read `tuning/selected_params.json` for the selected ParamSet (`values`), independently "
        "of source defaults. Relate measurements and fixed partners to this context; it does "
        "not replace the tunable source, full domains, constraints or trial history.\n"
        if selected_params is not None else
        "Selected parameters were not supplied; the selected operating point is unknown.\n"
    )
    return source + selected
