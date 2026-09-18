"""Intent transport and synthetic CPU mask math; no GPU/kernel validity claim."""

from pathlib import Path

import pytest

from kernel_optimizer.agents.modules import ParameterizerAgent
from kernel_optimizer.models.reports import BottleneckReport
from scripts.experiments import c2_local_agents as agents
from scripts.experiments.c2_retune import RetuneInputs
from tests.test_c2_opportunity_bridge import bridge as existing_bridge

bridge = existing_bridge


def test_child_three_argument_constructor_defaults_to_absent_intent(bridge):
    # Given / When: an existing caller constructs the original three-field Child.
    shared, inputs, services, captured, calls, payloads = bridge
    child = agents.Child(Path("candidate.py"), shared.space, shared.backend)
    # Then: new metadata is optional and does not require a caller change.
    assert child.rewrite_intent is None


@pytest.mark.parametrize("pass_intent", [False, True])
def test_initial_and_downstream_intent_are_the_same_proposal_payload(bridge, monkeypatch, pass_intent):
    # Given: an existing concrete summary, with joint-region and companion details.
    shared, inputs, services, captured, calls, payloads = bridge
    summary = ("Target joint region N=256 with SM=64; keep the partner layout and pipeline conditions together.\n"
               "Reduce repeated materialization, retain fp32 accumulation, and report unexpressed regions as unknown.")
    proposal = agents.LegacyProposal(shared.source, shared.backend, summary, "H1", BottleneckReport(summary="fixture"))
    initial_intents = []
    seed = ParameterizerAgent.seed_sandbox

    def observe(self, parameterizer_inputs, sandbox):
        initial_intents.append(parameterizer_inputs.rewrite_intent)
        return seed(self, parameterizer_inputs, sandbox)

    monkeypatch.setattr(ParameterizerAgent, "seed_sandbox", observe)
    # When: one parameterization returns its typed downstream handoff.
    child = agents.parameterize_legacy_proposal(shared, services, proposal, pass_intent=pass_intent)
    downstream = RetuneInputs.model_validate({
        "task": shared.task, "source": child.path, "space": inputs.project_root / "space.json",
        "reference": inputs.project_root / "reference.py", "sampler_seed": 0, "evaluation_seed": 0,
        "output": services.store.run_dir / "retune", "backend": child.backend,
        "rewrite_intent": child.rewrite_intent,
    })
    # Then: intent is transported intact, or absent at both boundaries for the control.
    expected = summary if pass_intent else None
    assert initial_intents == [expected]
    assert child.rewrite_intent == downstream.rewrite_intent == expected
    intent_file = child.path.parent / "analysis/rewrite_intent.md"
    assert intent_file.exists() == pass_intent
    if pass_intent:
        assert intent_file.read_text() == summary
    assert calls == ["ParameterizationResult"]
    assert proposal.change_summary == summary and proposal.source == shared.source


def test_projection_k_tail_masks_rows_not_output_columns():
    # Given: the final [BK, BN] weight tile has 32 valid K rows and 64 valid N columns.
    k, bk, bn = 672, 64, 64
    start = (k // bk) * bk
    activations = [1 if start + row < k else 0 for row in range(bk)]
    weights = [[(start + row + 1) * (column + 1) for column in range(bn)] for row in range(bk)]
    row_mask = [[start + row < k for column in range(bn)] for row in range(bk)]
    column_mask = [[start + column < k for column in range(bn)] for row in range(bk)]
    reference = [sum((index + 1) * (column + 1) for index in range(start, k)) for column in range(bn)]
    # When: compute the same tail contribution with each broadcast orientation.
    row_result = [sum(activations[row] * weights[row][column] * row_mask[row][column]
                      for row in range(bk)) for column in range(bn)]
    column_result = [sum(activations[row] * weights[row][column] * column_mask[row][column]
                         for row in range(bk)) for column in range(bn)]
    # Then: equal mask shape/population cannot establish correctness; column masking loses valid output work.
    valid_rows = k - start
    assert sum(map(sum, row_mask)) == sum(map(sum, column_mask)) == valid_rows * bn
    assert row_result == reference
    assert column_result[:valid_rows] == reference[:valid_rows]
    assert column_result[valid_rows:] == [0] * (bn - valid_rows)
    assert all(value > 0 for value in reference[valid_rows:])
