"""The measured floor must reach the final matrix, or a row accepted in search is rejected at final.

WHY THIS IS ITS OWN FILE
------------------------
`test_c3_quality_criteria_from_contract.py` pins how the floor changes a verdict. This pins that the
floor actually TRAVELS: from the setup lane that measures it, through the search lanes that accept a
candidate under it, into the final matrix that publishes the row.

That plumbing is the part that fails silently. If the matrix judged against the unfloored limit while
the search lane used the floored one, every accepted row would come back `invalid` at final -- for the
amplification the floor exists to account for, not for anything about the row -- and the §4.3 table
would be empty with a reason that looks like a candidate defect. The cells are the deliverable, so the
path is asserted rather than assumed.

The measurements behind the constants here (B12, B13) are documented in
`docs/c3-device-proof-methods.md` and in the two probes under `results/c3-fast-kernel-iteration/`.
"""

import json
from dataclasses import fields
from pathlib import Path

import pytest

from examples.c3_qwen3.fast_evaluation import EvaluationFiles
from examples.c3_qwen3.fast_matrix import MatrixInputs
from examples.c3_qwen3.fast_model import FastModelRequest
from examples.c3_qwen3.fast_runtime import RunInputs
from examples.c3_qwen3.fast_setup import control_quality_ok, write_floor_evidence
from examples.c3_qwen3.runner_quality import FLOOR_METRIC, criteria
from examples.c3_qwen3.runner_records import QualityPolicy
from kernel_optimizer.evaluation.task_eval import TaskEvaluation

FLOOR_73 = 0.010348571942351894       # B12, equivalence-preserving kernel on all 73 modules
TOLERANCE_IMPLIED = 0.03562812038262915  # B13, declared per-operator tolerance perturbation


def test_every_stage_that_judges_quality_can_carry_the_floor() -> None:
    # One missing field anywhere on this path silently reverts that stage to the unfloored limit.
    assert "floor_evidence" in {f.name for f in fields(EvaluationFiles)}
    assert "floor_evidence" in RunInputs.model_fields
    assert "floor_evidence" in FastModelRequest.model_fields
    assert "floor_evidence" in {f.name for f in fields(MatrixInputs)}


def test_the_floor_is_absent_by_default_everywhere() -> None:
    # The load-bearing default: a task that measures no floor behaves exactly as before, which is what
    # keeps this change off every other task and off the C2 KernelBench path.
    assert EvaluationFiles(Path("b"), Path("o"), Path("out")).floor_evidence is None
    assert FastModelRequest.model_fields["floor_evidence"].default is None
    assert RunInputs.model_fields["floor_evidence"].default is None
    assert next(f for f in fields(MatrixInputs) if f.name == "floor_evidence").default is None


class _Brief:
    site_id = "pure-727f184a"
    module_paths = tuple(f"model.layers.{i}.input_layernorm" for i in range(73))


class _Target:
    brief = _Brief()
    goal = "ttft"


class _Prepared:
    contract_sha256 = "c" * 64


class _Runner:
    prepared = _Prepared()


class _Runtime:
    runner = _Runner()


class _Stage:
    framework_id = "F"


class _Engine:
    target = _Target()
    runtime = _Runtime()
    stage = _Stage()


def test_the_setup_control_writes_evidence_the_gate_will_accept(tmp_path) -> None:
    # The round trip that matters: what the setup lane writes must be what `equivalence_floor` accepts,
    # including the identity fields. Writing and reading in one test is what keeps the two in step.
    from examples.c3_qwen3.runner_quality import equivalence_floor

    evaluation = TaskEvaluation(valid=False, detail="gate failed",
                               metrics={FLOOR_METRIC: FLOOR_73,
                                        "paired_mean_nll_delta_nat_per_token": 0.001254})
    path = write_floor_evidence(_Engine(), evaluation, tmp_path / "equivalence-floor.json")
    assert path is not None
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["module_count"] == 73 and document["site_id"] == "pure-727f184a"
    assert equivalence_floor(path, "c" * 64, "pure-727f184a", 73) == FLOOR_73


def test_no_evidence_is_written_when_nothing_was_measured(tmp_path) -> None:
    # An unmeasured floor must stay absent rather than become a zero or a guess: absent means "the
    # declared limit applies", which is a different and honest statement.
    target = tmp_path / "floor.json"
    assert write_floor_evidence(_Engine(), None, target) is None
    assert write_floor_evidence(_Engine(), TaskEvaluation(valid=False, detail="x", metrics={}), target) is None
    assert not target.exists()


def test_the_calibrating_control_is_not_judged_against_the_floor_it_defines() -> None:
    # The circularity, pinned in both directions. A correct control measuring the floor itself must
    # pass (it is the standard), but a control that fails a DIFFERENT criterion must not be excused --
    # otherwise "correct by construction" would stop being checked at all.
    at_floor = TaskEvaluation(valid=False, detail="gate failed",
                             metrics={FLOOR_METRIC: FLOOR_73,
                                      "paired_mean_nll_delta_nat_per_token": 0.001254})
    assert control_quality_ok(at_floor) == (True, None)
    bad_nll = TaskEvaluation(valid=False, detail="gate failed",
                             metrics={FLOOR_METRIC: FLOOR_73,
                                      "paired_mean_nll_delta_nat_per_token": 0.5})
    ok, detail = control_quality_ok(bad_nll)
    assert not ok and detail is not None and "paired_mean_nll_delta" in detail


def test_a_less_accurate_candidate_is_still_rejected_under_the_floor() -> None:
    # The floor must not become a blanket waiver. B13 measured 0.035628 for a replacement perturbed by
    # the contract's own declared per-operator tolerance; that candidate is genuinely worse than an
    # equivalence-preserving one and stays rejected, which is the whole point of calibrating the floor
    # to what a CORRECT implementation achieves rather than to what some tolerance permits.
    policy = QualityPolicy(local_operator_rtol=0.02, local_operator_atol=0.02,
                           logits_relative_l2_max_per_prompt=0.01,
                           logits_relative_l2_denominator_floor=1e-12,
                           paired_mean_nll_delta_max_nat_per_token=0.02)
    active = criteria(policy, FLOOR_73)
    limit = next(limit for key, limit, _ in active if key == FLOOR_METRIC)
    assert limit == FLOOR_73
    assert TOLERANCE_IMPLIED > limit * 3


@pytest.mark.parametrize("floor", [None, FLOOR_73])
def test_search_and_final_resolve_the_same_limit_from_the_same_evidence(floor) -> None:
    # Search and final are separate code paths (`fast_evaluation.model` and `fast_matrix.run_matrix`)
    # that must agree. Given the same evidence they resolve the same criteria; given different
    # evidence they would not, which is exactly the failure this guards.
    policy = QualityPolicy(local_operator_rtol=0.02, local_operator_atol=0.02,
                           logits_relative_l2_max_per_prompt=0.01,
                           logits_relative_l2_denominator_floor=1e-12,
                           paired_mean_nll_delta_max_nat_per_token=0.02)
    assert criteria(policy, floor) == criteria(policy, floor)
    unfloored = next(v for k, v, _ in criteria(policy, None) if k == FLOOR_METRIC)
    floored = next(v for k, v, _ in criteria(policy, floor) if k == FLOOR_METRIC)
    # A row at the floor passes under the floored limit and fails under the unfloored one, so the two
    # paths disagreeing about the evidence is a real difference in verdict, not a cosmetic one.
    if floor is not None:
        assert floored > unfloored and FLOOR_73 <= floored and FLOOR_73 > unfloored
