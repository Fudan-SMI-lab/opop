"""A rejection must carry what it measured, against what limit, or the next attempt is guesswork.

THE OBSERVATION THAT MOTIVATED THIS
-----------------------------------
In the live F4 development lane, two INDEPENDENT candidates -- different agent calls, different source
files, different kernel structure -- were rejected by the frozen teacher-forced gate at
`logits_relative_l2_max` 0.011012 and 0.011807 against a limit of 0.01. Both were told only
"frozen teacher-forced logits/NLL gate failed", and the receipt recorded `metrics: {}` because the
failure path dropped the measurements it had already taken.

A candidate 10% over a tolerance and one 10x over need opposite responses: adjust the numerics, or
abandon the approach. "It failed" cannot distinguish them, so both attempts landed in the same place.
The gate's threshold is NOT the subject here -- B9 measured it as loose for equivalent rewrites
(1.7e-05) and separating for a wrong one (2.9e-03), so it stays exactly as it is. What changes is that
a rejection reports the measured value, the limit, and the distance between them.

These tests pin all three links of that chain, because breaking any one of them restores the
unactionable message: the verdict text, the receipt's metrics, and the repair feedback handed to the
next agent call.
"""

import json

import pytest

from examples.c3_qwen3.fast_generation import _repair_feedback
from examples.c3_qwen3.fast_records import Artifact, CandidateResult
from examples.c3_qwen3.runner_quality import CRITERIA, _verdict
from kernel_optimizer.evaluation.task_eval import TaskEvaluation

# The two live rejections, as measured.
LIVE_L2 = (0.011011857705366400, 0.011807271831267152)


def test_a_passing_candidate_has_no_detail() -> None:
    valid, detail = _verdict({"logits_relative_l2_max": 0.0005,
                              "paired_mean_nll_delta_nat_per_token": 0.0001})
    assert valid and detail is None


@pytest.mark.parametrize("measured", LIVE_L2)
def test_the_failure_names_the_metric_its_value_and_its_limit(measured: float) -> None:
    # Given: one of the two live rejections, which missed the logits limit by roughly 10-18%.
    valid, detail = _verdict({"logits_relative_l2_max": measured,
                              "paired_mean_nll_delta_nat_per_token": 0.000962})
    assert not valid and detail is not None
    # Then: the message carries the measured value, the limit, and the overshoot, so the next attempt
    # knows it is close rather than merely wrong.
    assert f"{measured:.6g}" in detail and "0.01" in detail
    assert "over by" in detail and "%" in detail
    # And: the criterion that PASSED is reported too. Its distance is the evidence that the defect is
    # in the logits path rather than a general accuracy loss -- the NLL delta was 20x inside tolerance.
    assert "passing" in detail and "paired_mean_nll_delta_nat_per_token" in detail


def test_the_gate_reports_every_breach_not_just_the_first() -> None:
    valid, detail = _verdict({"logits_relative_l2_max": 0.5,
                             "paired_mean_nll_delta_nat_per_token": 0.9})
    assert not valid and detail is not None
    for key, _limit, _what in CRITERIA:
        assert key in detail, key


def test_the_criteria_are_declared_once_so_the_message_cannot_drift_from_the_comparison() -> None:
    # The limits the gate enforces and the limits it reports come from the same tuple. This is what
    # keeps a future threshold change from producing a message that cites the old number.
    assert dict((key, limit) for key, limit, _ in CRITERIA) == {
        "logits_relative_l2_max": 0.01, "paired_mean_nll_delta_nat_per_token": 0.02}
    for key, limit, _ in CRITERIA:
        assert _verdict({key: limit})[0] is True, f"{key} at exactly its limit must pass"
        assert _verdict({key: limit * 1.000001})[0] is False, f"{key} above its limit must fail"


def test_a_metric_the_gate_did_not_measure_is_not_treated_as_passing_or_failing() -> None:
    # An evaluation that died before computing a criterion must not be silently accepted on it. The
    # verdict is only about what was measured; admission is the caller's separate decision.
    valid, detail = _verdict({})
    assert valid and detail is None
    assert _verdict({"logits_relative_l2_max": 0.02})[0] is False


def result(**overrides) -> CandidateResult:
    base = dict(version=1, status="model_failed",
                error='{"receipt": "/p/fast-model-x.json", "detail": "RunnerError: frozen '
                      'teacher-forced logits/NLL gate failed: worst-prompt relative L2 of the '
                      'continuation logits 0.0110119 exceeds logits_relative_l2_max limit 0.01 '
                      '(over by 10.1%)"}',
                model=TaskEvaluation(valid=False, detail="rejected",
                                     metrics={"logits_relative_l2_max": LIVE_L2[0],
                                              "paired_mean_nll_delta_nat_per_token": 0.000962}),
                failed_sandbox=None)
    return CandidateResult(**{**base, **overrides})


def test_the_repair_feedback_carries_the_measured_metrics() -> None:
    # Given: a rejected candidate whose evaluation recorded what it measured.
    feedback = json.loads(_repair_feedback(result()))
    # Then: the next agent call receives the numbers themselves, not only the failure text -- it does
    # not have to open a receipt file to learn how close it was.
    assert feedback["measured_metrics"]["logits_relative_l2_max"] == LIVE_L2[0]
    assert feedback["measured_metrics"]["paired_mean_nll_delta_nat_per_token"] == 0.000962
    assert feedback["status"] == "model_failed"
    assert "gate failed" in feedback["failure"]


def test_the_repair_feedback_omits_metrics_it_does_not_have() -> None:
    # A candidate rejected before any measurement (bad source, denied admission) has nothing to
    # report. Reporting an empty dict would read as "measured, and all zero", which is a fiction.
    feedback = json.loads(_repair_feedback(result(status="source_rejected", model=None,
                                                  error="source admission rejected")))
    assert feedback["measured_metrics"] is None and feedback["local_failures"] is None


def test_the_repair_feedback_carries_local_failure_detail() -> None:
    # A local (fixture-level) rejection is the other place a candidate dies with something measured.
    # Its detail is the device gate's or the numerics' own message, which is equally actionable.
    from pathlib import Path

    from examples.c3_qwen3.device_records import LocalReport, StageIdentity
    stage = StageIdentity(profile="c3_fast_device", stage="development", framework_id="F")
    report = LocalReport(stage=stage, valid=False, failure_stage="device_proof",
                         detail="declared device kernel was not compiled and launched",
                         bundle_sha256="b", params_sha256="p", source_hashes={}, fixtures=(),
                         counts={}, wall_ms=1.0, raw_path=Path("receipt.json"))
    feedback = json.loads(_repair_feedback(result(status="local_failed", model=None, locals=(report,),
                                                 error="local gate rejected")))
    assert feedback["local_failures"] == ["device_proof: declared device kernel was not compiled and launched"]


def test_confirmation_metrics_are_included_when_present() -> None:
    # A candidate can pass the first model evaluation and fail confirmation. Both sets of numbers
    # matter then, and the later one must not be lost behind the earlier.
    feedback = json.loads(_repair_feedback(result(
        confirmation=TaskEvaluation(valid=False, detail="confirmation rejected",
                                    metrics={"logits_relative_l2_max": 0.02}))))
    assert feedback["measured_metrics"]["logits_relative_l2_max"] == 0.02
    assert feedback["measured_metrics"]["paired_mean_nll_delta_nat_per_token"] == 0.000962


def test_an_artifact_bundle_is_named_so_the_agent_repairs_the_right_source() -> None:
    from pathlib import Path

    from kernel_optimizer.models.core import ParamSet
    artifact = Artifact(author="agent", version=1, session_id="s", bundle=Path("bundle.json"),
                        bundle_sha256="deadbeef", source_hashes={"operators.py": "h"},
                        params=ParamSet(values={}), params_sha256="p", kernels=())
    feedback = json.loads(_repair_feedback(result(artifact=artifact)))
    assert feedback["repair_of"] == "deadbeef"
