"""The numerical criteria come from the contract, and an absolute logits limit needs a measured floor.

TWO DEFECTS THIS PINS, BOTH MEASURED
------------------------------------
1. THE LIMITS WERE HARDCODED WHILE THE CONTRACT DECLARED THEM. `runner_quality` compared against
   literal 0.01 / 0.02 while `shared_policy.quality` in the approved contract declared the same two
   numbers separately. Nothing checked that they agreed, so the enforced limit and the published limit
   could silently diverge. They now come from one place: the hash-verified contract.

2. AN ABSOLUTE LOGITS LIMIT IS NOT A PROPERTY OF A CANDIDATE. It depends on how many modules the
   selected site group replaces. Measured on box4 against the real model and the real frozen oracle,
   with ONE equivalence-preserving Triton RMSNorm bound to a nested prefix of the site group (B12):

       modules      1        4        16         36         73
       logits L2    0.0      0.0      0.004489   0.004489   0.010349    <- same kernel throughout
       verdict      passes   passes   passes     passes     REJECTED

   And perturbing the ORIGINAL operator output by exactly the contract's own declared per-operator
   tolerance (rtol = 0.02) measured 0.035628 at the logits (B13), i.e. 3.6x the 0.01 logits limit. Even
   a tenth of the declared tolerance measured 0.014851, still rejected. So the contract's two criteria
   are mutually unsatisfiable on this 73-module group, with no kernel involved at all.

   The fix is the one the framework already uses for timing noise (`TargetSpec.reference_noise_us`):
   MEASURE the floor on the actual group and require the candidate to be no worse than it, rather than
   assume a constant. The floor can only ever RAISE the declared limit, is accepted only when its
   evidence names the same contract/site/module count, and is absent by default -- with
   `logits_relative_l2_floor_multiple` defaulting to 1.0 and no evidence supplied, behaviour is
   byte-identical to the declared absolute limit. That default is what keeps every other task, and the
   C2 KernelBench path (which never imports this module), unaffected.
"""

import json

import pytest

from examples.c3_qwen3.runner_quality import (CRITERIA, FLOOR_METRIC, _verdict, criteria,
                                              equivalence_floor)
from examples.c3_qwen3.runner_records import Contract, QualityPolicy, RunnerError

# The live rejections and the measured floor, as recorded.
LIVE = (0.011155061927544058, 0.011807271831267152, 0.014172352920463037)
FLOOR_73 = 0.010348571942351894      # B12: equivalence-preserving kernel, all 73 modules
FLOOR_16 = 0.004489136458973596      # B12: the same kernel, 16 modules
TOLERANCE_IMPLIED = 0.03562812038262915  # B13: rtol=0.02 perturbation of the original output
CONTRACT_SHA = "a" * 64
SITE = "pure-727f184a"


def policy(**overrides) -> QualityPolicy:
    base = dict(local_operator_rtol=0.02, local_operator_atol=0.02,
                logits_relative_l2_max_per_prompt=0.01, logits_relative_l2_denominator_floor=1e-12,
                paired_mean_nll_delta_max_nat_per_token=0.02)
    return QualityPolicy(**{**base, **overrides})


def limits(active) -> dict[str, float]:
    return {key: limit for key, limit, _ in active}


def test_the_contract_supplies_the_limits() -> None:
    # Given: a contract declaring limits that differ from this module's defaults.
    declared = policy(logits_relative_l2_max_per_prompt=0.05,
                      paired_mean_nll_delta_max_nat_per_token=0.5)
    # Then: the declared numbers are what gets enforced, not the module's literals.
    assert limits(criteria(declared)) == {FLOOR_METRIC: 0.05,
                                          "paired_mean_nll_delta_nat_per_token": 0.5}
    assert limits(criteria(None)) == {key: limit for key, limit, _ in CRITERIA}


def test_a_contract_without_a_policy_keeps_the_module_defaults() -> None:
    # The operator-path contract carries no `shared_policy`; it must still parse and behave as before.
    contract = Contract.model_validate({
        "interface_version": "c3-task-local-v1.0.0",
        "assets": {"repo_id": "r", "revision": "v", "local_path_both_hosts": "/p",
                   "corpus_file": "c.json", "corpus_sha256": "h"},
        "goals": [], "runtime_versions": {}})
    assert contract.quality_policy() is None
    assert criteria(contract.quality_policy()) == CRITERIA


def test_the_declared_policy_round_trips_from_real_contract_shaped_json() -> None:
    # The real contract's quality block carries many more keys than the criteria. Parsing must accept
    # them rather than reject the contract, or the gate cannot read its own policy at all.
    contract = Contract.model_validate({
        "interface_version": "c3-fast-inputs-v1.0.0",
        "assets": {"repo_id": "r", "revision": "v", "local_path_both_hosts": "/p",
                   "corpus_file": "c.json", "corpus_sha256": "h"},
        "goals": [], "runtime_versions": {},
        "shared_policy": {"quality": {
            "local_operator_rtol": 0.02, "local_operator_atol": 0.02,
            "shape_dtype_finite_and_state_checks": True,
            "logits_relative_l2_max_per_prompt": 0.01,
            "logits_relative_l2_denominator_floor": 1e-12,
            "paired_mean_nll_delta_max_nat_per_token": 0.02,
            "teacher_forced_reference_tokens": True,
            "calibration_prompt_ids": ["calibration-calibration-00"],
            "calibration_input_tokens": 64, "reference_continuation_tokens": 32,
            "reference_continuations_available": False, "whole_model_fp64_rescue": False}}})
    declared = contract.quality_policy()
    assert declared is not None
    assert declared.logits_relative_l2_max_per_prompt == 0.01
    assert declared.local_operator_rtol == 0.02
    # And the floor multiple defaults to "no worse than an equivalent implementation".
    assert declared.logits_relative_l2_floor_multiple == 1.0


def test_without_floor_evidence_nothing_changes() -> None:
    # The load-bearing default. A task that measures no floor sees exactly the declared limits, which
    # is what keeps this change from touching any other task.
    assert criteria(policy(), None) == criteria(policy())
    for value in LIVE:
        assert _verdict({FLOOR_METRIC: value}, criteria(policy()))[0] is False


def test_a_measured_floor_raises_the_limit_but_only_to_the_floor() -> None:
    # Given: the floor B12 measured for this 73-module group, and the default multiple of 1.0.
    active = criteria(policy(), FLOOR_73)
    assert limits(active)[FLOOR_METRIC] == FLOOR_73
    # Then: a candidate at the floor is admitted -- an equivalence-preserving kernel is not rejected
    # for the group's amplification -- while every live rejection, all of which are ABOVE the floor,
    # stays rejected. The floor admits the achievable, not the whole cluster.
    assert _verdict({FLOOR_METRIC: FLOOR_73}, active)[0] is True
    for value in LIVE:
        assert _verdict({FLOOR_METRIC: value}, active)[0] is False, value


def test_a_floor_multiple_is_what_admits_a_candidate_above_the_floor() -> None:
    # Allowing a candidate to be somewhat worse than the best equivalent implementation is a separate,
    # explicit decision: it takes a declared multiple, and the contract has to say so.
    tight, loose = criteria(policy(), FLOOR_73), criteria(policy(logits_relative_l2_floor_multiple=1.5), FLOOR_73)
    assert _verdict({FLOOR_METRIC: LIVE[0]}, tight)[0] is False
    assert _verdict({FLOOR_METRIC: LIVE[0]}, loose)[0] is True
    assert limits(loose)[FLOOR_METRIC] == pytest.approx(FLOOR_73 * 1.5)


def test_a_floor_below_the_declared_limit_never_tightens_it() -> None:
    # A group of few modules measures a tiny floor. That must not make the gate STRICTER than the
    # contract declared -- the floor is a correction for amplification, not a new threshold.
    active = criteria(policy(), FLOOR_16)
    assert limits(active)[FLOOR_METRIC] == 0.01
    assert _verdict({FLOOR_METRIC: 0.0099}, active)[0] is True


def test_the_floor_does_not_touch_the_other_criterion() -> None:
    # The NLL criterion was never the problem: every live rejection was 20x inside it. Raising the
    # logits limit must not relax it, or the two would stop being independent checks.
    active = criteria(policy(), FLOOR_73)
    assert limits(active)["paired_mean_nll_delta_nat_per_token"] == 0.02
    assert _verdict({FLOOR_METRIC: FLOOR_73, "paired_mean_nll_delta_nat_per_token": 0.03}, active)[0] is False


def test_the_tolerance_implied_error_is_still_rejected_by_a_measured_floor() -> None:
    # The most important thing the floor must NOT do. B13 measured that a replacement perturbed by the
    # contract's declared per-operator tolerance produces 0.035628 at the logits. That candidate is
    # genuinely less accurate than an equivalence-preserving one, and the floor must keep rejecting it:
    # the floor is calibrated to what a CORRECT implementation achieves, not to what the loosest
    # declared tolerance permits.
    active = criteria(policy(), FLOOR_73)
    assert _verdict({FLOOR_METRIC: TOLERANCE_IMPLIED}, active)[0] is False
    assert TOLERANCE_IMPLIED > FLOOR_73 * 3


def floor_document(**overrides) -> dict:
    return {"contract_sha256": CONTRACT_SHA, "site_id": SITE, "module_count": 73,
            "logits_relative_l2_max": FLOOR_73, **overrides}


def test_floor_evidence_must_match_the_group_it_is_applied_to(tmp_path) -> None:
    # The floor is a property of a specific site group -- the same kernel measured 0.0 at one module
    # and 0.010349 at 73 -- so evidence from another group says nothing about this one and must be
    # refused rather than silently reused.
    path = tmp_path / "floor.json"
    path.write_text(json.dumps(floor_document()), encoding="utf-8")
    assert equivalence_floor(path, CONTRACT_SHA, SITE, 73) == FLOOR_73
    for wrong in (("contract", "b" * 64, SITE, 73), ("site", CONTRACT_SHA, "other-site", 73),
                  ("count", CONTRACT_SHA, SITE, 36)):
        with pytest.raises(RunnerError, match="different contract/site/module count"):
            equivalence_floor(path, *wrong[1:])


def test_missing_or_malformed_floor_evidence_fails_loudly(tmp_path) -> None:
    # A declared-but-absent floor must not fall back to the declared limit: that would make a typo in
    # a path silently change which criterion was enforced.
    with pytest.raises(RunnerError, match="missing"):
        equivalence_floor(tmp_path / "absent.json", CONTRACT_SHA, SITE, 73)
    for bad in ({"logits_relative_l2_max": "0.01"}, {"logits_relative_l2_max": -1.0},
                {"logits_relative_l2_max": None}):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(floor_document(**bad)), encoding="utf-8")
        with pytest.raises(RunnerError, match="no finite measured logits error"):
            equivalence_floor(path, CONTRACT_SHA, SITE, 73)


def test_a_zero_floor_is_a_valid_measurement(tmp_path) -> None:
    # B12 measured exactly 0.0 at one and four modules. Zero is a real result, not a missing one, and
    # it correctly leaves the declared limit in force.
    path = tmp_path / "floor.json"
    path.write_text(json.dumps(floor_document(module_count=1, logits_relative_l2_max=0.0)), encoding="utf-8")
    assert equivalence_floor(path, CONTRACT_SHA, SITE, 1) == 0.0
    assert limits(criteria(policy(), 0.0))[FLOOR_METRIC] == 0.01
