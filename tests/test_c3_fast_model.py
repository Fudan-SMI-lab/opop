from examples.c3_qwen3.fast_model import FastModelRequest, evaluate_fast_model
from examples.c3_qwen3.fast_prepare import FastAccess
from examples.c3_qwen3.local_eval import evaluate_local
from tests.c3_device_fakes import case


def test_fast_model_when_current_local_proof_precedes_quality(tmp_path) -> None:
    # Given: a baseline oracle and a real numerical fixture-only pass for this exact source/config.
    runtime, request, fixture = case(tmp_path)
    ids = ("calibration-calibration-00", "calibration-calibration-01")
    oracle = runtime.runner.create_oracles(ids)
    local = evaluate_local(request, (fixture,), runtime)
    assert local.valid, local.detail
    goal = next(g for g in runtime.runner.prepared.contract.goals if g.id == "single")
    model = FastModelRequest(stage=request.stage, access=FastAccess(profile="c3_fast_device", phase="development"),
        bundle_path=request.bundle_path, params={}, goal="single", prompt_ids=goal.search_prompt_ids,
        oracle_refs=oracle, local_reports=(local,))
    # When: the explicitly separate full-model adapter is called with model admission.
    result = evaluate_fast_model(model, runtime)
    # Then: native model units/counts, not local latency, determine the official score.
    assert result.valid, result.detail
    assert result.metrics["output_tokens"] == 256
    assert result.metrics["completed_requests"] == 2
    assert runtime.admission.calls == [("development", "local"), ("development", "model")]


def test_fast_model_when_local_prerequisite_is_missing(tmp_path) -> None:
    # Given: a candidate has no matching device/local receipt.
    runtime, request, _ = case(tmp_path)
    goal = next(g for g in runtime.runner.prepared.contract.goals if g.id == "single")
    model = FastModelRequest(stage=request.stage, access=FastAccess(profile="c3_fast_device", phase="development"),
        bundle_path=request.bundle_path, params={}, goal="single", prompt_ids=goal.search_prompt_ids,
        oracle_refs=tmp_path / "unused.json")
    # When / Then: no full-model forward or manufactured score occurs.
    result = evaluate_fast_model(model, runtime)
    assert not result.valid
    assert result.score is None
    assert runtime.runner.backend.forward_calls == 0
