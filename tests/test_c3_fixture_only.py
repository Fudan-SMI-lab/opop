import importlib
from pathlib import Path

import pytest


def test_fixture_only_surface_when_device_lane_requested() -> None:
    # Given / When / Then: this is a separate opt-in evaluator, never the full-model helper default.
    path = Path(__file__).resolve().parents[1] / "examples/c3_qwen3/local_eval.py"
    assert path.is_file(), "fixture-only device evaluator absent"
    assert callable(importlib.import_module("examples.c3_qwen3.local_eval").evaluate_local)


def test_local_mode_when_real_numerical_fixture_is_supplied(tmp_path: Path) -> None:
    # Given: all full-model evaluation paths are forbidden in this local mode.
    from tests.c3_device_fakes import case
    from examples.c3_qwen3.local_eval import evaluate_local
    runtime, request, fixture = case(tmp_path)
    def forbidden(*args):
        raise AssertionError("full model evaluation forbidden")
    runtime.runner.quality = forbidden
    runtime.runner.measure = forbidden
    # When: one admitted bundle/config executes local correctness and paired repeats.
    report = evaluate_local(request, (fixture,), runtime)
    # Then: real outputs were compared; only a local score exists and baseline is restored.
    assert report.valid, report.detail
    assert report.official_model_score is None
    assert report.counts["full_model_forwards"] == 0
    assert report.counts["warmup_calls"] == 6
    assert report.counts["timed_calls"] == 40
    assert len(report.fixtures[0].reference_us) == len(report.fixtures[0].candidate_us) == 20
    assert runtime.runner.binding.modules["scale"].forward(3.) == 6.


@pytest.mark.parametrize("body", ["return call.args[0] * 7", "return call.args[0] * float('nan')",
                                 "return type(call.args[0])(call.args[0].values[:-1])"])
def test_wrong_operator_when_local_result_is_invalid(tmp_path: Path, body: str) -> None:
    # Given / When: a wrong candidate computes actual numerical output.
    from tests.c3_device_fakes import case
    from examples.c3_qwen3.local_eval import evaluate_local
    runtime, request, fixture = case(tmp_path, body)
    report = evaluate_local(request, (fixture,), runtime)
    # Then: failure consumes admission, performs no timed repeats, and cannot create model J.
    assert not report.valid
    assert report.latency_us is report.official_model_score is None
    assert report.counts["admitted"] == 1
    assert report.counts["timed_calls"] == 0


@pytest.mark.parametrize("failure", [{"compiled": False}, {"launches": 0}, {"output_used": False}, {"stored_output_args": ()},
    {"loaded_input_args": ()}, {"skip_control_rejected": False}, {"cuda_names": ("unrelated",)}])
def test_device_gate_when_launch_or_output_proof_missing(tmp_path: Path, failure) -> None:
    # Given / When: simulated native evidence is missing one required independent condition.
    from tests.c3_device_fakes import case
    from examples.c3_qwen3.local_eval import evaluate_local
    runtime, request, fixture = case(tmp_path)
    runtime.device.bad_proof = failure
    report = evaluate_local(request, (fixture,), runtime)
    # Then: no CPU/Torch/dummy substitute is promoted to device proof.
    assert not report.valid
    assert report.failure_stage == "compile_device_proof"


def test_source_gate_when_receipt_is_ineligible(tmp_path: Path) -> None:
    # Given / When: a CPU-only source gate refuses the artifact before CUDA initialization.
    from tests.c3_device_fakes import case
    from examples.c3_qwen3.local_eval import evaluate_local
    runtime, request, fixture = case(tmp_path)
    request = request.model_copy(update={"source_gate": request.source_gate.model_copy(update={"eligible": False})})
    report = evaluate_local(request, (fixture,), runtime)
    # Then: denied work cannot call even the device-ready boundary.
    assert not report.valid
    assert runtime.device.ready_calls == 0


def test_triton_runtime_when_actual_backend_is_exposed() -> None:
    # Given / When / Then: CPU imports expose a real lazy adapter, not a GPU-executing import.
    path = Path(__file__).resolve().parents[1] / "examples/c3_qwen3/local_device.py"
    assert path.is_file(), "native Triton proof/timing adapter absent"
    assert callable(importlib.import_module("examples.c3_qwen3.local_device").TorchLocalDevice)


def test_local_rpc_when_explicit_fixture_handler_is_supplied(tmp_path) -> None:
    # Given: the existing loopback helper, explicitly opted into the fixture-only callback.
    from examples.c3_qwen3 import search_helper
    from examples.c3_qwen3.local_eval import evaluate_local
    from tests.c3_device_fakes import case
    runtime, request, fixture = case(tmp_path)
    assert hasattr(search_helper, "request_fixture"), "fixture-only resident RPC route absent"
    # When: one request traverses the actual existing socket service.
    with search_helper.HelperService(fixture_handler=lambda r: evaluate_local(r, (fixture,), runtime)) as service:
        endpoint = {"host": "127.0.0.1", "port": service.server.server_address[1]}
        result = search_helper.request_fixture(endpoint, request.model_dump(mode="json"))
    # Then: reply is local-only and did not create or evaluate another model.
    assert result.valid, result.detail
    assert result.official_model_score is None
    assert result.counts["full_model_forwards"] == 0


def test_pair_order_when_caller_admission_parity_changes(tmp_path) -> None:
    # Given / When: stage admission sequence chooses the paired side order.
    from examples.c3_qwen3.local_eval import evaluate_local
    from tests.c3_device_fakes import case
    runtime, request, fixture = case(tmp_path)
    runtime.admission.ordinal = 1
    report = evaluate_local(request, (fixture,), runtime)
    # Then: equal repeat counts survive the reversed order.
    assert report.fixtures[0].pair_order == ("candidate", "reference")
    assert len(report.fixtures[0].candidate_us) == len(report.fixtures[0].reference_us) == 20


def test_stale_fixture_when_values_change_after_capture(tmp_path) -> None:
    # Given / When: the descriptor is unchanged but the frozen numerical payload was mutated.
    from examples.c3_qwen3.local_eval import evaluate_local
    from tests.c3_device_fakes import case
    runtime, request, fixture = case(tmp_path)
    fixture.original.call.args[0].values[0] = 99
    report = evaluate_local(request, (fixture,), runtime)
    # Then: stale fixture identity cannot enter compilation or timing.
    assert not report.valid
    assert report.counts["candidate_calls"] == 0
    assert "identity" in report.detail


def test_probe_quality_failure_preserves_observed_kernel(tmp_path) -> None:
    # Given: a device probe observed a real launch but reported failed numerical checks.
    from examples.c3_qwen3.local_eval import evaluate_local
    from tests.c3_device_fakes import case
    runtime, request, fixture = case(tmp_path)
    runtime.device.bad_proof = {"local_quality_passed": False, "quality_detail": "mismatched output"}
    # When / Then: reject with a numerical stage while retaining the observed kernel identity.
    report = evaluate_local(request, (fixture,), runtime)
    assert not report.valid
    assert report.failure_stage == "local_correctness"
    assert report.fixtures[0].device_evidence[0].kernel_name == "observed"
