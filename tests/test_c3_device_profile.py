import importlib
import json
from pathlib import Path

import pytest


def test_device_profile_surface_is_lazy() -> None:
    # Given / When / Then: the CUDA-correlated path exists separately from old CPU profiling.
    root = Path(__file__).resolve().parents[1]
    assert (root / "examples/c3_qwen3/device_profile.py").is_file(), "short device profile is absent"
    assert callable(importlib.import_module("examples.c3_qwen3.device_profile").profile_goal)


def test_target_fixture_surface_is_explicit() -> None:
    # Given / When / Then: target capture cannot silently use the old 64-token fixture API.
    root = Path(__file__).resolve().parents[1]
    assert (root / "examples/c3_qwen3/device_capture.py").is_file(), "target-shape capture is absent"
    assert callable(importlib.import_module("examples.c3_qwen3.device_capture").capture_target_fixture)


@pytest.mark.parametrize("target", [("ttft", 0, (1, 4096, 3), 1),
    ("single", 0, (1, 256, 3), 1), ("single", 127, (1, 1, 3), 128),
    ("multi", 0, (8, 512, 3), 1), ("multi", 127, (8, 1, 3), 128)])
def test_target_capture_when_actual_goal_position_runs(tmp_path, target) -> None:
    # Given: a numerical tensor module in the existing resident model, not a 64-token toy call.
    from tests.c3_device_fakes import case, tensor_backend
    from examples.c3_qwen3.device_capture import capture_target_fixture
    from examples.c3_qwen3.device_records import CaptureSpec
    goal, step, shape, forwards = target
    runtime, request, _ = case(tmp_path)
    tensor_backend(runtime)
    # When: the real goal prefix and every preceding decode step execute before capture.
    report = capture_target_fixture(CaptureSpec(stage=request.stage, goal=goal, site_id="scale",
                                               module_path="scale", decode_step=step), runtime)
    # Then: target shape/position and all setup forward costs are exact.
    assert report.valid, report.detail
    assert report.fixture.original.call.args[0].shape == shape
    assert json.loads(report.raw_path.read_text())["data"]["forward_calls"] == forwards
    assert runtime.runner.backend.rows == []


def test_short_profile_when_only_first_and_late_steps_are_collected(tmp_path) -> None:
    # Given: explicit CPU profiler stand-in with overlapping device-event fixtures.
    from tests.c3_device_fakes import Trace, case, tensor_backend
    from examples.c3_qwen3.device_profile import ProfileRequest, profile_goal
    runtime, request, _ = case(tmp_path)
    tensor_backend(runtime)
    # When: batch8 advances normally while only prefill and final decode are traced.
    path = profile_goal(ProfileRequest(request.stage, "multi"), runtime, trace_factory=Trace)
    # Then: no inclusive-CPU or summed-overlap critical-path fraction is fabricated.
    data = json.loads(path.read_text())["data"]
    assert data["valid"], data
    assert data["forward_calls"] == 129
    assert data["startup_forwards"] == 1
    assert data["profiled_forwards"] == 2
    assert {row["decode_step"] for row in data["calls"]} == {0, 127}
    assert data["critical_path_fraction"] is None


def test_oversized_fixture_when_cap_is_exceeded(tmp_path) -> None:
    # Given / When: a pure call requiring more than the frozen cap cannot be cropped.
    from tests.c3_device_fakes import case, tensor_backend
    from examples.c3_qwen3.device_capture import capture_target_fixture
    from examples.c3_qwen3.device_records import CaptureSpec
    runtime, request, _ = case(tmp_path)
    tensor_backend(runtime)
    runtime.device.tools.size = lambda call: 64 * 1024 * 1024 + 1
    report = capture_target_fixture(CaptureSpec(stage=request.stage, goal="ttft", site_id="scale",
                                               module_path="scale", decode_step=0), runtime)
    # Then: unsupported is explicit and no fixture is presented as valid.
    assert not report.valid
    assert report.fixture is None
    assert "cap" in report.detail


def test_profile_limit_when_caller_exceeds_three_windows(tmp_path) -> None:
    # Given: profile ordinal3 is beyond the fixed three-window allowance.
    from tests.c3_device_fakes import Trace, case, tensor_backend
    from examples.c3_qwen3.device_profile import ProfileRequest, profile_goal
    runtime, request, _ = case(tmp_path)
    tensor_backend(runtime)
    runtime.admission.ordinal = 3
    # When / Then: refusal records zero started windows and zero model work.
    path = profile_goal(ProfileRequest(request.stage, "single"), runtime, trace_factory=Trace)
    data = json.loads(path.read_text())["data"]
    assert not data["valid"]
    assert data["trace_windows"] == data["forward_calls"] == 0


def test_profile_when_shared_forward_class_has_distinct_sites(tmp_path) -> None:
    # Given: two same-class modules receive different real shapes in one model call.
    import numpy as np
    from tests.c3_device_fakes import Tensor, Trace, case
    from tests.c3_tiny_backend import Scale
    from examples.c3_qwen3.device_profile import ProfileRequest, profile_goal
    runtime, request, _ = case(tmp_path)
    backend, binding = runtime.runner.backend, runtime.runner.binding
    head = Scale()
    binding.modules["lm_head"] = head
    binding.originals["lm_head"] = head.forward
    binding.instance_forward["lm_head"] = True
    def forward(rows):
        backend.forward_calls += 1
        for history, row in zip(backend.rows, rows, strict=True):
            history.extend(row)
        hidden = backend.model.scale.forward(Tensor(np.repeat(np.asarray(rows, dtype=np.float32)[..., None], 3, axis=-1)))
        head.forward(Tensor(hidden.values[:, -1:, :]))
        return tuple((1., 0.) for _ in rows)
    backend.forward = forward
    # When: the profile observes both scopes instead of grouping by forward's source identity.
    path = profile_goal(ProfileRequest(request.stage, "ttft"), runtime, trace_factory=Trace)
    # Then: projection and head remain separate with full-prefix versus last-token shapes.
    calls = json.loads(path.read_text())["data"]["calls"]
    shapes = {c["module_path"]: c["tensors"][0]["shape"] for c in calls}
    assert shapes == {"scale": [1, 4096, 3], "lm_head": [1, 1, 3]}
