import importlib
import math
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pytest

from examples.c3_qwen3.runner_oracle import nll
from examples.c3_qwen3.runner_records import RunnerError
from tests.c3_quality_replay import ORACLE, Arithmetic, legacy_compare, legacy_nll, load_rows, python_calls


def comparison() -> Callable[[tuple[float, ...], Sequence[float], int], Arithmetic]:
    assert importlib.util.find_spec("examples.c3_qwen3.quality_arithmetic") is not None, "vectorized quality arithmetic absent"
    return importlib.import_module("examples.c3_qwen3.quality_arithmetic").compare_logits


@pytest.mark.parametrize("values,target", [((), 0), ((0.0,), -1), ((0.0,), 1),
    ((float("nan"),), 0), ((float("inf"),), 0), ((-float("inf"),), 0)])
def test_invalid_nll_when_logits_or_target_invalid(values: tuple[float, ...], target: int) -> None:
    # Given / When / Then: preserve the original explicit invalid boundary.
    with pytest.raises(RunnerError, match="invalid logits/target"):
        nll(values, target)


def test_nll_avoids_python_calls_per_vocabulary_element() -> None:
    # Given: deterministic vocabulary work; count calls rather than assert flaky elapsed time.
    values = (0.0,) * 4096
    # When: the real exported arithmetic runs under a diagnostic call observer.
    calls, value = python_calls(lambda: nll(values, 0))
    # Then: numerical result is preserved without per-element Python generator invocations.
    assert value == pytest.approx(math.log(4096), abs=1e-12)
    assert calls < 100, f"full-vocabulary Python work still present: {calls} calls"


@pytest.mark.parametrize("perturbation", [0.0, 0.009999, 0.010001])
def test_real_f32_equivalence_when_perturbation_crosses_l2_threshold(perturbation: float) -> None:
    # Given: both actual native prompts, all 32 positions and all 151936 vocabulary entries.
    compare = comparison()
    rows = load_rows()
    legacy = []
    current = []
    maximum_nll_error = 0.0
    maximum_metric_error = 0.0
    # When: compare a nonzero perturbation against the untouched float32 reference bytes.
    for row in rows:
        candidate = tuple((np.asarray(row.logits, dtype=np.float64) * (1 + perturbation)).tolist())
        expected = legacy_compare(candidate, row.logits, row.target)
        actual = compare(candidate, row.logits, row.target)
        for name in ("candidate_nll", "reference_nll", "difference2", "norm2"):
            assert getattr(actual, name) == pytest.approx(getattr(expected, name), rel=1e-12, abs=1e-10)
        legacy.append(expected)
        current.append(actual)
        maximum_nll_error = max(maximum_nll_error, abs(actual.candidate_nll - expected.candidate_nll),
                                abs(actual.reference_nll - expected.reference_nll))
    # Then: retain per-prompt pooled L2, paired NLL and the gate side, not mean row-wise errors.
    for offset in (0, 32):
        old, new = legacy[offset:offset + 32], current[offset:offset + 32]
        old_error = math.sqrt(sum(r.difference2 for r in old)) / max(math.sqrt(sum(r.norm2 for r in old)), 1e-12)
        new_error = math.sqrt(sum(r.difference2 for r in new)) / max(math.sqrt(sum(r.norm2 for r in new)), 1e-12)
        assert new_error == pytest.approx(old_error, abs=1e-10)
        assert (new_error <= .01) == (old_error <= .01)
        maximum_metric_error = max(maximum_metric_error, abs(new_error - old_error))
        assert math.fsum(r.candidate_nll - r.reference_nll for r in new) / 32 == pytest.approx(
            math.fsum(r.candidate_nll - r.reference_nll for r in old) / 32, abs=1e-10)
    print({"perturbation": perturbation, "maximum_nll_absolute_error": maximum_nll_error,
           "maximum_pooled_l2_absolute_error": maximum_metric_error})


def test_shape_mismatch_when_candidate_length_differs() -> None:
    # Given / When / Then: length mismatch remains invalid instead of NumPy broadcasting.
    with pytest.raises(RunnerError, match="shape mismatch"):
        comparison()((1.0,), (1.0, 2.0), 0)


@pytest.mark.parametrize("delta", [0.019999, 0.020001])
def test_nll_threshold_when_small_relative_change_affects_likelihood(delta: float) -> None:
    # Given: a common logit offset keeps relative L2 tiny while NLL crosses its own gate.
    reference = (1000.0, 1000.0)
    bias = math.log(2 * math.exp(delta) - 1)
    candidate = (1000.0, 1000.0 + bias)
    # When: paired likelihood is evaluated independently of the L2 gate.
    actual = comparison()(candidate, reference, 0)
    # Then: preserve both sides of the .02 threshold and the original double formula.
    assert actual.candidate_nll - actual.reference_nll == pytest.approx(delta, abs=1e-10)
    assert actual.candidate_nll == pytest.approx(legacy_nll(candidate, 0), abs=1e-10)
    assert (actual.candidate_nll - actual.reference_nll <= .02) == (delta <= .02)


@pytest.mark.parametrize("target", [False, True, 0.5])
def test_index_semantics_when_target_uses_python_index_protocol(target) -> None:
    # Given: preserve Python tuple indexing, including rejection of fractional indices.
    values = (1.0, 2.0)
    if target == 0.5:
        # When / Then: do not silently truncate a fractional target into a valid token.
        with pytest.raises(TypeError):
            nll(values, target)
    else:
        # When / Then: Boolean indices retain the original tuple-index behavior.
        assert nll(values, target) == pytest.approx(legacy_nll(values, target), abs=1e-12)


@pytest.mark.parametrize("invalid", [None, float("nan"), float("inf"), -float("inf")])
def test_public_quality_when_replaying_saved_logits(tmp_path: Path, invalid: float | None) -> None:
    # Given: the real runner and oracle, with CPU saved-logit replay instead of Qwen/GPU inference.
    from tests.test_c3_model_runner import runner
    resident, backend, _ = runner(tmp_path)
    rows = iter(load_rows())
    def cpu_logits(_logits):
        values = tuple(next(rows).logits)
        return ((invalid, *values[1:]),) if invalid is not None else (values,)
    backend.cpu_logits = cpu_logits
    ids = ("calibration-calibration-00", "calibration-calibration-01")
    # When: the public quality path retains active binding diagnostics and all arithmetic gates.
    report = resident.quality(ids, ORACLE)
    # Then: baseline replay is exact and nonfinite candidate data remains invalid.
    assert report.valid == (invalid is None)
    if invalid is None:
        assert backend.forward_calls == 64
        assert report.metrics["logits_relative_l2_max"] == 0.0
        assert report.metrics["paired_mean_nll_delta_nat_per_token"] == 0.0
    else:
        assert report.detail is not None
