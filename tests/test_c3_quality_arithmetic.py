import importlib
import math
from collections.abc import Callable, Sequence

import pytest

from examples.c3_qwen3.runner_oracle import nll
from examples.c3_qwen3.runner_records import RunnerError
from tests.c3_quality_replay import Arithmetic, legacy_nll, python_calls


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
