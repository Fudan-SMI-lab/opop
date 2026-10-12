"""The noise-aware acceptance rule and the full-model spread it reads.

These pin the two pieces the lane's new accept logic rests on, in isolation from the loop:
  * `meaningfully_beats` treats a gain inside `k x noise` as no win, in the score's own direction;
  * `score_waves` records `score_spread_frac` from the within-evaluation repeats, so the lane has a
    real full-model noise floor to compare against instead of the operator-level microbenchmark.
"""

from types import SimpleNamespace

import pytest

from examples.c3_qwen3.fast_acceptance import (FALLBACK_SPREAD_FRAC, NOISE_MULTIPLE,
                                               meaningfully_beats, score_spread_frac)
from examples.c3_qwen3.runner_measure import score_waves
from examples.c3_qwen3.runner_records import GoalSpec, RunnerError
from kernel_optimizer.tuning.objective import Objective


def wave(input_tokens, ready_offsets, accepted=0.0, warmup=False):
    """A minimal WaveRecord-shaped stand-in: one request, tokens ready at the given offsets."""
    n = len(ready_offsets)
    return SimpleNamespace(warmup=warmup, complete=True, tokens=((0,) * n,),
                           token_ready_s=tuple(accepted + o for o in ready_offsets),
                           accepted_s=accepted, finished_s=accepted + ready_offsets[-1],
                           input_tokens=(input_tokens,))


# --- meaningfully_beats -------------------------------------------------------------------------

def test_minimise_gain_must_clear_the_band():
    obj = Objective(direction="minimize", unit="ms")
    # baseline 282.82, band = 1 x 0.4% x 282.82 = 1.131 ms.
    assert not meaningfully_beats(obj, 282.82, 282.0, FALLBACK_SPREAD_FRAC)   # 0.82 ms gain, inside
    assert meaningfully_beats(obj, 282.82, 280.0, FALLBACK_SPREAD_FRAC)       # 2.82 ms gain, outside


def test_maximise_gain_must_clear_the_band():
    obj = Objective(direction="maximize", unit="tokens/s")
    assert not meaningfully_beats(obj, 21.243, 21.30, FALLBACK_SPREAD_FRAC)   # +0.3%, inside
    assert meaningfully_beats(obj, 21.243, 22.5, FALLBACK_SPREAD_FRAC)        # +5.9%, outside


def test_a_worse_candidate_never_beats():
    obj = Objective(direction="minimize", unit="ms")
    assert not meaningfully_beats(obj, 282.82, 285.0, 0.0)  # slower, even with a zero band


def test_the_multiple_scales_the_band():
    obj = Objective(direction="minimize", unit="ms")
    # A 1.5 ms gain against a 0.4% (1.131 ms) band clears at k=1 but not at k=2.
    assert meaningfully_beats(obj, 282.82, 281.32, FALLBACK_SPREAD_FRAC, multiple=1.0)
    assert not meaningfully_beats(obj, 282.82, 281.32, FALLBACK_SPREAD_FRAC, multiple=2.0)


# --- score_spread_frac fallback -----------------------------------------------------------------

def test_absent_evaluation_uses_the_fallback():
    assert score_spread_frac(None) == FALLBACK_SPREAD_FRAC


@pytest.mark.parametrize("bad", [None, -0.1, float("nan"), True])
def test_a_missing_or_invalid_spread_uses_the_fallback(bad):
    ev = SimpleNamespace(metrics={} if bad is None else {"score_spread_frac": bad})
    assert score_spread_frac(ev) == FALLBACK_SPREAD_FRAC


def test_a_real_spread_is_used():
    ev = SimpleNamespace(metrics={"score_spread_frac": 0.012})
    assert score_spread_frac(ev) == 0.012


# --- score_waves records the spread -------------------------------------------------------------

def test_ttft_records_the_half_range_spread():
    goal = GoalSpec(id="ttft", direction="minimize", unit="ms", requests=4, input_tokens=4096,
                    output_tokens=1, search_prompt_ids=(), final_prompt_groups=())
    # Four requests whose first-token latencies are 100/102/104/106 ms; median 103, half-range 3,
    # so the spread fraction is 3/103.
    waves = tuple(wave(4096, [x / 1000.0]) for x in (100.0, 102.0, 104.0, 106.0))
    ev = score_waves(goal, waves)
    assert ev.score == pytest.approx(103.0)
    assert ev.metrics["score_spread_frac"] == pytest.approx(3.0 / 103.0)


def test_multi_has_no_self_spread():
    goal = GoalSpec(id="multi", direction="maximize", unit="tokens/s", requests=8, input_tokens=4096,
                    output_tokens=128, search_prompt_ids=(), final_prompt_groups=())
    batch = SimpleNamespace(warmup=False, complete=True, tokens=tuple((0,) * 128 for _ in range(8)),
                            token_ready_s=tuple(x / 1000.0 for x in range(128)), accepted_s=0.0,
                            finished_s=2.0, input_tokens=(4096,) * 8)
    ev = score_waves(goal, (batch,))
    # A single-sample goal cannot estimate its own spread, so the field is absent and the acceptance
    # gate will fall back to the conservative floor.
    assert "score_spread_frac" not in ev.metrics
    assert score_spread_frac(ev) == FALLBACK_SPREAD_FRAC
