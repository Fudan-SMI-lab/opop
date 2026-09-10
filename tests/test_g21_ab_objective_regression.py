"""The G21 failure mode recurring inside the A/B script itself.

Worth a test of its own because it is the clearest evidence that G21's real hazard was never the four
original call sites: `scripts/g9_information_ab.py` was written AFTER the authoritative reader existed,
by someone who had just spent a day on exactly this bug, and it still guessed
`robust_ms`/`median_ms`/`mean_ms`. All three read None against the stored
{max, mean, median, min, n, samples, std}, so all six measured candidates reported `ms: null` -- while
the numbers sat correctly in the run's own `jobs/*.out.json`.

Consequence: an A/B whose every arm has a null objective cannot be ranked. The measurement cost was
already paid; only the reporting threw it away.

So this pins the behaviour on the SHAPE THE WORKER ACTUALLY WRITES, verified from
run g9-20260910-144249 on box 1.
"""
from __future__ import annotations

from kernel_optimizer.store.read import latency_ms_of


def _worker_result(median: float = 1.5575, mean: float = 1.62) -> dict:
    """An evaluator result with the REAL key set, copied from a real `*.out.json`."""
    return {
        "ok": True,
        "latency_ms": {
            "max": median + 0.9, "mean": mean, "median": median,
            "min": median - 0.02, "n": None, "samples": [median] * 20,
            "std": 0.21,
        },
    }


def test_the_ab_objective_is_read_from_the_stored_keys():
    res = _worker_result()
    assert latency_ms_of(res) == 1.5575, (
        "the A/B objective is not being read from the stored `median` key, which is how six measured "
        "candidates all reported ms: null")


def test_the_three_guessed_names_are_absent_from_the_real_shape():
    """Guard the fixture itself: if it grew the suffixed keys, this test would pass on broken code."""
    lat = _worker_result()["latency_ms"]
    for guessed in ("robust_ms", "median_ms", "mean_ms"):
        assert guessed not in lat, (
            "%s appeared in the fixture, so this file would no longer detect the guess that caused "
            "the failure" % guessed)


def test_median_is_preferred_over_mean():
    """Not cosmetic: at n=20 the mean picks the faster of two configs 64.8% of the time, the median
    93.2%. Reading `mean` when `median` exists would make the A/B's ranking near a coin flip."""
    res = _worker_result(median=1.5575, mean=1.62)
    assert latency_ms_of(res) == 1.5575

    # With no median, the mean is the correct fallback rather than None.
    del res["latency_ms"]["median"]
    assert latency_ms_of(res) == 1.62


def test_a_result_with_no_latency_reads_as_none_not_zero():
    """A failed/untimed candidate must be distinguishable from a very fast one."""
    assert latency_ms_of({"ok": True, "latency_ms": {"n": 0}}) is None
    assert latency_ms_of({"ok": False}) is None
