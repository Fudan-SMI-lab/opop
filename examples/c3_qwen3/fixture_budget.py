"""The one place that decides how large a fixture may be.

Every cap here used to be a literal: 64 MiB on a single cloned call, 128 MiB on a snapshot and on the
resident set. Those numbers predate any measurement -- 128 MiB is 0.52% of a 4090 and 0.13% of an
A800, while the resident model whose operators we are capturing is already several GiB on the same
device. The effect was not safety but silent site exclusion: `model.layers.0.self_attn.o_proj` at the
ttft goal needs 3a+2r = 136 MiB and was refused inside `evaluate_local`'s source gate, so no
candidate for that site could ever produce a passing local report and the lane was structurally
incapable of accepting anything. Raising the literal would only move the arbitrary line, so the caps
are derived from the device's own free memory instead, which also makes them correct on a 24 GiB
4090, an 80 GiB A800 and CPU-only test doubles without a per-box edit.

The budget is read once per process and cached: `mem_get_info` moves as the resident model allocates,
and a cap that shrank mid-run would make the same fixture legal at capture time and illegal at
evaluation time -- the two checks must agree or a lane dies halfway with a contradiction on disk.
"""

import importlib
import os

from .runner_records import RunnerError

# What one fixture may claim of the free memory measured at first use. A fixture costs at most
# 3a+2r bytes (see `evaluate_local`) and this fraction bounds that total, not one term of it.
FRACTION = 0.10
# Below this the derivation is refused rather than silently applied: a box with under 2 GiB free
# cannot host the resident model the fixtures are captured from, so a tiny cap here would be a
# confusing second failure on top of the real one. It also carries the monotonicity guarantee --
# see MONOTONE below.
MINIMUM_FREE_BYTES = 2 * 1024 * 1024 * 1024
# The largest of the four literals this module replaces (the resident/snapshot cap; the per-call one
# was 64 MiB). Nothing that passed under the literals may be refused by the derivation.
HISTORICAL_CAP_BYTES = 128 * 1024 * 1024
# Monotonicity is structural rather than clamped: the smallest budget the CUDA path can return is
# MINIMUM_FREE_BYTES * FRACTION, and that already exceeds every literal replaced here. A `max(...)`
# against the cap would read like a safeguard while being unreachable, so the invariant is asserted
# at import instead -- lowering FRACTION or MINIMUM_FREE_BYTES then fails loudly rather than quietly
# reintroducing the site exclusions this module exists to remove.
MONOTONE = int(MINIMUM_FREE_BYTES * FRACTION) >= HISTORICAL_CAP_BYTES
if not MONOTONE:
    raise RunnerError("fixture budget policy would refuse configurations the literals admitted: "
                      f"{int(MINIMUM_FREE_BYTES * FRACTION)} < {HISTORICAL_CAP_BYTES}")
# Used when no CUDA device is visible (unit tests, CPU-only hosts). It keeps the old behaviour
# exactly: the fake codecs in the test suite assert against the 128 MiB boundary.
CPU_BUDGET_BYTES = HISTORICAL_CAP_BYTES
# Escape hatch for a box whose free memory is not a good proxy -- for instance a shared host where
# another tenant is about to allocate. Set it to a byte count to pin the budget.
OVERRIDE_ENV = "C3_FIXTURE_BUDGET_BYTES"

_cached: int | None = None


def _derive() -> int:
    override = os.environ.get(OVERRIDE_ENV)
    if override:
        try:
            pinned = int(override)
        except ValueError:
            raise RunnerError(f"{OVERRIDE_ENV} must be a byte count") from None
        if pinned < HISTORICAL_CAP_BYTES:
            raise RunnerError(f"{OVERRIDE_ENV} must be at least {HISTORICAL_CAP_BYTES} bytes")
        return pinned
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return CPU_BUDGET_BYTES
    if not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return CPU_BUDGET_BYTES
    free, _total = torch.cuda.mem_get_info()
    if free < MINIMUM_FREE_BYTES:
        raise RunnerError(f"fixture budget needs at least {MINIMUM_FREE_BYTES} bytes free; saw {free}")
    return int(free * FRACTION)


def fixture_budget_bytes() -> int:
    """The cap on one fixture's total bytes, measured once and then constant for the process."""
    global _cached
    if _cached is None:
        _cached = _derive()
    return _cached


def budget_evidence() -> dict[str, int | float | str | None]:
    """What the budget was derived from, for the receipt. A reader must never have to guess."""
    return {"budget_bytes": fixture_budget_bytes(), "fraction": FRACTION,
            "historical_cap_bytes": HISTORICAL_CAP_BYTES, "override": os.environ.get(OVERRIDE_ENV)}


def require_within_budget(actual: int, what: str) -> None:
    budget = fixture_budget_bytes()
    if actual > budget:
        raise RunnerError(f"unsupported fixture: {what} {actual} exceeds the "
                          f"{budget}-byte device fixture budget")
