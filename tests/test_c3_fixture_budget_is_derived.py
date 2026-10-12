"""The fixture caps are derived from the device, not written down.

Four separate literals used to decide which operator sites a goal could be evaluated on: 64 MiB on a
single cloned call (`TorchFixtureCodec.clone`, `TorchPureTools.require_pure`), 128 MiB on a capture
snapshot (`device_capture`), 128 MiB on the resident set (`binding_runtime`) and 128 MiB on snapshots
plus working bytes (`local_eval`). None was measured. On box4 the consequence was a lane that could
not accept any candidate: `self_attn.o_proj` at the ttft goal costs 3a+2r = 136 MiB, so every
`evaluate_local` call died in the source gate before the device was even touched, while the same
fixture had already been captured successfully by a path using a different formula.

These tests pin the properties the derivation must have, not the number it produces:
  * it never refuses what the literals admitted -- structurally, because the smallest budget the
    CUDA path can return already exceeds every literal replaced here, not by a clamp,
  * every one of the four call sites consults the same budget, so two gates cannot disagree,
  * it is derived from free device memory and can be pinned by an operator,
  * the sites that were structurally excluded become admissible, and the arithmetic that excluded
    them is reproduced here from the real tensor shapes rather than trusted.
"""

import ast
import importlib
from pathlib import Path

import pytest

from examples.c3_qwen3 import fixture_budget
from examples.c3_qwen3.fixture_budget import (
    CPU_BUDGET_BYTES, FRACTION, HISTORICAL_CAP_BYTES, MINIMUM_FREE_BYTES, MONOTONE, OVERRIDE_ENV,
    budget_evidence, fixture_budget_bytes, require_within_budget,
)
from examples.c3_qwen3.runner_records import RunnerError

MiB = 1024 * 1024
SOURCE = Path(__file__).resolve().parents[1] / "examples/c3_qwen3"
# Every site that must route its capacity decision through the shared budget.
GATED_FILES = ("torch_fixtures.py", "pure_tensors.py", "device_capture.py",
               "local_eval.py", "binding_runtime.py", "fast_setup.py")


@pytest.fixture(autouse=True)
def _uncached(monkeypatch):
    """The budget caches on first use; each test derives it again from a known environment."""
    monkeypatch.setattr(fixture_budget, "_cached", None)
    monkeypatch.delenv(OVERRIDE_ENV, raising=False)
    yield
    monkeypatch.setattr(fixture_budget, "_cached", None)


class FakeCuda:
    def __init__(self, free: int, *, available: bool = True, initialized: bool = True) -> None:
        self.free, self._available, self._initialized = free, available, initialized

    def is_available(self) -> bool:
        return self._available

    def is_initialized(self) -> bool:
        return self._initialized

    def mem_get_info(self) -> tuple[int, int]:
        return self.free, self.free * 2


def with_free(monkeypatch, free: int, **kwargs):
    fake = type("FakeTorch", (), {"cuda": FakeCuda(free, **kwargs)})()
    monkeypatch.setattr(importlib, "import_module", lambda name: fake if name == "torch" else importlib.__dict__)
    return fake


# --- the derivation itself ---------------------------------------------------------------------

def test_budget_is_a_fraction_of_measured_free_memory(monkeypatch):
    # Given: a box reporting 14633 MiB free, which is what box4 measured while two lanes ran.
    with_free(monkeypatch, 14633 * MiB)
    # Then: the cap is that fraction of what is actually free, not a written-down constant.
    assert fixture_budget_bytes() == int(14633 * MiB * FRACTION)
    assert fixture_budget_bytes() > HISTORICAL_CAP_BYTES


def test_the_policy_cannot_refuse_what_the_literals_admitted(monkeypatch):
    # The monotonicity guarantee is structural, not a clamp: the smallest budget the CUDA path can
    # return is the refusal threshold times the fraction, and that must already exceed every literal
    # replaced here. A `max(floor, ...)` would have looked like a safeguard while being unreachable.
    assert MONOTONE and int(MINIMUM_FREE_BYTES * FRACTION) >= HISTORICAL_CAP_BYTES
    with_free(monkeypatch, MINIMUM_FREE_BYTES)
    assert fixture_budget_bytes() >= HISTORICAL_CAP_BYTES


def test_tiny_free_memory_is_refused_rather_than_silently_tiny(monkeypatch):
    with_free(monkeypatch, MINIMUM_FREE_BYTES - 1)
    with pytest.raises(RunnerError, match="at least"):
        fixture_budget_bytes()


@pytest.mark.parametrize("kwargs", [{"available": False}, {"initialized": False}])
def test_without_a_live_cuda_device_the_budget_is_the_historical_cap(monkeypatch, kwargs):
    # Given: a CPU-only host or a process that has not initialized CUDA.
    with_free(monkeypatch, 80 * 1024 * MiB, **kwargs)
    # Then: behaviour is exactly what it was before this change.
    assert fixture_budget_bytes() == CPU_BUDGET_BYTES == HISTORICAL_CAP_BYTES


def test_absent_torch_is_the_historical_cap(monkeypatch):
    def no_torch(name):
        raise ImportError(name)
    monkeypatch.setattr(importlib, "import_module", no_torch)
    assert fixture_budget_bytes() == CPU_BUDGET_BYTES


def test_operator_can_pin_the_budget(monkeypatch):
    # Given: a shared host where free memory is not a good proxy for what we may claim.
    with_free(monkeypatch, 80 * 1024 * MiB)
    monkeypatch.setenv(OVERRIDE_ENV, str(300 * MiB))
    assert fixture_budget_bytes() == 300 * MiB


@pytest.mark.parametrize("value", ["not-a-number", "", "12.5"])
def test_a_malformed_override_fails_closed(monkeypatch, value):
    with_free(monkeypatch, 80 * 1024 * MiB)
    monkeypatch.setenv(OVERRIDE_ENV, value)
    if value == "":
        assert fixture_budget_bytes() == int(80 * 1024 * MiB * FRACTION)  # empty = unset
    else:
        with pytest.raises(RunnerError, match=OVERRIDE_ENV):
            fixture_budget_bytes()


def test_an_override_below_the_floor_is_refused(monkeypatch):
    # Otherwise an operator could quietly make the framework stricter than it has ever been and
    # reproduce exactly the silent site exclusion this change exists to remove.
    with_free(monkeypatch, 80 * 1024 * MiB)
    monkeypatch.setenv(OVERRIDE_ENV, str(HISTORICAL_CAP_BYTES - 1))
    with pytest.raises(RunnerError, match="at least"):
        fixture_budget_bytes()


def test_the_budget_is_measured_once_so_two_gates_cannot_disagree(monkeypatch):
    # Given: free memory that falls as the resident model allocates -- the real situation.
    fake = with_free(monkeypatch, 14633 * MiB)
    first = fixture_budget_bytes()
    fake.cuda.free = MINIMUM_FREE_BYTES
    # Then: a fixture admitted at capture time is still admissible at evaluation time.
    assert fixture_budget_bytes() == first


def test_the_refusal_names_the_budget_that_refused_it(monkeypatch):
    with_free(monkeypatch, 14633 * MiB)
    budget = fixture_budget_bytes()
    with pytest.raises(RunnerError, match=f"{budget}-byte") as raised:
        require_within_budget(budget + 1, "snapshots plus working input/output")
    assert "snapshots plus working input/output" in str(raised.value)
    require_within_budget(budget, "exactly at the budget")  # the boundary is inclusive


def test_evidence_records_what_the_budget_was_derived_from(monkeypatch):
    with_free(monkeypatch, 14633 * MiB)
    evidence = budget_evidence()
    assert evidence == {"budget_bytes": int(14633 * MiB * FRACTION), "fraction": FRACTION,
                        "historical_cap_bytes": HISTORICAL_CAP_BYTES, "override": None}


# --- the four call sites all consult it -------------------------------------------------------

def test_no_capacity_literal_survives_outside_the_budget_module():
    # A literal left behind in one gate is exactly the defect that made the proof gate re-reject
    # candidates the floor had admitted: two places deciding, one of them silently.
    offenders = {}
    for name in GATED_FILES:
        text = (SOURCE / name).read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(text)):
            if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult)
                    and {getattr(n, "value", None) for n in ast.walk(node) if isinstance(n, ast.Constant)} & {1024}):
                literal = ast.unparse(node)
                if "1024 * 1024" in literal:
                    offenders.setdefault(name, []).append(literal)
    assert offenders == {}


def test_every_gated_file_imports_the_shared_budget():
    for name in GATED_FILES:
        tree = ast.parse((SOURCE / name).read_text(encoding="utf-8"))
        imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                    and node.module == "fixture_budget" for alias in node.names}
        assert imported & {"require_within_budget"}, f"{name} decides capacity without the shared budget"


def test_the_budget_module_is_part_of_the_framework_identity():
    # A campaign frozen under one budget policy must not silently continue under another.
    text = (SOURCE / "fast_artifacts.py").read_text(encoding="utf-8")
    assert '"fixture_budget"' in text and '"torch_fixtures"' in text


# --- what this unlocks, computed from the real shapes -----------------------------------------

# Qwen3-4B, bf16. `a` is the operator's input bytes and `r` its output bytes at the goal's token
# count; both were read back from real receipts on box4 (o_proj ttft: a=32 MiB, r=20 MiB, and the
# recorded resident_bytes of 88080384 equals 2a+r to the byte).
SITES = {"model.norm": (2560, 2560), "q_proj": (2560, 4096), "o_proj": (4096, 2560),
         "up_proj": (2560, 9728), "down_proj": (9728, 2560)}


def costs(site: str, tokens: int) -> dict[str, int]:
    a_last, r_last = SITES[site]
    a, r = tokens * a_last * 2, tokens * r_last * 2
    return {"call": a, "snapshot": 2 * a + r, "clone_of_result": a + r, "snapshot_plus_working": 3 * a + 2 * r}


def test_the_recorded_resident_bytes_match_the_snapshot_formula():
    # Guards the arithmetic these tests rely on against the receipts actually on disk.
    assert costs("o_proj", 4096)["snapshot"] == 88080384
    assert costs("o_proj", 256)["snapshot"] == 5505024


@pytest.mark.parametrize("site", ["o_proj", "up_proj", "down_proj"])
def test_the_structurally_excluded_sites_were_refused_by_the_literals(site):
    # Given: the caps as they were. Then: each of the three largest sites was refused, each by a
    # different one of the four gates -- which is why raising only one of them was not enough.
    cost = costs(site, 4096)
    assert (cost["call"] > 64 * MiB or cost["clone_of_result"] > 64 * MiB
            or cost["snapshot"] > 128 * MiB or cost["snapshot_plus_working"] > 128 * MiB)


@pytest.mark.parametrize("site", list(SITES))
def test_every_site_is_admissible_under_a_budget_derived_from_a_24gib_card(monkeypatch, site):
    # Given: box4's measured free memory with both lanes resident.
    with_free(monkeypatch, 14633 * MiB)
    # Then: no gate refuses any candidate site, including the one ranked first by CUDA time.
    for what, value in costs(site, 4096).items():
        require_within_budget(value, what)


def test_the_largest_site_still_costs_a_small_share_of_the_card(monkeypatch):
    # The point is not that the budget is generous; it is that it is proportionate.
    with_free(monkeypatch, 14633 * MiB)
    worst = costs("down_proj", 4096)["snapshot_plus_working"]
    assert worst / (14633 * MiB) < 0.02
    assert worst <= fixture_budget_bytes()


def test_a_site_far_beyond_the_budget_is_still_refused(monkeypatch):
    # The derivation must keep refusing things, or it is not a guard. act_fn at ttft needs 608 MiB
    # of snapshot+working, which a box with 4 GiB free must not admit.
    with_free(monkeypatch, 4 * 1024 * MiB)
    with pytest.raises(RunnerError, match="exceeds"):
        require_within_budget(608 * MiB, "snapshots plus working input/output")
