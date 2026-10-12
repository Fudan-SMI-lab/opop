"""The matrix lane must register each optimised row's site before it binds anything.

WHAT THIS PROTECTS AGAINST -- MEASURED, NOT HYPOTHETICAL
--------------------------------------------------------
Campaign `F4-campaign-0921-053151` produced an accepted `ttft` row (273.433 ms against a 282.818 ms
baseline, device-proven on 6/6 fixtures) and then lost **all nine of its matrix cells** to:

    RunnerError: unresolved traced site: pure-727f184a931d6918df621c6385fe123d7dd248cb470c207fd...

`ModelBinding.bind` raises that when a bundle declares a site the binding was never told about, and
only `register_site` tells it. Every other lane registers through `FastEvaluator.capture`; the matrix
lane has no capture step -- the sealed phase refuses local feedback outright -- so `run_matrix` has to
register the sites itself.

The failure mode is nasty because it is SILENTLY ASYMMETRIC: baseline rows bind a zero-site bundle and
pass, so the table renders, is labelled "partial", and shows the optimised row as unmeasurable. That
reads like a kernel that could not be evaluated rather than a missing registration, which is exactly
the misreading these tests exist to prevent.

These drive the real `register_row_sites` against the production `ModelBinding`; only the module tree
is a stand-in, because what is under test is which sites get registered, not what the modules compute.
"""

from pathlib import Path

import pytest

from examples.c3_qwen3.fast_matrix import capture_original_fixtures, register_row_sites
from examples.c3_qwen3.fast_records import Framework, FrozenFramework, TargetSpec
from examples.c3_qwen3.runner_records import RunnerError

RESULTS = Path(__file__).parents[1] / "results/c3-fast-kernel-iteration"


def native_target() -> TargetSpec:
    """A `TargetSpec` written by the native `build_target`, whichever campaign tree holds one."""
    for candidate in sorted(RESULTS.glob("**/target.json")):
        return TargetSpec.model_validate_json(candidate.read_bytes())
    raise AssertionError(f"no native target.json anywhere under {RESULTS}")


class FakeBinding:
    """The registration surface of `ModelBinding`, with its real rules.

    `register_site` is reproduced rather than imported because the production class needs a resident
    torch model. Its three refusals -- unknown module, duplicate site ID, module already in another
    group -- are what the caller has to respect, so they are enforced here too.
    """

    def __init__(self, modules: tuple[str, ...]) -> None:
        self.modules = {path: object() for path in modules}
        self.sites: dict[str, tuple[str, ...]] = {}
        self.bundle = None

    def register_site(self, site_id: str, module_paths: tuple[str, ...]) -> None:
        if self.bundle is not None or not module_paths or site_id in self.sites:
            raise RunnerError("register unique nonempty sites before binding")
        occupied = {p for paths in self.sites.values() for p in paths}
        if any(p not in self.modules or p in occupied for p in module_paths):
            raise RunnerError("site module missing or belongs to another group")
        self.sites[site_id] = module_paths


class FakeRunner:
    def __init__(self, modules: tuple[str, ...]) -> None:
        self.binding = FakeBinding(modules)


def frozen_with(targets: dict) -> FrozenFramework:
    """A `FrozenFramework` carrying these targets.

    Only `targets` is read by `register_row_sites`, and no frozen-framework record is mirrored into
    the local results tree, so the surrounding fields are filled with placeholders rather than making
    these tests depend on an artifact that only exists on the GPU host.
    """
    return FrozenFramework(
        framework=Framework(revision="test", implementation_sha256="0" * 64, schema_sha256="0" * 64,
                            prompt_sha256="0" * 64, agent_config_sha256="0" * 64,
                            contract_sha256="0" * 64),
        development_result=Path("development/result.json"), development_result_sha256="0" * 64,
        targets=targets)


def test_an_accepted_row_gets_its_site_registered_from_the_frozen_target() -> None:
    # Given: a ttft lane that accepted a candidate, and the frozen target that says which 73 modules
    # the candidate's site_id covers.
    target = native_target()
    runner = FakeRunner(target.brief.module_paths)
    frozen = frozen_with({"ttft": target})

    # When: the matrix lane registers its rows' sites.
    registered = register_row_sites(runner, frozen, {"ttft": (object(), ())})

    # Then: the site the bundle will declare is known to the binding, with exactly the frozen
    # module set -- which is what stops `bind` raising "unresolved traced site".
    assert registered == (target.brief.site_id,)
    assert runner.binding.sites[target.brief.site_id] == target.brief.module_paths


def test_three_goals_sharing_one_site_register_it_once() -> None:
    # Given: this campaign's three goals all resolved to the SAME site group, and `register_site`
    # refuses a duplicate ID. Registering per row without checking would raise on the second goal.
    target = native_target()
    runner = FakeRunner(target.brief.module_paths)
    frozen = frozen_with({goal: target for goal in ("ttft", "single", "multi")})

    registered = register_row_sites(
        runner, frozen, {goal: (object(), ()) for goal in ("ttft", "single", "multi")})

    # Then: registered once, and the shared registration is accepted rather than treated as a clash.
    assert registered == (target.brief.site_id,)
    assert len(runner.binding.sites) == 1


def test_a_baseline_fallback_row_registers_nothing() -> None:
    # Given: two lanes fell back to baseline (measured: `single` and `multi` in this campaign). Those
    # rows bind a zero-site bundle, so registering a site for them would be registering a group
    # nothing will ever bind.
    target = native_target()
    runner = FakeRunner(target.brief.module_paths)
    frozen = frozen_with({goal: target for goal in ("ttft", "single", "multi")})

    registered = register_row_sites(runner, frozen, {"ttft": (object(), ())})

    assert registered == (target.brief.site_id,)
    assert len(runner.binding.sites) == 1


def test_an_accepted_row_without_a_frozen_target_is_refused_before_the_model_loads() -> None:
    # Given: a row that has an accepted candidate but no frozen target to site it. Its bundle would
    # declare a site nothing can resolve, so failing here -- before 14 GB of weights -- is the point.
    target = native_target()
    runner = FakeRunner(target.brief.module_paths)
    frozen = frozen_with({"single": target})

    with pytest.raises(RunnerError, match="no frozen target"):
        register_row_sites(runner, frozen, {"ttft": (object(), ())})


def test_registration_is_refused_once_a_bundle_is_bound() -> None:
    # Given: `register_site` refuses after a bind ("register unique nonempty sites before binding"),
    # and the matrix lane binds the baseline while generating its reference oracles. So the
    # registration has to happen BEFORE `final_oracles` -- and, because `attach_fast` calls
    # `sites.clear()`, AFTER `attach_fast`. This pins the second half of that sandwich; getting it
    # wrong the other way silently discarded the registration and cost a whole row nine cells.
    target = native_target()
    runner = FakeRunner(target.brief.module_paths)
    runner.binding.bundle = object()  # as `final_oracles` leaves it
    frozen = frozen_with({"ttft": target})

    with pytest.raises(RunnerError, match="before binding"):
        register_row_sites(runner, frozen, {"ttft": (object(), ())})


def test_clearing_sites_discards_a_registration_so_order_matters() -> None:
    # Given: `attach_fast` clears `binding.sites` when it swaps the prepared task. This is the exact
    # mechanism that made the first fix ineffective, so it is pinned as behaviour rather than left
    # as a comment: register, clear, and the site is gone.
    target = native_target()
    runner = FakeRunner(target.brief.module_paths)
    frozen = frozen_with({"ttft": target})

    register_row_sites(runner, frozen, {"ttft": (object(), ())})
    assert runner.binding.sites  # registered

    runner.binding.sites.clear()  # what attach_fast does
    assert not runner.binding.sites, "a registration made before attach_fast does not survive it"

    # And registering again afterwards works, which is why the call belongs after attach_fast.
    assert register_row_sites(runner, frozen, {"ttft": (object(), ())}) == (target.brief.site_id,)


def test_frozen_targets_disagreeing_about_a_shared_site_is_an_error() -> None:
    # Given: two goals whose targets claim the same site_id but different module sets. Silently
    # keeping the first would measure one row against the other's group.
    target = native_target()
    fewer = target.model_copy(update={
        "brief": target.brief.model_copy(update={"module_paths": target.brief.module_paths[:4]})})
    runner = FakeRunner(target.brief.module_paths)
    frozen = frozen_with({"ttft": target, "single": fewer})

    with pytest.raises(RunnerError, match="disagree about the modules"):
        register_row_sites(runner, frozen, {"ttft": (object(), ()), "single": (object(), ())})


class FixtureRunner:
    """The `capture_fixtures` surface, with the prompt-length rule the production runner enforces."""

    def __init__(self, prompt_lengths: dict[str, int], produce: int = 2) -> None:
        self.prepared = type("P", (), {"prompts_by_id": {
            pid: type("Q", (), {"input_ids": tuple(range(n))})() for pid, n in prompt_lengths.items()}})()
        self.binding = type("B", (), {"fixtures": {}, "restore": lambda self: None})()
        self.captured: tuple[str, ...] = ()
        self._produce = produce

    def capture_fixtures(self, prompt_ids: tuple[str, ...]) -> None:
        for pid in prompt_ids:
            if len(self.prepared.prompts_by_id[pid].input_ids) != 64:
                raise RunnerError("local fixtures use the frozen 64-token calibration inputs")
        self.captured = prompt_ids
        self.binding.fixtures = {(f"site-{i}", "prefill"): object() for i in range(self._produce)}


def test_original_fixtures_are_captured_from_the_64_token_calibration_prompts() -> None:
    # Given: the sealed corpus holds the two frozen 64-token calibration prompts alongside its
    # held-out blocks (verified on the real corpus: 58 prompts, both calibration ids present).
    runner = FixtureRunner({"calibration-calibration-00": 64, "calibration-calibration-01": 64,
                            "heldout-ttft-00": 4096, "heldout-multi-00": 512})

    captured = capture_original_fixtures(runner, {"ttft": (object(), ())})

    # Then: only the 64-token prompts are used -- a held-out 4096-token prompt would be refused by
    # `capture_fixtures`, and using held-out data for a calibration capture would be wrong anyway.
    assert set(runner.captured) == {"calibration-calibration-00", "calibration-calibration-01"}
    assert captured == 2


def test_no_fixtures_are_captured_when_every_row_is_a_baseline_fallback() -> None:
    # Given: a matrix where no lane accepted anything. Nothing binds a site, so the per-operator
    # check is never reached and a capture would be pure cost.
    runner = FixtureRunner({"calibration-calibration-00": 64})

    assert capture_original_fixtures(runner, {}) == 0
    assert runner.captured == ()


def test_a_corpus_without_a_calibration_prompt_fails_loudly() -> None:
    # Given: a sealed corpus with no 64-token prompt. Proceeding would fail every optimised cell
    # individually with a message about fixtures -- exactly how this defect first presented -- so it
    # has to fail once, here, before the measurements start.
    runner = FixtureRunner({"heldout-ttft-00": 4096})

    with pytest.raises(RunnerError, match="no 64-token calibration prompt"):
        capture_original_fixtures(runner, {"ttft": (object(), ())})


def test_a_capture_that_produces_nothing_fails_loudly() -> None:
    # Given: the prompts exist but the capture yields no fixture. A silent zero here means the
    # per-operator check rejects every optimised row later, for a reason that looks like the kernel.
    runner = FixtureRunner({"calibration-calibration-00": 64}, produce=0)

    with pytest.raises(RunnerError, match="produced nothing"):
        capture_original_fixtures(runner, {"ttft": (object(), ())})

    # Given: two goals whose targets claim the same site_id but different module sets. Silently
    # keeping the first would measure one row against the other's group.
    target = native_target()
    fewer = target.model_copy(update={
        "brief": target.brief.model_copy(update={"module_paths": target.brief.module_paths[:4]})})
    runner = FakeRunner(target.brief.module_paths)
    frozen = frozen_with({"ttft": target, "single": fewer})

    with pytest.raises(RunnerError, match="disagree about the modules"):
        register_row_sites(runner, frozen, {"ttft": (object(), ()), "single": (object(), ())})
