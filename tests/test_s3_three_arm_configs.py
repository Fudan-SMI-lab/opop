"""Step 3's three arms: the resolved configs must differ in exactly the switches, and nothing else.

This is step 3's own stated success criterion, and it is a test rather than a review because the
three files are hand-written ~130-line documents describing the same experiment three times. A budget
or a tolerance that drifted between them would make the primary outcome unattributable with nothing
to notice -- all three files load, all three runs complete, and the difference only surfaces
afterwards as a number nobody can explain.

The three arms:

    arm 1  control      mode: label    2e off   box 1
    arm 2  treatment 1  mode: vector   2e off   box 2
    arm 3  treatment 2  mode: vector   2e ON    box 1, after arm 1

Reuses `_resolved_diff`, `_MACHINE_KEYS` and `load_config` from `test_config_strictness.py` rather
than restating them: a machine key added there must apply here too, and a second copy of the list
would drift.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_config_strictness import _MACHINE_KEYS, _resolved_diff, load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
ARM1 = CONFIGS / "experiments_s3_arm1_control_box1.yaml"
ARM2 = CONFIGS / "experiments_s3_arm2_vector_box2.yaml"
ARM3 = CONFIGS / "experiments_s3_arm3_wall_box1.yaml"

_MODE = "v3.diagnosis.mode"
_WALL = frozenset({"v3.wall_attribution.enabled", "v3.wall_attribution.in_prompt"})


def _load(p: Path):
    if not p.exists():
        pytest.skip(f"{p.name} is not present in this checkout")
    return load_config(p)


def test_arm1_and_arm2_differ_only_in_the_mode_switch_and_machine_paths():
    """Control vs treatment 1: `mode` alone, plus the paths that must differ because the boxes do."""
    a, b = _load(ARM1), _load(ARM2)
    diffs = set(_resolved_diff(a.model_dump(), b.model_dump()))
    unexpected = diffs - {_MODE} - _MACHINE_KEYS
    assert not unexpected, (
        "arms 1 and 2 differ in something that is neither the mode switch nor a machine path, so a "
        "difference in the result could not be attributed to the switch: %s" % sorted(unexpected))
    assert _MODE in diffs, "arms 1 and 2 do not differ in `mode` -- arm 2 is not treating"


def test_arm2_and_arm3_differ_only_in_the_two_2e_switches():
    """Treatment 1 vs treatment 2: the 2e switches alone, and NOT `mode`.

    This is the comparison A3 rests on, and it is the strictest of the three: both arms are `vector`,
    so if `mode` shows up in this diff the arm is testing two things at once. No machine keys are
    allowed either -- arms 1 and 3 share box 1, so arm 3's paths equal arm 1's, which are the ones
    that differ from box 2's. Any machine key appearing here is therefore expected and comes from the
    box difference, so it is permitted; `mode` is not.
    """
    b, c = _load(ARM2), _load(ARM3)
    diffs = set(_resolved_diff(b.model_dump(), c.model_dump()))
    unexpected = diffs - _WALL - _MACHINE_KEYS
    assert not unexpected, (
        "arms 2 and 3 differ beyond the 2e switches, so A3 would be confounded: %s"
        % sorted(unexpected))
    assert _MODE not in diffs, (
        "arms 2 and 3 differ in `mode` as well as in 2e -- the arm tests two variables at once")
    missing = _WALL - diffs
    assert not missing, "arm 3 does not actually turn 2e on: %s unchanged" % sorted(missing)


def test_arm1_and_arm3_share_the_box_so_no_machine_key_may_differ():
    """Arms 1 and 3 both run on box 1, sequentially. Their machine paths must be IDENTICAL.

    A different `runs_dir` or venv between them would mean arm 3 ran against a different toolchain
    than the control it is compared with -- the silent failure the venv trap produces on these boxes,
    where each has two venvs and the matched one is named differently on each.
    """
    a, c = _load(ARM1), _load(ARM3)
    diffs = set(_resolved_diff(a.model_dump(), c.model_dump()))
    machine_diffs = diffs & _MACHINE_KEYS
    assert not machine_diffs, (
        "arms 1 and 3 are on the SAME box but their machine paths differ: %s" % sorted(machine_diffs))
    unexpected = diffs - {_MODE} - _WALL
    assert not unexpected, (
        "arms 1 and 3 differ beyond mode + 2e: %s" % sorted(unexpected))


def test_all_three_arms_agree_on_device_budgets_evaluation_and_agents():
    """The four blocks that decide what is measured and when a run stops.

    `device:` is written verbatim into every agent sandbox's docs/device.md and is exposed to
    agent-authored constraint expressions, so arms disagreeing there are being told they are on
    different hardware -- and boxes 1 and 2 are the same card, verified before pairing them. Budgets
    decide when a run ends, and every completed L3 run so far was ended by the wall clock, so an hour
    of difference is not a rounding error.

    Compared as whole blocks rather than key by key, so a key added to any of them later is covered
    without editing this test.
    """
    cfgs = [(ARM1.name, _load(ARM1)), (ARM2.name, _load(ARM2)), (ARM3.name, _load(ARM3))]
    first_name, first = cfgs[0]
    for name, cfg in cfgs[1:]:
        for block in ("device", "budgets", "evaluation", "agents"):
            assert getattr(cfg, block).model_dump() == getattr(first, block).model_dump(), (
                f"{name} disagrees with {first_name} on `{block}:` -- the arms are not comparable")


def test_the_two_arms_without_2e_have_it_fully_off():
    """`enabled: false` AND `in_prompt: false` in arms 1 and 2.

    Checked separately from the diff because the diff only proves the arms DIFFER. If arm 2 had
    `enabled: true` with `in_prompt: false`, it would spend GPU time on probes that reach no prompt:
    the diff against arm 3 would still show one switch, and the wall clock the arms are compared on
    would silently differ.
    """
    for p in (ARM1, ARM2):
        wa = _load(p).v3.wall_attribution
        assert wa.enabled is False, f"{p.name} probes walls but is not the 2e arm"
        assert wa.in_prompt is False, f"{p.name} puts walls in the prompt but is not the 2e arm"


def test_no_step3_arm_turns_on_the_s2d_ledger():
    """S2d is E1's variable (step 4), not step 3's.

    Leaving it on in any arm would put two independent variables in one comparison -- and E1 measured
    that this switch is not cosmetic: the arms only separated once a ledger actually reached a
    rewriter (2 deliveries vs 0).
    """
    for p in (ARM1, ARM2, ARM3):
        assert _load(p).v3.diagnosis.expectation_ledger is False, (
            f"{p.name} enables the S2d ledger, which belongs to E1 and not to step 3")


def test_arm3_is_the_only_arm_reading_walls_into_a_prompt():
    """Exactly one of the three arms may have `in_prompt: true`.

    A guard against the failure mode where a later edit "fixes" a null result by turning the
    treatment on in more than one arm, which removes the contrast rather than strengthening it.
    """
    on = [p.name for p in (ARM1, ARM2, ARM3) if _load(p).v3.wall_attribution.in_prompt]
    assert on == [ARM3.name], f"expected only arm 3 to read walls into a prompt, got {on}"
