"""Step 3's three arms: the resolved configs must differ in exactly the switches, and nothing else.

This is step 3's own stated success criterion, and it is a test rather than a review because the
three files are hand-written ~150-line documents describing the same experiment three times. A budget
or a tolerance that drifted between them would make the primary outcome unattributable with nothing
to notice -- all three files load, all three runs complete, and the difference only surfaces
afterwards as a number nobody can explain.

TOPOLOGY (changed 2026-09-12: box 2 was taken by another tenant after being powered down for the
clone, so both treatment arms moved to box 4, which has two 4090s):

    arm 1  control      mode: label    2e off   BOX 1
    arm 2  treatment 1  mode: vector   2e off   BOX 4, GPU 1
    arm 3  treatment 2  mode: vector   2e ON    BOX 4, GPU 0

Two arms now SHARE a box, which the project's standing rule forbade. The rule's reason was "each
would time the other's kernels" -- true of one card, and measured NOT to hold across two
(`probe_dual_card_interference.py`: worst median shift +1.20%, against a +-2-4% re-eval gap and a
4.72% within-arm spread). Sharing a box brings three new ways for the arms to contaminate each other
that no earlier pair could have, and none of them is caught by the diff assertions -- the values
legitimately DIFFER, so a diff cannot tell "correctly isolated" from "accidentally different". They
get their own explicit tests below.

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
ARM2 = CONFIGS / "experiments_s3_arm2_vector_box4gpu1.yaml"
ARM3 = CONFIGS / "experiments_s3_arm3_wall_box4gpu0.yaml"

_MODE = "v3.diagnosis.mode"
_WALL = frozenset({"v3.wall_attribution.enabled", "v3.wall_attribution.in_prompt"})
# Free-form dict, so `_MACHINE_KEYS` cannot cover it: XDG_DATA_HOME is a per-arm isolation value
# that legitimately differs between the two co-resident arms.
_XDG = "opencode.server_env.XDG_DATA_HOME"


def _load(p: Path):
    if not p.exists():
        pytest.skip(f"{p.name} is not present in this checkout")
    return load_config(p)


def test_arm1_and_arm2_differ_only_in_the_mode_switch_and_machine_paths():
    """Control vs treatment 1: `mode` alone, plus the paths that must differ because the boxes do."""
    a, b = _load(ARM1), _load(ARM2)
    diffs = set(_resolved_diff(a.model_dump(), b.model_dump()))
    unexpected = diffs - {_MODE, _XDG} - _MACHINE_KEYS
    assert not unexpected, (
        "arms 1 and 2 differ in something that is neither the mode switch nor a machine path, so a "
        "difference in the result could not be attributed to the switch: %s" % sorted(unexpected))
    assert _MODE in diffs, "arms 1 and 2 do not differ in `mode` -- arm 2 is not treating"


def test_arm2_and_arm3_differ_only_in_the_two_2e_switches():
    """Treatment 1 vs treatment 2: the 2e switches, plus the per-arm isolation values.

    This is the comparison A3 rests on, and it is the strictest of the three: both arms are `vector`,
    so if `mode` shows up in this diff the arm is testing two things at once.

    The two arms share a BOX, so the machine keys that may differ here are only the per-arm isolation
    ones (runs_dir, triton_cache_dir, XDG). Everything else about the machine -- venv, kernelbench
    paths, launch_cwd, sandbox config -- must be IDENTICAL, and that is asserted separately below;
    permitting `_MACHINE_KEYS` wholesale here would let a venv difference through.
    """
    b, c = _load(ARM2), _load(ARM3)
    diffs = set(_resolved_diff(b.model_dump(), c.model_dump()))
    allowed = _WALL | {_XDG, "run.runs_dir", "wsl.triton_cache_dir"}
    unexpected = diffs - allowed
    assert not unexpected, (
        "arms 2 and 3 differ beyond the 2e switches and the per-arm isolation values, so A3 would "
        "be confounded: %s" % sorted(unexpected))
    assert _MODE not in diffs, (
        "arms 2 and 3 differ in `mode` as well as in 2e -- the arm tests two variables at once")
    missing = _WALL - diffs
    assert not missing, "arm 3 does not actually turn 2e on: %s unchanged" % sorted(missing)


def test_the_two_co_resident_arms_agree_on_everything_about_the_machine():
    """Arms 2 and 3 are on the SAME box, so the venv and the kernelbench paths must match exactly.

    The venv is the trap this project keeps hitting: each box has two, the matched one is named
    differently on each box, and pointing an arm at the wrong one breaks the pairing while every
    other check still passes. Two arms on ONE box using different venvs would be that failure with
    no box difference to blame it on.
    """
    b, c = _load(ARM2), _load(ARM3)
    assert b.wsl.venv == c.wsl.venv, "co-resident arms use different venvs"
    assert b.wsl.kernelbench_src == c.wsl.kernelbench_src
    assert b.kernelbench_root == c.kernelbench_root
    assert b.opencode.launch_cwd == c.opencode.launch_cwd
    assert b.opencode.sandbox_config_path == c.opencode.sandbox_config_path


def test_the_two_co_resident_arms_isolate_runs_cache_and_opencode_state():
    """The three things that MUST differ between arms sharing a box, each for a measured reason.

    1. `runs_dir`: the GPU lock is `store.run_dir/jobs/gpu.lock` (wiring.py:109) -- per RUN, not per
       machine. Distinct run dirs are what keep two co-resident runs from sharing a lock, a jobs
       directory and an events log. (Distinct dirs are also what the harness needs anyway, since
       `RunStore.create` refuses an existing run dir.)
    2. `triton_cache_dir`: a shared cache lets the second arm hit the first arm's compilations, so
       `compile_s` and the wall clock stop being comparable -- and the wall clock is what ends every
       completed run in this project.
    3. `XDG_DATA_HOME`: `~/.local/share/opencode/opencode.db` is one SQLite file per box (180 MB on
       this one) and both arms' servers would write it.

    Asserted as INEQUALITY, which no diff test can do: the diff proves the arms differ somewhere,
    but a missing isolation value shows up as "one key fewer in the diff", which reads as MORE
    similar and therefore better.
    """
    b, c = _load(ARM2), _load(ARM3)
    assert b.run.runs_dir != c.run.runs_dir, (
        "co-resident arms share a runs_dir: they would share a GPU lock and an events log")
    assert b.wsl.triton_cache_dir != c.wsl.triton_cache_dir, (
        "co-resident arms share a triton cache: each would hit the other's compilations")
    x2 = b.opencode.server_env.get("XDG_DATA_HOME")
    x3 = c.opencode.server_env.get("XDG_DATA_HOME")
    assert x2 and x3, "a co-resident arm does not set XDG_DATA_HOME; both opencode servers would " \
                      "write one shared SQLite db"
    assert x2 != x3, "co-resident arms point XDG_DATA_HOME at the same directory"


def test_arm1_needs_no_xdg_isolation_but_must_not_collide_if_it_has_one():
    """Arm 1 is alone on box 1, so it needs no XDG override -- but if one is ever added it must not
    duplicate a box 4 value, because a reader comparing the three files would then see two arms
    apparently sharing state.
    """
    a, b, c = _load(ARM1), _load(ARM2), _load(ARM3)
    x1 = a.opencode.server_env.get("XDG_DATA_HOME")
    if x1 is not None:
        assert x1 not in {b.opencode.server_env.get("XDG_DATA_HOME"),
                          c.opencode.server_env.get("XDG_DATA_HOME")}


def test_all_three_arms_agree_on_device_budgets_evaluation_and_agents():
    """The four blocks that decide what is measured and when a run stops.

    `device:` is written verbatim into every agent sandbox's docs/device.md and is exposed to
    agent-authored constraint expressions, so arms disagreeing there are being told they are on
    different hardware -- and box 1 and box 4 are the same card, verified by identical resource-map
    digests over 162 configurations. Budgets decide when a run ends, and every completed L3 run so
    far was ended by the wall clock, so an hour of difference is not a rounding error.

    Compared as whole blocks rather than key by key, so a key added to any of them later is covered
    without editing this test.
    """
    cfgs = [(ARM1.name, _load(ARM1)), (ARM2.name, _load(ARM2)), (ARM3.name, _load(ARM3))]
    first_name, first = cfgs[0]
    for name, cfg in cfgs[1:]:
        for block in ("device", "budgets", "evaluation", "agents"):
            assert getattr(cfg, block).model_dump() == getattr(first, block).model_dump(), (
                f"{name} disagrees with {first_name} on `{block}:` -- the arms are not comparable")


def test_every_arm_keeps_the_output_token_ceiling():
    """`OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX` has no config-file route -- it is env-only -- and the
    first agent call on this model was truncated at 32000 tokens before it acted.

    Adding XDG_DATA_HOME to `server_env` means editing the same dict this lives in, so an arm could
    lose the ceiling while gaining isolation, and nothing else here would notice.
    """
    for p in (ARM1, ARM2, ARM3):
        env = _load(p).opencode.server_env
        assert env.get("OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"), (
            f"{p.name} lost the output-token ceiling; agent calls will truncate mid-action")


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
