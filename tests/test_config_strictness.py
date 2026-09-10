"""A mistyped config key must FAIL, not silently leave the switch at its default.

Written because the control run for S2/S2d was about to be launched from the first `v3:` block
this project has ever put in a YAML file, and the YAML->pydantic path for those switches had
never been exercised. Measured before the fix: `v3.diagnosis.modes: vector` validated cleanly
and left `mode="label"`. The treatment arm would have been a second control arm, the two 12 h
runs would have agreed, and the recorded conclusion would have been "the vector shape changes
nothing".

Every test here drives the real `load_config` (D-5) rather than constructing models by hand: the
defect lives in the composition of `yaml.safe_load` -> `_apply_override` -> `model_validate`, and
a test that validates a model directly would pass on the broken version.
"""

from __future__ import annotations

import pytest
import yaml
from pydantic import BaseModel, ValidationError

from kernel_optimizer.config import AppConfig, StrictConfig, load_config


def _write(tmp_path, body: str):
    p = tmp_path / "c.yaml"
    p.write_text(body, encoding="utf-8")
    return p


# --- the switch actually arrives ------------------------------------------------------------


def test_a_correctly_spelled_v3_block_reaches_the_config(tmp_path):
    """The positive control. Without this, every test below could pass on a config loader that
    rejects EVERYTHING, which would be a different defect with the same test outcome."""
    cfg = load_config(_write(tmp_path, """
v3:
  diagnosis:
    mode: vector
    expectation_ledger: true
"""))
    assert cfg.v3.diagnosis.mode == "vector"
    assert cfg.v3.diagnosis.expectation_ledger is True


def test_an_omitted_v3_block_is_the_control_arm(tmp_path):
    """`load_config` reads ONE file with no base layer, so the control arm is the absence of the
    block -- not a block that spells the defaults out. Asserted so that the two arms of the run
    are known to differ in exactly the two keys above."""
    cfg = load_config(_write(tmp_path, "run:\n  seed: 0\n"))
    assert cfg.v3.diagnosis.mode == "label"
    assert cfg.v3.diagnosis.expectation_ledger is False
    assert cfg.v3.search.deweight_unconditional_failures is False
    assert cfg.v3.search.declare_infeasible_out_of_space is False
    assert cfg.v3.diagnosis.access_pattern_walls is False


# --- the defect itself ---------------------------------------------------------------------


@pytest.mark.parametrize("body,typo", [
    ("v3:\n  diagnosis:\n    modes: vector\n", "modes"),
    ("v3:\n  diagnosis:\n    mode_: vector\n", "mode_"),
    ("v3:\n  diagnosis:\n    expectation_ledgers: true\n", "expectation_ledgers"),
    ("v3:\n  diagnosic:\n    mode: vector\n", "diagnosic"),
    ("v3:\n  serach:\n    deweight_unconditional_failures: true\n", "serach"),
    ("v33:\n  diagnosis:\n    mode: vector\n", "v33"),
])
def test_a_mistyped_switch_key_is_rejected_not_dropped(tmp_path, body, typo):
    """Each of these validated cleanly before the fix and left the switch OFF.

    The assertion is on the ERROR, not merely on `raises`: the message has to name the key, or a
    reader who typed `modes` is told only that the config is invalid and has to bisect a file.
    """
    with pytest.raises(ValidationError) as exc:
        load_config(_write(tmp_path, body))
    assert typo in str(exc.value)


def test_a_mistyped_key_in_a_block_that_is_otherwise_correct_is_still_rejected(tmp_path):
    """The dangerous shape: `mode` is right, so the block LOOKS like it took effect, and the
    second switch is silently off. This is the one that produces a half-treatment arm -- vector
    prompts with no ledger -- which is neither of the two arms the run is meant to compare."""
    with pytest.raises(ValidationError) as exc:
        load_config(_write(tmp_path, """
v3:
  diagnosis:
    mode: vector
    expectation_ledgers: true
"""))
    assert "expectation_ledgers" in str(exc.value)


def test_a_wrong_value_for_a_correct_key_is_also_rejected(tmp_path):
    """`mode` is a closed Literal, so a plausible-but-wrong VALUE ("vectors", "Vector") must fail
    too. Forbidding unknown keys does nothing about this axis, and the failure looks identical
    from the outside: the run proceeds on `label`."""
    for bad in ("vectors", "Vector", "VECTOR", "true"):
        with pytest.raises(ValidationError):
            load_config(_write(tmp_path, "v3:\n  diagnosis:\n    mode: %s\n" % bad))


# --- the CLI path, which builds the same dict a different way -------------------------------


def test_a_mistyped_dotted_override_is_rejected(tmp_path):
    """`_apply_override` walks with `setdefault`, so it CREATES whatever path it is given -- it
    cannot reject a typo itself. Forbidding at the model layer is what closes this path, which is
    why the fix is there and not in a key list the two paths would each have to consult."""
    with pytest.raises(ValidationError) as exc:
        load_config(None, overrides=["v3.diagnosis.modes=vector"])
    assert "modes" in str(exc.value)


def test_a_correct_dotted_override_still_works(tmp_path):
    """The `-o` flag is how the treatment arm can be run WITHOUT a second YAML file, so it has to
    keep working -- a fix that closed the typo path by breaking overrides would be worse than the
    defect."""
    cfg = load_config(None, overrides=["v3.diagnosis.mode=vector",
                                       "v3.diagnosis.expectation_ledger=true"])
    assert cfg.v3.diagnosis.mode == "vector"
    assert cfg.v3.diagnosis.expectation_ledger is True


def test_an_override_onto_a_loaded_file_is_rejected_for_a_typo_too(tmp_path):
    """Overrides are applied ON TOP of the file, so a typo'd override against a correct file must
    fail rather than leave the file's value in place looking like it was overridden."""
    good = _write(tmp_path, "v3:\n  diagnosis:\n    mode: vector\n")
    with pytest.raises(ValidationError):
        load_config(good, overrides=["v3.diagnosis.expectation_ledgers=true"])


# --- the fix has to cover every block, not the one that prompted it -------------------------


def test_every_config_model_forbids_unknown_keys():
    """Generalisation check. The `v3:` block is what exposed this, but a fix applied only there
    would leave `budgets:`, `evaluation:` and `device:` -- which decide wall clock, the
    correctness gate and what every agent is told about the GPU -- silently droppable. An omitted
    `device:` block has ALREADY told every L3 agent its GPU was "unknown".

    Walks the model graph rather than listing class names, so a config block added later is
    covered without editing this test.
    """
    seen: set[type] = set()

    def walk(model: type[BaseModel], path: str):
        if model in seen:
            return
        seen.add(model)
        assert model.model_config.get("extra") == "forbid", (
            "%s (%s) ignores unknown keys, so a typo in that block silently falls back to the "
            "field defaults" % (model.__name__, path))
        for name, f in model.model_fields.items():
            ann = f.annotation
            if isinstance(ann, type) and issubclass(ann, BaseModel):
                walk(ann, path + "." + name)

    walk(AppConfig, "")
    assert len(seen) >= 12, "walked only %d models; the graph should cover every block" % len(seen)


def test_the_shipped_configs_all_still_load(tmp_path):
    """Forbidding unknown keys can only be landed if it rejects nothing that was already in use.
    Every file in `configs/` must load -- and a file that fails must name the offending key, so
    the fix is actionable rather than a wall.

    This is the test that would have caught the fix being too aggressive, and it is why the
    strictness could be turned on at all: measured at the time, all 10 files were clean.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "configs"
    files = sorted(root.glob("*.yaml"))
    assert len(files) >= 8, "expected the shipped config set, found %d" % len(files)
    for p in files:
        try:
            load_config(p)
        except ValidationError as exc:  # pragma: no cover -- fails only on a real regression
            pytest.fail("shipped config %s no longer loads:\n%s" % (p.name, exc))


def test_the_strict_base_is_what_the_blocks_inherit():
    """Asserts the mechanism, not just the outcome: the blocks get their strictness by inheriting
    `StrictConfig`, so a block added later that subclasses it is strict by construction. A per-
    class `model_config` would satisfy the test above while letting the next block be added
    without one -- which is exactly how this defect arrived.
    """
    from kernel_optimizer import config as cfgmod

    blocks = [v for v in vars(cfgmod).values()
              if isinstance(v, type) and issubclass(v, BaseModel)
              and v.__module__ == cfgmod.__name__ and v is not StrictConfig]
    assert blocks, "no config blocks found -- the module layout changed"
    for b in blocks:
        assert issubclass(b, StrictConfig), (
            "%s is declared in config.py but does not inherit StrictConfig" % b.__name__)


def test_device_block_typo_is_rejected(tmp_path):
    """`device:` is imported from models/core.py, not declared in config.py, so the inheritance
    check above cannot reach it -- and it is the block with the worst silent failure on record
    (agents told the wrong shared-memory limit run kernels that cannot compile, with no error).
    """
    with pytest.raises(ValidationError) as exc:
        load_config(_write(tmp_path, """
device:
  name: NVIDIA A800 80GB PCIe (sm_80)
  max_shared_bytes_optin_: 166912
"""))
    assert "max_shared_bytes_optin_" in str(exc.value)


def test_free_form_blocks_stay_open(tmp_path):
    """The counter-direction. Two fields are deliberately arbitrary maps -- `server_env` (opencode
    reads settings that have no config-file route) and `sandbox_extra_config` (merged verbatim
    into the sandbox's opencode.json). Forbidding keys THERE would break the A800 config, which
    sets OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX that way. Asserted so a later tightening does not
    quietly take the provider config with it.
    """
    cfg = load_config(_write(tmp_path, """
opencode:
  server_env:
    OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX: "131072"
    ANYTHING_ELSE: "1"
  sandbox_extra_config:
    provider: {whatever: {models: {}}}
"""))
    assert cfg.opencode.server_env["ANYTHING_ELSE"] == "1"
    assert "whatever" in cfg.opencode.sandbox_extra_config["provider"]


def test_the_two_arms_of_the_control_run_differ_in_exactly_two_keys(tmp_path):
    """Guards the run itself rather than the loader: builds both arms the way the run will and
    asserts that the ONLY fields differing between the resolved configs are the two switches.

    J2-5 asks whether the final result got worse. If the treatment file also drifted a budget or a
    tolerance, a difference could not be attributed, and the drift would be invisible -- both
    files load, both runs complete. The diff is computed over the full resolved model, so a change
    anywhere in the file is caught, not only in blocks this test knows to look at.
    """
    control = _write(tmp_path, "budgets:\n  wall_clock_hours: 12\n")
    treat = tmp_path / "t.yaml"
    treat.write_text("budgets:\n  wall_clock_hours: 12\n"
                     "v3:\n  diagnosis:\n    mode: vector\n    expectation_ledger: true\n",
                     encoding="utf-8")

    a = load_config(control).model_dump()
    b = load_config(treat).model_dump()

    def diff(x, y, path=""):
        if isinstance(x, dict) and isinstance(y, dict):
            out = []
            for k in set(x) | set(y):
                out += diff(x.get(k), y.get(k), path + "." + str(k))
            return out
        return [] if x == y else [path.lstrip(".")]

    assert sorted(diff(a, b)) == ["v3.diagnosis.expectation_ledger", "v3.diagnosis.mode"]


def _resolved_diff(a: dict, b: dict) -> list[str]:
    def diff(x, y, path=""):
        if isinstance(x, dict) and isinstance(y, dict):
            out = []
            for k in set(x) | set(y):
                out += diff(x.get(k), y.get(k), path + "." + str(k))
            return out
        return [] if x == y else [path.lstrip(".")]

    return sorted(diff(a, b))


def test_the_shipped_a800_arm_pair_differs_in_exactly_the_two_switches():
    """The same assertion as above, against the REAL FILES the control run will load.

    The test above builds its own minimal pair, so it proves the diff machinery works; it says
    nothing about the files on disk. This one is the one that guards the experiment: it is the only
    check that `experiments_l3_glm_a800_s2.yaml` -- a hand-copied duplicate of a 205-line config --
    did not pick up an edit anywhere in the 100 lines it restates. Both files load either way, both
    runs would complete either way, and a drifted budget or tolerance would make J2-5
    unattributable with nothing to notice.

    Skips rather than fails if either file is absent, so the pair can be retired after the run
    without leaving a red test behind -- but as long as BOTH exist they must differ in exactly the
    two switches.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "configs"
    control = root / "experiments_l3_glm_a800.yaml"
    treat = root / "experiments_l3_glm_a800_s2.yaml"
    if not (control.exists() and treat.exists()):
        pytest.skip("the A800 arm pair is not present in this checkout")

    diffs = _resolved_diff(load_config(control).model_dump(), load_config(treat).model_dump())
    assert diffs == ["v3.diagnosis.expectation_ledger", "v3.diagnosis.mode"], (
        "the two arms of the control run differ in more than the two switches: %s" % diffs)


def test_the_treatment_arm_actually_turns_both_switches_on():
    """Direction check on the pair above. A diff of exactly two keys is satisfied by a treatment
    file that turns the switches the WRONG way (or by a control file that has them on and a
    treatment that has them off), which would invert the arms while passing every other check here.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "configs"
    control = root / "experiments_l3_glm_a800.yaml"
    treat = root / "experiments_l3_glm_a800_s2.yaml"
    if not (control.exists() and treat.exists()):
        pytest.skip("the A800 arm pair is not present in this checkout")

    c = load_config(control).v3.diagnosis
    t = load_config(treat).v3.diagnosis
    assert (c.mode, c.expectation_ledger) == ("label", False), "the CONTROL arm is not v2 behaviour"
    assert (t.mode, t.expectation_ledger) == ("vector", True), "the TREATMENT arm is not S2+S2d"


# --- the cross-box pair, where paths legitimately differ and nothing else may -----------------

# Keys that MUST differ between two boxes: they are where each machine keeps the same things.
# Everything not listed here is arm drift. Listed explicitly rather than pattern-matched on the
# word "path" so that adding a machine-specific key is a deliberate edit to this list.
_MACHINE_KEYS = frozenset({
    "run.runs_dir",
    "opencode.launch_cwd",
    "opencode.sandbox_config_path",
    "kernelbench_root",
    "wsl.venv",
    "wsl.kernelbench_src",
    "wsl.triton_cache_dir",
})

_SWITCH_KEYS = frozenset({"v3.diagnosis.mode", "v3.diagnosis.expectation_ledger"})


def test_the_cross_box_4090_pair_differs_only_in_switches_and_machine_paths():
    """The box1/box2 pair spans TWO machines, so the same-box guarantee is gone and every part of
    it has to be checked explicitly.

    That pairing rests on a measurement: scripts/sweep_resource_map.py returned a byte-identical
    digest (dd3eca5a..., 6 launch failures) on both 4090s over 162 configurations, with box 3
    differing (d565c810..., 0 failures) as the negative control. What that measurement does NOT
    cover is the config files: they are hand-copied 130-line documents, and a budget or tolerance
    that drifted between them would make J2-5's "did the final result get worse" unattributable,
    with nothing to notice -- both files load, both runs complete.

    So: the resolved diff must be exactly the two switches plus the machine paths. Anything else
    fails, and the assertion names it.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "configs"
    box1 = root / "experiments_l3_glm_box1.yaml"
    box2 = root / "experiments_l3_glm_box2_s2.yaml"
    if not (box1.exists() and box2.exists()):
        pytest.skip("the cross-box 4090 pair is not present in this checkout")

    diffs = set(_resolved_diff(load_config(box1).model_dump(), load_config(box2).model_dump()))
    unexpected = diffs - _SWITCH_KEYS - _MACHINE_KEYS
    assert not unexpected, (
        "the two arms differ in something that is neither a switch nor a machine path, so a "
        "difference in the final result could not be attributed to the switch: %s"
        % sorted(unexpected))
    missing = _SWITCH_KEYS - diffs
    assert not missing, (
        "the arms do NOT differ in %s -- the treatment arm is not actually treating"
        % sorted(missing))


def test_the_cross_box_pair_agrees_on_the_device_block_and_every_budget():
    """The two checks a cross-box pair needs that a same-box pair got for free.

    `device:` is written verbatim into every agent sandbox's docs/device.md and exposed to
    agent-authored constraint expressions, so two arms disagreeing there are being told they are on
    different hardware -- and these two boxes are the same card (24564 MiB, 450 W, 3105 MHz,
    cc 8.9), verified before pairing them. Budgets decide when a run stops, and 5 of 5 completed L3
    runs were ended by the wall clock, so an hour of difference is not a rounding error.

    Asserted as whole blocks rather than key by key: a key added to either block later is then
    covered without editing this test.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "configs"
    box1 = root / "experiments_l3_glm_box1.yaml"
    box2 = root / "experiments_l3_glm_box2_s2.yaml"
    if not (box1.exists() and box2.exists()):
        pytest.skip("the cross-box 4090 pair is not present in this checkout")

    a, b = load_config(box1), load_config(box2)
    assert a.device.model_dump() == b.device.model_dump(), (
        "the two arms describe DIFFERENT hardware to their agents; these boxes are the same card")
    assert a.budgets.model_dump() == b.budgets.model_dump(), "the arms have different budgets"
    assert a.evaluation.model_dump() == b.evaluation.model_dump(), (
        "the arms have different correctness/timing settings, so their results are not comparable")
    assert a.agents.model_dump() == b.agents.model_dump(), (
        "the arms use different agent models or candidate counts")


def test_the_two_4090_arms_point_at_different_venvs_on_purpose():
    """The trap this pair is most exposed to, and the one a version-number check would miss.

    The matched torch 2.13.0+cu129 / triton 3.7.1 environment is `orch-venv` on box 1 but
    `kernel-opt-venv` on box 2 -- the names are SWAPPED between the machines. Copying the venv path
    from one file to the other yields a valid-looking config that silently breaks the pairing: box 1
    would run torch 2.9.1/triton 3.5.1, or box 2 torch 2.14.0/triton 3.8.0, and the resource-map
    digests would no longer match while every other check here still passed.

    This asserts the two paths are DIFFERENT, which is the counter-intuitive direction and
    therefore the one somebody will 'fix'.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "configs"
    box1 = root / "experiments_l3_glm_box1.yaml"
    box2 = root / "experiments_l3_glm_box2_s2.yaml"
    if not (box1.exists() and box2.exists()):
        pytest.skip("the cross-box 4090 pair is not present in this checkout")

    v1 = str(load_config(box1).wsl.venv)
    v2 = str(load_config(box2).wsl.venv)
    assert v1.endswith("orch-venv"), (
        "box 1's matched venv is orch-venv (torch 2.13.0/triton 3.7.1); kernel-opt-venv there is "
        "torch 2.9.1/triton 3.5.1 and would break the pairing. Got %s" % v1)
    assert v2.endswith("kernel-opt-venv"), (
        "box 2's matched venv is kernel-opt-venv (torch 2.13.0/triton 3.7.1); orch-venv there is "
        "torch 2.14.0/triton 3.8.0 and would break the pairing. Got %s" % v2)


def test_neither_4090_arm_writes_into_a_v2_runs_directory():
    """Five completed v2 runs on box 1 and four on box 2 are the corpus several conclusions rest on
    (including the only copy of the S1b hard-gate counter-evidence). A v3 run writing into
    `runs-l3` would mix v3 events into it, and events.jsonl is append-only -- there is no undo.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "configs"
    for name in ("experiments_l3_glm_box1.yaml", "experiments_l3_glm_box2_s2.yaml"):
        p = root / name
        if not p.exists():
            continue
        runs = str(load_config(p).run.runs_dir)
        assert not runs.rstrip("/").endswith("runs-l3"), (
            "%s writes into the v2 corpus directory %s" % (name, runs))

