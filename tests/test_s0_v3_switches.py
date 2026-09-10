"""Every v3 stage must be switchable, and every switch must default to the pre-v3 behaviour (S0).

WHY THIS COMES FIRST. Each v3 stage needs its own control run -- same task, same budget, one switch
flipped -- because S1 and S2 both change what the tuner and the agent see, so a run from before a
stage landed is not comparable with one from after it. A stage that cannot be turned off cannot be
measured, so the switches are a prerequisite for the acceptance criteria rather than a convenience.

WHAT THIS FILE GUARDS, and each item is a failure this project has actually had:

  DEFAULTS ARE PRODUCTION. `load_config` reads ONE yaml with no base layer, so a key absent from the
  file silently falls back to the default in `config.py` -- `configs/default.yaml` is not an
  underlay. An omitted `device:` block once left every L3 agent being told its GPU was "unknown".
  So the defaults here are what runs, and every v3 default must be the OLD behaviour.

  A SWITCH THAT DOES NOT SWITCH. The counterpart risk: a flag that is read nowhere, or one whose
  "off" state is not actually the previous behaviour. The first is checked by the stages' own tests
  as they land; what is checked here is that the flag EXISTS, is spelled as the plan says, and can be
  set from a config file and from a `--set` override -- the two ways a control run will flip it.

  THE OVERRIDE PATH IS SEPARATE FROM THE FILE PATH. `_apply_override` builds nested dicts by hand and
  YAML-parses the value, so `v3.diagnosis.mode=vector` reaching the right field is not implied by the
  file path working. A control run that silently kept the default would produce two runs that differ
  in nothing, reported as if they were compared -- the credible-no-op failure shape.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from kernel_optimizer.config import AppConfig, load_config

# The switches the revised plan names, with the value each must hold when nothing is configured.
# Spelled out as literal dotted paths rather than derived from the model, so a RENAME breaks this
# test instead of silently passing: a config key is an external interface -- the yaml files and the
# runbook use these names.
EXPECTED_DEFAULTS = {
    "v3.search.declare_infeasible_out_of_space": False,
    "v3.search.deweight_unconditional_failures": False,
    "v3.diagnosis.mode": "label",
    "v3.diagnosis.expectation_ledger": False,
    "v3.diagnosis.access_pattern_walls": False,
}


def _get(cfg, dotted: str):
    node = cfg
    for part in dotted.split("."):
        node = getattr(node, part)
    return node


def test_every_v3_switch_exists_and_defaults_to_the_old_behaviour():
    """One assertion per switch, driven through the real model's defaults."""
    cfg = AppConfig()
    for dotted, expected in EXPECTED_DEFAULTS.items():
        assert _get(cfg, dotted) == expected, (
            "%s defaults to %r, not %r. Because load_config reads a single file with no base layer, "
            "this default IS the production behaviour for every config that omits the key -- so a "
            "v3 stage would be silently ON in runs that never asked for it, and the pre-v3 control "
            "arm would be unobtainable." % (dotted, _get(cfg, dotted), expected))


def test_a_config_file_that_mentions_no_v3_section_still_loads(tmp_path):
    """The existing configs do not have a `v3:` block and must keep working unchanged."""
    p = tmp_path / "c.yaml"
    p.write_text("run:\n  runs_dir: runs\n", encoding="utf-8")
    cfg = load_config(p)
    for dotted, expected in EXPECTED_DEFAULTS.items():
        assert _get(cfg, dotted) == expected, (
            "%s is not at its default for a config with no v3 section" % dotted)


def test_a_switch_can_be_set_from_a_config_file(tmp_path):
    """The path a control run's second arm will use."""
    p = tmp_path / "c.yaml"
    p.write_text(
        "v3:\n"
        "  search:\n"
        "    declare_infeasible_out_of_space: true\n"
        "  diagnosis:\n"
        "    mode: vector\n"
        "    expectation_ledger: true\n",
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.v3.search.declare_infeasible_out_of_space is True
    assert cfg.v3.diagnosis.mode == "vector"
    assert cfg.v3.diagnosis.expectation_ledger is True
    # Unset siblings must NOT be dragged along: the whole point is flipping one thing at a time.
    assert cfg.v3.search.deweight_unconditional_failures is False, (
        "setting one v3 switch changed another, so no control run isolates a single stage")
    assert cfg.v3.diagnosis.access_pattern_walls is False


def test_a_switch_can_be_set_from_a_command_line_override():
    """`--set v3.diagnosis.mode=vector`, the other way an arm gets flipped.

    Checked separately from the file path because `_apply_override` is a different code path: it
    builds the nested dicts itself and YAML-parses the value, so "true" becoming the string "true"
    (which is truthy either way, and would look like it worked) is a real possibility.
    """
    cfg = load_config(None, ["v3.diagnosis.mode=vector",
                             "v3.search.declare_infeasible_out_of_space=true"])
    assert cfg.v3.diagnosis.mode == "vector"
    val = cfg.v3.search.declare_infeasible_out_of_space
    assert val is True, (
        "the override produced %r rather than the boolean True; a truthy string would satisfy every "
        "`if cfg...:` in the codebase while being impossible to set back to False from the command "
        "line" % (val,))


def test_the_diagnosis_mode_is_constrained_to_the_two_arms():
    """A typo in the arm name must fail loudly, not select a third behaviour.

    `mode` is the switch the S2 control run turns, and its two values ARE the two arms. A free-form
    string would let `--set v3.diagnosis.mode=vecotr` run silently as neither arm, and the run would
    look completed.
    """
    with pytest.raises(Exception):
        load_config(None, ["v3.diagnosis.mode=vecotr"])


def test_the_switches_are_recorded_in_the_run_manifest(tmp_path):
    """A finished run must state which arm it was, from its own files.

    The manifest already carries `cfg.model_dump(mode="json")`, so this holds today -- it is asserted
    because it is what makes the control runs auditable after the fact, and because a future
    slimming of the manifest would break the comparison without breaking anything visible. Runs are
    read months later by `report`, and "which arm was this" cannot be reconstructed from latency.
    """
    from kernel_optimizer.store.run_store import RunStore

    cfg = load_config(None, ["v3.diagnosis.mode=vector"])
    store = RunStore.create(tmp_path, "run-arm-test", {"config": cfg.model_dump(mode="json")})
    manifest = (Path(store.run_dir) / "manifest.json").read_text(encoding="utf-8")
    assert '"mode": "vector"' in manifest, (
        "the run manifest does not record the v3 switches, so a completed run cannot say which arm "
        "of a control experiment it was")
