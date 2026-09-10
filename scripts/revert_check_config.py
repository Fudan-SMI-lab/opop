"""Revert-check for the config-strictness fix: every test must FAIL on the wrong implementation.

Same contract and the same hazards as `revert_check_s2.py` -- see that file's `run()` docstring for
why `-B` is required and why the outcome is read from a line that names the test rather than from a
summary line a flag can silence.

This one exists because the defect it closes was found the day the control run was to be launched,
in a code path that had never been exercised, and the fix is three lines. A three-line fix with no
failing control is how "thought it was fixed" happens -- and this project has that on record for
G21, which regressed the same day it was fixed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "src" / "kernel_optimizer" / "config.py"
CORE = ROOT / "src" / "kernel_optimizer" / "models" / "core.py"
ARM = ROOT / "configs" / "experiments_l3_glm_a800_s2.yaml"
B1 = ROOT / "configs" / "experiments_l3_glm_box1.yaml"
B2 = ROOT / "configs" / "experiments_l3_glm_box2_s2.yaml"
TESTS = [ROOT / "tests" / "test_config_strictness.py"]

VARIANTS: list[tuple[str, Path, str, str, list[str], str]] = [
    (
        "pydantic's default: unknown keys ignored (the pre-fix state)",
        CFG,
        '    model_config = ConfigDict(extra="forbid")\n\n\nclass RunConfig(StrictConfig):',
        "    pass\n\n\nclass RunConfig(StrictConfig):",
        ["test_a_mistyped_switch_key_is_rejected_not_dropped[v3:\\n  diagnosis:\\n    modes: vector\\n-modes]",
         "test_a_mistyped_switch_key_is_rejected_not_dropped[v3:\\n  diagnosic:\\n    mode: vector\\n-diagnosic]",
         "test_a_mistyped_key_in_a_block_that_is_otherwise_correct_is_still_rejected",
         "test_a_mistyped_dotted_override_is_rejected",
         "test_an_override_onto_a_loaded_file_is_rejected_for_a_typo_too",
         "test_every_config_model_forbids_unknown_keys"],
        "the state MEASURED on the day the control run was to be launched: "
        "`v3.diagnosis.modes: vector` validates and leaves mode='label', so the treatment arm is a "
        "SECOND CONTROL ARM. Two 12 h runs agree, and the recorded conclusion is 'the vector shape "
        "changes nothing' -- the diagnosis inverted",
    ),
    (
        "strictness declared per-class instead of on a shared base",
        CFG,
        "class V3DiagnosisConfig(StrictConfig):",
        "class V3DiagnosisConfig(BaseModel):",
        ["test_a_mistyped_switch_key_is_rejected_not_dropped[v3:\\n  diagnosis:\\n    modes: vector\\n-modes]",
         "test_a_mistyped_key_in_a_block_that_is_otherwise_correct_is_still_rejected",
         "test_every_config_model_forbids_unknown_keys",
         "test_the_strict_base_is_what_the_blocks_inherit"],
        "ONE block loses strictness while every other block keeps it, so the config looks protected "
        "and the v3 switches -- the only ones the run depends on -- are not. This is the shape the "
        "defect arrived in: a block added without the guard the others have",
    ),
    (
        "the device block left open (the fix applied only where it was noticed)",
        CORE,
        'model_config = ConfigDict(frozen=True, extra="forbid")',
        "model_config = ConfigDict(frozen=True)",
        ["test_every_config_model_forbids_unknown_keys",
         "test_device_block_typo_is_rejected"],
        "`device:` is imported from models/core.py, so a fix confined to config.py misses it. A "
        "dropped `max_shared_bytes_optin` tells every agent on the A800 it has 101376 B instead of "
        "166912 B -- silently forbidding the tile sizes that box exists to explore. This variant is "
        "not hypothetical: the first version of this fix WAS confined to config.py, and the "
        "generalisation test is what caught it",
    ),
    (
        "free-form maps forbidden along with everything else (over-application)",
        CFG,
        "    sandbox_extra_config: dict = Field(default_factory=dict)",
        "    sandbox_extra_config: StrictConfig = Field(default_factory=StrictConfig)",
        ["test_free_form_blocks_stay_open"],
        "the opposite error, and the reason `test_free_form_blocks_stay_open` exists: "
        "`sandbox_extra_config` is merged verbatim into each sandbox's opencode.json and "
        "`server_env` carries settings that have NO config-file route. Forbidding keys there breaks "
        "the provider block, and every agent call fails with ProviderModelNotFoundError. "
        "NOTE `test_the_shipped_configs_all_still_load` is deliberately NOT named here: no shipped "
        "config sets `sandbox_extra_config` (the A800 file uses `sandbox_config_path` and "
        "`server_env`), so it survives this variant CORRECTLY. It is evidence that the fix does not "
        "reject configs already in use -- not evidence about this variant",
    ),
    (
        "the closed Literal on `mode` widened to a free string",
        CFG,
        '    mode: Literal["label", "vector"] = "label"',
        '    mode: str = "label"',
        ["test_a_wrong_value_for_a_correct_key_is_also_rejected"],
        "the key is spelled right and the VALUE is wrong: `mode: vectors` validates, and every "
        "`!= \"vector\"` comparison in the orchestrator takes the label branch. Forbidding unknown "
        "keys does nothing about this axis, and from the outside the failure is identical",
    ),
    (
        "a switch renamed in the model while the YAML keeps the old name",
        CFG,
        "    expectation_ledger: bool = False",
        "    expectation_ledger_enabled: bool = False",
        ["test_the_two_arms_of_the_control_run_differ_in_exactly_two_keys",
         "test_a_correctly_spelled_v3_block_reaches_the_config",
         "test_an_omitted_v3_block_is_the_control_arm"],
        "the drift this pair of files is most exposed to: the model moves and the arm YAML does not, "
        "so the treatment file no longer sets the ledger. WITH the fix this is loud (the file is "
        "rejected); the variant's purpose is to prove the arm-comparison test is not VACUOUS -- it "
        "computes its diff over the whole resolved model, so if a switch stops arriving, or if the "
        "two arms ever came to differ in more than these two keys, the test says so. Without that, "
        "a treatment file that also moved a budget would be invisible (both files load, both runs "
        "complete) and J2-5's 'did the final result get worse' would be unattributable. "
        "NOTE a changed DEFAULT is not a usable variant here: it shifts both arms equally, so no "
        "diff appears -- measured, not assumed",
    ),
    # --- the SHIPPED arm pair, which is what the run actually loads ---------------------------
    (
        "the treatment arm drifts from its control in a second key",
        ARM,
        "  wall_clock_hours: 12",
        "  wall_clock_hours: 11",
        ["test_the_shipped_a800_arm_pair_differs_in_exactly_the_two_switches"],
        "the failure mode of a hand-copied 205-line config: BOTH files load, BOTH runs complete, "
        "and the treatment arm silently had an hour less wall clock -- on a task where 5 of 5 "
        "completed runs were ended by the wall clock, so an hour is not a rounding error. J2-5 "
        "('did the final result get worse') would then be measuring the budget, not the switch",
    ),
    (
        "the treatment arm has the switches the wrong way round",
        ARM,
        "    mode: vector",
        "    mode: label",
        ["test_the_treatment_arm_actually_turns_both_switches_on"],
        "the arms invert while the key-level diff STILL shows exactly two keys, so every other "
        "check here passes. The run would be labelled treatment and be a control, which is worse "
        "than not running it: the result would be recorded with the sign of the effect flipped",
    ),
    # --- the CROSS-BOX 4090 pair, where paths differ legitimately and nothing else may ---------
    (
        "the cross-box treatment arm drifts in a budget",
        B2,
        "  wall_clock_hours: 12",
        "  wall_clock_hours: 11",
        ["test_the_cross_box_4090_pair_differs_only_in_switches_and_machine_paths",
         "test_the_cross_box_pair_agrees_on_the_device_block_and_every_budget"],
        "the failure mode of two hand-copied 130-line configs. BOTH load, BOTH runs complete, and "
        "the treatment arm silently had an hour less wall clock -- on a task where 5 of 5 completed "
        "runs were ended by the wall clock. J2-5 would then be measuring the budget, not the switch",
    ),
    (
        "the cross-box arms describe different hardware to their agents",
        B2,
        "  max_shared_bytes_optin: 101376",
        "  max_shared_bytes_optin: 166912",
        ["test_the_cross_box_pair_agrees_on_the_device_block_and_every_budget",
         "test_the_cross_box_4090_pair_differs_only_in_switches_and_machine_paths"],
        "the A800's figure pasted into a 4090 config -- the plausible copy error, since the A800 "
        "file is the one this pair was derived from. `device:` is written verbatim into every agent "
        "sandbox's docs/device.md and exposed to agent-authored constraint expressions, so one arm "
        "would be inventing tiles that cannot launch on the card it is actually running",
    ),
    (
        "the venv paths 'corrected' to match each other",
        B2,
        "  venv: /root/autodl-tmp/kernel-opt-venv",
        "  venv: /root/autodl-tmp/orch-venv",
        ["test_the_two_4090_arms_point_at_different_venvs_on_purpose"],
        "the trap a version-number check cannot see and the one somebody will 'tidy up': the "
        "matched torch 2.13.0+cu129 / triton 3.7.1 environment is `orch-venv` on box 1 but "
        "`kernel-opt-venv` on box 2 -- the names are SWAPPED between the machines. Making the two "
        "files agree gives box 2 torch 2.14.0 / triton 3.8.0, the resource-map digests stop "
        "matching, and the pairing is void while every other check still passes",
    ),
    (
        "a v3 arm writing into the v2 corpus directory",
        B1,
        "  runs_dir: /root/autodl-tmp/opop-workspace/opop-glm/runs-v3",
        "  runs_dir: /root/autodl-tmp/opop-workspace/opop-glm/runs-l3",
        ["test_neither_4090_arm_writes_into_a_v2_runs_directory"],
        "events.jsonl is APPEND-ONLY, so there is no undo: a v3 run pointed at `runs-l3` mixes v3 "
        "events into the five completed v2 runs on box 1 (and four on box 2, which hold the only "
        "copy of the S1b hard-gate counter-evidence). Several recorded conclusions read that corpus",
    ),
]


def run(names: list[str]) -> tuple[set[str], set[str], bool, str]:
    proc = subprocess.run(
        [sys.executable, "-B", "-m", "pytest", *[str(t) for t in TESTS], "-v", "--no-header",
         "--tb=no", "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": "src", "PYTHONDONTWRITEBYTECODE": "1"})
    out = proc.stdout + proc.stderr
    failed: set[str] = set()
    skipped: set[str] = set()
    for line in out.splitlines():
        for n in names:
            if f"::{n} " not in line:
                continue
            if "FAILED" in line or "ERROR" in line:
                failed.add(n)
            elif "SKIPPED" in line:
                skipped.add(n)
    return failed, skipped, proc.returncode == 0, out


def apply_variant(path: Path, text: str, old: str, new: str, names: list[str]):
    path.write_text(text.replace(old, new), encoding="utf-8")
    try:
        failed, skipped, _, out = run(names)
    finally:
        path.write_text(text, encoding="utf-8")
    return failed, skipped, out


def main() -> int:
    originals = {p: p.read_text(encoding="utf-8") for p in (CFG, CORE, ARM, B1, B2)}
    ok = True
    unverified = 0
    unstable: list[str] = []
    try:
        _, _, green, out = run([])
        if not green:
            print("BASELINE IS NOT GREEN -- nothing below means anything\n" + out[-3000:])
            return 2
        print("baseline: green (%s)\n" % out.strip().splitlines()[-1])

        for label, path, old, new, must_fail, why in VARIANTS:
            text = originals[path]
            if text.count(old) != 1:
                print("**SKIPPED** %s: anchor occurs %d times, not once -- the variant would not be "
                      "the change it claims to be" % (label, text.count(old)))
                ok = False
                continue

            # A variant that does not parse fails every test for the WRONG reason, which would read
            # as discrimination -- the easiest way for this harness to produce a false `ok`. Scoped
            # to Python sources: two variants below patch a YAML config, which `compile()` would
            # reject as invalid Python and report as a malformed variant. YAML is checked with
            # yaml.safe_load for the same purpose.
            patched = text.replace(old, new)
            try:
                if path.suffix == ".py":
                    compile(patched, str(path), "exec")
                else:
                    yaml.safe_load(patched)
            except (SyntaxError, yaml.YAMLError) as exc:
                print("**SKIPPED** %s: the patched file does not parse (%s), so any failure it "
                      "produced would be for the wrong reason" % (label, exc))
                ok = False
                continue

            failed, was_skipped, out = apply_variant(path, text, old, new, must_fail)
            missing = [n for n in must_fail if n not in failed]
            unrun = [n for n in missing if n in was_skipped]
            wrongly_passed = [n for n in missing if n not in was_skipped]

            if wrongly_passed:
                failed2, _, _ = apply_variant(path, text, old, new, must_fail)
                still = [n for n in wrongly_passed if n not in failed2]
                if len(still) != len(wrongly_passed):
                    unstable.append(label)
                    ok = False
                    print("**UNSTABLE** %s" % label)
                    print("        two identical runs DISAGREED; neither answer can be reported")
                    print("        wrong version: %s\n" % why)
                    continue
                ok = False
                print("**FAIL** %s" % label)
                print("        these tests PASSED on the wrong implementation, so they are not")
                print("        evidence for it: %s" % ", ".join(wrongly_passed))
            elif unrun and len(unrun) == len(must_fail):
                unverified += 1
                print("UNVERIF %s" % label)
                print("        every named test SKIPPED here: %s" % ", ".join(unrun))
            else:
                verified = [n for n in must_fail if n in failed]
                print("ok      %s" % label)
                print("        %d/%d named tests failed as required%s" % (
                    len(verified), len(must_fail),
                    "" if not unrun else " (%d skipped here)" % len(unrun)))
            print("        wrong version: %s" % why)
            print()
    finally:
        for p, text in originals.items():
            p.write_text(text, encoding="utf-8")

    _, _, green, out = run([])
    if not green:
        print("!! RESTORE DID NOT COME BACK GREEN -- check git status\n" + out[-2000:])
        return 2
    print("restored, suite green again")
    if unstable:
        print("\nVERDICT: harness UNSTABLE on %d variant(s); no verdict is trustworthy: %s"
              % (len(unstable), ", ".join(unstable)))
        return 1
    if not ok:
        print("\nVERDICT: at least one test is not evidence -- see **FAIL** above")
        return 1
    if unverified:
        print("\nVERDICT: every variant this box could check discriminates; %d UNVERIFIED here"
              % unverified)
        return 0
    print("\nVERDICT: every test fails on the implementation it was written against")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
