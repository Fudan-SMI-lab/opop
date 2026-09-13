"""Do two arms' CONFIGS differ in exactly the intended places, and nowhere else?

WHY THIS FILE EXISTS AT ALL. Five committed files referenced `scripts/audit_arm_comparability.py` --
`check_arm_search_parity.py`'s own docstring leans on it ("verified the two arms' CONFIGS match
field-by-field"), and `analyze_s7_pair.py` PRINTS it as the next command for the operator to run. It was
never committed: `git log --all --diff-filter=A` finds no commit that ever added it. So the claim that
comparability had been checked rested on a script that does not exist, and the analysis sequence told
the operator to run a missing file.

WHAT IT CHECKS, and why a diff is the right shape. A paired control run's validity rests on ONE
difference. Any second difference is a second independent variable, and the failure mode is silent: a
stray `trials_per_space`, a different `wall_clock_hours`, one arm's `soft_wall.in_prompt` left off, and
the pair still runs to completion and still produces two numbers. Nothing surfaces the problem later --
`check_arm_search_parity.py` measures the SEARCH the arms got, `compare_calibrations.py` the
DENOMINATOR, and neither reads the configs.

Three mandatory differences are expected and named, not tolerated silently: `run.runs_dir`,
`opencode.server_env.XDG_DATA_HOME` and `wsl.triton_cache_dir` MUST differ, because two arms sharing a
runs dir, an opencode data dir or a Triton cache are not isolated. Their ABSENCE is reported as a
failure for exactly that reason -- an isolation path that is accidentally equal is worse than an extra
variable, and a checker that only looked for "too many differences" would pass it.

    python scripts/audit_arm_comparability.py <control.yaml> <treatment.yaml> \
        [--expect v3.slope_guide.enabled] ...

Every `--expect` names a key that is ALLOWED to differ. Anything else that differs is reported and the
exit code is non-zero.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# APPEND, never insert at 0. Inserting this script's own parent-of-parent/src at the front of sys.path
# is what made the first live run of this file silently wrong: copied to /root/probe-clean and invoked
# from inside the repo, `parents[1]/src` resolved to a DIFFERENT checkout of kernel_optimizer -- box4's
# /root/autodl-tmp/opop-workspace/opop at commit 14fab15, which predates S7 and has no
# `V3Config.slope_guide` at all. Every v3.* key then read None on BOTH sides, the key count came out 96
# instead of 119, and the audit reported "EXPECTED DIFFERENCE ABSENT -- the arms are running the SAME
# configuration". A plausible, alarming, entirely wrong verdict about a pair that was fine.
#
# Appending means an already-importable kernel_optimizer (the venv's, or one on PYTHONPATH) wins, and
# this fallback only applies when there is none. The assertion below then states which one was used, so
# the answer is never anonymous.
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from kernel_optimizer.config import load_config  # noqa: E402
from kernel_optimizer import config as _config_mod  # noqa: E402

# Two arms sharing any of these are not isolated. Required to DIFFER.
ISOLATION_KEYS = (
    "run.runs_dir",
    "opencode.server_env.XDG_DATA_HOME",
    "wsl.triton_cache_dir",
)


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Dotted-path view of the loaded config.

    Reads the LOADED config rather than the YAML text, which is the whole point: `default.yaml` is not
    a base layer, so an omitted key falls back silently and two arms can differ in a value that appears
    in neither file. Comparing the parsed objects sees those.
    """
    out: dict[str, Any] = {}
    items = obj.model_dump() if hasattr(obj, "model_dump") else obj
    if not isinstance(items, dict):
        return {prefix.rstrip("."): items}
    for key, val in items.items():
        path = f"{prefix}{key}"
        if isinstance(val, dict):
            out.update(_flatten(val, path + "."))
        else:
            out[path] = val
    return out


def compare(left: dict[str, Any], right: dict[str, Any], expected: set[str]) -> dict[str, Any]:
    keys = sorted(set(left) | set(right))
    differ = [k for k in keys if left.get(k) != right.get(k)]
    unexpected = [k for k in differ if k not in expected and k not in ISOLATION_KEYS]
    missing_expected = [k for k in expected if k not in differ]
    not_isolated = [k for k in ISOLATION_KEYS if k not in differ]
    return {
        "n_keys": len(keys),
        "differ": differ,
        "unexpected": unexpected,
        "missing_expected": missing_expected,
        "not_isolated": not_isolated,
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("control", type=Path)
    ap.add_argument("treatment", type=Path)
    ap.add_argument("--expect", action="append", default=[],
                    help="a dotted config key that is ALLOWED to differ (repeatable)")
    args = ap.parse_args(argv)

    left = _flatten(load_config(args.control))
    right = _flatten(load_config(args.treatment))
    expected = set(args.expect)
    res = compare(left, right, expected)

    # WHICH kernel_optimizer answered. Printed unconditionally, because the one failure this script has
    # already had was importing a checkout that predates the field being audited -- and the output was
    # indistinguishable from a real finding. A key named in --expect that is absent from BOTH configs is
    # the signature of exactly that, so it is called out rather than reported as "no variable".
    print("config module: %s" % _config_mod.__file__)
    absent_both = [k for k in expected if k not in left and k not in right]
    if absent_both:
        print()
        print("!! KEY NOT PRESENT IN EITHER CONFIG: %s" % ", ".join(absent_both))
        print("   That is usually not a pair problem -- it is this script reading a DIFFERENT checkout")
        print("   of kernel_optimizer than the one the run used (see the config module path above).")
        print("   Set PYTHONPATH to the running experiment's src/ and re-run before believing any")
        print("   verdict below.")
    print("control   %s" % args.control)
    print("treatment %s" % args.treatment)
    print("%d keys compared, %d differ" % (res["n_keys"], len(res["differ"])))
    print()
    for k in res["differ"]:
        if k in ISOLATION_KEYS:
            tag = "isolation (required)"
        elif k in expected:
            tag = "THE VARIABLE"
        else:
            tag = "!! UNEXPECTED"
        print("  %-20s %-46s %r  vs  %r" % (tag, k, left.get(k), right.get(k)))
    print()

    okay = True
    if res["unexpected"]:
        okay = False
        print("SECOND VARIABLE(S) PRESENT -- the pair cannot attribute a difference to one switch:")
        for k in res["unexpected"]:
            print("    %s" % k)
    if res["missing_expected"]:
        okay = False
        print("EXPECTED DIFFERENCE ABSENT -- the arms are running the SAME configuration, so the")
        print("experiment has no independent variable at all:")
        for k in res["missing_expected"]:
            print("    %s (both %r)" % (k, left.get(k)))
    if res["not_isolated"]:
        okay = False
        print("NOT ISOLATED -- these must DIFFER or the arms share state and contaminate each other:")
        for k in res["not_isolated"]:
            print("    %s (both %r)" % (k, left.get(k)))

    if okay:
        print("COMPARABLE -- exactly the intended difference(s) plus the three isolation paths.")
        print("This settles the CONFIG half only. `check_arm_search_parity.py` settles whether the")
        print("wall clock bought comparable SEARCH; for a CROSS-BOX pair `compare_calibrations.py`")
        print("settles the denominator (a same-box pair sharing one calibration.json gets that free).")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
