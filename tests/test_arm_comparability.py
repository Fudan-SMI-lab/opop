"""The comparability audit: the checks, and the two failure directions a "count the diffs" version misses.

WHY THESE TESTS. `audit_arm_comparability.py` was cited by five committed files and never existed, so it
arrives with no track record at all -- and it is the script whose "OK" would be quoted as evidence that a
paired run is valid. A checker that reports clean is not evidence it works.

The obvious implementation counts differences and fails when there are too many. That passes both of the
directions that actually destroy a pair:

  * the expected difference is ABSENT -- both arms run the same configuration, so the experiment has no
    independent variable and the two numbers are two samples of one thing. A diff count sees FEWER
    differences and is happy.
  * an ISOLATION path is accidentally equal -- the arms share a runs dir, an opencode data dir or a
    Triton cache, so they contaminate each other. Again fewer differences, again happy.

Both are asserted below, in the direction that must FAIL.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "aac", Path("scripts/audit_arm_comparability.py"))
aac = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(aac)

# A minimal pair in the shape `_flatten` produces: the isolation paths differ, one variable differs.
_L = {
    "run.runs_dir": "/runs/control",
    "opencode.server_env.XDG_DATA_HOME": "/xdg/c",
    "wsl.triton_cache_dir": "/triton/c",
    "v3.slope_guide.enabled": False,
    "budgets.trials_per_space": 40,
    "budgets.wall_clock_hours": 12,
}
_R = dict(_L, **{
    "run.runs_dir": "/runs/treatment",
    "opencode.server_env.XDG_DATA_HOME": "/xdg/t",
    "wsl.triton_cache_dir": "/triton/t",
    "v3.slope_guide.enabled": True,
})
_EXPECT = {"v3.slope_guide.enabled"}


def test_a_clean_pair_is_comparable():
    res = aac.compare(_L, _R, _EXPECT)
    assert res["unexpected"] == []
    assert res["missing_expected"] == []
    assert res["not_isolated"] == []
    assert set(res["differ"]) == _EXPECT | set(aac.ISOLATION_KEYS)


def test_a_second_variable_is_caught():
    """The direction a diff count DOES catch, kept as the baseline that makes the others meaningful."""
    bad = dict(_R, **{"budgets.trials_per_space": 20})
    res = aac.compare(_L, bad, _EXPECT)
    assert res["unexpected"] == ["budgets.trials_per_space"]


def test_the_missing_independent_variable_is_caught():
    """FEWER differences, and fatal: both arms have slope_guide off, so there is nothing to measure.

    A checker that only fails on "too many differences" passes this, and the run would produce two
    numbers that look like a result.
    """
    same = dict(_R, **{"v3.slope_guide.enabled": False})
    res = aac.compare(_L, same, _EXPECT)
    assert res["missing_expected"] == ["v3.slope_guide.enabled"]
    assert res["unexpected"] == []


def test_a_shared_isolation_path_is_caught():
    """Also FEWER differences, also fatal: a shared Triton cache means one arm's compilations serve the
    other's, so the arms are not independent runs.
    """
    shared = dict(_R, **{"wsl.triton_cache_dir": "/triton/c"})
    res = aac.compare(_L, shared, _EXPECT)
    assert res["not_isolated"] == ["wsl.triton_cache_dir"]


def test_every_isolation_path_is_checked_not_just_one():
    """All three, together -- so the check cannot pass by testing whichever one happens to differ."""
    shared = dict(_R, **{
        "run.runs_dir": "/runs/control",
        "opencode.server_env.XDG_DATA_HOME": "/xdg/c",
        "wsl.triton_cache_dir": "/triton/c",
    })
    res = aac.compare(_L, shared, _EXPECT)
    assert set(res["not_isolated"]) == set(aac.ISOLATION_KEYS)


def test_a_key_present_on_only_one_side_counts_as_a_difference():
    """`default.yaml` is not a base layer, so an omitted key falls back silently. A key that exists on
    one side only must therefore read as a difference rather than as absence.
    """
    extra = dict(_R, **{"v3.something_new": 7})
    res = aac.compare(_L, extra, _EXPECT)
    assert "v3.something_new" in res["unexpected"]


def test_flatten_produces_dotted_paths_from_a_nested_mapping():
    flat = aac._flatten({"a": {"b": {"c": 1}}, "d": 2})
    assert flat == {"a.b.c": 1, "d": 2}


def test_a_key_absent_from_both_configs_is_distinguished_from_a_missing_variable():
    """The failure this script actually had: it imported a checkout of kernel_optimizer that predates
    the audited field, so `v3.slope_guide.enabled` was absent from BOTH sides. `compare` correctly
    reports it under `missing_expected` -- but the CAUSE is a wrong import, not a badly configured pair,
    and the two need different actions. The distinguishing signature is that the key is in neither
    mapping, which `main` checks and reports before the verdict.
    """
    left = {k: v for k, v in _L.items() if k != "v3.slope_guide.enabled"}
    right = {k: v for k, v in _R.items() if k != "v3.slope_guide.enabled"}
    res = aac.compare(left, right, _EXPECT)
    assert res["missing_expected"] == ["v3.slope_guide.enabled"]
    # the signature main() keys off
    assert "v3.slope_guide.enabled" not in left and "v3.slope_guide.enabled" not in right


def test_a_present_but_equal_variable_is_a_real_pair_defect():
    """The contrast case, and it is what makes the test above meaningful: the key IS present on both
    sides and simply has the same value. That IS a badly configured pair, and must not be excused as an
    import problem.
    """
    same = dict(_R, **{"v3.slope_guide.enabled": False})
    res = aac.compare(_L, same, _EXPECT)
    assert res["missing_expected"] == ["v3.slope_guide.enabled"]
    assert "v3.slope_guide.enabled" in _L and "v3.slope_guide.enabled" in same
