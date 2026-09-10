"""G27: the conversion verdict must actually receive resource profiles.

MEASURED FAILURE, 2026-09-10. `evaluation/conversion.py` (G3's fix) was correct code that never
received data. `BestRecord` had only candidate_id/params/latency_ms, so the orchestrator's
`getattr(family.best, "profile", None)` on both sides of a rewrite round returned None every time:

    getattr(family.best, "profile", None)              -> None
    conversion_verdict(3.0, 2.99, None, None, 2.0)      -> {'conversion': 'flat', ...}
                                                           and NO 'resource_deltas' key at all

Consequences: `resource_deltas` had never been produced in any run, and `no_conversion` -- resources
improved while latency did not, the one verdict the module exists to name -- was unreachable in
production. v3 therefore had no data at all on resource-to-performance conversion, which is S4's
entire subject.

Why the existing guard missed it: tests/test_improvements.py asserts the STRING "profile_before"
appears in the orchestrator source and that its assignment precedes _do_rewrite. Both were true. Its
own failure message ("no resource delta is possible and every round would report 'flat'") described
exactly what production was doing. A source-text assertion encoding the bug it was meant to catch.

So these tests drive the real objects end to end.
"""
from __future__ import annotations

from kernel_optimizer.control.families import FamilyManager
from kernel_optimizer.evaluation.conversion import conversion_verdict
from kernel_optimizer.models.core import ParamSet, ProfileRecord


def _profile(n_regs=64, shared=8192, occ=0.5, spills=0) -> ProfileRecord:
    return ProfileRecord(n_regs=n_regs, n_spills=spills, shared_bytes=shared,
                         occupancy={"achieved": occ})


_SOURCE = "PARAMS = {'B': 32}\n\n\nclass ModelNew:\n    pass\n"


def _mgr_with_family():
    """A manager holding one registered seed candidate, via the real registration path."""
    mgr = FamilyManager()
    cand = mgr.register_candidate(
        source=_SOURCE,
        origin="seed", parent_ids=[], backend="triton", approach="baseline tiling")
    assert cand is not None, "registration returned None (treated as a structural duplicate)"
    return mgr, cand


def test_best_record_carries_the_profile():
    """The missing field. Without it every conversion verdict is latency-only."""
    mgr, cand = _mgr_with_family()
    prof = _profile(n_regs=100)
    improved = mgr.update_best(cand.family_id, cand.candidate_id,
                               ParamSet(values={"B": 32}), 3.0, profile=prof)
    assert improved
    best = mgr.families[cand.family_id].best
    assert best is not None
    assert getattr(best, "profile", None) is not None, (
        "family.best has no profile, so the orchestrator's getattr(..., 'profile', None) returns "
        "None and resource_deltas can never be computed -- the exact production defect")
    assert best.profile.n_regs == 100


def test_resource_deltas_are_produced_when_profiles_are_present():
    """The payload that had never once been generated in a real run."""
    before = _profile(n_regs=200, shared=32768, occ=0.20)
    after = _profile(n_regs=96, shared=8192, occ=0.60)
    out = conversion_verdict(3.0, 2.0, before, after, 2.0)
    assert "resource_deltas" in out, (
        "no resource_deltas key: the verdict is still latency-only, which is what every recorded "
        "round contained before this fix")
    deltas = out["resource_deltas"]
    assert deltas, "resource_deltas is empty"
    # The dimensions that changed must be named with before/after, not just a direction.
    names = {d.get("dimension") for d in deltas} if isinstance(deltas, list) else set(deltas)
    assert any("reg" in str(n) for n in names), names


def test_no_conversion_is_reachable():
    """Resources improved, latency did not -- the verdict the module exists to name.

    Unreachable in production before this fix, because with both profiles None there was no
    resource change to observe.
    """
    before = _profile(n_regs=200, shared=32768, occ=0.20)
    after = _profile(n_regs=96, shared=8192, occ=0.60)   # clearly better resources
    out = conversion_verdict(3.0, 2.995, before, after, 2.0)  # latency essentially unchanged
    assert out["conversion"] == "no_conversion", (
        "resources improved substantially (regs 200->96, shared 32768->8192, occupancy 0.20->0.60) "
        "while latency moved 0.17%% -- that is precisely 'no_conversion', and reporting it as %r "
        "counts a non-improvement as a success" % out["conversion"])
    assert out["resources_improved"], out


def test_a_missing_profile_reads_as_unknown_not_as_no_change():
    """Direction of the failure: absent data must never look like a measured zero delta."""
    out = conversion_verdict(3.0, 2.0, None, None, 2.0)
    assert "resource_deltas" not in out or not out["resource_deltas"], out
    assert out["conversion"] != "no_conversion", (
        "with no profiles at all the verdict claimed resources did not convert, which is an "
        "assertion about data it never had")


def test_update_best_is_still_monotonic_and_keeps_the_winning_profile():
    """A worse round must not replace the incumbent -- nor its profile."""
    mgr, cand = _mgr_with_family()
    mgr.update_best(cand.family_id, cand.candidate_id, ParamSet(values={}), 2.0,
                    profile=_profile(n_regs=64))
    kept = mgr.update_best(cand.family_id, cand.candidate_id, ParamSet(values={}), 5.0,
                           profile=_profile(n_regs=255))
    assert kept is False, "a slower result replaced the family best"
    best = mgr.families[cand.family_id].best
    assert best.latency_ms == 2.0
    assert best.profile.n_regs == 64, (
        "the incumbent kept its latency but took the loser's profile, so the next conversion "
        "verdict would compare a resource state that never produced that latency")
