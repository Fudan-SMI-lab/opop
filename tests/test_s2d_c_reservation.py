"""S2d(c) is testable only if a ledger can reach a rewriter prompt. This is that reservation.

THE FINDING BEHIND IT (`docs/result-s2d-c-never-administered.md`, measured, not reasoned): the ledger
is keyed per FAMILY, so a rewriter sees one only when the SAME family gets a second round after its
first was reconciled. Across every run this project has made -- 9 of 9 families in finished runs plus
both paired arms -- each family received exactly ONE round. `ledger_entries` was therefore `[]` on
every rewriter call ever made, in BOTH arms, regardless of `expectation_ledger`. The arithmetic is off
by one family every time: 4 seed families, 3 rounds per 12 h, and `active_families` rule 1 puts every
never-rewritten family first, so the queue never empties.

WHAT MUST NOT BREAK. Rule 1 exists for a measured reason: on `run-l3-43-20260904-093730` the
best-ranked family stalled at [19.6, 19.6, 19.6] while the second-ranked went [19.5, 17.9, 17.9] and
produced the run's winner -- with `max_families_active=1`, ranking on latency would have DELETED the
winner. So the reservation yields ONE slot of several and only while an unproven family still gets
one. The tests below assert that boundary in both directions, because a reservation that quietly
starves rule 1 would trade a measurable S2d(c) for an unmeasurable regression in search quality.

Every test drives the REAL `active_families` on a REAL `FamilyManager`, with families built through
`register_candidate`/`record_best` rather than hand-assembled dicts.
"""

from __future__ import annotations

from kernel_optimizer.control.families import FamilyManager
from kernel_optimizer.models.core import BestRecord, ParamSet

# STRUCTURALLY distinct, not just different PARAMS values. `structural_signature` zeroes the PARAMS
# literal before hashing (that is its whole job -- two tunings of one kernel are one family), so
# sources differing only in `{'A': i}` are duplicates and `register_candidate` correctly returns None
# for all but the first. Varying the body is what makes these separate families, and getting that
# wrong is what the `assert cand is not None` in `_seed` caught.
_SRC = ("PARAMS = {'A': 1}\n\n\nclass ModelNew:\n    def forward(self, x):\n"
        "        return x %s PARAMS['A']\n")
_BODIES = ["*", "+", "-", "/", "**", "%%"]


def _mgr(active: int = 2, reserve: bool = False) -> FamilyManager:
    return FamilyManager(max_families_active=active, max_families_total=6,
                         max_families_total_hard=6, reserve_round_for_reconciled=reserve)


def _seed(m: FamilyManager, i: int, ms: float, rounds: int = 0) -> str:
    """One seed family with a correct incumbent, so `active_families` does not filter it out."""
    cand = m.register_candidate(_SRC % _BODIES[i], "seed", [], "triton", "a%d" % i)
    assert cand is not None
    fam = m.families[cand.family_id]
    # Fields copied from the model, not guessed: `BestRecord.params` is a ParamSet (whose own field
    # is `values`), `latency_ms` is a plain float, and there is no `space_id`/`trial_id`. Pydantic
    # said so loudly, which is the good case.
    fam.best = BestRecord(candidate_id=cand.candidate_id,
                          params=ParamSet(values={"A": i}), latency_ms=ms)
    fam.best_history = [ms]
    fam.rewrite_rounds_used = rounds
    return cand.family_id


def test_without_the_switch_rule_1_is_unconditional_and_s2d_c_never_fires():
    """The state the three finished runs and both arms were in: 4 unproven families, 2 slots, so the
    reconciled family is never revisited and the ledger never reaches a prompt."""
    m = _mgr(active=2, reserve=False)
    done = _seed(m, 0, 3.0, rounds=1)
    for i in (1, 2, 3):
        _seed(m, i, 3.5 + i)
    m.families_with_a_ledger.add(done)

    got = [f.family_id for f in m.active_families()]
    assert done not in got, (
        "rule 1 must keep the reconciled family out while unproven families remain: %s" % got)
    assert all(m.families[f].rewrite_rounds_used == 0 for f in got), got


def test_with_the_switch_a_reconciled_family_gets_one_of_the_slots():
    """The whole point: the ledger can now reach a rewriter prompt."""
    m = _mgr(active=2, reserve=True)
    done = _seed(m, 0, 3.0, rounds=1)
    for i in (1, 2, 3):
        _seed(m, i, 3.5 + i)
    m.families_with_a_ledger.add(done)

    got = [f.family_id for f in m.active_families()]
    assert done in got, "the reconciled family must get a slot: %s" % got
    assert len(got) == 2, got


def test_an_unproven_family_still_gets_a_slot_in_the_same_round():
    """Rule 1's protected case. The reservation yields ONE slot, never all of them -- otherwise it
    reintroduces exactly the early pruning that cost run-l3-43-20260904-093730 its winner."""
    m = _mgr(active=2, reserve=True)
    done = _seed(m, 0, 3.0, rounds=1)
    fresh = [_seed(m, i, 3.5 + i) for i in (1, 2, 3)]
    m.families_with_a_ledger.add(done)

    got = [f.family_id for f in m.active_families()]
    unproven = [f for f in got if m.families[f].rewrite_rounds_used == 0]
    assert unproven, "at least one never-rewritten family must still be activated: %s" % got
    assert set(got) & set(fresh), got


def test_with_only_one_unproven_family_left_the_reservation_declines():
    """The boundary that matters most. Swapping here would leave ZERO unproven families, which is
    the early-pruning failure. Postponing S2d(c) by a round is the correct trade."""
    m = _mgr(active=2, reserve=True)
    done = _seed(m, 0, 3.0, rounds=1)
    only = _seed(m, 1, 3.4)
    m.families_with_a_ledger.add(done)

    got = [f.family_id for f in m.active_families()]
    assert only in got, "the single unproven family must not be displaced: %s" % got
    # With 2 slots and 2 eligible families both are chosen anyway, which is the right outcome:
    # nothing was starved and the reconciled family is present.
    assert set(got) == {done, only}, got


def test_a_single_slot_never_yields_it():
    """`max_families_active=1` is the configuration where rule 1 is load-bearing on its own: giving
    the ONLY slot away deletes rule 1 rather than bending it."""
    m = _mgr(active=1, reserve=True)
    done = _seed(m, 0, 3.0, rounds=1)
    fresh = _seed(m, 1, 3.4)
    m.families_with_a_ledger.add(done)

    got = [f.family_id for f in m.active_families()]
    assert got == [fresh], "with one slot the unproven family keeps it: %s" % got


def test_a_family_with_no_ledger_is_not_reserved_for():
    """The reservation reads `families_with_a_ledger`, not `rewrite_rounds_used >= 1`. A family that had
    a round but was never reconciled has no ledger to carry, so revisiting it tests nothing."""
    m = _mgr(active=2, reserve=True)
    had_round = _seed(m, 0, 3.0, rounds=1)          # a round, but NOT reconciled
    for i in (1, 2, 3):
        _seed(m, i, 3.5 + i)

    got = [f.family_id for f in m.active_families()]
    assert had_round not in got, (
        "without a reconciliation there is no ledger to deliver, so rule 1 stands: %s" % got)


def test_an_empty_family_is_still_excluded_with_the_switch_on():
    """The other load-bearing exclusion: a family with no correct candidate cannot be rewritten at
    all, and activating one ended run-l3-21-20260905-071312 at 2.05 h of 12 h with 4 rounds unspent.
    The reservation must not smuggle one in."""
    m = _mgr(active=2, reserve=True)
    done = _seed(m, 0, 3.0, rounds=1)
    m.families_with_a_ledger.add(done)
    empty = m.register_candidate(_SRC % _BODIES[5], "seed", [], "triton", "empty")
    assert empty is not None
    m.families[empty.family_id].best = None        # nothing correct
    m.families_with_a_ledger.add(empty.family_id)     # and reconciled, so only `best` excludes it
    _seed(m, 1, 3.4)

    got = [f.family_id for f in m.active_families()]
    assert empty.family_id not in got, "a family with no correct candidate must stay excluded: %s" % got
    # And the empty family must be a family the filter would OTHERWISE have chosen, or this asserts
    # nothing: with `best=None` it sorts to the end on `_incumbent` (inf), so removing the filter
    # would not change a 2-slot slate that already has 2 better families. Give it the BEST possible
    # rank instead -- unproven (0 rounds) and, once the filter is gone, an incumbent that would put
    # it first -- so the exclusion is the only thing keeping it out.
    m2 = _mgr(active=1, reserve=True)
    lone = m2.register_candidate(_SRC % _BODIES[0], "seed", [], "triton", "empty-only")
    assert lone is not None
    m2.families[lone.family_id].best = None
    assert m2.active_families() == [], (
        "with its only family having nothing correct, the slate must be empty rather than "
        "containing an unrewritable family: %s" % [f.family_id for f in m2.active_families()])


def test_the_slate_never_exceeds_max_families_active():
    """A reservation that ADDED a slot would silently raise the rewrite budget, which is a different
    experiment from the one this switch declares."""
    m = _mgr(active=2, reserve=True)
    a = _seed(m, 0, 3.0, rounds=1)
    b = _seed(m, 1, 3.1, rounds=1)
    for i in (2, 3, 4):
        _seed(m, i, 3.5 + i)
    m.families_with_a_ledger.update({a, b})

    got = m.active_families()
    assert len(got) <= 2, [f.family_id for f in got]


def test_no_family_appears_twice_in_the_slate():
    """The swap builds a new list by slicing; an off-by-one there would activate the same family
    twice in one round, which double-charges its rewrite budget."""
    m = _mgr(active=3, reserve=True)
    done = _seed(m, 0, 3.0, rounds=1)
    for i in (1, 2, 3, 4):
        _seed(m, i, 3.5 + i)
    m.families_with_a_ledger.add(done)

    ids = [f.family_id for f in m.active_families()]
    assert len(ids) == len(set(ids)), ids

def test_selection_cannot_see_what_the_ledger_SAYS():
    """J2d-8, enforced behaviourally rather than by the text ban alone.

    The text ban in `test_j2d_8_the_expectation_field_is_absent_from_every_selection_module` greps the
    selection modules for the word, and this change is precisely the sort that would tempt someone to
    relax it. So the invariant is also asserted by execution: the reservation reads SET MEMBERSHIP and
    nothing else, so a family whose ledger is all hits and one whose ledger is all misses must produce
    the IDENTICAL slate. If a hit rate ever leaks into ranking, this fails.

    Two managers built identically, with the same families in the same order -- only the imagined
    content of the ledger differs, and the manager has no channel to see it.
    """
    slates = []
    for _ledger_quality in ("all hits", "all misses"):
        m = _mgr(active=2, reserve=True)
        done = _seed(m, 0, 3.0, rounds=1)
        for i in (1, 2, 3):
            _seed(m, i, 3.5 + i)
        # The same membership either way. There is deliberately no argument by which the caller
        # could pass a hit count, which is the point: the type is `set[str]`.
        m.families_with_a_ledger.add(done)
        # Compared by ROLE, not by family_id: ids are uuid-random per manager, so comparing them
        # across two managers compares the uuid generator. The identity that matters is "which
        # families, by incumbent latency and by whether they have a ledger".
        slates.append([(f.family_id == done, f.best.latency_ms, f.rewrite_rounds_used)
                       for f in m.active_families()])

    assert slates[0] == slates[1], (
        "the slate differed between an all-hit and an all-miss ledger, so accuracy reached "
        "allocation: %s vs %s" % tuple(slates))
    assert any(has_ledger for has_ledger, _ms, _r in slates[0]), (
        "the test is vacuous unless the ledger family is actually in the slate: %s" % slates[0])


def test_the_set_is_only_ids():
    """A `set[str]` cannot carry a direction, a hit count or a rate. Asserted so a later change to
    `dict[str, ...]` -- which would make a hit rate reachable from allocation -- fails here."""
    m = _mgr(active=2, reserve=True)
    assert isinstance(m.families_with_a_ledger, set), type(m.families_with_a_ledger)
    m.families_with_a_ledger.add("fam-1")
    assert m.families_with_a_ledger == {"fam-1"}
    assert all(isinstance(x, str) for x in m.families_with_a_ledger)


def test_a_missing_set_cannot_damage_the_ledger_it_describes():
    """The defect this change actually had, pinned.

    The orchestrator's set update sat INSIDE `_record_reconciliation`'s
    `except Exception` -- and after the `self.ledger[...].append(entry)`. A `families` collaborator
    without the attribute (several tests use a SimpleNamespace) raised AttributeError, the blanket
    except swallowed it, and the round's SECOND ledger entry was never appended: an unrelated wiring
    test dropped from 2 entries to 1 with no error anywhere. A support field must never be able to
    damage the thing it describes.

    Drives the real `_record_reconciliation` with a `families` object that has no such attribute.
    """
    import pytest
    pytest.importorskip("optuna")     # the orchestrator imports the tuner
    from types import SimpleNamespace

    from kernel_optimizer.control.orchestrator import Orchestrator, _DeclaredExpectations
    from kernel_optimizer.models.reports import ResourceExpectation

    class _Store:
        def __init__(self):
            self.events = []

        def append(self, t, p):
            self.events.append((t, p))

    # Fields copied from `test_s2d_wiring._orch`, which drives this same method: `runs` is what
    # per-candidate attribution reads, and omitting it raised
    # "'Orchestrator' object has no attribute 'runs'" -- swallowed by the same blanket except and
    # reported as a reconciliation failure, which is exactly the confusion this test is about.
    o = Orchestrator.__new__(Orchestrator)
    o.store = _Store()
    o.ledger = {}
    o.round_expectations = {}
    o.runs = {}
    # No `families_with_a_ledger` anywhere on it, which is the scenario under test.
    o.deps = SimpleNamespace(families=SimpleNamespace(families={}))
    o.cfg = SimpleNamespace(
        budgets=SimpleNamespace(min_improvement_pct=2.0),
        v3=SimpleNamespace(diagnosis=SimpleNamespace(expectation_ledger=True)))
    for cid, hyp, direction in (("cand-a", "H1", "down"), ("cand-b", "H2", "up")):
        o.round_expectations.setdefault("fam-1", []).append(
            _DeclaredExpectations(
                hypothesis_id=hyp, change_summary="c", candidate_id=cid,
                expectation=ResourceExpectation(dimension="n_regs", expect=direction, why="w")))
    conv = {"conversion": "improved", "latency_gain_pct": 5.0,
            "latency_ms_before": 3.2, "latency_ms_after": 3.0,
            "resource_deltas": {"n_regs": {"before": 155.0, "after": 140.0,
                                           "delta": -15.0, "rel": 0.0968}}}

    o._record_reconciliation("fam-1", 0, conv, SimpleNamespace(n_regs=155))

    entries = o.ledger.get("fam-1") or []
    assert len(entries) == 2, (
        "a missing families_with_a_ledger must not cost the round an entry; got %d and "
        "RECONCILE_FAILED=%r" % (len(entries),
                                 [p for t, p in o.store.events if "FAILED" in t]))
    assert not [t for t, _ in o.store.events if t == "EXPECTATIONS_RECONCILE_FAILED"], (
        "and it must not be reported as a reconciliation failure either: %s" % o.store.events)

def test_the_LIVE_reconciliation_marks_the_family_in_production():
    """Every test above populates `families_with_a_ledger` directly, so all of them pass on an
    orchestrator that never fills it -- the switch would then be inert in production and green in CI.
    This drives the real `_record_reconciliation` against a real `FamilyManager`.
    """
    import pytest
    pytest.importorskip("optuna")
    from types import SimpleNamespace

    from kernel_optimizer.control.orchestrator import Orchestrator, _DeclaredExpectations
    from kernel_optimizer.models.reports import ResourceExpectation

    class _Store:
        def __init__(self):
            self.events = []

        def append(self, t, p):
            self.events.append((t, p))

    m = _mgr(active=2, reserve=True)
    fid = _seed(m, 0, 3.0, rounds=1)

    o = Orchestrator.__new__(Orchestrator)
    o.store = _Store()
    o.ledger = {}
    o.runs = {}
    o.round_expectations = {fid: [_DeclaredExpectations(
        hypothesis_id="H1", change_summary="c", candidate_id="cand-a",
        expectation=ResourceExpectation(dimension="n_regs", expect="down", why="w"))]}
    o.deps = SimpleNamespace(families=m)
    o.cfg = SimpleNamespace(
        budgets=SimpleNamespace(min_improvement_pct=2.0),
        v3=SimpleNamespace(diagnosis=SimpleNamespace(expectation_ledger=True)))

    assert fid not in m.families_with_a_ledger, "precondition: nothing reconciled yet"

    o._record_reconciliation(fid, 0, {
        "conversion": "improved", "latency_gain_pct": 5.0,
        "latency_ms_before": 3.2, "latency_ms_after": 3.0,
        "resource_deltas": {"n_regs": {"before": 155.0, "after": 140.0,
                                       "delta": -15.0, "rel": 0.0968}}},
        SimpleNamespace(n_regs=155))

    assert fid in m.families_with_a_ledger, (
        "production must mark the family, or the reservation can never fire in a real run: %s"
        % m.families_with_a_ledger)


def test_the_reservation_survives_a_resume():
    """Without a restore, the reservation reverts to rule 1 on the resumed half of a run and the run
    reports S2d(c) as administered having administered it only before the interrupt -- nothing in the
    log distinguishes the halves. Drives the real `_restore_family_control_state` off a real store.
    """
    import tempfile
    from pathlib import Path

    import pytest
    pytest.importorskip("optuna")
    from types import SimpleNamespace

    from kernel_optimizer.control.orchestrator import Orchestrator
    from kernel_optimizer.store.run_store import RunStore

    tmp = Path(tempfile.mkdtemp())
    store = RunStore.create(tmp, "run-s2dc", {})
    m = _mgr(active=2, reserve=True)
    fid = _seed(m, 0, 3.0, rounds=1)

    o = Orchestrator.__new__(Orchestrator)
    o.store = store
    o.runs = {}
    o.failed_hypotheses = {}
    o.round_expectations = {}
    o.round_hypotheses = {}
    o.ledger = {}
    o.deps = SimpleNamespace(families=m)

    store.append("FAMILY_ROUND_RECORDED", {"family_id": fid, "best_ms": 3.0, "round": 1})
    store.append("EXPECTATIONS_RECONCILED", {
        "family_id": fid, "round": 1, "candidate_id": "cand-a", "id": "H1", "change": "c",
        "reconciliation": {"hypothesis_id": "H1", "hits": 1, "misses": 0, "vacuous": 0,
                           "per_dimension": []},
        "conversion": "improved", "latency_gain_pct": 5.0, "n_declared": 1})

    o._restore_family_control_state()

    assert o.ledger.get(fid), "the ledger itself must be restored"
    assert fid in m.families_with_a_ledger, (
        "and the reservation must know about it, or S2d(c) is half-administered across the "
        "interrupt: %s" % m.families_with_a_ledger)


def test_the_switch_needs_a_ledger_to_take_effect(monkeypatch):
    """`reserve_round_for_reconciled` without `expectation_ledger` would change the search order for a
    run that cannot use a ledger -- comparability spent for nothing.

    Drives the REAL wiring. A first version recomputed the `and` in the test body, which asserted my
    own arithmetic and passed on any `wiring.py` whatsoever: its revert-variant patched the real
    conjunction out and the test never noticed. `build_orchestrator` needs a live opencode server, so
    this calls it with a stubbed `FamilyManager` and reads the kwarg the wiring actually passed,
    aborting the rest of the construction immediately afterwards.
    """
    import pytest
    pytest.importorskip("optuna")

    from kernel_optimizer import wiring as wiring_mod
    from kernel_optimizer.config import AppConfig

    from pathlib import Path

    from kernel_optimizer.models.core import TaskSpec

    seen: dict[str, object] = {}

    class _Stop(Exception):
        pass

    class _FakeStore:
        run_dir = Path(".")

    class _FakeRuntime:
        client = object()

    def _fake_manager(**kw):
        seen.update(kw)
        raise _Stop()          # nothing after this line matters to the assertion

    def _build(reserve: bool, ledger: bool):
        seen.clear()
        cfg = AppConfig()
        cfg.v3.diagnosis.reserve_round_for_reconciled = reserve
        cfg.v3.diagnosis.expectation_ledger = ledger
        monkeypatch.setattr(wiring_mod, "FamilyManager", _fake_manager)
        monkeypatch.setattr(wiring_mod, "build_gpu_stack",
                            lambda *a, **k: (None, None, None, None))
        monkeypatch.setattr(wiring_mod, "SpaceValidator", lambda *a, **k: None)
        monkeypatch.setattr(wiring_mod, "TuningStatsAnalyzer", lambda *a, **k: None)
        try:
            wiring_mod.build_orchestrator(
                cfg, _FakeStore(),
                TaskSpec(level=3, problem_id=43, name="43_MinGPTCausalAttention",
                         ref_path=Path("ref.py"), ref_src_sha="0" * 64),
                _FakeRuntime())
        except _Stop:
            pass
        assert "reserve_round_for_reconciled" in seen, (
            "the wiring must pass the flag at all: %s" % sorted(seen))
        return seen["reserve_round_for_reconciled"]

    assert _build(reserve=True, ledger=False) is False, (
        "no ledger means no reservation, or a run changes its search order for nothing")
    assert _build(reserve=True, ledger=True) is True
    assert _build(reserve=False, ledger=True) is False

    # And the defaults must be off on both, or the three finished runs stop being comparable
    # with the next one.
    fresh = AppConfig()
    assert fresh.v3.diagnosis.reserve_round_for_reconciled is False
    assert fresh.v3.diagnosis.expectation_ledger is False
