"""Resume must restore memory-only control state from the event log.

Regression for the level3:43 run: a mid-rewrite crash + resume left every
family with crun.report is None and an empty best_history, so Loop C silently
burned rewrite rounds without ever calling the rewriter (spurious
budget_exhausted) and the `converged` stop_kind became unreachable. Both
bottleneck reports and family round history are in-memory-only pipeline
products; resume rebuilds them from BOTTLENECK_REPORTED / FAMILY_ROUND_RECORDED.
"""

from kernel_optimizer.control.families import FamilyManager
from kernel_optimizer.control.orchestrator import CandidateRun, Orchestrator
from kernel_optimizer.models.core import Candidate, Family
from kernel_optimizer.store.run_store import RunStore


def _orch(store: RunStore, families: FamilyManager) -> Orchestrator:
    orch = Orchestrator.__new__(Orchestrator)  # skip real wiring/GPU/agents
    orch.store = store
    orch.runs = {}
    # S2d state the restore fills. Set here rather than in each test so a test that forgets one gets
    # an AttributeError from the real code path instead of quietly exercising a different branch.
    orch.failed_hypotheses = {}
    orch.round_expectations = {}
    orch.ledger = {}
    orch.round_hypotheses = {}

    class _Deps:
        pass

    deps = _Deps()
    deps.families = families
    orch.deps = deps  # type: ignore[attr-defined]
    return orch


def _candidate(cid: str, fam_id: str) -> Candidate:
    return Candidate(candidate_id=cid, family_id=fam_id, origin="seed",
                     backend="triton", source_sha="x", structural_signature="sig")


def test_restore_pipeline_recovers_bottleneck_report(tmp_path):
    store = RunStore.create(tmp_path, "run-r", {})
    families = FamilyManager()
    orch = _orch(store, families)

    cand = _candidate("cand-1", "fam-1")
    crun = CandidateRun(candidate=cand, source="x = 1\n")
    # The analyst emitted a report in the pre-crash process; only the event survives.
    store.append("BOTTLENECK_REPORTED", {
        "candidate_id": "cand-1",
        "report": {"summary": "register bound", "suggested_action": "rewrite"},
    })

    orch._restore_pipeline(crun, store.replay())

    assert crun.report is not None, "report must be recovered so Loop C can rewrite"
    assert crun.report.summary == "register bound"
    assert crun.report.suggested_action == "rewrite"


def test_restore_pipeline_report_none_when_no_event(tmp_path):
    store = RunStore.create(tmp_path, "run-r", {})
    orch = _orch(store, FamilyManager())
    crun = CandidateRun(candidate=_candidate("cand-1", "fam-1"), source="x = 1\n")
    orch._restore_pipeline(crun, store.replay())
    assert crun.report is None


def test_restore_family_control_state_rebuilds_history_and_rounds(tmp_path):
    store = RunStore.create(tmp_path, "run-r", {})
    families = FamilyManager()
    families.families["fam-1"] = Family(
        family_id="fam-1", anchor_candidate_id="cand-1", member_ids=["cand-1"])
    orch = _orch(store, families)

    # Two completed rewrite rounds were persisted before the crash.
    store.append("FAMILY_ROUND_RECORDED", {"family_id": "fam-1", "best_ms": 30.0, "round": 1})
    store.append("FAMILY_ROUND_RECORDED", {"family_id": "fam-1", "best_ms": 29.0, "round": 2})

    orch._restore_family_control_state()

    fam = families.families["fam-1"]
    assert fam.best_history == [30.0, 29.0]
    assert fam.rewrite_rounds_used == 2, "budget accounting must survive resume"


def test_restore_family_control_state_no_rounds_is_zero(tmp_path):
    """A crash *inside* the first rewrite (before FAMILY_ROUND_RECORDED) must leave
    rewrite_rounds_used at 0 so the round is retried, not silently counted."""
    store = RunStore.create(tmp_path, "run-r", {})
    families = FamilyManager()
    families.families["fam-1"] = Family(
        family_id="fam-1", anchor_candidate_id="cand-1", member_ids=["cand-1"])
    orch = _orch(store, families)

    orch._restore_family_control_state()

    fam = families.families["fam-1"]
    assert fam.best_history == []
    assert fam.rewrite_rounds_used == 0


# --------------------------------------------------------------------------------------------------
# S2d: declarations from an IN-FLIGHT round, which a resume could not recover at all
# --------------------------------------------------------------------------------------------------

def _rewrite_produced(cand: str, fam: str, hyp: str, *pairs: tuple[str, str]) -> dict:
    """A `REWRITE_PRODUCED` payload, field names copied from the emitter (orchestrator.py:2437)."""
    return {"candidate_id": cand, "family_id": fam, "hypothesis_id": hyp,
            "change_summary": "fuse the projections",
            "expectations": [{"dimension": d, "expect": e, "why": "because"} for d, e in pairs]}


def test_declarations_from_an_unreconciled_round_are_restored(tmp_path):
    """The gap that made box 2's pooled-ledger defect unfixable by restarting.

    `EXPECTATIONS_RECONCILED` is written in the same breath that POPS `round_expectations`, so an
    in-flight round has declarations and no reconciliation -- and restoring only from that event
    recovers nothing. The declarations are in `REWRITE_PRODUCED`, journalled at declaration time.

    Without this the resumed round reconciles against an empty buffer: `n_declared=0`, no hits, no
    misses, and a log indistinguishable from an agent that declared nothing. Measured: box 2's
    `run-l3-43-20260911-052630` held 16 declarations -- S2d's only production set -- in memory alone.
    """
    store = RunStore.create(tmp_path, "run-r", {})
    families = FamilyManager()
    families.families["fam-1"] = Family(
        family_id="fam-1", anchor_candidate_id="cand-1", member_ids=["cand-1"])
    orch = _orch(store, families)

    store.append("REWRITE_PRODUCED", _rewrite_produced(
        "cand-win", "fam-1", "H1+H3", ("shared_bytes", "up"), ("n_regs", "unknown")))
    store.append("REWRITE_PRODUCED", _rewrite_produced(
        "cand-other", "fam-1", "H2", ("shared_bytes", "unchanged")))

    orch._restore_family_control_state()

    decl = orch.round_expectations["fam-1"]
    assert len(decl) == 3, "all three declarations, across BOTH candidates of the round"
    # Attributed per candidate, which is what reconciliation groups by -- a restore that lost the
    # candidate id would put the round straight back into the pooled defect.
    assert {d.candidate_id for d in decl} == {"cand-win", "cand-other"}
    assert {d.hypothesis_id for d in decl} == {"H1+H3", "H2"}
    # Rehydrated as the real model, so a restored declaration reconciles by the same path as a live
    # one. A dict shim passes `reconcile` and diverges in the orchestrator.
    from kernel_optimizer.models.reports import ResourceExpectation
    assert all(isinstance(d.expectation, ResourceExpectation) for d in decl)
    assert {d.expectation.expect for d in decl} == {"up", "unknown", "unchanged"}
    assert store.replay().events[-1].type == "EXPECTATIONS_RESTORED"


def test_an_already_reconciled_candidate_is_not_restored(tmp_path):
    """The other half, and the one that matters more: re-restoring a reconciled candidate would score
    the NEXT round against this round's predictions -- the exact defect the `pop` exists to prevent,
    reintroduced through the resume path."""
    store = RunStore.create(tmp_path, "run-r", {})
    families = FamilyManager()
    families.families["fam-1"] = Family(
        family_id="fam-1", anchor_candidate_id="cand-1", member_ids=["cand-1"])
    orch = _orch(store, families)

    store.append("REWRITE_PRODUCED", _rewrite_produced(
        "cand-done", "fam-1", "H1", ("shared_bytes", "up")))
    store.append("REWRITE_PRODUCED", _rewrite_produced(
        "cand-live", "fam-1", "H2", ("n_regs", "down")))
    store.append("FAMILY_ROUND_RECORDED", {"family_id": "fam-1", "best_ms": 3.0, "round": 0})
    store.append("EXPECTATIONS_RECONCILED", {
        "family_id": "fam-1", "round": 0, "candidate_id": "cand-done", "id": "H1",
        "change": "c", "reconciliation": {"hits": 1, "misses": 0, "vacuous": 0}, "n_declared": 1})

    orch._restore_family_control_state()

    decl = orch.round_expectations.get("fam-1", [])
    assert [d.candidate_id for d in decl] == ["cand-live"], (
        "the reconciled candidate is retired; only the un-reconciled one comes back")
    assert orch.ledger["fam-1"][0]["candidate_id"] == "cand-done"


def test_a_pre_fix_pooled_entry_retires_its_familys_declarations(tmp_path):
    """A resumed OLD run's ledger entries name no candidate, so there is no per-candidate rule to
    apply. Retiring the whole family's declarations is the conservative direction: re-scoring a round
    already in the ledger is worse than losing an in-flight one, because a wrong ledger reaches the
    agent while a missing one only reads as `n_declared=0`."""
    store = RunStore.create(tmp_path, "run-r", {})
    families = FamilyManager()
    families.families["fam-1"] = Family(
        family_id="fam-1", anchor_candidate_id="cand-1", member_ids=["cand-1"])
    orch = _orch(store, families)

    store.append("REWRITE_PRODUCED", _rewrite_produced(
        "cand-a", "fam-1", "H1", ("shared_bytes", "up")))
    store.append("EXPECTATIONS_RECONCILED", {           # no candidate_id: the pre-fix shape
        "family_id": "fam-1", "round": 0, "id": "H1", "change": "c",
        "reconciliation": {"hits": 1, "misses": 0}, "n_declared": 1})

    orch._restore_family_control_state()

    assert orch.round_expectations.get("fam-1", []) == []
    assert "EXPECTATIONS_RESTORED" not in [e.type for e in store.replay().events]


def test_a_rewrite_that_declared_nothing_restores_nothing(tmp_path):
    """An empty `expectations` list is a real state -- "declared nothing" -- and must not produce a
    restore event, which would make the event's own presence meaningless as a signal."""
    store = RunStore.create(tmp_path, "run-r", {})
    families = FamilyManager()
    families.families["fam-1"] = Family(
        family_id="fam-1", anchor_candidate_id="cand-1", member_ids=["cand-1"])
    orch = _orch(store, families)

    store.append("REWRITE_PRODUCED", _rewrite_produced("cand-a", "fam-1", "H1"))
    orch._restore_family_control_state()

    assert orch.round_expectations == {}
    assert "EXPECTATIONS_RESTORED" not in [e.type for e in store.replay().events]


def test_the_rounds_attempted_hypotheses_are_restored_even_with_no_expectations(tmp_path):
    """`round_hypotheses` decides which hypotheses a failed round may mark as failed, so losing it on
    resume lets an idea nobody implemented be retired permanently.

    Deliberately with an EMPTY expectations list: the two are independent, and a rewrite that declared
    nothing still IMPLEMENTED its hypothesis.
    """
    store = RunStore.create(tmp_path, "run-r", {})
    families = FamilyManager()
    families.families["fam-1"] = Family(
        family_id="fam-1", anchor_candidate_id="cand-1", member_ids=["cand-1"])
    orch = _orch(store, families)

    store.append("REWRITE_PRODUCED", _rewrite_produced("cand-a", "fam-1", "H1"))
    store.append("REWRITE_PRODUCED", _rewrite_produced("cand-b", "fam-1", "H3"))
    orch._restore_family_control_state()

    assert orch.round_hypotheses["fam-1"] == {"H1", "H3"}


def test_a_closed_round_does_not_leak_its_attempts_into_the_next(tmp_path):
    """The retirement signal is END OF ROUND, not `HYPOTHESES_FAILED`.

    `HYPOTHESES_FAILED` fires only when a round fails to IMPROVE. An improving round emits none, so
    keying retirement on it would leave round 0's attempts in the set -- and then a genuinely untried
    hypothesis in round 1 would look "attempted" and be markable as failed. That is the same
    never-tried-marked-failed defect, arriving by a different route.
    """
    store = RunStore.create(tmp_path, "run-r", {})
    families = FamilyManager()
    families.families["fam-1"] = Family(
        family_id="fam-1", anchor_candidate_id="cand-1", member_ids=["cand-1"])
    orch = _orch(store, families)

    # Round 0 IMPROVED: recorded, no HYPOTHESES_FAILED.
    store.append("REWRITE_PRODUCED", _rewrite_produced("cand-a", "fam-1", "H1"))
    store.append("FAMILY_ROUND_RECORDED", {"family_id": "fam-1", "best_ms": 3.0, "round": 0})
    # Round 1 is in flight, and implemented only H2.
    store.append("REWRITE_PRODUCED", _rewrite_produced("cand-b", "fam-1", "H2"))

    orch._restore_family_control_state()

    assert orch.round_hypotheses["fam-1"] == {"H2"}, (
        "round 0's H1 must not still be in the set: round 1 never implemented it")


def test_a_round_that_evaluated_nothing_also_retires_its_attempts(tmp_path):
    """`FAMILY_ROUND_NOT_EVALUATED` closes a round too -- it consumed budget. Leaving the set behind
    would carry a failed round's attempts into the next one."""
    store = RunStore.create(tmp_path, "run-r", {})
    families = FamilyManager()
    families.families["fam-1"] = Family(
        family_id="fam-1", anchor_candidate_id="cand-1", member_ids=["cand-1"])
    orch = _orch(store, families)

    store.append("REWRITE_PRODUCED", _rewrite_produced("cand-a", "fam-1", "H1"))
    store.append("FAMILY_ROUND_NOT_EVALUATED", {"family_id": "fam-1", "round": 0, "best_ms": 3.0})

    orch._restore_family_control_state()

    assert orch.round_hypotheses.get("fam-1", set()) == set()


# --------------------------------------------------------------------------------------------------
# Which hypotheses a failed round may mark FAILED
# --------------------------------------------------------------------------------------------------

def _hyps(*ids):
    from kernel_optimizer.models.reports import Hypothesis
    return [Hypothesis(id=i, change="change %s" % i, expected_effect="faster") for i in ids]


def test_a_failed_round_marks_only_the_hypotheses_it_implemented():
    """Measured 3 of 3 on the rounds in the corpus that reached this branch:

        fam-c4d585ab  rewriter produced H1,H2  ->  marked H1,H2,H3 failed  ->  H3 never tried
        fam-d5c28c50  rewriter produced H1,H2  ->  marked H1,H2,H3 failed  ->  H3 never tried
        fam-4e6e7d9e  rewriter produced H1,H1  ->  marked H1,H2   failed  ->  H2 never tried

    The analyst proposes 2-4 and the rewriter implements 2, so the whole list is never the right
    answer. `failed_hypotheses` is journalled, replayed, and read by the rewriter as "already tried,
    did NOT help" -- so a wrong entry retires a live idea permanently, on the strength of a sibling
    candidate's failure.
    """
    from kernel_optimizer.control.orchestrator import _split_attempted

    tried, untried = _split_attempted(_hyps("H1", "H2", "H3"), {"H1", "H2"}, 1)
    assert [t["id"] for t in tried] == ["H1", "H2"]
    assert untried == ["H3"]
    # The payload keeps the change text, which is what the rewriter actually reads.
    assert tried[0]["change"] == "change H1" and tried[0]["round"] == 1


def test_both_candidates_implementing_the_same_hypothesis_retires_only_that_one():
    """`fam-4e6e7d9e` produced H1 twice. Two attempts at one idea is still evidence about one idea --
    and H2, which the analyst proposed and nobody wrote, must survive."""
    from kernel_optimizer.control.orchestrator import _split_attempted

    tried, untried = _split_attempted(_hyps("H1", "H2"), {"H1"}, 2)
    assert [t["id"] for t in tried] == ["H1"]
    assert untried == ["H2"]


def test_no_declared_hypothesis_id_leaves_the_list_untouched():
    """Seen twice in the corpus: both candidates of a round carried `hypothesis_id: ""`.

    With no declaration there is no basis for saying which idea was tried, so nothing is filtered and
    the pre-fix behaviour stands. Over-marking is the lesser error HERE, and only here: the round did
    fail, and its ideas came from this report. Marking none would lose the round's evidence as surely
    as marking all of them fabricates it.
    """
    from kernel_optimizer.control.orchestrator import _split_attempted

    tried, untried = _split_attempted(_hyps("H1", "H2", "H3"), set(), 0)
    assert [t["id"] for t in tried] == ["H1", "H2", "H3"]
    assert untried == []


def test_an_attempt_naming_a_hypothesis_the_analyst_never_proposed_marks_nothing_extra():
    """The rewriter is free to answer with an id of its own. That must not inject a hypothesis into
    `failed_hypotheses` that has no `change` text -- the rewriter would read a blank entry."""
    from kernel_optimizer.control.orchestrator import _split_attempted

    tried, untried = _split_attempted(_hyps("H1", "H2"), {"H1", "H9"}, 3)
    assert [t["id"] for t in tried] == ["H1"]
    assert untried == ["H2"]


def test_the_live_and_resume_paths_agree_on_the_attempt_set(tmp_path):
    """`round_hypotheses` is populated TWICE by independent code: `_do_rewrite` fills it live from the
    rewriter's answer, and `_restore_family_control_state` rebuilds it from `REWRITE_PRODUCED`. Two
    readers of the same fact drift, and a drift here is silent -- the live run marks one set failed and
    a resumed run marks another.

    The live line needs a real rewriter agent to reach, so it cannot be driven directly. What CAN be
    checked is that both derive the set from the same field of the same event: the resume path is run
    for real, and the live path's contribution is reproduced from the events it would have written.
    Asserting equality of the two is weaker than driving both, and it is stated as such rather than
    dressed up -- but it is what catches the two paths reading different fields.
    """
    store = RunStore.create(tmp_path, "run-r", {})
    families = FamilyManager()
    families.families["fam-1"] = Family(
        family_id="fam-1", anchor_candidate_id="cand-1", member_ids=["cand-1"])
    orch = _orch(store, families)

    events = [_rewrite_produced("cand-a", "fam-1", "H1", ("n_regs", "down")),
              _rewrite_produced("cand-b", "fam-1", "H3")]
    for p in events:
        store.append("REWRITE_PRODUCED", p)

    orch._restore_family_control_state()

    # What the live path adds, per its own line: `hypothesis_id` of each produced rewrite.
    live = {p["hypothesis_id"] for p in events}
    assert orch.round_hypotheses["fam-1"] == live

    # And the set feeds the filter identically either way.
    from kernel_optimizer.control.orchestrator import _split_attempted
    from_resume, _ = _split_attempted(_hyps("H1", "H2", "H3"),
                                      {h for h in orch.round_hypotheses["fam-1"] if h}, 0)
    from_live, _ = _split_attempted(_hyps("H1", "H2", "H3"), {h for h in live if h}, 0)
    assert [t["id"] for t in from_resume] == [t["id"] for t in from_live] == ["H1", "H3"]
