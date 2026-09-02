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
