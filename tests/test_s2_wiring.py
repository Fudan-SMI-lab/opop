"""S2 wiring: G28 at the real prompt boundary, and the mode switch at the real call site.

The unit tests in `test_s2_dimensions.py` drive `digest.for_prompt`. That is not enough for G28: the
defect is in `agents/modules.py`, which renders `verdict.evidence` key by key into the analyst's
document and writes `## Verdict: **{kind}**`. A gate tested only on `for_prompt` would leave that
render untouched and every test still green -- the same shape as asserting on source text instead of
behaviour, which this repo has a recorded instance of (a test that passed on the buggy version and
FAILED after the fix).

So these tests drive `_bottleneck_doc` and `BottleneckAnalystAgent.seed_sandbox` -- the functions that
actually build what the agent reads.
"""

from __future__ import annotations

from kernel_optimizer.agents.modules import AnalystInputs, _bottleneck_doc
from kernel_optimizer.evaluation.digest import digest, for_prompt
from kernel_optimizer.evaluation.dimensions import state_from_evidence, unreachable_ceilings
from tests.test_s2_dimensions import A800, l3_48_evidence


class _Verdict:
    """The shape `_bottleneck_doc` reads off a BottleneckVerdict. A stand-in rather than the real
    class only because constructing one needs a full DevicePeaks; every field read here is present
    with the production name, so a rename still breaks these tests."""

    def __init__(self, evidence: dict, unmeasured=()):
        self.kind = "memory_bound"
        self.evidence = evidence
        self.suggests = "the kernel is near this GPU's measured DRAM ceiling"
        self.disagreement = ""
        self.unmeasured = tuple(unmeasured)


def _digest_text() -> str:
    ev = l3_48_evidence()
    return for_prompt(digest(state_from_evidence(ev, device_limits=A800),
                             unreachable=unreachable_ceilings(ev)))


# --------------------------------------------------------------------------------------------------
# G28 / J2-3: label mode is the OLD behaviour, and vector mode must not carry it
# --------------------------------------------------------------------------------------------------

def test_label_mode_still_renders_the_label_and_its_evidence():
    """The baseline, asserted so the next test means something.

    `v3.diagnosis.mode: label` is v2's behaviour and the control arm. If this ever stopped rendering
    the label, the control run would not be comparing the new shape against the old one -- it would
    be comparing it against a third thing.
    """
    doc = _bottleneck_doc(_Verdict(l3_48_evidence()), None, None, None)
    assert "## Verdict: **memory_bound**" in doc
    assert "pct_of_dram_peak" in doc, "label mode renders the evidence keys; that is the baseline"


def test_vector_mode_removes_the_label_and_every_raw_evidence_key():
    """G28. The rendered document in vector mode must contain neither the single `kind` nor the
    key-by-key evidence dump.

    Revert-checked against `_bottleneck_doc` ignoring its `digest_text` argument: the label and the
    keys both survive and this test fails on all four assertions.
    """
    doc = _bottleneck_doc(_Verdict(l3_48_evidence()), None, None, _digest_text())
    assert "## Verdict:" not in doc
    assert "memory_bound" not in doc
    for key in ("pct_of_dram_peak", "occupancy_limiter", "shared_used_frac", "threads_launched="):
        assert key not in doc, f"raw evidence key {key} reached the agent document"
    # ...and the digested form IS there, so this is a change of form and not a deletion (D-9).
    assert "one line per dimension" in doc


def test_vector_mode_keeps_the_unmeasurable_list():
    """"unknown on this box" and "measured and fine" are different states, and an agent told the
    first reasons differently from one told the second.

    This is the one part of the label block that must survive the switch: dropping it would remove a
    caveat rather than change a form, and the control run would then differ in two ways at once.
    """
    v = _Verdict(l3_48_evidence(), unmeasured=["bank conflicts", "warp stall reasons"])
    doc = _bottleneck_doc(v, None, None, _digest_text())
    assert "CANNOT see" in doc
    assert "bank conflicts" in doc
    assert "not treat their absence as evidence" in doc


def test_the_analyst_sandbox_document_is_the_digested_one_in_vector_mode(tmp_path):
    """The end of the real path: what the file in the sandbox actually contains.

    Drives `seed_sandbox`, so a future change that renders the vector somewhere else in the sandbox
    (a second file, an extra prompt section) is caught here rather than passing because
    `_bottleneck_doc` alone stayed clean.
    """
    from kernel_optimizer.agents.modules import BottleneckAnalystAgent
    from kernel_optimizer.agents.sandbox import Sandbox
    from kernel_optimizer.models.core import DeviceLimits, TaskSpec
    from kernel_optimizer.models.reports import TuningStats

    sb = Sandbox(root=tmp_path / "sb")
    agent = BottleneckAnalystAgent.__new__(BottleneckAnalystAgent)
    inputs = AnalystInputs(
        task=TaskSpec(level=3, problem_id=48, name="48_Mamba2ReturnFinalState",
                      ref_path="KernelBench/level3/48.py", ref_src_sha="0" * 64),
        candidate_source="PARAMS = {}\n",
        stats=TuningStats(space_id="s", candidate_id="cand-1", n_trials=0,
                          n_complete=0, n_fail=0),
        trials_csv="a,b\n",
        device=DeviceLimits(**A800),
        bottleneck_verdict=_Verdict(l3_48_evidence()),
        digest_text=_digest_text(),
    )
    agent.seed_sandbox(inputs, sb)

    text = (sb.root / "analysis" / "bottleneck.md").read_text(encoding="utf-8")
    assert "## Verdict:" not in text
    assert "pct_of_dram_peak" not in text
    assert "one line per dimension" in text
    # Nothing else the agent is given may carry the raw vector either -- a leak through a second
    # file would satisfy every other assertion here.
    seen = 0
    for p in sb.root.rglob("*"):
        if p.is_file():
            seen += 1
            body = p.read_text(encoding="utf-8", errors="replace")
            assert "pct_of_dram_peak" not in body, f"raw vector leaked via {p.name}"
    assert seen >= 5, "seed_sandbox wrote almost nothing; the sweep above would be vacuous"


# --------------------------------------------------------------------------------------------------
# The mode switch, at the orchestrator's call site
# --------------------------------------------------------------------------------------------------

class _Store:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def append(self, kind, payload):
        self.events.append((kind, payload))

    def kinds(self):
        return [k for k, _ in self.events]

    def payload(self, kind):
        return next(p for k, p in self.events if k == kind)


def _orch(mode: str):
    """An Orchestrator with only what `_dimension_digest` touches, so the test exercises the real
    method rather than a re-implementation of it (D-5).

    `importorskip("optuna")` because importing `orchestrator` pulls in `tuning/tpe.py`. The Windows
    host has no optuna or torch by design, so these six tests SKIP there and run on the A800, which
    is the authoritative suite. Skipping is right and asserting-around is not: a hand-built stand-in
    for the orchestrator would test the stand-in.

    `task_cost` and `calibration` are set because a real Orchestrator always has them (assigned in
    `__init__`) -- S3 made `_dimension_digest` read both, and a fixture missing them produced
    `DIMENSION_STATE_FAILED` on the A800 while the local run skipped. That is the G41 shape again: a
    fixture that is not production-shaped. They are set to None rather than to objects, because None
    is a REAL state (an uncalibrated box, a run whose task cost failed) and the bounds must then read
    `source="none"` rather than raising -- so this fixture exercises the degraded path, and
    `test_the_s3_bound_reaches_the_journalled_record` covers the populated one.
    """
    import pytest

    pytest.importorskip("optuna")

    from types import SimpleNamespace

    from kernel_optimizer.control.orchestrator import Orchestrator
    from kernel_optimizer.models.core import DeviceLimits

    o = Orchestrator.__new__(Orchestrator)
    o.store = _Store()
    o.task_cost = None
    o.calibration = None
    o.cfg = SimpleNamespace(
        device=DeviceLimits(**A800),
        v3=SimpleNamespace(diagnosis=SimpleNamespace(mode=mode)))
    return o


def _crun():
    from types import SimpleNamespace
    return SimpleNamespace(candidate=SimpleNamespace(
        candidate_id="cand-1", structural_signature="sig-1"))


def test_the_vector_is_journalled_in_label_mode_but_does_not_reach_the_prompt():
    """The asymmetry that makes the control run possible with ONE switch.

    Recording is unconditional (it changes no agent's behaviour, and it is what makes J2-1 replayable
    from the control arm's own log); prompting is switched. Revert-check: making the whole method
    return early under label mode drops `DIMENSION_STATE` and this test fails -- and the control arm
    would then have no vector to compare against.
    """
    o = _orch("label")
    text = o._dimension_digest(_crun(), _Verdict(l3_48_evidence()))
    assert text is None, "label mode must not put the digest in the prompt"
    assert "DIMENSION_STATE" in o.store.kinds(), "the vector must be journalled in both modes"
    p = o.store.payload("DIMENSION_STATE")
    assert p["prompt_mode"] == "label"
    assert p["binding"] == ["occupancy"]
    assert p["n_binding"] == 1
    assert len(p["records"]) == 8


def test_vector_mode_journals_and_also_returns_prompt_text():
    o = _orch("vector")
    text = o._dimension_digest(_crun(), _Verdict(l3_48_evidence()))
    assert text and "one line per dimension" in text
    assert o.store.payload("DIMENSION_STATE")["prompt_mode"] == "vector"


def test_the_journalled_state_is_keyed_on_the_structural_signature_not_the_family():
    """J2-7 at the wiring level. `families.py` is not consulted here at all, by construction."""
    o = _orch("vector")
    o._dimension_digest(_crun(), _Verdict(l3_48_evidence()))
    p = o.store.payload("DIMENSION_STATE")
    assert p["candidate_id"] == "cand-1"
    assert p["structural_signature"] == "sig-1"
    assert "family_id" not in p


def test_a_broken_evidence_dict_journals_a_failure_and_does_not_raise():
    """A diagnostic defect must never look like a candidate defect.

    That inversion is what several of these fixes exist to prevent, so the method swallows and
    journals. Revert-check: without the try/except this raises AttributeError and kills the
    candidate's analysis step -- which is how run-l1-42 died at its first analyst call on a different
    field.
    """
    o = _orch("vector")

    class Hostile:
        kind = "x"
        unmeasured = ()

        @property
        def evidence(self):
            return {"occupancy": 0.1}

    class Exploding(Hostile):
        @property
        def evidence(self):
            raise RuntimeError("evidence accessor is broken")

    text = o._dimension_digest(_crun(), Exploding())
    assert text is None
    assert "DIMENSION_STATE_FAILED" in o.store.kinds()
    assert "broken" in o.store.payload("DIMENSION_STATE_FAILED")["error"]


def test_no_verdict_means_no_vector_rather_than_an_empty_one():
    """An absent verdict must not produce a document full of `unknown` -- that would report a
    collection failure as a measured clean bill of health."""
    o = _orch("vector")
    assert o._dimension_digest(_crun(), None) is None
    assert o.store.kinds() == []


def test_applicability_notes_are_journalled_and_empty_on_a_healthy_vector():
    """P4's check runs on the real path, and reads clean on the production evidence shape.

    Non-empty here would mean the records the run journals are internally inconsistent -- and the
    first version of the check DID fire on both aten dimensions, which is what forced the ceiling
    clause. A check that cries wolf on every candidate teaches a reader to ignore it.
    """
    o = _orch("vector")
    o._dimension_digest(_crun(), _Verdict(l3_48_evidence()))
    assert o.store.payload("DIMENSION_STATE")["applicability_notes"] == []


def test_the_s3_bound_and_provenance_reach_the_journalled_record():
    """S3 through the REAL orchestrator method, with a real task cost and calibration.

    This is the test that was missing when S3 landed. The wiring tests all built an Orchestrator
    without `task_cost`/`calibration`, so on the A800 `_dimension_digest` raised
    `AttributeError: 'Orchestrator' object has no attribute 'task_cost'` and four tests failed on a
    FIXTURE defect -- the G41 shape a third time. Two things follow, and both are asserted here:

      * the populated path is exercised, so a floor and a provenance actually reach the event log;
      * `DIMENSION_STATE_FAILED` must NOT be journalled, because the previous version's failure was
        invisible to every assertion the file had.

    Note what made this diagnosable at all: G40's handler turned the AttributeError into a journalled
    error string instead of a dead analysis step, so the cause was two commands away.
    """
    from kernel_optimizer.evaluation.task_cost import TaskCost

    class Cal:
        measured_at = "2026-09-11T00:00:00Z"

        def identity(self):
            return "NVIDIA A800 80GB PCIe|8.0|108|2.8.0|550.54|3.4.0"

    o = _orch("vector")
    o.task_cost = TaskCost(flop_count=0, compulsory_bytes=1_351_000_000,
                           reference_bytes=1_351_000_000, op_count=12)
    o.calibration = Cal()
    text = o._dimension_digest(_crun(), _Verdict(l3_48_evidence()))

    assert "DIMENSION_STATE_FAILED" not in o.store.kinds(), o.store.events
    p = o.store.payload("DIMENSION_STATE")
    aten = next(r for r in p["records"] if r["dimension_id"] == "candidate_aten_bytes")
    assert aten["bound"]["floor"] == 1_351_000_000.0
    assert aten["bound"]["source"] == "task_lower_bound"
    regs = next(r for r in p["records"] if r["dimension_id"] == "n_regs")
    assert regs["bound"]["floor"] is None, "an invented floor is worse than none"
    assert p["compute_ceiling_provenance"]["source"] in ("measured", "none")
    # And the room-left figures must be in the prompt text, not only in the log.
    assert text and "How much room is left" in text


def test_a_run_without_a_calibration_or_task_cost_still_produces_a_vector():
    """Both are legitimately None -- an uncalibrated box, or a run whose task cost measurement failed
    -- and the vector must degrade to `source="none"` bounds rather than failing.

    Revert-check: making `lower_bound` require a task cost turns this into `DIMENSION_STATE_FAILED`,
    i.e. a box that cannot be calibrated would lose its whole resource vector.
    """
    o = _orch("vector")
    text = o._dimension_digest(_crun(), _Verdict(l3_48_evidence()))
    assert "DIMENSION_STATE_FAILED" not in o.store.kinds()
    p = o.store.payload("DIMENSION_STATE")
    aten = next(r for r in p["records"] if r["dimension_id"] == "candidate_aten_bytes")
    assert aten["bound"]["floor"] is None
    assert "not that the floor is zero" in aten["bound"]["reason"]
    assert text
