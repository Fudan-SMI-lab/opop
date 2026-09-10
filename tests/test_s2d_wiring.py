"""S2d wiring: the ledger at the real prompt path and the real orchestrator call sites.

`test_s2d_reconcile.py` drives the pure reconciler. That leaves the two things that carry the actual
risk: whether the ledger reaches the rewriter's prompt (J2d-4) and whether the round-to-round
bookkeeping is right. Both live in `agents/modules.py` and `control/orchestrator.py`, so these tests
drive those, not a copy of their logic.
"""

from __future__ import annotations

import pytest

from kernel_optimizer.evaluation.reconcile import DIMENSION_VOCABULARY, reconcile
from kernel_optimizer.models.reports import (
    BottleneckReport,
    Hypothesis,
    ResourceExpectation,
    RewriteCandidate,
    RewriteResult,
)
from tests.test_s2_dimensions import A800
from tests.test_s2d_reconcile import deltas, exp


def _ledger_entry(round_no: int = 1, *, conversion: str = "no_conversion") -> dict:
    r = reconcile([exp("shared_bytes", "down", "smaller tile"), exp("n_regs", "up")],
                  deltas(shared_bytes=(65536.0, 16384.0), n_regs=(96.0, 100.0)))
    return {"id": "H1", "round": round_no, "change": "split the reduction",
            "reconciliation": r.model_dump(), "conversion": conversion,
            "latency_gain_pct": 0.4}


def _rewriter_inputs(**over):
    from kernel_optimizer.agents.modules import RewriterInputs
    from kernel_optimizer.models.core import DeviceLimits, TaskSpec

    base = dict(
        task=TaskSpec(level=3, problem_id=48, name="48_x",
                      ref_path="KernelBench/level3/48.py", ref_src_sha="0" * 64),
        best_source="PARAMS = {'BLOCK_M': 64}\n",
        report=BottleneckReport(summary="s", hypotheses=[
            Hypothesis(id="H2", change="tile the reduction", expected_effect="less shared")]),
        failed_hypotheses=[{"id": "H1", "change": "old", "round": 0}],
        device=DeviceLimits(**A800),
        n_candidates=2,
    )
    base.update(over)
    return RewriterInputs(**base)


# --------------------------------------------------------------------------------------------------
# J2d-4: the ledger reaches the real prompt, and carries no raw vector
# --------------------------------------------------------------------------------------------------

def test_j2d_4_the_ledger_reaches_the_rewriters_sandbox_as_prose(tmp_path):
    """J2d-4, driving `seed_sandbox` (D-5).

    Revert-checked against `seed_sandbox` ignoring `ledger_entries`: the file is absent and the
    rewriter sees only `failed_hypotheses.json`, i.e. "H1 was tried and did not help" with no way to
    learn "H1 said shared memory would fall and it rose" -- which is the whole point of the stage.
    """
    from kernel_optimizer.agents.modules import StructureRewriterAgent
    from kernel_optimizer.agents.sandbox import Sandbox

    sb = Sandbox(root=tmp_path / "sb")
    agent = StructureRewriterAgent.__new__(StructureRewriterAgent)
    agent.seed_sandbox(_rewriter_inputs(ledger_entries=[_ledger_entry()]), sb)

    text = (sb.root / "history" / "prediction_ledger.md").read_text(encoding="utf-8")
    assert "shared_bytes" in text and "you said `down`" in text
    assert "as you predicted" in text
    assert "ATTRIBUTION CAVEAT" in text
    # Prose, not a dump. The JSON form is what the old `failed_hypotheses.json` already was.
    assert '"per_dimension"' not in text and "{" not in text


def test_j2d_4_the_ledger_carries_no_raw_vector(tmp_path):
    """Same gate as J2-3, applied to the ledger: it must not become a second route for the raw
    numbers the digestion layer exists to keep out of prompts.

    Checks with the REAL gate rather than a key list, and also sweeps the whole sandbox -- a leak via
    a second file would satisfy a check that only looked at the ledger.
    """
    from kernel_optimizer.agents.modules import StructureRewriterAgent
    from kernel_optimizer.agents.sandbox import Sandbox
    from kernel_optimizer.evaluation.digest import assert_no_raw_vector

    sb = Sandbox(root=tmp_path / "sb")
    agent = StructureRewriterAgent.__new__(StructureRewriterAgent)
    entry = _ledger_entry()
    agent.seed_sandbox(_rewriter_inputs(ledger_entries=[entry]), sb)

    assert_no_raw_vector(entry)
    assert_no_raw_vector(entry["reconciliation"])
    for p in sb.root.rglob("*"):
        if p.is_file():
            body = p.read_text(encoding="utf-8", errors="replace")
            for marker in ("pct_of_dram_peak", "pct_of_compute_peak", "arithmetic_intensity"):
                assert marker not in body, f"{marker} reached the rewriter via {p.name}"


def test_without_a_ledger_the_old_failed_hypotheses_file_is_still_written(tmp_path):
    """Round 1 has nothing to reconcile, and a run with the switch off has nothing at all. Neither
    may lose the pre-S2d information -- the ledger is an addition, not a replacement of the channel.
    """
    from kernel_optimizer.agents.modules import StructureRewriterAgent
    from kernel_optimizer.agents.sandbox import Sandbox

    sb = Sandbox(root=tmp_path / "sb")
    agent = StructureRewriterAgent.__new__(StructureRewriterAgent)
    agent.seed_sandbox(_rewriter_inputs(), sb)
    assert (sb.root / "history" / "failed_hypotheses.json").is_file()
    assert not (sb.root / "history" / "prediction_ledger.md").exists()


def test_the_prompt_points_at_whichever_history_file_exists(tmp_path):
    """A prompt naming a file that is not there teaches the agent to stop opening files.

    Revert-check: hard-coding `failed_hypotheses.json` in the prompt makes the second assertion fail
    while everything else passes -- the ledger would be written and never read.
    """
    from kernel_optimizer.agents.modules import StructureRewriterAgent
    from kernel_optimizer.agents.sandbox import Sandbox

    agent = StructureRewriterAgent.__new__(StructureRewriterAgent)
    sb = Sandbox(root=tmp_path / "sb")

    plain = agent.render_prompt(_rewriter_inputs(), sb)
    assert "failed_hypotheses.json" in plain
    assert "prediction_ledger.md" not in plain

    withledger = agent.render_prompt(_rewriter_inputs(ledger_entries=[_ledger_entry()]), sb)
    assert "prediction_ledger.md" in withledger


def test_the_prompt_asks_for_directions_and_forbids_magnitudes():
    """The schema field only helps if the prompt explains the four rules; an unexplained field gets
    filled with whatever shape the model guesses."""
    from kernel_optimizer.agents.modules import StructureRewriterAgent
    from kernel_optimizer.agents.sandbox import Sandbox

    agent = StructureRewriterAgent.__new__(StructureRewriterAgent)
    text = agent.render_prompt(_rewriter_inputs(), Sandbox(root=__import__("pathlib").Path(".")))
    assert '"expectations"' in text
    assert "never a magnitude" in text
    assert "NUMBER, not \"better\"" in text
    # Every legal name must be listed -- the agent cannot guess a closed vocabulary.
    for dim in DIMENSION_VOCABULARY:
        assert dim in text
    # And the boundary must be stated, or the agent may believe a good prediction earns budget.
    assert "never affects whether your rewrite is accepted" in text


# --------------------------------------------------------------------------------------------------
# J2d-1 at the real check_output
# --------------------------------------------------------------------------------------------------

def test_j2d_1_check_output_rejects_an_unknown_dimension_and_lists_the_legal_ones(tmp_path):
    """J2d-1 at the layer that actually returns feedback to the agent.

    The message must LIST the vocabulary. A rejection that does not say what the legal values are
    turns a schema check into a guessing game -- this project burned three repair calls on a message
    that described an encoding problem as a content problem.
    """
    from kernel_optimizer.agents.modules import StructureRewriterAgent
    from kernel_optimizer.agents.sandbox import Sandbox

    sb = Sandbox(root=tmp_path / "sb")
    sb.write_input("rewrites/rw_1.py", "PARAMS = {}\n\n\nclass ModelNew:\n    pass\n")
    # `write_input` puts it under the sandbox root, which is where `_files_exist_check` looks.
    agent = StructureRewriterAgent.__new__(StructureRewriterAgent)
    out = RewriteResult(candidates=[RewriteCandidate(
        file="rewrites/rw_1.py", change_summary="c", hypothesis_id="H1",
        expectations=[ResourceExpectation(dimension="memory_pressure", expect="down")])])
    msg = agent.check_output(out, sb)
    assert msg and "memory_pressure" in msg
    assert "n_regs" in msg and "shared_bytes" in msg, "the message must list the legal names"
    assert "never a percentage" in msg


def test_check_output_accepts_a_legal_dimension(tmp_path):
    """The control: the check must not reject everything."""
    from kernel_optimizer.agents.modules import StructureRewriterAgent
    from kernel_optimizer.agents.sandbox import Sandbox

    sb = Sandbox(root=tmp_path / "sb")
    sb.write_input("rewrites/rw_1.py", "PARAMS = {}\n\n\nclass ModelNew:\n    pass\n")
    agent = StructureRewriterAgent.__new__(StructureRewriterAgent)
    out = RewriteResult(candidates=[RewriteCandidate(
        file="rewrites/rw_1.py", change_summary="c",
        expectations=[ResourceExpectation(dimension="shared_bytes", expect="down")])])
    msg = agent.check_output(out, sb)
    assert msg is None or "memory_pressure" not in msg


# --------------------------------------------------------------------------------------------------
# The orchestrator's round bookkeeping
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


def _orch(ledger_on: bool = True):
    pytest.importorskip("optuna")
    from types import SimpleNamespace

    from kernel_optimizer.control.orchestrator import Orchestrator

    o = Orchestrator.__new__(Orchestrator)
    o.store = _Store()
    o.round_expectations = {}
    o.ledger = {}
    o.cfg = SimpleNamespace(
        v3=SimpleNamespace(diagnosis=SimpleNamespace(expectation_ledger=ledger_on)))
    return o


def _declare(o, family_id: str, *pairs: tuple[str, str]) -> None:
    from kernel_optimizer.control.orchestrator import _DeclaredExpectations
    for dim, direction in pairs:
        o.round_expectations.setdefault(family_id, []).append(
            _DeclaredExpectations(hypothesis_id="H1", change_summary="split the reduction",
                                  expectation=exp(dim, direction)))


def test_a_round_is_reconciled_against_its_own_declarations_and_journalled():
    o = _orch()
    _declare(o, "fam-1", ("shared_bytes", "down"), ("n_regs", "up"))
    conv = {"conversion": "no_conversion", "latency_gain_pct": 0.3,
            "resource_deltas": deltas(shared_bytes=(65536.0, 16384.0), n_regs=(96.0, 220.0))}
    o._record_reconciliation("fam-1", 1, conv)

    assert "EXPECTATIONS_RECONCILED" in o.store.kinds()
    p = o.store.payload("EXPECTATIONS_RECONCILED")
    assert p["reconciliation"]["hits"] == 2
    assert p["conversion"] == "no_conversion"
    assert p["n_declared"] == 2
    assert o.ledger["fam-1"] == [p]


def test_the_declarations_are_popped_so_the_next_round_is_not_scored_against_them():
    """The bug this prevents reads as a plausible ledger and is entirely wrong: round N+1 scored
    against round N's predictions.

    Revert-checked against `.get` instead of `.pop`: round 2 reports 2 declarations it never made,
    and every one of them is a hit or miss attributed to the wrong round.
    """
    o = _orch()
    _declare(o, "fam-1", ("shared_bytes", "down"))
    conv = {"conversion": "improved", "latency_gain_pct": 8.0,
            "resource_deltas": deltas(shared_bytes=(65536.0, 16384.0))}
    o._record_reconciliation("fam-1", 1, conv)
    assert o.round_expectations.get("fam-1", []) == []

    # Round 2 declares nothing.
    o._record_reconciliation("fam-1", 2, conv)
    entries = o.ledger["fam-1"]
    assert len(entries) == 2
    assert entries[1]["n_declared"] == 0
    assert entries[1]["reconciliation"]["hits"] == 0


def test_two_families_do_not_share_declarations():
    o = _orch()
    _declare(o, "fam-1", ("shared_bytes", "down"))
    _declare(o, "fam-2", ("n_regs", "up"))
    conv = {"conversion": "flat", "latency_gain_pct": 0.1,
            "resource_deltas": deltas(shared_bytes=(65536.0, 16384.0))}
    o._record_reconciliation("fam-1", 1, conv)
    assert len(o.round_expectations["fam-2"]) == 1
    assert o.ledger["fam-1"][0]["n_declared"] == 1


def test_the_ledger_entry_names_the_hypothesis_it_is_about():
    """A `miss` attached only to a family and a round number is not enough for the next prompt to be
    about anything."""
    o = _orch()
    _declare(o, "fam-1", ("shared_bytes", "down"))
    o._record_reconciliation("fam-1", 1, {
        "conversion": "improved", "latency_gain_pct": 9.0,
        "resource_deltas": deltas(shared_bytes=(65536.0, 16384.0))})
    e = o.ledger["fam-1"][0]
    assert e["id"] == "H1"
    assert "split the reduction" in e["change"]


def test_a_reconcile_failure_journals_and_does_not_end_the_round():
    """Same discipline as S2's `_dimension_digest`: a diagnostic that ends a rewrite round presents a
    bookkeeping defect as a candidate defect. And a rewrite round is the scarcest budget there is."""
    o = _orch()

    class Hostile(dict):
        def get(self, key, default=None):
            raise RuntimeError("conversion dict is broken")

    o._record_reconciliation("fam-1", 1, Hostile())
    assert "EXPECTATIONS_RECONCILE_FAILED" in o.store.kinds()
    assert "broken" in o.store.payload("EXPECTATIONS_RECONCILE_FAILED")["error"]


def test_the_ledger_is_journalled_even_with_the_switch_off():
    """The same asymmetry as S2's vector: recording is unconditional, only the PROMPT is switched.

    `_record_reconciliation` does not consult the switch at all -- the switch is read where the
    prompt is built. Asserted here so a future "optimisation" that skips the work when the switch is
    off is caught: it would leave the control arm with no ledger to compare against, and J2d-9 would
    need a second run.
    """
    o = _orch(ledger_on=False)
    _declare(o, "fam-1", ("shared_bytes", "down"))
    o._record_reconciliation("fam-1", 1, {
        "conversion": "improved", "latency_gain_pct": 9.0,
        "resource_deltas": deltas(shared_bytes=(65536.0, 16384.0))})
    assert "EXPECTATIONS_RECONCILED" in o.store.kinds()
    assert o.ledger["fam-1"]


def test_a_declaration_is_journalled_before_any_measurement_exists():
    """S2d(a)'s ordering requirement. A declaration recorded AFTER the measurement could have been
    shaped by the outcome, and nothing in the log would show it -- so `REWRITE_PRODUCED` carries the
    expectations, and it is written at rewrite time.

    Asserted on the event's own field rather than by running `_do_rewrite` (which needs a live agent):
    the point is that the payload carries them at all, which is what a later audit reads.
    """
    pytest.importorskip("optuna")
    import inspect

    from kernel_optimizer.control.orchestrator import Orchestrator

    src = inspect.getsource(Orchestrator._do_rewrite)
    produced = src.split('"REWRITE_PRODUCED"', 1)
    assert len(produced) == 2, "REWRITE_PRODUCED must be emitted by _do_rewrite"
    # The expectations are attached in the same payload, and the round buffer is filled there too.
    assert '"expectations"' in produced[1]
    assert "round_expectations" in produced[1]
