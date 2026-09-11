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
    # `runs` and `families` are what per-candidate attribution reads: each candidate's own trials give
    # its own profile. Empty by default, so a test that declares nothing about candidates gets the
    # `unmeasured` path rather than a silent fallback to another candidate's numbers.
    o.runs = {}
    o.deps = SimpleNamespace(families=SimpleNamespace(families={}))
    o.cfg = SimpleNamespace(
        budgets=SimpleNamespace(min_improvement_pct=2.0),
        v3=SimpleNamespace(diagnosis=SimpleNamespace(expectation_ledger=ledger_on)))
    return o


def _declare(o, family_id: str, *pairs: tuple[str, str], candidate_id: str = "cand-a",
             hypothesis_id: str = "H1") -> None:
    from kernel_optimizer.control.orchestrator import _DeclaredExpectations
    for dim, direction in pairs:
        o.round_expectations.setdefault(family_id, []).append(
            _DeclaredExpectations(hypothesis_id=hypothesis_id,
                                  change_summary="split the reduction",
                                  expectation=exp(dim, direction),
                                  candidate_id=candidate_id))


def _prof(**fields):
    """A parent profile as `_rewrite_round` captures it: a ProfileRecord-shaped object, read by
    attribute the way `conversion._read` reads it."""
    from types import SimpleNamespace
    return SimpleNamespace(**fields)


def _measured(o, cand_id: str, ms: float, **prof):
    """Give a candidate a winning trial, so `_best_profile` has something to return.

    Shaped from what `_best_profile` actually reads -- `status`, `latency_ms.robust_ms`, `profile` --
    rather than from what the reader would find convenient, because a fixture invented to match the
    reader proves nothing about either.
    """
    from types import SimpleNamespace
    o.runs[cand_id] = SimpleNamespace(
        best_ms=ms,
        trials=[SimpleNamespace(status="complete",
                                latency_ms=SimpleNamespace(robust_ms=ms),
                                profile=SimpleNamespace(**prof))])
    return o.runs[cand_id]


def test_a_round_is_reconciled_against_its_own_declarations_and_journalled():
    """`_measured` is not decoration: reconciliation is PER CANDIDATE, so a declaration is judged
    against the profile of the candidate that made it. Without a measurement this candidate correctly
    reads `unmeasured` -- which is what the fixture said before per-candidate attribution existed, when
    an entry was scored against the round's deltas no matter which code they came from."""
    o = _orch()
    _declare(o, "fam-1", ("shared_bytes", "down"), ("n_regs", "up"))
    _measured(o, "cand-a", 3.0, shared_bytes=16384, n_regs=220)
    conv = {"conversion": "no_conversion", "latency_gain_pct": 0.3,
            "latency_ms_before": 3.01, "latency_ms_after": 3.0,
            "resource_deltas": deltas(shared_bytes=(65536.0, 16384.0), n_regs=(96.0, 220.0))}
    o._record_reconciliation("fam-1", 1, conv,
                             _prof(shared_bytes=65536, n_regs=96))

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


def test_two_candidates_in_one_round_are_reconciled_SEPARATELY():
    """The pooling defect, on box 2's own numbers (`run-l3-43-20260911-052630`, round 0).

    Every measured rewrite round produced TWO candidates (9 of 9 across the completed L3 runs), asked
    for different hypotheses, so their declarations contradict each other about the same dimension by
    design. H1+H3 said `shared_bytes: up` and won; H2 said `unchanged` about the same dimension for
    different code. Pooled, the log said 6 hits / 6 misses; separately it is 4/1 for the winner and
    2/5 for H2.

    Revert-checked against the pooled implementation: ONE entry appears with hits=6, misses=6,
    n_declared=16, so both assertions on the winner's entry fail. That is the whole defect -- the
    strongest S2d evidence in the run, five declarations with every falsifiable one correct, diluted
    to a coin flip, and H2 charged with misses for another candidate's changes.
    """
    o = _orch()
    WIN, OTHER = "cand-2d8eaf9a", "cand-3760b4d7"
    _declare(o, "fam-1", ("shared_bytes", "up"), ("n_regs", "unknown"),
             candidate_id=WIN, hypothesis_id="H1+H3")
    _declare(o, "fam-1", ("shared_bytes", "unchanged"), ("n_regs", "down"),
             candidate_id=OTHER, hypothesis_id="H2")

    # The winner IS the family incumbent, so it reconciles against the round's own conversion.
    from types import SimpleNamespace
    o.deps.families.families["fam-1"] = SimpleNamespace(
        best=SimpleNamespace(candidate_id=WIN))
    # The other candidate's own measurement: shared memory did NOT move, exactly as H2 said.
    _measured(o, OTHER, 3.30, n_regs=140, shared_bytes=17408)

    conv = {"conversion": "improved", "latency_gain_pct": 9.833,
            "latency_ms_before": 3.2128, "latency_ms_after": 2.8969,
            "resource_deltas": deltas(shared_bytes=(17408.0, 32768.0), n_regs=(155.0, 155.0))}
    o._record_reconciliation("fam-1", 0, conv,
                             SimpleNamespace(n_regs=155, shared_bytes=17408))

    entries = o.ledger["fam-1"]
    assert len(entries) == 2, "one entry per candidate, not one per round"
    by_cand = {e["candidate_id"]: e for e in entries}
    assert set(by_cand) == {WIN, OTHER}

    win = by_cand[WIN]["reconciliation"]
    assert win["hits"] == 1 and win["misses"] == 0, (
        "the winner said shared_bytes would rise and it rose; pooled this reads as 6/6")
    assert win["vacuous"] == 1                      # n_regs declared `unknown`
    assert by_cand[WIN]["n_declared"] == 2

    oth = by_cand[OTHER]["reconciliation"]
    rows = {r["dimension"]: r for r in oth["per_dimension"]}
    assert rows["shared_bytes"]["match"] == "hit", (
        "H2 said its own shared_bytes would not move, and its own measurement agrees -- charging it "
        "a miss for the OTHER candidate's 17408->32768 is the defect")
    assert rows["shared_bytes"]["before"] == 17408 and rows["shared_bytes"]["after"] == 17408


def test_a_candidates_entry_is_never_scored_against_another_candidates_deltas():
    """The narrower half: a candidate with NO measurement of its own must read `unmeasured`, not
    inherit the round's map.

    This is the mechanism behind the test above, isolated. A rewrite whose parameterization was
    rejected has declarations and no profile; scoring it against the round's deltas would report hits
    and misses for code that was never measured, and those are indistinguishable in the log from real
    ones.
    """
    o = _orch()
    _declare(o, "fam-1", ("shared_bytes", "up"), candidate_id="cand-never-tuned")
    conv = {"conversion": "improved", "latency_gain_pct": 9.0,
            "latency_ms_before": 3.2, "latency_ms_after": 2.9,
            "resource_deltas": deltas(shared_bytes=(17408.0, 32768.0))}
    o._record_reconciliation("fam-1", 0, conv, None)

    e = o.ledger["fam-1"][0]
    r = e["reconciliation"]
    assert r["hits"] == 0 and r["misses"] == 0, "nothing was measured, so nothing is judged"
    assert list(r["dimensions_unmeasured"]) == ["shared_bytes"]
    assert e["conversion"] is None, (
        "an unmeasured candidate has no conversion verdict of its own; carrying the round's would "
        "credit it with another candidate's latency gain")


def test_the_ledger_heading_names_the_candidate_so_two_sections_are_distinguishable():
    """`render_ledger` is the only thing the agent sees, so per-candidate entries have to be
    per-candidate THERE too.

    Revert-checked against a heading that names only the round: `shared_bytes` then appears twice
    under one title, once `up` and correct and once `unchanged` and wrong, with nothing saying they
    describe different code. The prose is the whole treatment in S2d(c) -- a reader who cannot tell
    the two apart is worse off than one given no ledger.
    """
    from kernel_optimizer.evaluation.reconcile import render_ledger

    a = reconcile([exp("shared_bytes", "up")],
                  deltas(shared_bytes=(17408.0, 32768.0)), hypothesis_id="H1+H3")
    b = reconcile([exp("shared_bytes", "unchanged")],
                  deltas(shared_bytes=(17408.0, 17408.0)), hypothesis_id="H2")
    text = render_ledger([
        {"id": "H1+H3", "round": 0, "candidate_id": "cand-2d8eaf9a", "change": "fuse projections",
         "reconciliation": a.model_dump(), "conversion": "improved", "latency_gain_pct": 9.8},
        {"id": "H2", "round": 0, "candidate_id": "cand-3760b4d7", "change": "reshape registers",
         "reconciliation": b.model_dump(), "conversion": "flat", "latency_gain_pct": 0.1},
    ])
    assert "cand-2d8eaf9a" in text and "cand-3760b4d7" in text
    assert text.count("## Round 0") == 2
    # The same dimension appears in both sections with DIFFERENT readings, which is why the heading
    # has to separate them: `up` / 17408 -> 32768 for the winner, `unchanged` / flat for H2. Both are
    # hits (H2's own shared memory really did not move), so the discriminator is the numbers, not a
    # hit-vs-miss contrast -- asserting one of each here would have been asserting the pooled bug.
    # `_fmt` renders >=1000 as 3 significant figures, so the assertion is on its output, not on the
    # raw integer: matching "32768" fails on a working renderer.
    assert "3.28e+04" in text and text.count("shared_bytes") == 2
    assert "you said `up`" in text and "you said `unchanged`" in text


def test_render_ledger_still_renders_an_entry_with_no_candidate_id():
    """A resumed run replays entries journalled before `candidate_id` existed. They must render as the
    older shape rather than printing `None` into the agent's prompt."""
    from kernel_optimizer.evaluation.reconcile import render_ledger

    r = reconcile([exp("shared_bytes", "down")], deltas(shared_bytes=(65536.0, 16384.0)))
    text = render_ledger([{"id": "H1", "round": 1, "change": "old",
                           "reconciliation": r.model_dump(), "conversion": "improved"}])
    assert "## Round 1 — H1" in text
    assert "None" not in text


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


def test_the_round_payload_carries_the_conversion_verdict_and_the_ledger_reuses_it():
    """The BEHAVIOURAL half of `test_rewrite_rounds_record_their_conversion_verdict`, which is a
    source-text assertion.

    That test broke on S2d for a reason worth recording: hoisting `conversion_verdict(...)` out of the
    `store.append(...)` call -- so the ledger could reuse the same `resource_deltas` -- moved the call
    outside the slice it was reading, and it failed on unchanged behaviour. Its scope is fixed, but a
    source-text assertion cannot notice if the verdict stops REACHING the payload, and this repo has a
    recorded case of an assertion on source text that passed on buggy code and failed after the fix.
    So this drives the values.

    Uses `conversion_verdict` and `reconcile` on the same input the round would, and asserts the two
    halves agree about what moved -- which is the actual invariant the hoist was for.
    """
    from kernel_optimizer.evaluation.conversion import conversion_verdict

    conv = conversion_verdict(2.00, 1.50, None, None, min_improvement_pct=2.0)
    assert "conversion" in conv, "the payload spread must include the verdict itself"
    assert conv["conversion"] == "improved"

    # With profiles absent there are no deltas, and the ledger must then report `unmeasured` rather
    # than inventing a flat reading -- the same rule on both sides of the hoist.
    r = reconcile([exp("n_regs", "down")], conv.get("resource_deltas"))
    assert r.dimensions_unmeasured == ("n_regs",)
    assert r.hits == 0 and r.misses == 0
