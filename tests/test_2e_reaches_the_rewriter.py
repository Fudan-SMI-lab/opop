"""A3's precondition: the wall measurement reaches the agent that ACTS on it.

`test_2e_wall_attribution.py`'s 29 tests all passed while `wall_text` reached ONLY the analyst. That
is the shape of failure the project has recorded as "a clean run is not evidence a checker works":
the suite was green because nothing in it asked this question.

Why it matters for the third arm specifically. A3 is "does a rewrite aimed at an attributed wall
actually free that dimension". The rewriter's only view of the diagnosis is
`analysis/bottleneck.json` -- the analyst's OUTPUT. Without a direct route, the measured wall reaches
the rewriter only if the analyst elects to restate it, i.e. laundered through the one agent whose
claims on this exact subject were measured at 8 confirmed out of 53. A negative A3 would then be
unattributable between "the wall is not a real bottleneck" and "the analyst dropped it" -- which is
precisely the confound this arm is run to avoid.
"""

from __future__ import annotations

import pytest


def _rw_inputs(**over):
    from kernel_optimizer.agents.modules import RewriterInputs
    from kernel_optimizer.models.core import DeviceLimits, TaskSpec
    from kernel_optimizer.models.reports import BottleneckReport, Hypothesis
    from tests.test_s2_dimensions import A800

    base = dict(
        task=TaskSpec(level=3, problem_id=43, name="43_x",
                      ref_path="KernelBench/level3/43.py", ref_src_sha="0" * 64),
        best_source="PARAMS = {'BLOCK_M': 256}\n",
        report=BottleneckReport(summary="s", hypotheses=[
            Hypothesis(id="H1", change="tile it", expected_effect="less shared")]),
        failed_hypotheses=[],
        device=DeviceLimits(**A800),
        n_candidates=2,
    )
    base.update(over)
    return RewriterInputs(**base)


def _attributed_wall_text() -> str:
    """Rendered by the REAL renderer from a wall carrying a real verdict, not hand-written prose.

    Hand-written prose would test the plumbing against a fixture invented to match the reader, and
    the thing most likely to break is the renderer's own contract (the at-the-optimum condition).
    """
    from kernel_optimizer.evaluation import wall_attribution as wa

    w = wa.Wall(param="BLOCK_M", refused_value=512.0, ran_values=[64.0, 128.0, 256.0], side="high",
                tail_values=[64.0, 128.0, 256.0], tail_latencies=[3.8221, 3.3807, 3.1836],
                monotone=True, tail_gain_pct=16.7, verdict="attributed",
                max_shared=122880, limit=101376, over_ratio=122880 / 101376)
    text = wa.for_prompt([w])
    assert text, "the renderer must speak for an attributed wall, or this test proves nothing"
    return text


def test_the_wall_measurement_reaches_the_rewriters_sandbox(tmp_path):
    """Revert-checked: with the `if inputs.wall_text` block removed from `seed_sandbox`, the file is
    absent and the rewriter is left with only the analyst's `parameter_limits`.
    """
    from kernel_optimizer.agents.modules import StructureRewriterAgent
    from kernel_optimizer.agents.sandbox import Sandbox

    sb = Sandbox(root=tmp_path / "sb")
    agent = StructureRewriterAgent.__new__(StructureRewriterAgent)
    agent.seed_sandbox(_rw_inputs(wall_text=_attributed_wall_text()), sb)

    path = sb.root / "analysis" / "resource_walls.md"
    assert path.is_file(), "the attributed wall never reached the rewriter"
    body = path.read_text(encoding="utf-8")
    assert "BLOCK_M" in body and "512" in body
    # The conditional reading must survive the trip. Unconditional ("BLOCK_M is capped by shared
    # memory") would send the rewriter after a wall that only exists at this optimum: from the space
    # default only 1 of 6 walls attributes to a single knob, against 6 of 6 from the optimum.
    assert "最优参数点" in body, "the at-the-optimum condition did not survive the trip"


def test_the_wall_file_is_separate_from_the_analysts_report(tmp_path):
    """The measured fact and the analyst's guess about the same subject must stay distinguishable.

    Merging them into `bottleneck.json` would leave the rewriter unable to tell which of two claims
    about "what blocks this knob" was measured -- and that difference IS the third arm.
    """
    from kernel_optimizer.agents.modules import StructureRewriterAgent
    from kernel_optimizer.agents.sandbox import Sandbox

    sb = Sandbox(root=tmp_path / "sb")
    agent = StructureRewriterAgent.__new__(StructureRewriterAgent)
    agent.seed_sandbox(_rw_inputs(wall_text=_attributed_wall_text()), sb)

    bottleneck = (sb.root / "analysis" / "bottleneck.json").read_text(encoding="utf-8")
    assert "BLOCK_M" not in bottleneck, "the wall was merged into the analyst's report"


def test_the_prompt_names_the_wall_file_only_when_it_exists(tmp_path):
    """A prompt pointing at a file the sandbox lacks teaches the agent the harness is unreliable,
    and it is what a naive f-string interpolation produces.

    Both arms are checked, because the failure is symmetric: the control arm mentioning the file is
    as wrong as the treatment arm omitting it, and only one of those would be noticed in a run.
    """
    from kernel_optimizer.agents.modules import StructureRewriterAgent
    from kernel_optimizer.agents.sandbox import Sandbox

    agent = StructureRewriterAgent.__new__(StructureRewriterAgent)

    sb_on = Sandbox(root=tmp_path / "on")
    inputs_on = _rw_inputs(wall_text=_attributed_wall_text())
    agent.seed_sandbox(inputs_on, sb_on)
    prompt_on = agent.render_prompt(inputs_on, sb_on)
    assert "analysis/resource_walls.md" in prompt_on
    # It has to be marked as measured, or it reads as one more opinion beside `parameter_limits`.
    assert "measured" in prompt_on

    sb_off = Sandbox(root=tmp_path / "off")
    inputs_off = _rw_inputs()
    agent.seed_sandbox(inputs_off, sb_off)
    prompt_off = agent.render_prompt(inputs_off, sb_off)
    assert "resource_walls" not in prompt_off, "the control arm was told about a file it lacks"
    assert not (sb_off.root / "analysis" / "resource_walls.md").exists()


def test_the_wall_text_carries_no_raw_vector(tmp_path):
    """The same gate as the ledger's: a new file into the rewriter's sandbox is a new route for the
    raw numbers the digestion layer exists to keep out of prompts.

    Swept over the WHOLE sandbox, because a leak via a second file would satisfy a check that only
    read this one.
    """
    from kernel_optimizer.agents.modules import StructureRewriterAgent
    from kernel_optimizer.agents.sandbox import Sandbox

    sb = Sandbox(root=tmp_path / "sb")
    agent = StructureRewriterAgent.__new__(StructureRewriterAgent)
    agent.seed_sandbox(_rw_inputs(wall_text=_attributed_wall_text()), sb)

    for p in sb.root.rglob("*"):
        if p.is_file():
            body = p.read_text(encoding="utf-8", errors="replace")
            for marker in ("pct_of_dram_peak", "pct_of_compute_peak", "arithmetic_intensity"):
                assert marker not in body, f"{marker} reached the rewriter via {p.name}"


def test_the_orchestrator_carries_wall_text_on_the_candidate():
    """The field must exist on `CandidateRun` and default to None.

    `_stats_and_analysis` sets it and `_rewrite_round` reads it, ~1100 lines apart, so the DEFAULT is
    what the control and `vector` arms actually run with. A default of `""` would keep
    `if inputs.wall_text` falsy, but any later `is not None` check would then flip both other arms
    into the treatment path -- silently, and only visible in the prompts.

    Skipped where optuna is absent (the orchestrator imports the tuner at module scope), which is
    why the two links this test covers are ALSO pinned structurally below -- a skip is not a verdict,
    and on Windows this test is a skip.
    """
    pytest.importorskip("optuna")
    from kernel_optimizer.control.orchestrator import CandidateRun
    from kernel_optimizer.models.core import Candidate

    crun = CandidateRun(candidate=Candidate(
        candidate_id="cand-1", family_id="fam-1", origin="seed", backend="triton",
        source_sha="0" * 64, structural_signature="s"))
    assert crun.wall_text is None


# --------------------------------------------------------------------------------------------------
# The two orchestrator-side links, pinned WITHOUT importing the orchestrator.
#
# A revert check over the five wiring edits caught only three: deleting `crun.wall_text = wall_text`
# or `wall_text=parent_crun.wall_text` left the suite green, because the only test covering them
# needs optuna and therefore SKIPS on the box where the suite is usually run. Both deletions are
# silent in production too -- the arm still runs, the walls are still journalled, and only the
# prompts differ -- so a green suite must not be reachable without them.
#
# An AST scan rather than a substring search: a substring can be satisfied by the same text inside a
# comment, and this file's own comments discuss both statements by name.
# --------------------------------------------------------------------------------------------------

def _orchestrator_ast():
    import ast
    import pathlib

    src = pathlib.Path("src/kernel_optimizer/control/orchestrator.py").read_text(encoding="utf-8")
    return ast.parse(src)


def test_the_candidate_runs_wall_text_defaults_to_none_structurally():
    """The DEFAULT is what the two non-2e arms run with, and it must be None rather than `""`.

    Both are falsy, so `if inputs.wall_text` behaves the same and every behavioural test above stays
    green either way -- the revert check confirmed that. The difference bites on any later
    `is not None` reader, which would flip the control and `vector` arms into the treatment path
    silently. Pinned here as well as in the runtime test above because that one needs optuna and
    therefore skips on Windows, where this suite is usually run.
    """
    import ast

    for node in ast.walk(_orchestrator_ast()):
        if not (isinstance(node, ast.ClassDef) and node.name == "CandidateRun"):
            continue
        for stmt in node.body:
            if (isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
                    and stmt.target.id == "wall_text"):
                assert isinstance(stmt.value, ast.Constant) and stmt.value.value is None, \
                    "CandidateRun.wall_text must default to None, not to a falsy string"
                return
        raise AssertionError("CandidateRun has no wall_text field")
    raise AssertionError("CandidateRun not found -- this test is anchored on the wrong name")


def test_stats_and_analysis_stores_the_wall_text_on_the_candidate():
    """`crun.wall_text = wall_text` must exist as a real assignment, not as prose about one."""
    import ast

    found = [
        node for node in ast.walk(_orchestrator_ast())
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == "wall_text"
                and isinstance(t.value, ast.Name) and t.value.id == "crun"
                for t in node.targets)
    ]
    assert found, ("nothing assigns `crun.wall_text`, so `_rewrite_round` always reads None and the "
                   "third arm silently becomes the second")


def test_the_rewriter_call_passes_the_parent_candidates_wall_text():
    """`RewriterInputs(..., wall_text=parent_crun.wall_text)` must be a real keyword argument.

    The parent is the candidate whose space was probed, so its walls describe the source being
    rewritten; passing the child's (always None at that point) would look identical in the config and
    produce an empty file forever.
    """
    import ast

    calls = [
        node for node in ast.walk(_orchestrator_ast())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "RewriterInputs"
    ]
    assert calls, "no RewriterInputs call found -- this test is anchored on the wrong name"
    for call in calls:
        kw = {k.arg: k.value for k in call.keywords}
        assert "wall_text" in kw, "the rewriter is constructed without wall_text"
        v = kw["wall_text"]
        assert isinstance(v, ast.Attribute) and v.attr == "wall_text", \
            "wall_text must be passed through, not hard-coded"
        assert isinstance(v.value, ast.Name) and v.value.id == "parent_crun", \
            "the wall must come from the PARENT candidate, whose space was the one probed"
