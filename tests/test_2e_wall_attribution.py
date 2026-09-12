"""2e: shared-memory wall attribution.

The measurements these tests encode, all from box 2's two finished L3:43 runs (18 variants, one
worker process, 8.8 s):
  - from the candidate's optimum, 6 of 6 walls attribute to a single knob; the negative control
    (theta* itself) passed 6 of 6;
  - from the space's DEFAULT corner, only 1 of 6 does -- so the verdict is conditional, and that is
    the single most load-bearing fact in the design;
  - 3 of the 6 walls have latency getting WORSE toward the wall (worst -54.8%), so a wall is not
    automatically an opportunity;
  - the analyst's own `parameter_limits` agreed on 8 of 53 claims.
"""

from __future__ import annotations

import ast
import io
from pathlib import Path

import pytest

from kernel_optimizer.evaluation import wall_attribution as wa
from kernel_optimizer.models.reports import ParamStat, TuningStats

SRC = Path("src/kernel_optimizer")


def _stats(*param_stats: ParamStat) -> TuningStats:
    return TuningStats(candidate_id="cand-x", space_id="sp-x", n_complete=10, n_fail=0,
                       param_stats=list(param_stats))


def _ps(name: str, by_value: dict[str, float]) -> ParamStat:
    return ParamStat(name=name, best_value=next(iter(by_value)), latency_by_value=by_value)


# ---------------------------------------------------------------------------------------------
# find_walls: what counts as a wall at all
# ---------------------------------------------------------------------------------------------


def test_a_refused_value_beyond_the_measured_range_is_a_wall():
    """box2 cand-2dc6a2ff: ran 64/128/256, refused 512 -- the shape 2e exists for."""
    stats = _stats(_ps("BLOCK_M", {"64": 3.8221, "128": 3.3807, "256": 3.1836}))
    walls = wa.find_walls(stats, [{"BLOCK_M": 512}])
    assert len(walls) == 1
    w = walls[0]
    assert w.param == "BLOCK_M"
    assert w.refused_value == 512.0
    assert w.side == "high"
    assert w.monotone is True
    assert w.tail_gain_pct == pytest.approx(16.65, abs=0.1)


def test_a_refused_value_INSIDE_the_measured_range_is_not_a_wall():
    """158 of 164 refusals on box 2 were of this kind: the tuner reached both sides, nothing was cut.

    Counting these as walls would report a truncation where none exists, and would have turned
    box 2's 6 walls into 164.
    """
    stats = _stats(_ps("BLOCK_M", {"64": 3.8, "128": 3.4, "256": 3.2}))
    assert wa.find_walls(stats, [{"BLOCK_M": 128}]) == []


def test_a_boolean_refused_value_is_not_treated_as_a_number():
    """`float(True)` is 1.0, so without the bool guard an off-switch becomes a numeric axis.

    The reachable shape is a MIXED domain -- an agent writing `[False, 2, 4, 8]` for "off, or a tile
    size", which the parameterizer is free to produce. `latency_by_value` keys are stringified, so
    the numeric side is 2/4/8 (three values, past the floor), while the refused set comes from a
    TrialRecord's raw `params.values` and therefore carries a real `False`. Read as 0.0 that lands
    below the measured range and is reported as a low wall at `K = False`, complete with a tail
    slope -- a wall that does not exist, on an axis that does not exist.

    The first version of this test used a bool knob with string keys ("True"/"False"). It passed,
    but via the three-value floor rather than the bool guard, so the guard could have been deleted
    with the suite still green -- which is exactly what the revert check reported.
    """
    stats = _stats(_ps("K", {"False": 5.0, "2": 4.0, "4": 3.5, "8": 3.2}))
    assert wa.find_walls(stats, [{"K": False}]) == []
    # Positive control on the same knob: a genuine numeric refusal IS still found, so the assertion
    # above is about the bool and not about the knob being skipped wholesale.
    walls = wa.find_walls(stats, [{"K": 16}])
    assert [w.param for w in walls] == ["K"] and walls[0].refused_value == 16.0


def test_a_categorical_knob_is_never_a_wall():
    """The 'edge' of a categorical domain is an artifact of the order the agent wrote it in.

    A slope over dtype names measures the list, not the hardware. This project already banned the
    same reasoning for `at_boundary` on categorical domains.
    """
    stats = _stats(_ps("COMPUTE_DTYPE", {"fp16": 3.1, "bf16": 3.4, "fp32": 4.0}))
    assert wa.find_walls(stats, [{"COMPUTE_DTYPE": "tf32"}]) == []


def test_fewer_than_three_measured_values_gives_no_wall():
    """Two points are monotone through any two points; three is the fewest that can be non-monotone.

    Without this floor a knob measured at exactly two values would always report `monotone=True`.
    """
    stats = _stats(_ps("BLOCK_M", {"64": 3.8, "128": 3.4}))
    assert wa.find_walls(stats, [{"BLOCK_M": 256}]) == []


def test_no_refusals_means_no_walls():
    stats = _stats(_ps("BLOCK_M", {"64": 3.8, "128": 3.4, "256": 3.2}))
    assert wa.find_walls(stats, []) == []


def test_the_nearest_refused_value_is_the_one_reported():
    """box2 cand-2dc6a2ff refused BOTH 512 and 1024. 512 is the smallest step that hits the wall.

    Reporting 1024 would describe a rewrite nobody asked for and would inflate `over_ratio`.
    """
    stats = _stats(_ps("BLOCK_M", {"64": 3.8, "128": 3.4, "256": 3.2}))
    walls = wa.find_walls(stats, [{"BLOCK_M": 1024}, {"BLOCK_M": 512}])
    assert walls[0].refused_value == 512.0


def test_a_wall_below_the_measured_range_is_found_and_its_tail_reads_inward():
    """A `min`-side wall exists too, and its tail must be read from far to near the wall.

    Reading the tail left-to-right for a low wall would invert the slope's sign.
    """
    stats = _stats(_ps("NUM_STAGES", {"2": 3.9, "3": 3.5, "4": 3.2}))
    walls = wa.find_walls(stats, [{"NUM_STAGES": 1}])
    assert len(walls) == 1
    w = walls[0]
    assert w.side == "low"
    assert w.tail_values == [4.0, 3.0, 2.0]      # far -> near the wall
    assert w.tail_latencies == [3.2, 3.5, 3.9]
    assert w.monotone is False                    # latency WORSENS toward this wall
    assert w.tail_gain_pct < 0


# ---------------------------------------------------------------------------------------------
# select_for_probing: a wall is not automatically an opportunity
# ---------------------------------------------------------------------------------------------


def test_a_wall_with_a_worsening_tail_is_not_probed():
    """box2 cand-2d8eaf9a GEMM_BK: capped, and latency is 54.8% WORSE toward the cap.

    3 of box 2's 6 walls were like this. Probing them would spend GPU time to produce a rewrite
    brief for a non-problem.
    """
    stats = _stats(_ps("GEMM_BK", {"16": 3.7279, "32": 2.8616, "64": 3.0054, "128": 4.4283}))
    walls = wa.find_walls(stats, [{"GEMM_BK": 256}])
    assert len(walls) == 1
    probe, skipped = wa.select_for_probing(walls, 8)
    assert probe == []
    assert skipped == []


def test_probing_order_is_by_tail_gain_descending():
    stats = _stats(_ps("A", {"1": 10.0, "2": 9.5, "3": 9.0}),      # ~10%
                   _ps("B", {"1": 10.0, "2": 7.0, "3": 5.0}))      # ~50%
    walls = wa.find_walls(stats, [{"A": 4, "B": 4}])
    probe, _ = wa.select_for_probing(walls, 8)
    assert [w.param for w in probe] == ["B", "A"]


def test_the_probe_cap_skips_rather_than_drops():
    """The cap bounds DIAGNOSTIC depth. A dropped wall would be indistinguishable from no wall."""
    stats = _stats(_ps("A", {"1": 10.0, "2": 9.0, "3": 8.0}),
                   _ps("B", {"1": 10.0, "2": 8.0, "3": 6.0}),
                   _ps("C", {"1": 10.0, "2": 9.5, "3": 9.2}))
    walls = wa.find_walls(stats, [{"A": 4, "B": 4, "C": 4}])
    probe, skipped = wa.select_for_probing(walls, 1)
    assert len(probe) == 1 and len(skipped) == 2
    assert len(probe) + len(skipped) == 3, "every worth-probing wall must still be accounted for"


# ---------------------------------------------------------------------------------------------
# the ablation itself
# ---------------------------------------------------------------------------------------------


def test_the_ablation_moves_exactly_one_knob():
    theta = {"BLOCK_M": 256, "BLOCK_N": 128, "NUM_STAGES": 3, "COMPUTE_DTYPE": "fp16"}
    stats = _stats(_ps("BLOCK_M", {"64": 3.8, "128": 3.4, "256": 3.2}))
    wall = wa.find_walls(stats, [{"BLOCK_M": 512}])[0]
    out = wa.ablation_params(theta, wall)
    assert out["BLOCK_M"] == 512
    assert {k: v for k, v in out.items() if k != "BLOCK_M"} == \
           {k: v for k, v in theta.items() if k != "BLOCK_M"}
    assert theta["BLOCK_M"] == 256, "the caller's dict must not be mutated"


def test_an_integral_refused_value_is_written_back_as_an_int():
    """`BLOCK_M = 512.0` is not `BLOCK_M = 512`: it changes the source text, and a float where a
    `tl.constexpr` int is expected can change what the kernel does. The refused value arrives as a
    float because it came through numeric parsing.
    """
    stats = _stats(_ps("BLOCK_M", {"64": 3.8, "128": 3.4, "256": 3.2}))
    wall = wa.find_walls(stats, [{"BLOCK_M": 512}])[0]
    assert wall.refused_value == 512.0                      # parsed as float
    assert wa.ablation_params({"BLOCK_M": 256}, wall)["BLOCK_M"] == 512
    assert isinstance(wa.ablation_params({"BLOCK_M": 256}, wall)["BLOCK_M"], int)


# ---------------------------------------------------------------------------------------------
# the verdict mapping -- where a collapse would be a defect
# ---------------------------------------------------------------------------------------------


def test_an_unanswered_probe_is_undecidable_and_never_not_attributed():
    """`cached_shared_verdict` is three-valued for a reason. Collapsing None into "not attributed"
    would let a worker timeout read as evidence that the knob is innocent -- the same shape of bug
    as caching a probe failure, which cost 0.71 h on box 3.
    """
    assert wa.verdict_from(None) == "undecidable"
    assert wa.verdict_from(True) == "not_attributed"    # it fits => the knob alone is not the cause
    assert wa.verdict_from(False) == "attributed"       # it cannot launch => the knob alone suffices


def test_summarize_counts_only_decided_walls():
    stats = _stats(_ps("A", {"1": 10.0, "2": 9.0, "3": 8.0}))
    w = wa.find_walls(stats, [{"A": 4}])[0]
    assert wa.summarize([w]) == {"attributed": 0, "not_attributed": 0, "undecidable": 0}
    w.verdict = "attributed"
    assert wa.summarize([w])["attributed"] == 1


# ---------------------------------------------------------------------------------------------
# for_prompt: the conditionality is not optional
# ---------------------------------------------------------------------------------------------


def _attributed_wall(second_origin: str | None = None) -> wa.Wall:
    stats = _stats(_ps("BLOCK_M", {"64": 3.8221, "128": 3.3807, "256": 3.1836}))
    w = wa.find_walls(stats, [{"BLOCK_M": 512}])[0]
    w.verdict = "attributed"
    w.max_shared, w.limit, w.over_ratio = 122880, 101376, 122880 / 101376
    w.second_origin = second_origin
    return w


def test_the_prompt_text_states_the_conclusion_is_at_the_optimum():
    """The measured reason: from the default corner only 1 of 6 walls attributes, against 6 of 6
    from the optimum. An unconditional "BLOCK_M is capped by shared memory" would send the rewriter
    after a wall that is not there at the point it rewrites from.
    """
    text = wa.for_prompt([_attributed_wall()])
    assert text is not None
    assert "最优参数点" in text, "the text must say the verdict holds AT THE OPTIMUM"


def test_a_disagreeing_second_origin_is_stated_in_the_prompt():
    text = wa.for_prompt([_attributed_wall(second_origin="not_attributed")])
    assert "默认配置" in text and "其他 knob" in text


def test_only_attributed_walls_reach_the_prompt():
    """A `not_attributed` or `undecidable` wall is not a fact about one knob, so presenting it as a
    rewrite brief would be a guess wearing a measurement's clothes.
    """
    w = _attributed_wall()
    w.verdict = "not_attributed"
    assert wa.for_prompt([w]) is None
    w.verdict = "undecidable"
    assert wa.for_prompt([w]) is None
    stats = _stats(_ps("A", {"1": 10.0, "2": 9.0, "3": 8.0}))
    assert wa.for_prompt(wa.find_walls(stats, [{"A": 4}])) is None   # verdict still None


def test_the_prompt_carries_no_resource_vector_and_no_candidate_latency():
    """`pct_of_dram_peak` / `pct_of_compute_peak` are 1/latency rescaled -- measured constant to
    0.07-0.36% while gpu_ms moved 4.6x -- so putting them in a prompt tells the agent how fast this
    candidate is, and any later "the agent optimized resources" conclusion becomes circular.
    `assert_no_raw_vector` blocks them for the dimension digest; this section must not reintroduce
    them by another door. A per-knob TREND is a property of the axis, not a ranking of the candidate.
    """
    text = wa.for_prompt([_attributed_wall()])
    for banned in ("pct_of_dram_peak", "pct_of_compute_peak", "achieved_tbs", "achieved_tflops"):
        assert banned not in text


def test_the_prompt_marks_the_direction_as_a_hint_not_an_instruction():
    """The agent, not the harness, decides whether a restructure is possible. This project already
    measured what happens when a hand-written resource formula is presented as truth: agent-written
    shared-memory constraints came in at a median 32% of the real limit.
    """
    text = wa.for_prompt([_attributed_wall()])
    assert "提示" in text and "不是指令" in text


# ---------------------------------------------------------------------------------------------
# structural guards on the wiring
# ---------------------------------------------------------------------------------------------


def test_the_attribution_never_shrinks_a_domain():
    """2e records a wall; it must never enforce one. The user's standing constraint is that the
    search space is not narrowed to control cost, and `declare_infeasible_out_of_space` is the
    separate, switched, off-by-default stage that is allowed to touch domains.
    """
    text = io.open(SRC / "evaluation" / "wall_attribution.py", encoding="utf-8").read()
    for banned in ("choices =", "choices.remove", "domains =", "del ", ".pop("):
        assert banned not in text, f"wall_attribution must not mutate a space ({banned})"


def test_the_orchestrator_probes_walls_in_one_batch():
    """A probe in its own worker process costs a median 16.7 s, almost all of it process start; 18
    variants in one process measured 8.8 s. Probing per wall would make the diagnostic cost more
    than the waste it describes, so the batch entry point is the one that must be called.
    """
    text = io.open(SRC / "control" / "orchestrator.py", encoding="utf-8").read()
    tree = ast.parse(text)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_probe_walls")
    body = ast.get_source_segment(text, fn) or ""
    assert "prescreen_batch" in body
    assert "compile_screen(" not in body, \
        "compile_screen is one-probe-per-call; use the batch entry point"


def test_wall_attribution_runs_before_the_analyst_is_called():
    """The point of measuring is to replace the analyst's unchecked guess in the SAME round. If the
    attribution ran afterwards it could only ever inform the next candidate, and `in_prompt` would
    silently do nothing.
    """
    text = io.open(SRC / "control" / "orchestrator.py", encoding="utf-8").read()
    tree = ast.parse(text)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_stats_and_analysis")
    body = ast.get_source_segment(text, fn) or ""
    i_attr = body.find("_attribute_resource_walls")
    i_analyst = body.find("self.deps.analyst.invoke")
    assert i_attr != -1 and i_analyst != -1
    assert i_attr < i_analyst, "attribution must precede the analyst call"


def test_the_diagnostic_cannot_end_a_candidate():
    """Same rule as `_reconcile_round`: a diagnostic that raised would present a bookkeeping defect
    as a candidate defect. The whole body is wrapped, and the failure is journalled rather than
    swallowed in silence.
    """
    text = io.open(SRC / "control" / "orchestrator.py", encoding="utf-8").read()
    tree = ast.parse(text)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_attribute_resource_walls")
    body = ast.get_source_segment(text, fn) or ""
    assert "except Exception" in body
    assert "RESOURCE_WALL_ATTRIBUTION_FAILED" in body


def test_theta_star_uses_the_projects_own_objective():
    """`robust_ms` (median, else mean) is this project's measured objective: 93.2% rank-correctness
    against the mean's 64.8% at 20 samples, and `min` is biased +9.8% to +156%. A second rule here
    could name a different winner than the one the framework selected and the agent was told about.

    Matched on CODE only. The first version of this test scanned the whole function text and failed
    on the word `.mean` inside the comment explaining why `.mean` is not used -- a guard that forbids
    naming the thing it forbids is unmaintainable, and the next person would have deleted the
    explanation to make it pass.
    """
    text = io.open(SRC / "control" / "orchestrator.py", encoding="utf-8").read()
    tree = ast.parse(text)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_theta_star")
    # Attribute accesses on the latency object, from the AST -- comments cannot reach this.
    attrs = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
    assert "robust_ms" in attrs
    assert "mean" not in attrs, "selecting on the mean ranks 20-sample pairs correctly only 64.8%"
    assert "median" not in attrs, "read the objective through robust_ms, not the raw field"
    calls = {n.func.id for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "min" not in calls and "max" not in calls, \
        "min over samples reports the luckiest sample, not the cost"


def test_2e_is_off_by_default_and_the_prompt_path_is_off_too():
    """Both switches default off: the probes cost GPU time, so a run with them on is not
    byte-comparable with the finished runs; and `in_prompt` changes what the agent sees, which is
    the thing that needs a control arm. `probe_second_origin` defaults ON because the 1-of-6 result
    means a verdict without it cannot be read correctly.
    """
    from kernel_optimizer.config import AppConfig

    cfg = AppConfig()
    assert cfg.v3.wall_attribution.enabled is False
    assert cfg.v3.wall_attribution.in_prompt is False
    assert cfg.v3.wall_attribution.probe_second_origin is True
    assert cfg.v3.wall_attribution.max_probes_per_candidate == 8


def test_the_report_section_is_absent_when_2e_never_ran():
    """A run without 2e must read exactly as it did before, byte for byte in this section."""
    from kernel_optimizer.reporting.wall_report import wall_lines

    assert wall_lines([]) == []
    assert wall_lines([{"type": "TRIAL_DONE", "payload": {}}]) == []


def test_the_report_reads_events_of_either_shape():
    """`report.py` passes event OBJECTS with `.type`/`.payload`; the offline probes pass dicts. A
    reader that handles only one silently returns an empty section for the other, which reads as
    "2e found nothing" -- this project's recurring failure mode.
    """
    from kernel_optimizer.reporting.wall_report import wall_lines

    payload = {"candidate_id": "c", "walls_found": 1, "walls_probed": 1,
               "counts": {"attributed": 1, "not_attributed": 0, "undecidable": 0},
               "walls": [{"param": "BLOCK_M", "refused_value": 512.0,
                          "ran_values": [64.0, 128.0, 256.0], "tail_gain_pct": 16.7,
                          "monotone": True, "verdict": "attributed", "max_shared": 122880,
                          "limit": 101376, "over_ratio": 1.21, "second_origin": "not_attributed"}]}

    class Ev:
        def __init__(self, t, p):
            self.type, self.payload = t, p

    as_dicts = wall_lines([{"type": "RESOURCE_WALL_ATTRIBUTED", "payload": payload}])
    as_objs = wall_lines([Ev("RESOURCE_WALL_ATTRIBUTED", payload)])
    assert as_dicts and as_objs
    assert as_dicts == as_objs


def test_the_report_separates_worthless_walls_from_unattributed_ones():
    """"we found no wall" and "we found six worthless walls" are different states. A wall with
    `verdict: None` was never probed (slope filter or cap) and must not appear in the verdict TABLE
    as though the probe had cleared it.

    Checked on the table rows, not on the whole section: the summary line legitimately reads
    "ATTRIBUTED 0", and the first version of this test failed on exactly that -- it would have been
    "fixed" by removing a correct count from the report.
    """
    from kernel_optimizer.reporting.wall_report import wall_lines

    payload = {"candidate_id": "c", "walls_found": 1, "walls_probed": 0, "walls_worthless": 1,
               "walls": [{"param": "GEMM_BK", "refused_value": 256.0,
                          "ran_values": [16.0, 32.0, 64.0, 128.0], "tail_gain_pct": -54.8,
                          "monotone": False, "verdict": None}]}
    out = wall_lines([{"type": "RESOURCE_WALL_ATTRIBUTED", "payload": payload}])
    joined = "\n".join(out)
    assert "不值钱" in joined, "a capped-but-worthless wall must still be reported"
    # No verdict table at all, because nothing was probed. A row would be a fabricated verdict.
    assert not any(line.startswith("| `c`") for line in out), \
        "an unprobed wall must not appear as a table row"
    assert "GEMM_BK" in joined, "and it must still be named somewhere"
