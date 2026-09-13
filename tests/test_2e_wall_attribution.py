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

def test_refusals_with_no_walls_is_reported_as_a_finding_not_an_empty_table():
    """"20 refusals, 0 walls" and "0 refusals" are different states and must read differently.

    Measured live on arm 3: two candidates with 12 and 8 refused configurations produced 0 walls,
    because every refused value had also RUN successfully in some other combination -- nothing was
    truncated out of its range. A summary printing only "found 0" cannot be told apart from "this
    card never refused anything", and the two call for opposite conclusions: the first says the
    mechanism had input and legitimately had nothing to say, the second says the arm is on the wrong
    hardware (the A800's 166912 B against the 4090's 101376 B).
    """
    from kernel_optimizer.reporting.wall_report import wall_lines

    had_input = {"candidate_id": "cand-a", "n_refused_configs": 12, "walls_found": 0,
                 "walls_probed": 0, "walls_worthless": 0, "walls": []}
    out = "\n".join(wall_lines([{"type": "RESOURCE_WALL_ATTRIBUTED", "payload": had_input}]))
    assert "12" in out, "the refusal count is not reported, so 0 walls is uninterpretable"
    assert "有输入但没有墙" in out, "a run with input and no walls is not distinguished"
    assert "没有任何输入" not in out, "a run WITH refusals was described as having no input"

    no_input = dict(had_input, n_refused_configs=0)
    out2 = "\n".join(wall_lines([{"type": "RESOURCE_WALL_ATTRIBUTED", "payload": no_input}]))
    assert "没有任何输入" in out2, "a run with no refusals at all is not distinguished"
    assert "有输入但没有墙" not in out2, "a run with no input was described as having had input"


def test_neither_zero_message_appears_when_walls_were_found():
    """The two explanatory lines are for the empty cases only.

    A run that DID find walls must not also carry "there was no wall", which would contradict its own
    table -- the kind of thing that happens when a new branch is added without an else.
    """
    from kernel_optimizer.reporting.wall_report import wall_lines

    payload = {"candidate_id": "cand-b", "n_refused_configs": 40, "walls_found": 1,
               "walls_probed": 1, "walls_worthless": 0,
               "counts": {"attributed": 1},
               "walls": [{"param": "BLOCK_M", "refused_value": 512.0,
                          "ran_values": [64.0, 128.0, 256.0], "tail_gain_pct": 16.7,
                          "monotone": True, "verdict": "attributed",
                          "max_shared": 122880, "limit": 101376, "over_ratio": 1.21}]}
    out = "\n".join(wall_lines([{"type": "RESOURCE_WALL_ATTRIBUTED", "payload": payload}]))
    assert "有输入但没有墙" not in out
    assert "没有任何输入" not in out
    assert "40" in out, "the refusal count should still be reported when walls WERE found"


# ---------------------------------------------------------------------------
# A3: did the rewrite the wall text steered actually FREE the walled dimension?
#
# The fixtures below are built from arm 3's real log (run-l3-43-20260912-214326): candidate
# cand-dc87a93a, knob BLOCK_N, refused 128, probe measured 122880 B against the 101376 B limit, and
# two descendants that ran BLOCK_N=128 at 77824 B and 45056..98304 B. Field spellings are copied from
# the emitters -- `params.values`, `profile.shared_bytes`, `parent_ids`,
# `failure_kind == "infeasible_shared_memory"` -- not invented, because a fixture shaped to match
# the reader proves only that the reader reads itself.
# ---------------------------------------------------------------------------

def _wall_ev(cid="cand-dc87a93a", knob="BLOCK_N", refused=128.0, verdict="attributed"):
    return {"type": "RESOURCE_WALL_ATTRIBUTED",
            "payload": {"candidate_id": cid, "n_refused_configs": 12, "walls_found": 1,
                        "walls_probed": 1, "walls_worthless": 0,
                        "counts": {"attributed": 1 if verdict == "attributed" else 0},
                        "walls": [{"param": knob, "refused_value": refused,
                                   "ran_values": [16.0, 32.0, 64.0], "tail_gain_pct": 7.5,
                                   "monotone": True, "verdict": verdict,
                                   "max_shared": 122880, "limit": 101376, "over_ratio": 1.212}]}}


def _child(cid, parent):
    return {"type": "CANDIDATE_REGISTERED",
            "payload": {"candidate": {"candidate_id": cid, "family_id": "fam-8e54d0af",
                                      "parent_ids": [parent], "origin": "rewrite"}}}


def _trial(cid, values, status="complete", shared=77824, failure_kind=None):
    trial = {"candidate_id": cid, "params": {"values": values}, "status": status,
             "latency_ms": {"mean": 3.9, "median": 3.84, "std": 0.1, "min": 3.7, "max": 4.1,
                            "n_samples": 20}}
    if status == "complete":
        trial["profile"] = {"shared_bytes": shared, "n_regs": 200, "n_spills": 0,
                            "compile_s": 0.42, "num_warps": 8, "num_stages": 2}
    else:
        trial["failure_kind"] = failure_kind
    return {"type": "TRIAL_DONE", "payload": {"trial": trial}}


def test_a3_reports_FREED_when_a_descendant_runs_the_refused_value():
    """The positive case, and the one measured live: the compiler accepted what it refused."""
    from kernel_optimizer.reporting.wall_report import wall_lines

    out = "\n".join(wall_lines([
        _wall_ev(),
        _child("cand-5bd8ddd3", "cand-dc87a93a"),
        _trial("cand-5bd8ddd3", {"BLOCK_M": 64, "BLOCK_N": 128, "NUM_WARPS": 8}),
    ]))
    assert "A3" in out, "the A3 section did not render at all"
    assert "FREED" in out
    assert "cand-5bd8ddd3" in out
    assert "77824" in out, "the footprint AT the wall's value is the evidence and must be shown"
    assert "占用已降到上限内" in out, (
        "77824 is under the 101376 limit, so the section must say the footprint fell -- otherwise a "
        "reader cannot tell the mechanism worked from a coincidence")


def test_a3_matches_a_string_knob_value_against_a_float_refused_value():
    """The trap that would fake a negative result.

    `refused_value` comes back from `find_walls` as a FLOAT (128.0) -- it passes through that
    module's numeric coercion. The trial's `params.values` holds whatever the SPACE declared, and
    spaces really do declare numeric-looking knobs with `kind: "str"` (COMPUTE_DTYPE is a str knob;
    a tile size declared the same way arrives as "128"). `"128" == 128.0` is False in Python, so a
    plain `==` prints "未曾尝试" for a dimension that was demonstrably freed -- indistinguishable
    from a real negative A3, with the mechanism silently reported as ineffective. This is the reason
    `_same_value` coerces before comparing.

    NOT tested with int 128 against float 128.0: Python already answers True for that pair, so such a
    test passes with or without `_same_value` and pins nothing. Found by the revert check, which
    reported the suite still green after replacing `_same_value`'s body with `a == b`.
    """
    from kernel_optimizer.reporting.wall_report import wall_lines

    as_str = "\n".join(wall_lines([
        _wall_ev(refused=128.0),
        _child("kid", "cand-dc87a93a"),
        _trial("kid", {"BLOCK_N": "128"}),        # str in the trial, as a str-kind knob yields
    ]))
    as_float = "\n".join(wall_lines([
        _wall_ev(refused=128.0),
        _child("kid", "cand-dc87a93a"),
        _trial("kid", {"BLOCK_N": 128.0}),
    ]))
    assert "FREED" in as_str, (
        'a str trial value ("128") did not match a float refused_value (128.0)')
    assert "FREED" in as_float
    assert "未曾尝试" not in as_str


def test_a3_does_not_match_values_that_merely_look_similar():
    """The other half of `_same_value`: coercion must not turn a DIFFERENT value into a match.

    Without this, "loosen the comparison until the test passes" would be a valid way to make the
    section report FREED for a wall nothing freed -- a false positive on the project's own claim,
    which is worse than the false negative above.
    """
    from kernel_optimizer.reporting.wall_report import wall_lines

    out = "\n".join(wall_lines([
        _wall_ev(refused=128.0),
        _child("kid", "cand-dc87a93a"),
        _trial("kid", {"BLOCK_N": 64}),
        _trial("kid", {"BLOCK_N": "64"}),
        _trial("kid", {"BLOCK_N": 1280}),
    ]))
    assert "FREED" not in out, "a non-matching value was reported as having freed the wall"
    assert "未曾尝试" in out


def test_a3_distinguishes_still_walled_from_dimension_gone_from_not_tried():
    """Three non-FREED outcomes that must never be collapsed into one.

    Collapsing them is the actual risk: "维度已消失" counted as failure slanders a rewrite that
    removed the constraint by removing the tile loop, and counted as success credits one for deleting
    the evidence. "未曾尝试" is neither -- TPE is not a uniform sampler.
    """
    from kernel_optimizer.reporting.wall_report import wall_lines

    still = "\n".join(wall_lines([
        _wall_ev(),
        _child("kid", "cand-dc87a93a"),
        _trial("kid", {"BLOCK_N": 128}, status="fail",
               failure_kind="infeasible_shared_memory"),
    ]))
    assert "仍被拒" in still and "FREED" not in still

    gone = "\n".join(wall_lines([
        _wall_ev(),
        _child("kid", "cand-dc87a93a"),
        _trial("kid", {"TILE": 64}),   # the knob is not in this child's space at all
    ]))
    assert "维度已消失" in gone and "FREED" not in gone

    untried = "\n".join(wall_lines([
        _wall_ev(),
        _child("kid", "cand-dc87a93a"),
        _trial("kid", {"BLOCK_N": 64}),   # knob present, but never at the refused value
    ]))
    assert "未曾尝试" in untried and "FREED" not in untried


def test_a3_says_so_when_the_walled_candidate_has_no_descendant_yet():
    """A wall found late in a run has no rewrite to judge. That is not a negative result, and a blank
    row would read as one."""
    from kernel_optimizer.reporting.wall_report import wall_lines

    out = "\n".join(wall_lines([_wall_ev()]))
    assert "尚无后代" in out
    assert "FREED" not in out


def test_a3_uses_the_footprint_at_the_wall_not_at_the_childs_optimum():
    """The distinction the whole section rests on, pinned with arm 3's real confound.

    cand-e5172b8a's optimum used 73728 B -- IDENTICAL to its walled parent's -- while its trials at
    BLOCK_N=128 ran at 45056..98304 B. A section that reported the optimum's footprint would show "no
    change" for a candidate that demonstrably freed the wall. So the number in the row must come from
    the trials AT the refused value, and the fastest trial (here a low-BLOCK_N one at 73728) must NOT
    be what gets reported.
    """
    from kernel_optimizer.reporting.wall_report import wall_lines

    out = "\n".join(wall_lines([
        _wall_ev(),
        _child("cand-e5172b8a", "cand-dc87a93a"),
        # The optimum: fastest trial, but at a DIFFERENT BLOCK_N, and at the parent's footprint.
        _trial("cand-e5172b8a", {"BLOCK_N": 32}, shared=73728),
        # The evidence: trials at the wall's value, with a lower footprint.
        _trial("cand-e5172b8a", {"BLOCK_N": 128}, shared=45056),
        _trial("cand-e5172b8a", {"BLOCK_N": 128}, shared=98304),
    ]))
    assert "45056..98304" in out, (
        "the row must report the footprint span AT the refused value, not the optimum's")
    assert "FREED" in out


def test_a3_is_silent_when_no_wall_was_attributed():
    """A found-but-unattributed wall has no delivered measurement to have steered anything, so there
    is nothing for A3 to check -- and an empty A3 table would read as a negative."""
    from kernel_optimizer.reporting.wall_report import wall_lines

    out = "\n".join(wall_lines([
        _wall_ev(verdict="not_attributed"),
        _child("kid", "cand-dc87a93a"),
        _trial("kid", {"BLOCK_N": 128}),
    ]))
    assert "A3" not in out, "A3 rendered for a wall that was never attributed"


def test_a3_names_the_counterfactual_it_cannot_supply():
    """The section must not read as proof on its own. "the rewriter relieves shared memory anyway"
    explains the same observation, and only an arm with 2e OFF can rule it out -- measured 0 of 7 on
    arm 2 against 1 of 1 on arm 3. A reader who takes the FREED row as evidence of the steer without
    that comparison has over-read it, so the caveat ships inside the section.
    """
    from kernel_optimizer.reporting.wall_report import wall_lines

    out = "\n".join(wall_lines([
        _wall_ev(),
        _child("kid", "cand-dc87a93a"),
        _trial("kid", {"BLOCK_N": 128}),
    ]))
    assert "不能单独证明" in out
    assert "find_walls" in out, "the caveat must name how to run the counterfactual"


def test_a3_walks_the_whole_lineage_not_just_direct_children():
    """A wall can be freed two rewrites later. `descendants` is transitive for that reason, and a
    direct-children-only reader would report "未曾尝试" for a grandchild that freed it.
    """
    from kernel_optimizer.reporting.wall_report import wall_lines

    out = "\n".join(wall_lines([
        _wall_ev(),
        _child("kid", "cand-dc87a93a"),
        _child("grandkid", "kid"),
        _trial("kid", {"BLOCK_N": 64}),                  # child did not try it
        _trial("grandkid", {"BLOCK_N": 128}),            # grandchild freed it
    ]))
    assert "grandkid" in out, "the grandchild never appeared in the table"
    assert "FREED" in out
