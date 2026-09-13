"""Item 2: the register-spill SOFT wall.

The measurements these tests encode, from 2788 completed trials / 56 candidates across a 4090 and an
A800, plus a re-run of the SHIPPING function over the 153-candidate local corpus
(`scripts/probes/soft_wall_acceptance.py`):

  - `n_spills` passes the independence threshold (largest |rho| 0.339 against n_regs, -0.048 against
    the tile product) AND the stricter test that separated it from occupancy: the winner sits at the
    BOTTOM of its own spill range on 56 of 56 candidates (median position 0.00), so "reduce spilling"
    never points away from the measured optimum. Occupancy is PEAKED on 37 of 56 with the winner at a
    median position 0.33, which is why it is not implemented and must not be re-proposed.
  - applicability is NOT a constant: 43% (24/56) on the five-run corpus, 23.5% (36/153) on the local
    one, and 0 of 8 on the L3:48 runs. The shape travels -- many winners do not spill at all -- the
    percentage does not, so every report publishes its denominator.
  - the shipping criterion produces 50 (candidate, knob) walls over 22 candidates, median tail gain
    +17.8%.
  - the shape is peaked on 27 of 56 candidates, so the criterion is applied PER KNOB and never as a
    global monotone rule.
"""

from __future__ import annotations

import ast
import io
from pathlib import Path

from kernel_optimizer.evaluation import soft_wall as sw
from kernel_optimizer.models.core import LatencyStats, ParamSet, ProfileRecord, TrialRecord
from kernel_optimizer.models.reports import ParamStat, TuningStats

SRC = Path("src/kernel_optimizer")


def _stats(*names: str) -> TuningStats:
    """`find_soft_walls` reads only the knob NAMES off the stats; the curves come from the trials."""
    return TuningStats(candidate_id="cand-x", space_id="sp-x", n_complete=10, n_fail=0,
                       param_stats=[ParamStat(name=n, best_value="?") for n in names])


def _trial(values: dict, ms: float, spills: int | None, *, status: str = "complete",
           regs: int = 200, limiter: str | None = None) -> TrialRecord:
    """A TrialRecord shaped like the emitter's.

    Field spellings copied from `models/core.py` rather than invented: `LatencyStats` requires
    mean/std/min/max/n_samples and exposes `robust_ms` as a PROPERTY (never serialized), and
    `occupancy` is a NESTED dict whose limiter key is `limiter` -- not `occupancy_limiter`, which is
    the name of the model's property that reads it.
    """
    occ = None
    if limiter is not None:
        occ = {"occupancy": 0.25, "limiter": limiter, "active_warps": 8, "max_warps_per_sm": 48}
    return TrialRecord(
        trial_id=f"t-{sorted(values.items())}-{ms}", candidate_id="cand-x", space_id="sp-x",
        params=ParamSet(values=values), status=status,
        latency_ms=LatencyStats(mean=ms, std=0.05, min=ms, max=ms, n_samples=20, median=ms),
        profile=ProfileRecord(n_regs=regs, n_spills=spills, shared_bytes=32768, num_warps=4,
                              num_stages=2, compile_s=0.4, occupancy=occ))


def _ladder(spills: list[int], lats: list[float], knob: str = "BLOCK_M",
            values: list[int] | None = None, limiter: str | None = None) -> list[TrialRecord]:
    """One trial per value, spills and latency given per value."""
    vals = values or [16, 32, 64, 128][:len(spills)]
    return [_trial({knob: v}, ms, sp, limiter=limiter)
            for v, sp, ms in zip(vals, spills, lats)]


# ---------------------------------------------------------------------------------------------
# the applicability gate, which is the first thing the criterion decides
# ---------------------------------------------------------------------------------------------


def test_a_winner_that_does_not_spill_is_NOT_APPLICABLE_rather_than_no_wall():
    """The distinction the whole reporting rests on. 103 of 153 candidates in the local corpus are in
    this state: their fastest trial spills zero, so no knob can be walled by spilling AT THE POINT THE
    AGENT REWRITES FROM. Reporting that as "no wall found" would make a structural non-event look like
    a negative result about the mechanism.
    """
    # Fastest trial is the last one (1.0 ms) and it does not spill.
    trials = _ladder([8, 4, 0], [3.0, 2.0, 1.0])
    scan = sw.find_soft_walls(_stats("BLOCK_M"), trials)
    assert scan.applicable is False
    assert scan.walls == []
    assert "does not spill" in scan.reason
    assert scan.spills_at_best == 0.0


def test_a_spilling_winner_is_applicable():
    trials = _ladder([0, 4, 12], [3.0, 2.0, 1.0])
    scan = sw.find_soft_walls(_stats("BLOCK_M"), trials)
    assert scan.applicable is True
    assert scan.spills_at_best == 12.0


def test_the_gate_reads_the_BEST_trial_and_not_any_trial():
    """Same rule as the hard wall's ablation origin: the question is whether a step is available at the
    point the agent is rewriting from. A candidate with one slow spilling trial and a fast clean one is
    NOT applicable, however much spilling exists elsewhere in its history.
    """
    trials = [_trial({"BLOCK_M": 16}, 9.0, 400),      # slow and spilling heavily
              _trial({"BLOCK_M": 32}, 8.0, 300),
              _trial({"BLOCK_M": 64}, 1.0, 0)]        # the winner, clean
    scan = sw.find_soft_walls(_stats("BLOCK_M"), trials)
    assert scan.applicable is False


def test_a_candidate_with_no_completed_timed_trial_is_not_applicable():
    trials = [_trial({"BLOCK_M": 16}, 3.0, 8, status="fail")]
    scan = sw.find_soft_walls(_stats("BLOCK_M"), trials)
    assert scan.applicable is False
    assert "no completed" in scan.reason


def test_a_missing_n_spills_is_unmeasured_and_not_zero():
    """None must never be coerced to 0. An unmeasured spill count read as zero would declare every
    candidate on a backend that reports no spills "not applicable" for the wrong reason -- the reader
    would have manufactured a measurement.
    """
    trials = _ladder([0, 4, None], [3.0, 2.0, 1.0])   # winner's spills unmeasured
    scan = sw.find_soft_walls(_stats("BLOCK_M"), trials)
    assert scan.applicable is False
    assert "no n_spills" in scan.reason
    assert scan.spills_at_best is None, "an unmeasured field must not be reported as a number"


# ---------------------------------------------------------------------------------------------
# the criterion's three conditions
# ---------------------------------------------------------------------------------------------


def test_a_monotone_rise_into_spilling_with_latency_improving_is_a_wall():
    """The positive control. Without one, a criterion that silently never fires would produce the same
    "no walls" output as a criterion that correctly found none -- and this project has recorded five
    negative "results" that turned out to be broken probes.
    """
    trials = _ladder([0, 0, 16, 72], [4.0, 3.5, 3.0, 2.6], limiter="registers")
    scan = sw.find_soft_walls(_stats("BLOCK_M"), trials)
    assert [w.param for w in scan.walls] == ["BLOCK_M"]
    w = scan.walls[0]
    assert w.spills_by_value == [0.0, 0.0, 16.0, 72.0]
    assert w.onset_value == 64.0, "the onset is where spilling FIRST becomes non-zero"
    assert w.tail_gain_pct > 0
    assert w.limiter_at_best == "registers"


def test_a_peaked_spill_curve_does_not_fire():
    """27 of 56 candidates are peaked, so the criterion is applied PER KNOB rather than as a global
    monotone rule. A peaked knob simply does not fire; forcing it into a monotone reading would report
    a wall where the relation does not have one.
    """
    trials = _ladder([0, 40, 8, 72], [4.0, 3.5, 3.0, 2.6])
    scan = sw.find_soft_walls(_stats("BLOCK_M"), trials)
    assert scan.applicable is True, "the winner does spill, so this is applicable"
    assert scan.walls == []
    assert scan.n_knobs_with_spill_variation == 1, \
        "the knob must be counted as examined, or 'peaked' and 'no data' would look the same"


def test_a_worsening_tail_does_not_fire():
    """The same slope filter as the hard wall, where it removed 3 of 6 walls (worst -54.8%). A knob
    whose latency gets WORSE toward the spilling end is capped and worthless: telling the rewriter to
    free it would spend a rewrite on a non-problem.
    """
    trials = _ladder([0, 4, 16, 72], [2.0, 2.4, 3.0, 3.6])   # latency rising with spills
    scan = sw.find_soft_walls(_stats("BLOCK_M"), trials)
    # The winner here is the FIRST value, which does not spill => not applicable, which is itself
    # correct. Force applicability with a spilling winner and a worsening tail:
    trials = [_trial({"BLOCK_M": 16}, 3.0, 30), _trial({"BLOCK_M": 32}, 2.0, 40),
              _trial({"BLOCK_M": 64}, 2.5, 50), _trial({"BLOCK_M": 128}, 3.4, 90)]
    scan = sw.find_soft_walls(_stats("BLOCK_M"), trials)
    assert scan.applicable is True
    assert scan.walls == [], "a knob whose latency worsens toward the spilling end is worthless"


def test_a_flat_spill_curve_does_not_fire():
    """No knob moved the spilling, so nothing is attributable to a knob -- the soft analogue of a
    refused value that lies INSIDE the measured range."""
    trials = _ladder([8, 8, 8, 8], [4.0, 3.5, 3.0, 2.6])
    scan = sw.find_soft_walls(_stats("BLOCK_M"), trials)
    assert scan.applicable is True
    assert scan.walls == []
    assert scan.n_knobs_with_spill_variation == 0
    assert "no knob" in scan.reason


def test_fewer_than_three_values_gives_no_wall():
    """Two points are monotone through any two points. Same floor as the hard wall."""
    trials = _ladder([0, 40], [3.0, 2.0], values=[16, 32])
    scan = sw.find_soft_walls(_stats("BLOCK_M"), trials)
    assert scan.walls == []


def test_a_categorical_knob_is_never_a_soft_wall():
    """A precision switch's "first" and "last" choices are an artifact of list order, so a slope over
    them measures the order the agent wrote. Same `_as_num` exclusion as the hard wall, and reused from
    it rather than reimplemented.

    The fixture pairs the categorical knob with a NUMERIC one that has the same curve, so a variant
    that let non-numeric values through cannot pass by collapsing every string to a single bucket -- the
    revert-check caught exactly that hole in the first version of this test, which used a lone
    categorical knob and stayed green when the numeric filter was removed.
    """
    trials = [_trial({"DTYPE": "fp16", "BLOCK": 16}, 3.0, 0),
              _trial({"DTYPE": "bf16", "BLOCK": 32}, 2.5, 20),
              _trial({"DTYPE": "tf32", "BLOCK": 64}, 2.0, 60)]
    scan = sw.find_soft_walls(_stats("DTYPE", "BLOCK"), trials)
    assert scan.applicable is True
    assert [w.param for w in scan.walls] == ["BLOCK"], \
        "the numeric knob must fire and the categorical one must not"
    assert scan.n_knobs_examined == 1, \
        "a categorical knob must not even be counted as examined; a distance over its list order " \
        "would be a measurement of the order the agent happened to write"


def test_walls_are_ordered_by_tail_gain():
    trials = [_trial({"A": 16, "B": 16}, 4.0, 0), _trial({"A": 32, "B": 32}, 3.9, 8),
              _trial({"A": 64, "B": 64}, 2.0, 40)]
    scan = sw.find_soft_walls(_stats("A", "B"), trials)
    assert len(scan.walls) == 2
    assert scan.walls[0].tail_gain_pct >= scan.walls[1].tail_gain_pct


def test_a_knob_absent_from_param_stats_does_not_enter():
    """The harness's own view of "this candidate's knobs" is `param_stats`. A knob present in the
    trials but not there would be a divergence between two views of the same space, and a wall reported
    on it could name a knob the report never lists.
    """
    trials = [_trial({"A": 16, "SECRET": 1}, 4.0, 0), _trial({"A": 32, "SECRET": 2}, 3.0, 8),
              _trial({"A": 64, "SECRET": 4}, 2.0, 40)]
    scan = sw.find_soft_walls(_stats("A"), trials)
    assert [w.param for w in scan.walls] == ["A"]


def test_the_scan_reports_its_denominators():
    """A count with no denominator is not a measurement: "0 walls" from a non-spilling winner and
    "0 walls" from flat curves are different facts, and applicability was measured at 43% on one corpus
    and 23.5% on another -- so the denominator has to travel with the count.
    """
    trials = _ladder([0, 0, 16, 72], [4.0, 3.5, 3.0, 2.6], limiter="registers")
    p = sw.find_soft_walls(_stats("BLOCK_M"), trials).payload()
    for key in ("applicable", "reason", "n_knobs_examined", "n_knobs_with_spill_variation",
                "spills_at_best", "limiter_at_best", "n_walls"):
        assert key in p, f"the payload must carry {key}"
    assert p["n_walls"] == 1 and p["n_knobs_examined"] == 1


# ---------------------------------------------------------------------------------------------
# reading a REPLAYED trial, where every field is a dict
# ---------------------------------------------------------------------------------------------


def _as_dict(t: TrialRecord) -> dict:
    return t.model_dump()


def test_the_criterion_reads_replayed_dict_trials_identically():
    """`report` and every offline probe pass dicts from events.jsonl; the orchestrator passes models. A
    reader that handled only one would silently return "no walls" for the other, which reads as a
    negative result -- this project's recurring failure shape.
    """
    trials = _ladder([0, 0, 16, 72], [4.0, 3.5, 3.0, 2.6], limiter="registers")
    from_models = sw.find_soft_walls(_stats("BLOCK_M"), trials)
    from_dicts = sw.find_soft_walls(_stats("BLOCK_M"), [_as_dict(t) for t in trials])
    assert from_dicts.payload() == from_models.payload()


def test_robust_ms_is_reproduced_and_not_looked_up_by_name():
    """`robust_ms` is a @PROPERTY and is NEVER serialized. A replayed trial has median/mean and no
    `robust_ms` key, so a reader that looked the name up would treat every trial as untimed and report
    "no completed, timed trial" for a full run.
    """
    trials = [_as_dict(t) for t in _ladder([0, 0, 16, 72], [4.0, 3.5, 3.0, 2.6])]
    for t in trials:
        assert "robust_ms" not in (t["latency_ms"] or {}), "fixture drifted from the emitter"
    assert sw.find_soft_walls(_stats("BLOCK_M"), trials).applicable is True


def test_the_limiter_key_is_the_one_the_emitter_writes():
    """`evaluation/statics.py` names the field `limiter`; `ProfileRecord.occupancy_limiter` is the
    PROPERTY that reads it. A reader looking for `occupancy_limiter` inside the dict finds nothing and
    reports None for a field that WAS measured -- `occupancy-is-nested-and-a-flat-read-fakes-unmeasured`
    one level deeper, and the first version of this module had exactly that bug.
    """
    trials = _ladder([0, 0, 16, 72], [4.0, 3.5, 3.0, 2.6], limiter="shared_memory")
    assert sw.find_soft_walls(_stats("BLOCK_M"), trials).limiter_at_best == "shared_memory"
    # ...and through the replayed dict path, where the property is not available at all.
    dicts = [_as_dict(t) for t in trials]
    assert dicts[-1]["profile"]["occupancy"]["limiter"] == "shared_memory", "fixture drifted"
    assert sw.find_soft_walls(_stats("BLOCK_M"), dicts).limiter_at_best == "shared_memory"


# ---------------------------------------------------------------------------------------------
# the rendered text
# ---------------------------------------------------------------------------------------------


def _one_wall_scan():
    return sw.find_soft_walls(
        _stats("GEMM_NUM_WARPS"),
        _ladder([0, 0, 16, 72], [4.0, 3.5, 3.0, 2.6], knob="GEMM_NUM_WARPS",
                values=[2, 4, 8, 16], limiter="registers"))


def test_the_text_says_there_was_no_independent_probe():
    """The load-bearing difference from the hard wall. The hard wall's attribution is checked by a
    second, independent question to the compiler (6 of 6 from the optimum); a soft wall has ONE
    observation of an already-measured field. A reader who takes it for a probed verdict has over-read
    it, so the caveat ships inside the text rather than in a document.
    """
    text = sw.for_prompt(_one_wall_scan())
    assert text is not None
    assert "未经独立探针确认" in text


def test_the_text_is_conditional_on_the_measured_points():
    text = sw.for_prompt(_one_wall_scan())
    assert "已测" in text, "an unconditional claim would read as an intrinsic limit of the knob"


def test_the_text_marks_the_direction_as_a_hint():
    """The agent decides whether a restructure is possible. A hand-written resource rule presented as
    truth measured a median 32% of the real limit in this project.
    """
    text = sw.for_prompt(_one_wall_scan())
    assert "提示" in text and "不是指令" in text


def test_the_text_carries_no_resource_vector_and_no_candidate_latency():
    """`pct_of_dram_peak` / `pct_of_compute_peak` are 1/latency rescaled, so putting them in a prompt
    tells the agent how fast this candidate is and makes any later "the agent optimized resources"
    conclusion circular. Same boundary the hard wall's text keeps.
    """
    text = sw.for_prompt(_one_wall_scan())
    for banned in ("pct_of_dram_peak", "pct_of_compute_peak", "achieved_tbs", "achieved_tflops"):
        assert banned not in text


def test_the_text_does_not_mention_occupancy_as_a_direction():
    """Occupancy failed the optimum test -- peaked on 37 of 56 with the winner at a median position
    0.33 -- so "raise occupancy" points away from the measured optimum. The limiter may be reported as
    the EXPLANATION of a spill wall; the word must never appear as advice.
    """
    text = sw.for_prompt(_one_wall_scan())
    assert "提高 occupancy" not in text
    assert "occupancy" not in text.replace("registers", ""), \
        "occupancy must not be presented to the rewriter as a direction"


def test_nothing_is_rendered_when_not_applicable_or_when_no_wall():
    assert sw.for_prompt(sw.find_soft_walls(_stats("A"), _ladder([8, 4, 0], [3.0, 2.0, 1.0]))) is None
    assert sw.for_prompt(sw.find_soft_walls(_stats("A"), _ladder([8, 8, 8], [3.0, 2.0, 1.0]))) is None


def test_every_payload_row_says_it_was_not_probe_confirmed():
    """Not only the prose. A reader of the raw event log never sees `for_prompt`, and the difference
    between a probed verdict and a single observation must survive into the machine-readable row.
    """
    p = _one_wall_scan().payload()
    assert p["walls"], "fixture produced no wall"
    for row in p["walls"]:
        assert row["probe_confirmed"] is False
        assert row["kind"] == "n_spills"
        assert "verdict" not in row, \
            "a soft wall has no probe, so a `verdict` field would invite a probed reading"
        assert "second_origin" not in row, \
            "there is no origin to re-probe; a lookalike field would hide that weakness"


# ---------------------------------------------------------------------------------------------
# structural guards
# ---------------------------------------------------------------------------------------------


def test_the_soft_wall_never_shrinks_a_domain():
    """Records, never enforces. The standing constraint is that the search space is not narrowed."""
    text = io.open(SRC / "evaluation" / "soft_wall.py", encoding="utf-8").read()
    tree = ast.parse(text)
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        elif isinstance(node, ast.Delete):
            targets = list(node.targets)
        for t in targets:
            name = getattr(t, "id", None) or getattr(t, "attr", None)
            assert name not in ("choices", "domains"), \
                f"soft_wall must not rebind {name!r} -- it records a wall, never enforces one"


def test_the_criterion_is_pure_and_touches_no_gpu():
    """Zero GPU cost is the reason this needs no control arm: it reads a field every trial already
    carries. An import of the evaluator or the worker would mean it could block or cost time.
    """
    text = io.open(SRC / "evaluation" / "soft_wall.py", encoding="utf-8").read()
    tree = ast.parse(text)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    for banned in ("subprocess", "kernel_optimizer.gpu.worker_client",
                   "kernel_optimizer.evaluation.correctness"):
        assert not any(i.startswith(banned) for i in imported), \
            f"soft_wall must stay pure; it imports {banned}"


def test_the_orderedness_rule_is_shared_with_the_hard_wall():
    """Two implementations of "is this knob numeric" are free to disagree, and then a soft wall could
    fire on a knob the hard wall treats as categorical. Checked behaviourally over the cases where a
    second implementation would plausibly differ.
    """
    from kernel_optimizer.evaluation import wall_attribution as wa

    for value in (16, 32.5, "128", "fp16", True, False, None, [1]):
        assert (sw._as_num(value) is None) == (wa._as_num(value) is None), \
            f"the two numeric rules disagree on {value!r}"


# ---------------------------------------------------------------------------------------------
# the wiring
# ---------------------------------------------------------------------------------------------


def test_off_by_default_and_the_prompt_path_is_off_too():
    """It costs no GPU, but it adds an event and (with in_prompt) changes what the agent sees, so a run
    with it on is not comparable with the finished ones. Same rule as every other v3 stage.
    """
    from kernel_optimizer.config import AppConfig

    cfg = AppConfig()
    assert cfg.v3.soft_wall.enabled is False
    assert cfg.v3.soft_wall.in_prompt is False
    assert cfg.v3.soft_wall.max_rows_in_prompt == 3


def test_the_scan_runs_before_the_analyst_is_called():
    """The point of measuring is to be able to replace the analyst's unchecked guess in the SAME round.
    Running afterwards could only inform the next candidate, and `in_prompt` would silently do nothing.
    """
    text = io.open(SRC / "control" / "orchestrator.py", encoding="utf-8").read()
    tree = ast.parse(text)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_stats_and_analysis")
    body = ast.get_source_segment(text, fn) or ""
    i_soft = body.find("_attribute_soft_walls")
    i_analyst = body.find("self.deps.analyst.invoke")
    assert i_soft != -1 and i_analyst != -1
    assert i_soft < i_analyst, "the soft-wall scan must precede the analyst call"


def test_the_soft_wall_cannot_end_a_candidate():
    """Same rule as the hard wall's attribution: a diagnostic that raised would present a bookkeeping
    defect as a candidate defect. The whole body is wrapped and the failure is journalled.
    """
    text = io.open(SRC / "control" / "orchestrator.py", encoding="utf-8").read()
    tree = ast.parse(text)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_attribute_soft_walls")
    body = ast.get_source_segment(text, fn) or ""
    assert "except Exception" in body
    assert "RESOURCE_SOFT_WALL_FAILED" in body


def test_the_soft_wall_is_journalled_even_when_it_finds_nothing():
    """"nothing found" is a result, and its absence would be indistinguishable from the switch being
    off. Driven through the REAL `_attribute_soft_walls` with a real store, because the structural
    version of this test (asserting the append precedes the render in the source) stayed green when the
    append was wrapped in `if scan.walls:` -- the revert-check caught that, and a source-order guard
    could not have.
    """
    from kernel_optimizer.config import AppConfig
    from kernel_optimizer.control import orchestrator as orch_mod

    class _Store:
        def __init__(self):
            self.events = []

        def append(self, t, p):
            self.events.append((t, p))

    def _run(trials):
        cfg = AppConfig()
        cfg.v3.soft_wall.enabled = True
        o = object.__new__(orch_mod.Orchestrator)
        o.cfg = cfg
        o.store = _Store()
        crun = object.__new__(orch_mod.CandidateRun)
        crun.candidate = type("C", (), {"candidate_id": "cand-x"})()
        crun.stats = _stats("BLOCK_M")
        crun.trials = trials
        text = o._attribute_soft_walls(crun)
        return o.store.events, text

    # (a) a candidate whose winner does not spill -- NOT applicable, and still journalled.
    events, text = _run(_ladder([8, 4, 0], [3.0, 2.0, 1.0]))
    assert [t for t, _ in events] == ["RESOURCE_SOFT_WALL"], \
        "a non-applicable candidate must still produce the event: 103 of 153 candidates are here, " \
        "and silence would read as the switch being off"
    p = events[0][1]
    assert p["applicable"] is False and p["n_walls"] == 0
    assert p["reason"], "the event must say WHY nothing was found"
    assert text is None

    # (b) applicable but every curve flat -- also journalled, and distinguishable from (a).
    events, _ = _run(_ladder([8, 8, 8], [3.0, 2.0, 1.0]))
    p2 = events[0][1]
    assert p2["applicable"] is True and p2["n_walls"] == 0
    assert p2["reason"] != p["reason"], \
        "'not applicable' and 'applicable, no wall' must not journal the same reason"

    # (c) a real wall -- journalled with the rows, and rendered.
    events, text3 = _run(_ladder([0, 0, 16, 72], [4.0, 3.5, 3.0, 2.6], limiter="registers"))
    p3 = events[0][1]
    assert p3["n_walls"] == 1 and p3["walls"][0]["param"] == "BLOCK_M"
    assert text3 is not None and "BLOCK_M" in text3


def test_the_soft_wall_is_silent_when_the_switch_is_off():
    """The disable path must emit NOTHING -- an event claiming 0 walls would put a zero in the corpus
    that reads as a failed scan rather than an absent one.
    """
    from kernel_optimizer.config import AppConfig
    from kernel_optimizer.control import orchestrator as orch_mod

    class _Store:
        def __init__(self):
            self.events = []

        def append(self, t, p):
            self.events.append((t, p))

    o = object.__new__(orch_mod.Orchestrator)
    o.cfg = AppConfig()                      # enabled defaults False
    o.store = _Store()
    crun = object.__new__(orch_mod.CandidateRun)
    crun.candidate = type("C", (), {"candidate_id": "cand-x"})()
    crun.stats = _stats("BLOCK_M")
    crun.trials = _ladder([0, 0, 16, 72], [4.0, 3.5, 3.0, 2.6])
    assert o._attribute_soft_walls(crun) is None
    assert o.store.events == []


def test_the_soft_wall_does_not_touch_the_tuning_loop():
    """Item 2's own switches cannot change the trial sequence -- the reason it needs no control arm.

    ORIGINALLY asserted as `"soft_wall" not in _tune`'s source, and S7 (item 3.2) made that assertion
    false without making the invariant false: S7 can consult the SAME criterion inside the loop, but only
    under its own `v3.slope_guide.enabled`, and even then only if `v3.soft_wall.enabled` is also on. So
    the guard is now behavioural -- what must hold is that turning item 2 on, by itself, leaves the
    tuning loop with no slope guide at all.

    Asserted by CALLING the decision rather than reading the source, because a text assertion cannot
    distinguish "the flags are combined correctly" from "the flag is mentioned" -- the trap recorded as
    `source-text-assertions-can-encode-the-bug`.
    """
    from kernel_optimizer.config import AppConfig
    from kernel_optimizer.control import orchestrator as orch_mod
    from kernel_optimizer.models.core import ParamDomain, ParameterSpace

    space = ParameterSpace(space_id="sp-x", candidate_id="cand-x", source_sha="0" * 8,
                           domains=[ParamDomain(name="BLOCK_M", kind="int",
                                                choices=[16, 32, 64, 128])])

    def _guide_for(cfg) -> object | None:
        o = object.__new__(orch_mod.Orchestrator)
        o.cfg = cfg
        return o._make_slope_guide(space)

    # Item 2 fully on, S7 untouched: the tuning loop gets NOTHING.
    cfg = AppConfig()
    cfg.v3.soft_wall.enabled = True
    cfg.v3.soft_wall.in_prompt = True
    assert _guide_for(cfg) is None, \
        "turning the soft wall on must not put anything into the tuning loop, or budget parity breaks"

    # And the converse, which is the part S7 adds: the soft criterion reaches the sampler only when
    # BOTH switches are on. S7 alone must not smuggle an unconfirmed signal into the loop.
    cfg = AppConfig()
    cfg.v3.slope_guide.enabled = True
    cfg.v3.slope_guide.use_soft_wall = True      # ...but v3.soft_wall.enabled stays False
    guide = _guide_for(cfg)
    assert guide is not None and guide.use_soft_wall is False, \
        "the spill wall has no independent probe confirmation, so S7 must not use it while item 2 is off"

    cfg.v3.soft_wall.enabled = True
    assert _guide_for(cfg).use_soft_wall is True


def test_the_prompt_gate_is_separate_from_the_scan_gate():
    """Two switches, not one boolean. `enabled` journals the scan at zero GPU cost and cannot change a
    decision; `in_prompt` changes what the agent sees, which is the thing that needs a control. Merging
    them would make the cheap, safe half unavailable on its own -- the same separation S1b uses.
    """
    text = io.open(SRC / "control" / "orchestrator.py", encoding="utf-8").read()
    tree = ast.parse(text)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_attribute_soft_walls")
    body = ast.get_source_segment(text, fn) or ""
    assert "cfg.enabled" in body, "the scan must be gated on `enabled`"
    # Attribute names from the AST, so this cannot match `max_rows_in_prompt` -- which it did on the
    # first run, as a substring. A guard that fires on an unrelated name gets relaxed until it fires on
    # nothing.
    fn_attrs = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
    assert "in_prompt" not in fn_attrs, \
        "the scan must not read `in_prompt`; delivery is gated at the call site"
    assert "max_rows_in_prompt" in fn_attrs, "the prompt cap is read where the text is rendered"
    stats_fn = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == "_stats_and_analysis")
    stats_body = ast.get_source_segment(text, stats_fn) or ""
    assert "self.cfg.v3.soft_wall.in_prompt" in stats_body


def test_the_soft_text_is_appended_after_the_hard_wall_and_not_merged():
    """One is confirmed by an independent compiler probe, the other is a single observation. Merging
    them into one block would let a reader carry the hard wall's confirmation over to the soft wall's
    rows -- which is exactly the over-reading the "未经独立探针确认" line exists to prevent.
    """
    text = io.open(SRC / "control" / "orchestrator.py", encoding="utf-8").read()
    tree = ast.parse(text)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_stats_and_analysis")
    body = ast.get_source_segment(text, fn) or ""
    i_hard = body.find("wall_attribution.for_prompt")
    i_soft = body.find("_attribute_soft_walls")
    assert i_hard < i_soft, "the hard wall's text must be built first so the soft text appends to it"
    assert 'f"{wall_text}\\n\\n{soft_text}"' in body, \
        "the two texts must be separate blocks, not interleaved rows"


# ---------------------------------------------------------------------------------------------
# the report section
# ---------------------------------------------------------------------------------------------


def _soft_ev(payload: dict) -> dict:
    return {"type": "RESOURCE_SOFT_WALL", "payload": payload}


def _real_payload(cid: str = "cand-x") -> dict:
    """Built by running the SHIPPING scan, so the report is tested against what the orchestrator
    actually writes. A hand-written payload shaped to match the reader would prove only that the reader
    reads itself.
    """
    scan = sw.find_soft_walls(
        _stats("GEMM_NUM_WARPS"),
        _ladder([0, 0, 16, 72], [4.0, 3.5, 3.0, 2.6], knob="GEMM_NUM_WARPS",
                values=[2, 4, 8, 16], limiter="registers"))
    return {"candidate_id": cid, **scan.payload()}


def test_the_report_section_is_absent_when_the_soft_wall_never_ran():
    from kernel_optimizer.reporting.wall_report import soft_wall_lines, wall_lines

    assert soft_wall_lines([]) == []
    assert soft_wall_lines([{"type": "TRIAL_DONE", "payload": {}}]) == []
    assert wall_lines([{"type": "TRIAL_DONE", "payload": {}}]) == []


def test_the_report_renders_the_soft_wall_even_when_2e_is_off():
    """The two mechanisms are independently switched. `wall_lines` returned [] as soon as there were no
    `RESOURCE_WALL_ATTRIBUTED` events, which would have dropped a section that has content -- a new
    reader silently reporting nothing for a run that produced something.
    """
    from kernel_optimizer.reporting.wall_report import wall_lines

    out = "\n".join(wall_lines([_soft_ev(_real_payload())]))
    assert "寄存器溢出墙" in out
    assert "GEMM_NUM_WARPS" in out


def test_the_report_publishes_the_denominators():
    """A count of walls cannot be read alone: 103 of 153 candidates in the local corpus have a
    non-spilling winner, so "not applicable" and "applicable, no wall" must both be visible, and the
    applicability rate has to come from THIS run (43% on one corpus, 23.5% on another, 0 of 8 on L3:48).
    """
    from kernel_optimizer.reporting.wall_report import soft_wall_lines

    hit = _real_payload("cand-hit")
    na = sw.find_soft_walls(_stats("A"), _ladder([8, 4, 0], [3.0, 2.0, 1.0])).payload()
    na["candidate_id"] = "cand-na"
    out = "\n".join(soft_wall_lines([_soft_ev(hit), _soft_ev(na)]))
    assert "候选数 2" in out
    assert "可适用)的 1 个" in out, out
    assert "最优 trial 根本不溢出" in out
    assert "不是阴性结果" in out
    assert "适用率不是常数" in out


def test_the_report_says_every_row_is_unconfirmed():
    from kernel_optimizer.reporting.wall_report import soft_wall_lines

    out = "\n".join(soft_wall_lines([_soft_ev(_real_payload())]))
    assert "未经独立探针确认" in out


def test_the_report_distinguishes_no_wall_from_not_applicable():
    """A candidate that spills at its optimum but whose curves are all flat is "applicable, no wall".
    Rendering that identically to "not applicable" would erase the only state that says the criterion
    ran on real input and legitimately had nothing to say.
    """
    from kernel_optimizer.reporting.wall_report import soft_wall_lines

    flat = sw.find_soft_walls(_stats("A"), _ladder([8, 8, 8], [3.0, 2.0, 1.0])).payload()
    flat["candidate_id"] = "cand-flat"
    assert flat["applicable"] is True and flat["n_walls"] == 0
    out = "\n".join(soft_wall_lines([_soft_ev(flat)]))
    assert "可适用)的 1 个" in out, "an applicable candidate must count toward the applicable total"
    assert "没有找到任何溢出墙" in out


def test_the_report_reads_events_of_either_shape():
    """`report.py` passes event OBJECTS; the offline probes pass dicts. A reader that handles one
    silently returns an empty section for the other, which reads as "found nothing"."""
    from kernel_optimizer.reporting.wall_report import soft_wall_lines

    payload = _real_payload()

    class Ev:
        def __init__(self, t, p):
            self.type, self.payload = t, p

    assert soft_wall_lines([_soft_ev(payload)]) == soft_wall_lines([Ev("RESOURCE_SOFT_WALL",
                                                                      payload)])
