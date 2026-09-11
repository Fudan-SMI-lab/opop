"""A1 wiring: the derived bound must reach the worker, on a fresh run AND on a resume.

The module-level derivation is tested in `test_plausibility_bound.py`. This file tests the two
things that have silently broken in this project before:

  1. **A resume that loses the bound.** `_baseline` returns EARLY on a resumed run, having
     restored the task cost from the log. A `set_plausibility` call placed inside it would then
     cover only the fresh path, and a resumed run would flag against nothing while looking
     identical in the event log. The same shape cost this project `launch_bound`: unreachable
     across 848 trials, unnoticed, because nothing asserted the number arrived.
  2. **A bound that is computed and then not passed to the job.** Recorded as
     `a-reused-measurement-leaves-no-artifact` and `a-correct-reader-does-not-prevent-the-guess`:
     the producing side working is not evidence the consuming side reads it.

Every test here drives the REAL `_install_plausibility` and the REAL job builders.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from kernel_optimizer.evaluation.plausibility import speedup_ceiling
from kernel_optimizer.models.core import Baseline, LatencyStats
from kernel_optimizer.store.run_store import RunStore

# The A800's own numbers, from run-l3-48-20260911-052647's events.jsonl.
A800_COST = {"flop_count": 30_366_760_960, "compulsory_bytes": 1_350_565_888,
             "reference_bytes": 54_672_711_734, "op_count": 130}


class _Cal:
    """A calibration with only the fields the bound reads. Deliberately MINIMAL.

    A full `Calibration` would keep passing if the code started reading some other field, while
    this fails loudly -- the `a-support-field-must-not-damage-what-it-describes` lesson: a minimal
    stub is an asset precisely because it breaks when a dependency changes.
    """

    dram_tbs = 1.6858134468378299
    fp32_tflops = 19.012
    tf32_tflops = 111.465
    fp16_tflops = 226.170
    bf16_tflops = 233.956
    fp32_triton_tflops = 0.0
    tf32_triton_tflops = 0.0
    fp16_triton_tflops = 0.0
    bf16_triton_tflops = 0.0
    l2_bytes = 41_943_040


class _Evaluator:
    """Records what `set_plausibility` was handed, and nothing else."""

    def __init__(self):
        self.installed = "never called"

    def set_plausibility(self, ceiling, derivation: str = ""):
        self.installed = ceiling


def _orch(store, evaluator, *, cal=None, baselines=None, cost=None):
    """A real Orchestrator with only the attributes `_install_plausibility` touches.

    Built with `__new__` so no GPU worker or opencode server is needed, but the METHOD under test
    is the shipped one -- a test that reimplemented the derivation in its own body would pass on
    any orchestrator at all, which is the defect `a-test-that-copies-the-loop-does-not-test-it`
    describes and which two of this project's own tests have had.
    """
    from kernel_optimizer.control.orchestrator import Orchestrator

    orch = Orchestrator.__new__(Orchestrator)
    orch.store = store
    orch.calibration = cal
    orch.baselines = baselines if baselines is not None else []
    orch.task_cost = cost

    class Deps:
        pass

    deps = Deps()
    deps.evaluator = evaluator
    orch.deps = deps
    return orch


def _eager(ms: float, kind: str = "eager") -> Baseline:
    return Baseline(kind=kind, latency_ms=LatencyStats(
        mean=ms, std=0.1, min=ms - 0.1, max=ms + 0.1, median=ms, n_samples=100))


def _cost():
    from kernel_optimizer.evaluation.task_cost import cost_from_worker

    return cost_from_worker({"task_cost": A800_COST})


def test_the_bound_is_derived_and_installed_on_the_evaluator():
    with tempfile.TemporaryDirectory() as td:
        store = RunStore.create(Path(td) / "runs", "run-pb", {"task": "level3:48"})
        ev = _Evaluator()
        _orch(store, ev, cal=_Cal(), baselines=[_eager(14.0)], cost=_cost())._install_plausibility()

        assert ev.installed != "never called", "set_plausibility was never called"
        assert ev.installed is not None, "the bound was derivable but None was installed"
        assert abs(ev.installed.ceiling_x - 17.48) < 0.05, ev.installed.ceiling_x
        # The measured 14.29x must be legal under it.
        assert ev.installed.threshold_x > 14.294


def test_the_bound_is_journalled_so_the_report_can_justify_a_flag():
    with tempfile.TemporaryDirectory() as td:
        store = RunStore.create(Path(td) / "runs", "run-pb", {"task": "level3:48"})
        _orch(store, _Evaluator(), cal=_Cal(), baselines=[_eager(14.0)],
              cost=_cost())._install_plausibility()

        evs = [e for e in store.replay().events if e.type == "PLAUSIBILITY_BOUND"]
        assert len(evs) == 1, [e.type for e in store.replay().events]
        assert evs[0].payload["derived"] is True
        bound = evs[0].payload["bound"]
        assert bound and abs(bound["ceiling_x"] - 17.48) < 0.05
        assert "1.6858" in bound["derivation"], (
            "the derivation must be journalled, or a flagged report cannot say what number it "
            "was flagged against -- which is what made the 10x constant cost three manual "
            "re-verifications")


def test_no_calibration_installs_None_and_says_why_rather_than_staying_silent():
    """The refusal path, end to end. A box with no calibration flags nothing -- and the log must
    say so, because an absent event reads exactly like a check that passed."""
    with tempfile.TemporaryDirectory() as td:
        store = RunStore.create(Path(td) / "runs", "run-pb", {"task": "level3:48"})
        ev = _Evaluator()
        _orch(store, ev, cal=None, baselines=[_eager(14.0)],
              cost=_cost())._install_plausibility()

        assert ev.installed is None, (
            "with no measured ceilings the evaluator must be given NO bound, not a constant")
        payload = [e.payload for e in store.replay().events
                   if e.type == "PLAUSIBILITY_BOUND"][-1]
        assert payload["derived"] is False
        assert payload["reason"], "a refusal with no reason cannot be acted on"


def test_no_baseline_yields_no_bound():
    with tempfile.TemporaryDirectory() as td:
        store = RunStore.create(Path(td) / "runs", "run-pb", {"task": "level3:48"})
        ev = _Evaluator()
        _orch(store, ev, cal=_Cal(), baselines=[], cost=_cost())._install_plausibility()
        assert ev.installed is None
        payload = [e.payload for e in store.replay().events
                   if e.type == "PLAUSIBILITY_BOUND"][-1]
        assert payload["reason"] == "no eager baseline"


def test_torch_compile_is_not_used_as_the_reference():
    """`torch.compile` is a competitor to beat, not the semantic reference.

    TWO SCENARIOS, and only the second actually tests the filter. When an eager baseline is
    present it is the SLOWER one, so `max()` picks it whether or not compile baselines were
    filtered out -- the first assertion below passes even with the filter deleted, which the
    revert check proved (`reference_is_torch_compile_not_eager` was NOT CAUGHT by an earlier
    version of this test). The filter earns its place in the case where there is no eager
    baseline at all: it must then REFUSE to derive a bound rather than quietly substitute
    torch.compile, which would shrink L3:48's ceiling from 17.48x to 10.7x and re-flag the very
    verified-correct winner this change exists to stop flagging.
    """
    with tempfile.TemporaryDirectory() as td:
        store = RunStore.create(Path(td) / "runs", "run-pb", {"task": "level3:48"})
        ev = _Evaluator()
        _orch(store, ev, cal=_Cal(),
              baselines=[_eager(14.0), _eager(8.59, kind="torch_compile"),
                         _eager(8.07, kind="torch_compile_tf32")],
              cost=_cost())._install_plausibility()
        # 14.0 / 0.8011, not 8.59 / 0.8011.
        assert abs(ev.installed.ceiling_x - 17.48) < 0.05, (
            "the ceiling was computed from a torch.compile baseline: %r" % ev.installed.ceiling_x)

    # The scenario that actually exercises the filter: compile baselines only.
    with tempfile.TemporaryDirectory() as td:
        store = RunStore.create(Path(td) / "runs", "run-pb2", {"task": "level3:48"})
        ev = _Evaluator()
        _orch(store, ev, cal=_Cal(),
              baselines=[_eager(8.59, kind="torch_compile"),
                         _eager(8.07, kind="torch_compile_tf32")],
              cost=_cost())._install_plausibility()
        assert ev.installed is None, (
            "with no eager baseline the bound must be REFUSED, not derived from torch.compile: "
            "a compile-based ceiling is smaller than the physical one by the compile speedup "
            "itself, which turns the flag into exactly the false-positive machine it replaced")
        payload = [e.payload for e in store.replay().events
                   if e.type == "PLAUSIBILITY_BOUND"][-1]
        assert payload["reason"] == "no eager baseline"


def test_the_slowest_eager_baseline_is_the_numerator():
    """Every uncertainty must WIDEN the bound. Both eager arms are legitimate references, and the
    slower one yields the larger ceiling, i.e. the quieter flag."""
    with tempfile.TemporaryDirectory() as td:
        store = RunStore.create(Path(td) / "runs", "run-pb", {"task": "level3:48"})
        ev = _Evaluator()
        _orch(store, ev, cal=_Cal(),
              baselines=[_eager(13.5, kind="eager_tf32"), _eager(14.0)],
              cost=_cost())._install_plausibility()
        assert abs(ev.installed.reference_ms - 14.0) < 1e-9, ev.installed.reference_ms


def test_the_bound_survives_a_resume():
    """The defect this method's placement exists to avoid.

    `_install_plausibility` is called AFTER `_baseline`, in `_run`, precisely so the resumed path
    -- which returns early from `_baseline` having restored the cost and baselines from the log --
    installs the same bound. Simulated here by building a second orchestrator from the same store
    with the state a resume would have restored.
    """
    with tempfile.TemporaryDirectory() as td:
        store = RunStore.create(Path(td) / "runs", "run-pb", {"task": "level3:48"})
        first = _Evaluator()
        _orch(store, first, cal=_Cal(), baselines=[_eager(14.0),
                                                  _eager(8.59, kind="torch_compile")],
              cost=_cost())._install_plausibility()

        resumed = _Evaluator()
        _orch(store, resumed, cal=_Cal(), baselines=[_eager(14.0),
                                                     _eager(8.59, kind="torch_compile")],
              cost=_cost())._install_plausibility()

        assert resumed.installed is not None, "a resumed run flagged against nothing"
        assert abs(resumed.installed.threshold_x - first.installed.threshold_x) < 1e-9, (
            "the two halves of a resumed run flagged against different thresholds")


def test_install_is_called_after_baseline_in_the_real_run_sequence():
    """And the ORDER is asserted, not assumed.

    Inside `_baseline` the call would cover only the fresh path; before `_calibrate` it would see
    no ceilings at all. Reading the source is the only way to check a call ORDER, and here the
    thing being asserted IS a textual property of the sequence -- unlike a behavioural claim,
    which this file drives through the real method everywhere else.
    """
    import ast

    src = Path("src/kernel_optimizer/control/orchestrator.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "_run")
    calls = [n.func.attr for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr in ("_calibrate", "_baseline", "_install_plausibility",
                                 "_generate_seeds")]
    assert calls[:4] == ["_calibrate", "_baseline", "_install_plausibility",
                         "_generate_seeds"], calls


def test_the_evaluator_puts_the_bound_into_every_timed_job():
    """The consuming half. A bound that is installed but never reaches a job flags nothing."""
    from kernel_optimizer.evaluation.correctness import CorrectnessEvaluator
    from kernel_optimizer.gpu.jobs import make_eval_job, make_relaxed_correctness_job

    ev = CorrectnessEvaluator.__new__(CorrectnessEvaluator)
    ev._plausibility = None
    assert ev._plausibility_fields() is None, "no bound must mean no job field at all"

    ev.set_plausibility(speedup_ceiling(
        compulsory_bytes=A800_COST["compulsory_bytes"], flop_count=A800_COST["flop_count"],
        reference_ms=14.0, dram_tbs=_Cal.dram_tbs, peak_tflops=_Cal.bf16_tflops,
        l2_bytes=_Cal.l2_bytes))
    fields = ev._plausibility_fields()
    assert fields and abs(fields["plausibility_threshold_x"] - 26.21) < 0.05, fields
    assert abs(fields["plausibility_reference_ms"] - 14.0) < 1e-9
    assert "1.6858" in fields["plausibility_derivation"]

    # And both job builders must carry them through.
    for job in (
        make_eval_job("ref.py", "k.py", measure_performance=True, num_correct_trials=5,
                      num_perf_trials=100, timing_method="cuda_event", backend="triton",
                      precision="fp32", seed=0, build_dir=None, collect_kernel_metadata=True,
                      plausibility=fields),
        make_relaxed_correctness_job("ref.py", "k.py", num_correct_trials=5, backend="triton",
                                     precision="fp32", seed=0, collect_kernel_metadata=True,
                                     relaxed_elem_tol=0.01, relaxed_pass_frac=0.99,
                                     cosine_min=0.99985, plausibility=fields),
    ):
        assert job["plausibility_threshold_x"] == fields["plausibility_threshold_x"], job
        assert job["plausibility_reference_ms"] == 14.0

    # A job built WITHOUT a bound must carry no threshold key: the worker then records
    # `plausibility_checked: False` instead of comparing against a constant nobody derived.
    bare = make_eval_job("ref.py", "k.py", measure_performance=True, num_correct_trials=5,
                         num_perf_trials=100, timing_method="cuda_event", backend="triton",
                         precision="fp32", seed=0, build_dir=None,
                         collect_kernel_metadata=True, plausibility=None)
    assert "plausibility_threshold_x" not in bare
    assert "excessive_speedup_threshold" not in bare, (
        "the legacy constant must not be re-added by default; it is an explicit override only")


def test_a_mutating_consumer_cannot_corrupt_the_installed_bound():
    """`_plausibility_fields` hands out a copy: a job builder that mutated its input would
    otherwise change the threshold every later job is flagged against."""
    from kernel_optimizer.evaluation.correctness import CorrectnessEvaluator

    ev = CorrectnessEvaluator.__new__(CorrectnessEvaluator)
    ev._plausibility = None
    ev.set_plausibility(speedup_ceiling(
        compulsory_bytes=A800_COST["compulsory_bytes"], flop_count=A800_COST["flop_count"],
        reference_ms=14.0, dram_tbs=_Cal.dram_tbs, peak_tflops=_Cal.bf16_tflops))
    first = ev._plausibility_fields()
    first["plausibility_threshold_x"] = 1.0
    assert ev._plausibility_fields()["plausibility_threshold_x"] != 1.0


def test_set_plausibility_with_None_clears_rather_than_keeping_a_stale_bound():
    """A second task in one process must not inherit the first task's ceiling."""
    from kernel_optimizer.evaluation.correctness import CorrectnessEvaluator

    ev = CorrectnessEvaluator.__new__(CorrectnessEvaluator)
    ev._plausibility = None
    ev.set_plausibility(speedup_ceiling(
        compulsory_bytes=A800_COST["compulsory_bytes"], flop_count=A800_COST["flop_count"],
        reference_ms=14.0, dram_tbs=_Cal.dram_tbs, peak_tflops=_Cal.bf16_tflops))
    assert ev._plausibility_fields() is not None
    ev.set_plausibility(None)
    assert ev._plausibility_fields() is None


def test_an_evaluator_without_the_setter_does_not_end_the_run():
    """A diagnostic must never be fatal. A collaborator predating this feature (a stub in another
    test, a replayed double) simply gets no bound."""
    with tempfile.TemporaryDirectory() as td:
        store = RunStore.create(Path(td) / "runs", "run-pb", {"task": "level3:48"})

        class Old:
            pass

        _orch(store, Old(), cal=_Cal(), baselines=[_eager(14.0)],
              cost=_cost())._install_plausibility()
        # No exception, and no event claiming a bound was applied.
        assert not [e for e in store.replay().events if e.type == "PLAUSIBILITY_BOUND"]
