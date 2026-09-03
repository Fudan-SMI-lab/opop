"""Tests for the harness improvements: Triton lint (C), novelty slot accounting (E),
repair failure-class guidance (F), and the relaxed-correctness slack gate (A)."""

import importlib.util

import pytest

from kernel_optimizer.agents.modules import _repair_guidance
from kernel_optimizer.control.families import FamilyManager
from kernel_optimizer.models.core import Candidate, Family
from kernel_optimizer.paramspace.triton_lint import lint_triton_source


# --- C: triton lint -----------------------------------------------------------

def test_lint_flags_next_power_of_2_in_device_code():
    src = """
import triton
import triton.language as tl
@triton.jit
def k(x_ptr, D: tl.constexpr):
    d = tl.arange(0, tl.next_power_of_2(D))
    return d
"""
    hard, _warn = lint_triton_source(src)
    assert len(hard) == 1
    assert "next_power_of_2" in hard[0]


def test_lint_allows_host_side_next_power_of_2():
    src = """
import triton
import triton.language as tl
def host(D):
    return triton.next_power_of_2(D)
@triton.jit
def k(x_ptr, DP: tl.constexpr):
    d = tl.arange(0, DP)
    return d
"""
    hard, _warn = lint_triton_source(src)
    assert hard == []


def test_lint_noop_on_non_triton():
    assert lint_triton_source("x = 1\n") == ([], [])


def test_lint_reports_syntax_error():
    hard, _warn = lint_triton_source("def bad(:\n")
    assert hard and "does not parse" in hard[0]


# --- E: novelty slot accounting ----------------------------------------------

def _seed(fm: FamilyManager, fid: str, dropped: bool) -> None:
    cid = f"cand-{fid}"
    cand = Candidate(candidate_id=cid, family_id=fid, origin="seed", backend="triton",
                     source_sha=fid, structural_signature=fid)
    fm.candidates[cid] = cand
    fm._sources[cid] = f"# {fid}\nx = 1\n"
    status = "frozen_budget" if dropped else "active"
    fm.families[fid] = Family(family_id=fid, anchor_candidate_id=cid, member_ids=[cid],
                              status=status)


def test_dropped_families_do_not_consume_novelty_budget():
    fm = FamilyManager(max_families_total=3, max_families_total_hard=6)
    # Three dead families (all seeds dropped, none has a best).
    for i in range(3):
        _seed(fm, f"fam-dead{i}", dropped=True)
    assert fm.productive_family_count() == 0
    # A novel seed must be accepted despite 3 families already existing.
    result = fm.accept_novel_seed("# distinct\ny = 2\n", "triton", "novel approach", "differs")
    assert isinstance(result, Candidate)


def test_productive_families_still_enforce_budget():
    fm = FamilyManager(max_families_total=3, max_families_total_hard=6)
    for i in range(3):
        _seed(fm, f"fam-live{i}", dropped=False)  # active -> productive
    assert fm.productive_family_count() == 3
    result = fm.accept_novel_seed("# distinct\ny = 2\n", "triton", "novel", "differs")
    assert not isinstance(result, Candidate)
    assert result.reason == "family_budget"


def test_hard_cap_bounds_total_families():
    fm = FamilyManager(max_families_total=3, max_families_total_hard=4)
    for i in range(4):
        _seed(fm, f"fam-dead{i}", dropped=True)  # 4 dead, 0 productive, but 4 total
    result = fm.accept_novel_seed("# distinct\nz = 3\n", "triton", "novel", "differs")
    assert not isinstance(result, Candidate)
    assert result.reason == "family_budget_hard"


# --- F: repair failure-class guidance ----------------------------------------

def test_repair_guidance_routes_by_failure_kind():
    assert "NUMERICAL" in _repair_guidance("correctness_mismatch")
    assert "COMPILE" in _repair_guidance("compile_error")
    assert "COMPILE" in _repair_guidance("runtime_error")
    assert "OUT-OF-MEMORY" in _repair_guidance("oom")
    # unknown kind still returns a usable generic hint
    assert _repair_guidance("weird") and "root cause" in _repair_guidance("weird")


# --- A: relaxed slack gate (worker-side helper, imported directly) -----------

def _load_worker_relaxed_close():
    """worker_main imports torch lazily; load the module and grab _relaxed_close.
    Skip if torch is unavailable on the host (it is not, on the Windows side)."""
    spec = importlib.util.find_spec("kernel_optimizer.gpu.worker_main")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._relaxed_close


def test_relaxed_close_semantics():
    torch = pytest.importorskip("torch")
    _relaxed_close = _load_worker_relaxed_close()
    ref = torch.ones(1000)
    # exact match passes
    assert _relaxed_close(ref, ref.clone(), 0.01, 0.99, 0.99985)
    # 0.5% of elements badly wrong -> still >99% within tol -> passes
    got = ref.clone(); got[:5] = 5.0
    assert _relaxed_close(ref, got, 0.01, 0.99, 0.99985)
    # 5% of elements wrong -> below 99% frac -> fails
    got2 = ref.clone(); got2[:50] = 5.0
    assert not _relaxed_close(ref, got2, 0.01, 0.99, 0.99985)
    # shape mismatch never passes
    assert not _relaxed_close(ref, torch.ones(999), 0.01, 0.99, 0.99985)


# --- A/decision-2: dual-precision baselines must construct as valid Baselines ---

def test_measure_baseline_dual_precision_records_valid_rows():
    """Regression: relaxed mode records eager/torch_compile at both ieee and tf32.
    The ieee rows keep the exact 'eager'/'torch_compile' kinds (speedup denominator);
    tf32 rows are suffixed. All must be valid Baseline objects (note is a str, kind
    accepts the suffixed form)."""
    from kernel_optimizer.config import EvalConfig
    from kernel_optimizer.evaluation.benchmark import Benchmarker
    from kernel_optimizer.models.core import Baseline, TaskSpec

    class FakeWorker:
        def run_job(self, job, timeout, tag, lock_mode="exclusive"):
            return {"ok": True,
                    "latency_ms": {"mean": 10.0, "std": 0.1, "min": 9.9, "max": 10.2, "n": 50}}

    cfg = EvalConfig(correctness_mode="dual_witness_relaxed", perf_trials=50)
    bench = Benchmarker(FakeWorker(), evaluator=None, cfg=cfg)
    task = TaskSpec(level=1, problem_id=19, name="relu", ref_path="x", ref_src_sha="deadbeef")
    baselines = bench.measure_baseline(task)

    kinds = {b.kind for b in baselines}
    assert {"eager", "torch_compile", "eager_tf32", "torch_compile_tf32"} <= kinds
    assert all(isinstance(b, Baseline) and isinstance(b.note, str) for b in baselines)
    # ieee rows (the speedup denominators) carry no note; tf32 rows are annotated.
    ieee = next(b for b in baselines if b.kind == "eager")
    tf32 = next(b for b in baselines if b.kind == "eager_tf32")
    assert ieee.note == "" and "tf32" in tf32.note


def test_measure_baseline_strict_mode_single_precision():
    """Strict mode keeps the original two-baseline behavior (ieee only)."""
    from kernel_optimizer.config import EvalConfig
    from kernel_optimizer.evaluation.benchmark import Benchmarker
    from kernel_optimizer.models.core import TaskSpec

    class FakeWorker:
        def run_job(self, job, timeout, tag, lock_mode="exclusive"):
            return {"ok": True,
                    "latency_ms": {"mean": 10.0, "std": 0.1, "min": 9.9, "max": 10.2, "n": 50}}

    cfg = EvalConfig(correctness_mode="strict", perf_trials=50)
    bench = Benchmarker(FakeWorker(), evaluator=None, cfg=cfg)
    task = TaskSpec(level=1, problem_id=19, name="relu", ref_path="x", ref_src_sha="deadbeef")
    baselines = bench.measure_baseline(task)
    assert {b.kind for b in baselines} == {"eager", "torch_compile"}


# --- H3: candidate precision detection + honest same-precision verdict --------

def test_detect_precision_from_params_knob():
    from kernel_optimizer.control.orchestrator import _detect_candidate_precision
    from kernel_optimizer.models.core import ParamSet

    src = 'acc = tl.dot(a, b, input_precision=PARAMS["DOT_PRECISION"])\n'
    assert _detect_candidate_precision(src, ParamSet(values={"DOT_PRECISION": "tf32"})) == "tf32"
    assert _detect_candidate_precision(
        src, ParamSet(values={"DOT_PRECISION": "ieee"})) == "ieee_fp32"


def test_detect_precision_from_source_literal():
    from kernel_optimizer.control.orchestrator import _detect_candidate_precision
    from kernel_optimizer.models.core import ParamSet

    empty = ParamSet(values={"BLOCK_M": 64})
    assert _detect_candidate_precision(
        'x = tl.dot(a, b, input_precision="ieee")', empty) == "ieee_fp32"
    assert _detect_candidate_precision(
        "x = tl.dot(a, b, input_precision='tf32')", empty) == "tf32"
    # bare tl.dot on fp32 inputs -> tf32 default path on this GPU generation
    assert _detect_candidate_precision("x = tl.dot(a, b)", empty) == "tf32"
    # no dot at all -> unknown
    assert _detect_candidate_precision("y = x + 1", empty) == "unknown"


def test_honest_verdict_compares_same_precision():
    from kernel_optimizer.control.orchestrator import _honest_verdict

    speedups = {
        "eager": 1.5, "eager_tf32": 0.9,
        "torch_compile": 1.2, "torch_compile_tf32": 0.6,
    }
    # tf32 candidate must be judged against torch_compile_tf32 (the honest rival)
    v_tf32 = _honest_verdict("tf32", speedups)
    assert v_tf32["compared_against"] == "torch_compile_tf32"
    assert v_tf32["same_precision_speedup"] == 0.6
    assert v_tf32["beats_same_precision_baseline"] is False
    # ieee candidate compares against the ieee torch.compile
    v_ieee = _honest_verdict("ieee_fp32", speedups)
    assert v_ieee["compared_against"] == "torch_compile"
    assert v_ieee["same_precision_speedup"] == 1.2
    assert v_ieee["beats_same_precision_baseline"] is True


def test_honest_verdict_falls_back_when_tf32_baseline_absent():
    from kernel_optimizer.control.orchestrator import _honest_verdict

    # strict mode: only ieee baselines recorded. A tf32 candidate falls back to
    # the untagged torch_compile rather than reporting nothing.
    speedups = {"eager": 1.5, "torch_compile": 1.2}
    v = _honest_verdict("tf32", speedups)
    assert v["compared_against"] == "torch_compile"
    assert v["same_precision_speedup"] == 1.2


# --- J: reference eval-semantics doc (train/eval mode injected as a task fact) ----

def test_eval_semantics_doc_train_mode_warns_batchnorm():
    from kernel_optimizer.agents.modules import _eval_semantics_doc
    doc = _eval_semantics_doc({
        "training": True,
        "norm_layers": [{"type": "BatchNorm2d", "training": True,
                         "has_running_stats": True, "track_running_stats": True,
                         "momentum": 0.1}],
    })
    assert "TRAIN mode" in doc
    assert "CURRENT BATCH" in doc
    assert "BatchNorm2d" in doc


def test_eval_semantics_doc_eval_mode():
    from kernel_optimizer.agents.modules import _eval_semantics_doc
    doc = _eval_semantics_doc({"training": False, "norm_layers": []})
    assert "EVAL mode" in doc


def test_eval_semantics_doc_missing_degrades_gracefully():
    from kernel_optimizer.agents.modules import _eval_semantics_doc
    # No probe result -> neutral, non-forcing note (does not assert train or eval).
    doc = _eval_semantics_doc(None)
    assert "Not probed" in doc
    doc_empty = _eval_semantics_doc({})
    assert "Not probed" in doc_empty


def test_probe_semantics_job_shape():
    from kernel_optimizer.gpu.jobs import make_probe_semantics_job
    job = make_probe_semantics_job("/path/to/ref.py")
    assert job["job_type"] == "probe_semantics"
    assert job["ref_src_path"] == "/path/to/ref.py"


# --- L: dtype-knob consistency lint warning (non-blocking) --------------------

def test_lint_warns_hardcoded_fp16_without_dtype_knob():
    src = """
import triton
import triton.language as tl
PARAMS = {"BLOCK_M": 64, "BLOCK_N": 64}
@triton.jit
def k(a_ptr, b_ptr, BLOCK_M: tl.constexpr):
    a = tl.load(a_ptr).to(tl.float16)
    b = tl.load(b_ptr).to(tl.float16)
    acc = tl.dot(a, b)
    return acc
"""
    hard, warns = lint_triton_source(src)
    assert hard == []                      # never a hard error (non-blocking)
    assert any("dtype" in w.lower() and "knob" in w.lower() for w in warns)


def test_lint_no_warn_when_dtype_knob_present():
    # name-agnostic: COMPUTE_DTYPE value "fp16" is recognized even though the key
    # is not "DOT_PRECISION".
    src = """
import triton
import triton.language as tl
PARAMS = {"BLOCK_M": 64, "COMPUTE_DTYPE": "fp16"}
@triton.jit
def k(a_ptr, b_ptr, BLOCK_M: tl.constexpr):
    a = tl.load(a_ptr).to(tl.float16)
    acc = tl.dot(a, a)
    return acc
"""
    hard, warns = lint_triton_source(src)
    assert hard == []
    assert not any("dtype" in w.lower() and "knob" in w.lower() for w in warns)


def test_lint_no_dtype_warn_for_plain_fp32_kernel():
    src = """
import triton
import triton.language as tl
PARAMS = {"BLOCK_M": 64}
@triton.jit
def k(a_ptr, BLOCK_M: tl.constexpr):
    a = tl.load(a_ptr)
    return a
"""
    hard, warns = lint_triton_source(src)
    assert hard == []
    assert warns == []


def test_detect_precision_fp16_from_compute_dtype_knob():
    from kernel_optimizer.control.orchestrator import _detect_candidate_precision
    from kernel_optimizer.models.core import ParamSet
    src = 'x = a.to(tl.float16)\n'
    assert _detect_candidate_precision(
        src, ParamSet(values={"COMPUTE_DTYPE": "fp16"})) == "fp16"
    assert _detect_candidate_precision(
        src, ParamSet(values={"COMPUTE_DTYPE": "bf16"})) == "bf16"


# --- K: boundary + idle-resource -> space expansion decision ------------------

def _param_stat(name, at_boundary, direction):
    from kernel_optimizer.models.reports import ParamStat
    return ParamStat(name=name, best_value=128, at_boundary=at_boundary,
                     boundary_direction=direction, effect_pct=5.0,
                     latency_by_value={}, failure_rate_by_value={})


def _stats(param_stats, regs_frac=None, shared_frac=None):
    from kernel_optimizer.models.reports import ResourceSnapshot, TuningStats
    res = None
    if regs_frac is not None or shared_frac is not None:
        res = ResourceSnapshot(n_regs=None, regs_frac_of_limit=regs_frac,
                               shared_bytes=None, shared_frac_of_limit=shared_frac,
                               n_spills=0)
    return TuningStats(candidate_id="c", space_id="s", n_complete=10, n_fail=0,
                       best=None, param_stats=param_stats, resource_at_best=res,
                       failure_clusters=[])


def test_expand_when_boundary_and_idle_resource():
    from kernel_optimizer.control.orchestrator import boundary_knobs_to_expand
    # BLOCK_M at max edge, shared only 40% used -> expandable
    stats = _stats([_param_stat("BLOCK_M", True, "max"),
                    _param_stat("BLOCK_N", False, None)],
                   regs_frac=1.0, shared_frac=0.4)
    out = boundary_knobs_to_expand(stats, idle_frac=0.8)
    assert out == [{"name": "BLOCK_M", "direction": "max"}]


def test_no_expand_when_all_resources_saturated():
    from kernel_optimizer.control.orchestrator import boundary_knobs_to_expand
    # boundary knob exists but every resource is saturated -> defer to rewrite
    stats = _stats([_param_stat("BLOCK_M", True, "max")],
                   regs_frac=1.0, shared_frac=0.98)
    assert boundary_knobs_to_expand(stats, idle_frac=0.8) == []


def test_no_expand_when_no_boundary_knob():
    from kernel_optimizer.control.orchestrator import boundary_knobs_to_expand
    stats = _stats([_param_stat("BLOCK_M", False, None)],
                   regs_frac=0.5, shared_frac=0.4)
    assert boundary_knobs_to_expand(stats, idle_frac=0.8) == []
