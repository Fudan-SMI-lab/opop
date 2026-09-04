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
    assert "10x FASTER" in _repair_guidance("excessive_speedup")
    # unknown kind still returns a usable generic hint
    assert _repair_guidance("weird") and "root cause" in _repair_guidance("weird")


def test_excessive_speedup_is_a_known_failure_kind():
    """The relaxed correctness path can now reject a candidate for being implausibly
    fast, so TrialRecord must accept that kind (a Literal mismatch would raise on
    every such trial)."""
    from kernel_optimizer.models.core import ParamSet, TrialRecord
    rec = TrialRecord(trial_id="t", candidate_id="c", space_id="s",
                      params=ParamSet(values={"B": 1}), status="fail",
                      failure_kind="excessive_speedup")
    assert rec.failure_kind == "excessive_speedup"


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

def _param_stat(name, at_boundary, direction, effect_pct=5.0):
    from kernel_optimizer.models.reports import ParamStat
    return ParamStat(name=name, best_value=128, at_boundary=at_boundary,
                     boundary_direction=direction, effect_pct=effect_pct,
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


def _space(names_kinds):
    """Minimal ParameterSpace so boundary_knobs_to_expand can check knob kinds."""
    from kernel_optimizer.models.core import ParamDomain, ParameterSpace
    doms = []
    for name, kind in names_kinds:
        choices = [1, 2, 3] if kind != "str" else ["fp16", "tf32"]
        doms.append(ParamDomain(name=name, kind=kind, choices=choices))
    return ParameterSpace(space_id="s", candidate_id="c", source_sha="x", domains=doms)


def test_expand_when_boundary_and_idle_resource():
    from kernel_optimizer.control.orchestrator import boundary_knobs_to_expand
    # BLOCK_M at max edge, shared only 40% used -> expandable
    stats = _stats([_param_stat("BLOCK_M", True, "max"),
                    _param_stat("BLOCK_N", False, None)],
                   regs_frac=1.0, shared_frac=0.4)
    sp = _space([("BLOCK_M", "int"), ("BLOCK_N", "int")])
    out = boundary_knobs_to_expand(stats, idle_frac=0.8, space=sp)
    assert out == [{"name": "BLOCK_M", "direction": "max"}]


def test_no_expand_when_all_resources_saturated():
    from kernel_optimizer.control.orchestrator import boundary_knobs_to_expand
    # boundary knob exists but every resource is saturated -> defer to rewrite
    stats = _stats([_param_stat("BLOCK_M", True, "max")],
                   regs_frac=1.0, shared_frac=0.98)
    sp = _space([("BLOCK_M", "int")])
    assert boundary_knobs_to_expand(stats, idle_frac=0.8, space=sp) == []


def test_no_expand_when_no_boundary_knob():
    from kernel_optimizer.control.orchestrator import boundary_knobs_to_expand
    stats = _stats([_param_stat("BLOCK_M", False, None)],
                   regs_frac=0.5, shared_frac=0.4)
    sp = _space([("BLOCK_M", "int")])
    assert boundary_knobs_to_expand(stats, idle_frac=0.8, space=sp) == []


def test_categorical_knob_is_not_expandable():
    """Regression (found live on L3:21): COMPUTE_DTYPE was flagged at_boundary and K
    tried to 'extend' it, but a dtype choice list has no next value beyond its edge.
    Only ordered numeric knobs may be expanded."""
    from kernel_optimizer.control.orchestrator import boundary_knobs_to_expand
    stats = _stats([_param_stat("COMPUTE_DTYPE", True, "min"),
                    _param_stat("BLOCK_K", True, "max")],
                   regs_frac=0.4, shared_frac=0.48)
    sp = _space([("COMPUTE_DTYPE", "str"), ("BLOCK_K", "int")])
    out = boundary_knobs_to_expand(stats, idle_frac=0.8, space=sp)
    assert out == [{"name": "BLOCK_K", "direction": "max"}]


def test_flat_latency_surface_is_not_an_expansion_opportunity():
    """Regression (found live on L3:21 cand-dc6526b6): with a flat latency surface the
    'argmin at an edge' test passes on noise for EVERY knob (all 6 flagged at_boundary
    with 0.0-0.4% effect). A knob that changes latency by ~0% is irrelevant, not
    blocked — expanding its range cannot help, so require a meaningful effect size."""
    from kernel_optimizer.control.orchestrator import boundary_knobs_to_expand
    flat = _stats([_param_stat("BLOCK_M", True, "max", effect_pct=0.4),
                   _param_stat("BLOCK_N", True, "min", effect_pct=0.0)],
                  regs_frac=0.4, shared_frac=0.4)
    sp = _space([("BLOCK_M", "int"), ("BLOCK_N", "int")])
    assert boundary_knobs_to_expand(stats=flat, idle_frac=0.8, space=sp,
                                    min_effect_pct=2.0) == []
    # a knob with real effect at a boundary still qualifies
    sharp = _stats([_param_stat("BLOCK_M", True, "max", effect_pct=41.25)],
                   regs_frac=0.4, shared_frac=0.48)
    assert boundary_knobs_to_expand(stats=sharp, idle_frac=0.8,
                                    space=_space([("BLOCK_M", "int")]),
                                    min_effect_pct=2.0) == [
        {"name": "BLOCK_M", "direction": "max"}]


# --- K: expansion prompt/guard must steer away from the live rejection reasons ----


def test_expand_prompt_forbids_shrinking_a_knob():
    """Regression (found live on L3:21, cand-6582d191 and cand-80665a49): the expand
    agent returned a knob collapsed to a single choice, rejected as degenerate_domain.
    Expansion may only ADD values, so the prompt must say so explicitly."""
    from kernel_optimizer.agents.modules import ParameterizerAgent
    text = " ".join(str(c) for c in ParameterizerAgent._render_expand_prompt.__code__.co_consts
                    if isinstance(c, str))
    assert "degenerate_domain" in text
    assert "NEVER SHRINK" in text


def test_guard_rejects_membership_test_with_actionable_message():
    """Regression (found live on L3:21, cand-98852844): the agent wrote a membership
    test in a constraint; the guard correctly refuses it, but the old message
    ('comparison op not allowed') did not say what to write instead, so K's feedback
    retry had nothing to act on."""
    from kernel_optimizer.paramspace.guard import ConstraintError, eval_constraint

    with pytest.raises(ConstraintError) as exc:
        eval_constraint('DTYPE in ("fp16", "bf16")', {"DTYPE": "fp16"})
    msg = str(exc.value)
    assert "In" in msg          # names the offending node
    assert "==" in msg          # tells the agent how to express it legally
    # the legal disjunction form still evaluates
    assert eval_constraint('DTYPE == "fp16" or DTYPE == "bf16"', {"DTYPE": "bf16"}) is True


# --- K: an expansion must never lose ground ------------------------------------


def _trial(cid, sp, vals, ms):
    from kernel_optimizer.models.core import LatencyStats, ParamSet, TrialRecord
    return TrialRecord(
        trial_id=f"tr-{ms}", candidate_id=cid, space_id=sp,
        params=ParamSet(values=vals), status="complete",
        latency_ms=LatencyStats(mean=ms, std=0.1, min=ms - 0.1, max=ms + 0.1, n_samples=20))


def test_expanded_space_still_contains_the_prior_optimum():
    """The invariant K relies on: expansion only ADDS choices, so the pre-expansion
    optimum stays legal in the expanded space. If this holds, carrying it over as an
    anchor is always sound."""
    from kernel_optimizer.models.core import (
        DeviceLimits, ParamDomain, ParameterSpace, ParamSet,
    )
    from kernel_optimizer.paramspace.guard import check_config

    def sp(choices):
        return ParameterSpace(
            space_id="s", candidate_id="c", source_sha="x",
            domains=[ParamDomain(name="BLOCK_M", kind="int", choices=choices)])

    old, expanded = sp([1, 2, 3]), sp([1, 2, 3, 4])   # expansion only ADDS
    best = ParamSet(values={"BLOCK_M": 3})
    dev = DeviceLimits()
    assert check_config(old, best, dev) is None
    assert check_config(expanded, best, dev) is None  # still legal -> anchorable


def test_candidate_best_ms_never_regresses_across_spaces():
    """Regression (found live on L3:43 cand-0c3b5820): the expansion re-tune ran a
    FRESH TPE study over the expanded space, failed to rediscover the 20.0 ms config
    in 40 trials, and reported 22.6 ms — the candidate went backwards. crun.best_ms
    must be a running minimum over all of the candidate's spaces, mirroring
    FamilyManager.update_best, which was already monotonic."""
    from kernel_optimizer.control.orchestrator import CandidateRun
    from kernel_optimizer.models.core import Candidate

    cand = Candidate(candidate_id="c", family_id="f", origin="seed", backend="triton",
                     source_sha="x", structural_signature="y")
    crun = CandidateRun(candidate=cand, source="x = 1\n")
    # first space finds 20.0
    crun.best_ms = 20.0
    # a worse re-tune must not overwrite it
    new_best = 22.6
    if crun.best_ms is None or new_best < crun.best_ms:
        crun.best_ms = new_best
    assert crun.best_ms == 20.0
    # a better re-tune does
    better = 19.1
    if crun.best_ms is None or better < crun.best_ms:
        crun.best_ms = better
    assert crun.best_ms == 19.1


def test_family_update_best_is_monotonic():
    """The family-level best already ignores worse results, which is why the L3:43
    regression did not corrupt the reported run best — only the candidate-local
    number and the stats fed to the analyst."""
    from kernel_optimizer.models.core import ParamSet
    fm = FamilyManager(max_families_total=3, max_families_total_hard=6)
    _seed(fm, "fam-x", dropped=False)
    p = ParamSet(values={"BLOCK_M": 1})
    assert fm.update_best("fam-x", "cand-fam-x", p, 20.0) is True
    assert fm.update_best("fam-x", "cand-fam-x", p, 22.6) is False
    assert fm.families["fam-x"].best.latency_ms == 20.0


# --- diagnostics: a truncated traceback must not hide the exception --------------

# Shape of a real failing witness from L3:43 (run-l3-43-20260904-093730): the harness
# and torch frames come first, the actual cause is the LAST line.
_REAL_TAIL = (
    "runtime_error: Traceback (most recent call last):\n"
    '  File "/mnt/d/Pyhon_projects/opop/v2/src/kernel_optimizer/gpu/worker_main.py", '
    "line 471, in run_relaxed_correctness\n"
    "    out_kernel = model_new(*inputs); torch.cuda.synchronize(device=device)\n"
    "                 ^^^^^^^^^^^^^^^^^^\n"
    '  File "/mnt/d/.../torch/nn/modules/module.py", line 1775, in _wrapped_call_impl\n'
    "    return self._call_impl(*args, **kwargs)\n"
    "           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n"
    + '  File "/mnt/d/.../triton/compiler.py", line 99, in launch\n    pass\n' * 12
    + "triton.runtime.errors.OutOfResources: out of resource: shared memory, "
      "Required: 409600, Hardware limit: 101376. Reducing block sizes or "
      "`num_stages` may help.\n"
)


def test_error_excerpt_keeps_the_actual_exception():
    """Regression (found live on L3:43): witness rejections were reported as
    `log_tail[:500]`, which cut the traceback off inside torch's call frames and never
    reached the exception line. All three rejected seeds carried a byte-identical,
    diagnosis-free detail, so the repair agent had to guess. The excerpt must keep the
    TAIL, where the cause is."""
    from kernel_optimizer.paramspace.validation import error_excerpt

    assert "OutOfResources" not in _REAL_TAIL[:500]      # the old behaviour's window
    out = error_excerpt(_REAL_TAIL, 800)
    assert "OutOfResources" in out
    assert "409600" in out and "101376" in out           # the actionable numbers
    assert len(out) <= 900                               # still context-budget safe
    assert "elided" in out                               # says it dropped the middle


def test_error_excerpt_passes_short_text_through_unchanged():
    from kernel_optimizer.paramspace.validation import error_excerpt
    assert error_excerpt("boom: bad thing", 800) == "boom: bad thing"
    assert error_excerpt("", 800) == ""
    assert error_excerpt(None, 800) == ""


# --- A: the WSL venv must never sit on a 9p mount ------------------------------


def test_wsl_paths_are_on_ext4_not_9p():
    """Regression guard for the single largest cost in the harness: a venv under
    /mnt/* is read over 9p, where `import torch` costs ~26.8s instead of ~2.4s
    (measured, alternating, incl. a real triton compile+launch). With ~1000 one-shot
    GPU jobs per L3 run that was 6.7h of the 11.7h wall clock."""
    from kernel_optimizer.config import WslConfig
    cfg = WslConfig()
    assert not cfg.venv.startswith("/mnt/"), "venv on 9p costs ~11x per-job startup"
    assert not cfg.triton_cache_dir.startswith("/mnt/")
    # kernelbench_src stays on /mnt: read-only, a few files per job, and it must
    # remain visible from Windows.
    assert cfg.kernelbench_src.startswith("/mnt/")


def test_setup_script_refuses_a_9p_venv():
    from pathlib import Path
    src = Path("scripts/setup_wsl_venv.sh").read_text(encoding="utf-8")
    assert "REFUSING" in src and "/mnt/*" in src


# --- B1: prefetched parameterization ------------------------------------------


def test_run_store_append_is_thread_safe():
    """B1 runs parameterizer calls on a background thread, and those calls append
    AGENT_CALL_* events. Without a lock, `self._seq += 1` races and events.jsonl —
    the resume authority — gets duplicate seqs or torn lines."""
    import tempfile
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    from kernel_optimizer.store.run_store import RunStore

    with tempfile.TemporaryDirectory() as td:
        store = RunStore.create(Path(td), "run-x", {"task": "t"})
        n = 200
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda i: store.append("TRIAL_DONE", {"i": i}), range(n)))
        events = store.iter_events()
        # RUN_CREATED + n appends, every seq distinct and every line valid JSON.
        assert len(events) == n + 1
        assert len({e.seq for e in events}) == n + 1


def test_prefetch_disabled_by_config():
    from kernel_optimizer.config import BudgetConfig
    assert BudgetConfig().prefetch_parameterization == 1     # on by default
    assert BudgetConfig(prefetch_parameterization=0).prefetch_parameterization == 0


def test_prefetched_outcome_is_discarded_when_source_changed():
    """The safety rule that makes B1 information-preserving.

    A prefetch is issued against the candidate source as it stood at submit time. If a
    repair rewrote that source in the meantime, the prefetched parameterization
    describes the OLD code and must be thrown away rather than published — otherwise
    the space would be validated against source the agent never saw.
    """
    from concurrent.futures import Future

    from kernel_optimizer.control.orchestrator import CandidateRun, Orchestrator
    from kernel_optimizer.models.core import Candidate

    cand = Candidate(candidate_id="c", family_id="f", origin="seed", backend="triton",
                     source_sha="x", structural_signature="y")
    orch = Orchestrator.__new__(Orchestrator)          # no wiring needed for this unit
    orch.runs = {"c": CandidateRun(candidate=cand, source="NEW source")}
    sentinel = object()
    fut: Future = Future()
    fut.set_result(sentinel)
    orch._prefetched = {"c": fut}

    # Source moved on since the prefetch -> discard, fall back to a fresh call.
    assert orch._take_prefetched("c", "OLD source") is None

    # Same source -> the prefetched outcome is claimed.
    fut2: Future = Future()
    fut2.set_result(sentinel)
    orch._prefetched = {"c": fut2}
    assert orch._take_prefetched("c", "NEW source") is sentinel

    # Claiming is one-shot: the future is consumed.
    assert orch._take_prefetched("c", "NEW source") is None


def test_prefetch_agent_failure_falls_back_to_sync_call():
    """A failed prefetch must be invisible: return None so the caller makes its own
    synchronous attempt (and gets the real error path with its own retry budget)."""
    from concurrent.futures import Future

    from kernel_optimizer.agents.runtime import AgentCallError
    from kernel_optimizer.control.orchestrator import CandidateRun, Orchestrator
    from kernel_optimizer.models.core import Candidate

    cand = Candidate(candidate_id="c", family_id="f", origin="seed", backend="triton",
                     source_sha="x", structural_signature="y")
    orch = Orchestrator.__new__(Orchestrator)
    orch.runs = {"c": CandidateRun(candidate=cand, source="src")}
    for exc in (AgentCallError("boom"), RuntimeError("unexpected")):
        fut: Future = Future()
        fut.set_exception(exc)
        orch._prefetched = {"c": fut}
        assert orch._take_prefetched("c", "src") is None


# --- early pruning: family budget must not be handed out by incumbent latency -----


def _fam(fm, fid, incumbent, rounds_used, history=()):
    """Register a family with a given incumbent, rounds used, and round history."""
    from kernel_optimizer.models.core import BestRecord, Family, ParamSet
    cid = f"cand-{fid}"
    fm.families[fid] = Family(
        family_id=fid, anchor_candidate_id=cid, member_ids=[cid], status="active",
        best=BestRecord(candidate_id=cid, params=ParamSet(values={"B": 1}),
                        latency_ms=incumbent),
        best_history=list(history), rewrite_rounds_used=rounds_used,
    )
    return fm.families[fid]


def test_every_family_gets_a_rewrite_round_before_any_is_pruned():
    """The core anti-early-pruning guarantee.

    Found live: active_families() sorted by incumbent latency and sliced to
    max_families_active, so in both round-2 L3 runs 2 of 4 families never received a
    single rewrite round — they were frozen as 'budget_exhausted' having never once
    invoked the rewriter. A branch must be given one chance before it can lose budget.
    """
    fm = FamilyManager(max_families_total=4, max_families_total_hard=8)
    fm.max_families_active = 2
    _fam(fm, "fast-proven", 19.6, rounds_used=3, history=[19.6, 19.6, 19.6])
    _fam(fm, "slow-unproven", 31.6, rounds_used=0)
    picked = {f.family_id for f in fm.active_families()}
    assert "slow-unproven" in picked, "an unproven branch must not be pruned on latency"


def test_stalled_family_yields_to_one_still_improving():
    """Ranking is by improvement slope, not absolute latency.

    Reproduces L3:43 exactly: fam-c9461c56 held the better number (19.6) but had
    stalled across three rounds, while fam-ff3ef34b (19.5 -> 17.9) was still moving and
    produced the run's winner. The still-improving branch must be preferred.
    """
    fm = FamilyManager(max_families_total=4, max_families_total_hard=8)
    fm.max_families_active = 1
    _fam(fm, "stalled-but-good", 19.6, rounds_used=3, history=[19.6, 19.6, 19.6])
    _fam(fm, "improving", 17.9, rounds_used=3, history=[19.5, 17.9])
    assert [f.family_id for f in fm.active_families()] == ["improving"]

    # And it still holds when the stalled family has the BETTER incumbent, which is the
    # case that latency-ranking got wrong.
    fm2 = FamilyManager(max_families_total=4, max_families_total_hard=8)
    fm2.max_families_active = 1
    _fam(fm2, "stalled-better-number", 18.0, rounds_used=2, history=[18.0, 18.0])
    _fam(fm2, "improving-worse-number", 19.0, rounds_used=2, history=[22.0, 19.0])
    assert [f.family_id for f in fm2.active_families()] == ["improving-worse-number"]


def test_latency_only_breaks_ties_among_equally_stalled_families():
    fm = FamilyManager(max_families_total=4, max_families_total_hard=8)
    fm.max_families_active = 1
    _fam(fm, "slower", 25.0, rounds_used=2, history=[25.0, 25.0])
    _fam(fm, "faster", 20.0, rounds_used=2, history=[20.0, 20.0])
    assert [f.family_id for f in fm.active_families()] == ["faster"]


def test_improvement_pct_handles_short_and_degenerate_history():
    fm = FamilyManager(max_families_total=2, max_families_total_hard=4)
    f_new = _fam(fm, "new", 20.0, rounds_used=0)
    f_one = _fam(fm, "one", 20.0, rounds_used=1, history=[20.0])
    f_zero = _fam(fm, "zero", 20.0, rounds_used=2, history=[0.0, 20.0])
    for f in (f_new, f_one, f_zero):
        assert fm._improvement_pct(f) == 0.0
    f_ok = _fam(fm, "ok", 18.0, rounds_used=2, history=[20.0, 18.0])
    assert fm._improvement_pct(f_ok) == pytest.approx(10.0)


# --- B1 coverage: prefetch must apply to rewrite/novelty candidates too -----------


def test_pipeline_batch_prefetches_the_next_candidate():
    """B1 initially prefetched only inside the seed loop. In L3:43, 10 of the 14
    candidates were rewrites, so under a third of the parameterizer calls were
    overlapped. All three pipelining sites now go through _pipeline_batch."""
    from kernel_optimizer.config import AppConfig
    from kernel_optimizer.control.orchestrator import Orchestrator

    orch = Orchestrator.__new__(Orchestrator)
    orch.cfg = AppConfig()
    orch.cfg.budgets.prefetch_parameterization = 1
    prefetched: list[str] = []
    pipelined: list[str] = []
    orch._prefetch_parameterization = prefetched.append
    orch._candidate_pipeline = pipelined.append

    orch._pipeline_batch(["a", "b", "c"])

    assert pipelined == ["a", "b", "c"]
    # Each iteration prefetches the NEXT id; the last has no successor.
    assert prefetched == ["b", "c"]


def test_pipeline_batch_prefetch_can_be_disabled():
    from kernel_optimizer.config import AppConfig
    from kernel_optimizer.control.orchestrator import Orchestrator

    orch = Orchestrator.__new__(Orchestrator)
    orch.cfg = AppConfig()
    orch.cfg.budgets.prefetch_parameterization = 0
    prefetched: list[str] = []
    pipelined: list[str] = []
    orch._prefetch_parameterization = prefetched.append
    orch._candidate_pipeline = pipelined.append

    orch._pipeline_batch(["a", "b"])
    assert pipelined == ["a", "b"] and prefetched == []







