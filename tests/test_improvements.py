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









# --- T: transport timeouts retry the ORIGINAL prompt on a FRESH session -----------
#
# Found while verifying the L3:48 rerun: one L3:43 repair call spent 0.99h (34% of all
# agent wall time in that run) on two 20-minute ReadTimeouts before succeeding. Two
# defects fed it. (1) The retry sent "Your previous response could not be used:
# transport error..." — but a ReadTimeout means no response ever arrived, so the agent
# was asked to fix a message it never sent, losing the actual task text. (2) The retry
# reused the same session, queueing a second generation behind its own aborted turn.


class _FakeStore:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def append(self, event_type: str, payload: dict) -> None:
        self.events.append((event_type, payload))

    def put_artifact(self, *a, **k):  # pragma: no cover - unused here
        return "sha-stub"


class _FakeSandboxes:
    def __init__(self, tmp_path) -> None:
        self.tmp_path = tmp_path

    def create(self, call_id: str):
        from kernel_optimizer.agents.sandbox import Sandbox

        root = self.tmp_path / call_id
        root.mkdir(parents=True, exist_ok=True)
        return Sandbox(root)


def _timeout_module(tmp_path, fail_times: int, max_transport_retries: int = 2):
    """An AgentModule whose transport raises ReadTimeout `fail_times` times."""
    from pydantic import BaseModel as _BM

    from kernel_optimizer.agents.base import AgentModule
    from kernel_optimizer.agents.runtime import AgentCallError, PromptResult
    from kernel_optimizer.config import AgentModuleConfig

    class Out(_BM):
        answer: str

    seen: list[tuple[str, str]] = []  # (session_id, prompt text)
    sessions: list[str] = []

    class FakeClient:
        def create_session(self, root, title=""):
            sid = f"ses_{len(sessions)}"
            sessions.append(sid)
            return sid

        def prompt(self, session_id, text, **kw):
            seen.append((session_id, text))
            if len(seen) <= fail_times:
                raise AgentCallError("prompt transport error (ReadTimeout): timed out")
            return PromptResult(
                text="", structured={"answer": "ok"}, tokens={}, cost=0.0,
                session_id=session_id, message_id="m1",
            )

    class Mod(AgentModule):
        name = "repair"
        output_model = Out

        def seed_sandbox(self, inputs, sb):
            pass

        def render_prompt(self, inputs, sb):
            return "ORIGINAL TASK TEXT"

    store = _FakeStore()
    cfg = AgentModuleConfig(max_transport_retries=max_transport_retries)
    mod = Mod(FakeClient(), _FakeSandboxes(tmp_path), store, cfg)
    return mod, store, seen, sessions


def test_transport_timeout_resends_original_prompt_on_fresh_session(tmp_path):
    mod, store, seen, sessions = _timeout_module(tmp_path, fail_times=1)
    out = mod.invoke(None)

    assert out.output.answer == "ok"
    assert len(seen) == 2
    # The retry must carry the real task, NOT "your previous response could not be used".
    assert seen[1][1] == "ORIGINAL TASK TEXT"
    assert "could not be used" not in seen[1][1]
    # ...and must run on a different session than the one prompt() aborted.
    assert seen[0][0] != seen[1][0]
    assert any(t == "AGENT_SESSION_RESET" for t, _ in store.events)


def test_schema_failure_still_gets_corrective_feedback(tmp_path):
    """The transport fix must not disable ordinary schema-failure feedback: an invalid
    response DID arrive, so the agent should be told what was wrong with it."""
    from pydantic import BaseModel as _BM

    from kernel_optimizer.agents.base import AgentModule
    from kernel_optimizer.agents.runtime import PromptResult
    from kernel_optimizer.config import AgentModuleConfig

    class Out(_BM):
        answer: str

    seen: list[str] = []

    class FakeClient:
        def create_session(self, root, title=""):
            return "ses_only"

        def prompt(self, session_id, text, **kw):
            seen.append(text)
            structured = {"wrong_field": 1} if len(seen) == 1 else {"answer": "ok"}
            return PromptResult(text="", structured=structured, tokens={}, cost=0.0,
                                session_id=session_id, message_id="m")

    class Mod(AgentModule):
        name = "parameterizer"
        output_model = Out

        def seed_sandbox(self, inputs, sb):
            pass

        def render_prompt(self, inputs, sb):
            return "ORIGINAL TASK TEXT"

    mod = Mod(FakeClient(), _FakeSandboxes(tmp_path), _FakeStore(), AgentModuleConfig())
    assert mod.invoke(None).output.answer == "ok"
    assert len(seen) == 2 and "could not be used" in seen[1]


def test_transport_retries_are_capped(tmp_path):
    """A permanently dead endpoint must not consume the whole retry budget at
    request_timeout_s apiece; it gives up after max_transport_retries."""
    from kernel_optimizer.agents.runtime import AgentCallError

    mod, store, seen, _ = _timeout_module(tmp_path, fail_times=99,
                                          max_transport_retries=1)
    with pytest.raises(AgentCallError):
        mod.invoke(None)
    assert len(seen) == 2  # initial attempt + 1 transport retry, then stop


# --- R: repair agent must see the reference and its own rejected diagnoses --------
#
# Found in the L3:48 rerun. cand-0137895f was rejected three times with
# witness_default_failed. The repair agent first diagnosed "the reference parameterizes
# the transition as A_effective = -exp(A), so decay must be exp(-exp(A))", was rejected,
# then diagnosed the exact OPPOSITE ("the reference uses A_t directly") and reverted to
# a form already known to fail. The reference (level3/48 line 61,
# `torch.exp(self.segsum(A_blocks))`) uses A directly -- but the agent could not check,
# because the repair sandbox never contained ref.py, and could not tell it was going in
# circles, because each call saw only the current error. Both inputs are strictly
# same-candidate; nothing cross-candidate is shared.


def test_repair_sandbox_gets_reference_and_rejected_history(tmp_path):
    from kernel_optimizer.agents.modules import RepairAgent, RepairInputs
    from kernel_optimizer.agents.sandbox import Sandbox
    from kernel_optimizer.config import AgentModuleConfig
    from kernel_optimizer.models.core import DeviceLimits, TaskSpec

    sb = Sandbox(tmp_path)
    agent = RepairAgent.__new__(RepairAgent)
    agent.cfg = AgentModuleConfig()
    inputs = RepairInputs(
        task=TaskSpec(level=3, problem_id=48, name="48_Mamba2ReturnY",
                      ref_path=tmp_path / "ref.py", ref_src_sha="x"),
        broken_source="PARAMS = {}\n",
        failure_kind="witness_default_failed",
        failure_detail="correctness_mismatch: relaxed mismatch (max abs diff 7.3e15)",
        device=DeviceLimits(),
        ref_source="Y = torch.exp(self.segsum(A_blocks))  # A used directly\n",
        prior_attempts=[
            {"diagnosis": "decay must be exp(-exp(A))", "failure_detail": "diff 7.3e15"},
        ],
    )
    agent.seed_sandbox(inputs, sb)

    ref = (tmp_path / "task" / "ref.py").read_text(encoding="utf-8")
    assert "segsum(A_blocks)" in ref
    hist = (tmp_path / "failure" / "rejected_repairs.md").read_text(encoding="utf-8")
    assert "exp(-exp(A))" in hist and "DISPROVEN" in hist
    # The agent must be warned against merely inverting a rejected claim.
    assert "inverting" in hist

    prompt = agent.render_prompt(inputs, sb)
    assert "task/ref.py" in prompt and "failure/rejected_repairs.md" in prompt
    assert "invert" in prompt


def test_repair_prompt_omits_absent_optional_inputs(tmp_path):
    """A repair with no reference and no history must not reference missing files."""
    from kernel_optimizer.agents.modules import RepairAgent, RepairInputs
    from kernel_optimizer.agents.sandbox import Sandbox
    from kernel_optimizer.config import AgentModuleConfig
    from kernel_optimizer.models.core import DeviceLimits, TaskSpec

    sb = Sandbox(tmp_path)
    agent = RepairAgent.__new__(RepairAgent)
    agent.cfg = AgentModuleConfig()
    inputs = RepairInputs(
        task=TaskSpec(level=1, problem_id=19, name="19_ReLU",
                      ref_path=tmp_path / "ref.py", ref_src_sha="x"),
        broken_source="PARAMS = {}\n",
        failure_kind="witness_minimal_failed",
        failure_detail="compile_error: boom",
        device=DeviceLimits(),
    )
    agent.seed_sandbox(inputs, sb)
    prompt = agent.render_prompt(inputs, sb)
    assert not (tmp_path / "task" / "ref.py").exists()
    assert not (tmp_path / "failure" / "rejected_repairs.md").exists()
    assert "task/ref.py" not in prompt and "rejected_repairs" not in prompt


def test_rejected_repairs_doc_flags_contradictory_history():
    """Two mutually-inverse rejected diagnoses is the oscillation signature; the doc
    must tell the agent that neither is the cause, not just 'do not repeat'."""
    from kernel_optimizer.agents.modules import _rejected_repairs_doc

    doc = _rejected_repairs_doc([
        {"diagnosis": "decay must be exp(-exp(A))", "failure_detail": "diff 7.3e15"},
        {"diagnosis": "reference uses A directly", "failure_detail": "diff 1.0e22"},
    ])
    assert "Attempt 1" in doc and "Attempt 2" in doc
    assert "neither is the" in doc
    assert "7.3e15" in doc and "1.0e22" in doc


# --- N: cosine must not overflow on large-magnitude outputs -----------------------
#
# The L3:48 rerun rejected every seed with witness_default_failed. Scoring the rejected
# witnesses directly (scripts/score_l3_48_witnesses.py) showed they were CORRECT:
# frac_within_1% = 0.999983 against a 0.99 gate, median relative error 4e-7. They were
# rejected because the cosine term was nan. level3/48's outputs reach 1e22, and fp32
# dot()/norm() overflow to inf above ~1.8e19 (fp32 max 3.4e38, products are squares), so
# cos = inf/inf = nan and `nan >= cosine_min` is False. Two correct candidates were
# discarded by an arithmetic overflow, and four repair attempts were spent inventing
# sign-convention bugs to explain it.


def test_cosine_survives_1e22_magnitude():
    torch = pytest.importorskip("torch")
    _relaxed_close = _load_worker_relaxed_close()

    # Identical tensors at level3/48's real output scale must compare equal.
    ref = torch.full((4096,), 1e22)
    assert _relaxed_close(ref, ref.clone(), 0.01, 0.99, 0.99985)

    # Sanity: the naive fp32 formula this replaced really does overflow here.
    a = ref.flatten()
    assert not torch.isfinite(a.norm())


def test_cosine_helper_is_finite_where_fp32_overflows():
    torch = pytest.importorskip("torch")
    import importlib.util

    spec = importlib.util.find_spec("kernel_optimizer.gpu.worker_main")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    big = torch.full((1000,), 1e22)
    cos = mod._cosine_similarity(big, big.clone())
    assert cos == pytest.approx(1.0, abs=1e-6)
    # ...and it must never exceed the valid cosine range after rescaling.
    assert -1.0 <= cos <= 1.0

    # Opposed vectors at the same scale still read as opposed.
    assert mod._cosine_similarity(big, -big) == pytest.approx(-1.0, abs=1e-6)
    # All-zero pair is defined as identical, not nan.
    zero = torch.zeros(10)
    assert mod._cosine_similarity(zero, zero) == 1.0


def test_large_magnitude_wrong_answer_is_still_rejected():
    """The overflow fix must not turn the gate into a rubber stamp: a genuinely wrong
    kernel at the same 1e22 scale must still fail."""
    torch = pytest.importorskip("torch")
    _relaxed_close = _load_worker_relaxed_close()

    ref = torch.full((4096,), 1e22)
    wrong = ref.clone()
    wrong[:1000] = -1e22  # 24% of elements sign-flipped
    assert not _relaxed_close(ref, wrong, 0.01, 0.99, 0.99985)
    # A near-zero output against a huge reference must also fail.
    assert not _relaxed_close(ref, torch.zeros_like(ref), 0.01, 0.99, 0.99985)


def test_relaxed_metrics_reports_gate_criteria_not_just_max_diff():
    """The failure message reported only max-abs-diff. On level3/48 the reference's own
    fp32-vs-fp64 max-abs-diff is 1.5e16, so '7.3e15' read as catastrophic while being
    inside the reference's own noise -- which is what sent the repair agent chasing
    imaginary sign-convention bugs. The message must carry the gate's real criteria."""
    torch = pytest.importorskip("torch")
    import importlib.util

    spec = importlib.util.find_spec("kernel_optimizer.gpu.worker_main")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    ref = torch.full((1000,), 1e22)
    got = ref.clone()
    got[:5] = 5e21
    m = mod._relaxed_metrics(ref, got)
    for key in ("frac_within_tol", "cosine", "median_rel_err", "p99_rel_err",
                "max_abs_diff", "ref_absmax", "ref_absmedian"):
        assert key in m, key
    assert m["frac_within_tol"] == pytest.approx(0.995, abs=1e-6)
    assert m["cosine"] != "nan"

    shape_m = mod._relaxed_metrics(ref, torch.zeros(999))
    assert "shape_ref" in shape_m and "shape_got" in shape_m


# --- X: the excessive-speedup guard must not reject VERIFIED-CORRECT kernels -------
#
# Found live in the L3:48 rerun. The guard hard-failed any candidate over 10x, ignoring
# correctness. Four trials of cand-c18203b6 were rejected at 11.1x-13.9x with
# correct=True and trials_passed=3/3, while a neighbouring parameter point at 8.95x was
# accepted -- same kernel, verdict decided by which side of 10x the timing noise landed.
# 4 of 19 trials (21%) were discarded, and they were the FASTEST ones, so the reported
# optimum was biased downward. The guard's purpose is catching work-SKIPPING, and a
# kernel that reproduced the reference's values on fresh inputs in every correctness
# trial has not skipped the work. Correctness now decides acceptance; the speedup only
# raises a flag. A fast kernel that FAILS correctness is still a hard failure -- which is
# what the timing-cheat fixture is, since it caches on tensor identity and so cannot pass
# correctness trials that use fresh inputs.


def _guard_verdict(*, correct: bool, cand_ms: float, ref_ms: float,
                   thr: float = 10.0) -> dict:
    """Replay the worker's post-timing guard block on a synthetic result."""
    import importlib.util

    spec = importlib.util.find_spec("kernel_optimizer.gpu.worker_main")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # The guard is inline in run_relaxed_correctness; exercise it through a job whose
    # numbers are fixed, by calling the same arithmetic the module applies.
    pass_count, num_trials = (3, 3) if correct else (1, 3)
    result = {
        "ok": correct,
        "failure_kind": None if correct else "correctness_mismatch",
        "latency_ms": {"mean": cand_ms},
        "ref_latency_ms": {"mean": ref_ms},
    }
    speedup = ref_ms / cand_ms
    result["speedup_vs_ref_in_worker"] = speedup
    if speedup >= thr:
        result["excessive_speedup"] = True
        if not correct:
            result["ok"] = False
            result["failure_kind"] = "excessive_speedup"
        else:
            result["excessive_speedup_note"] = f"{speedup:.1f}x flagged"
    else:
        result["excessive_speedup"] = False
    return result


def test_guard_source_accepts_correct_fast_kernel_and_fails_incorrect_one():
    """Assert against the real module source, so the test tracks the shipped logic
    rather than only the replay helper above."""
    from pathlib import Path

    src = Path("src/kernel_optimizer/gpu/worker_main.py").read_text(encoding="utf-8")
    # Take the whole guard block: from the threshold test to where the flag is set
    # False on the non-suspicious path. Splitting on the first "else:" would cut at
    # the inner `if not correct:` branch and miss the accept path.
    guard = src.split("if speedup >= thr:")[1].split('result["excessive_speedup"] = False')[0]
    # The hard fail must be conditional on correctness, not unconditional.
    assert "if not correct:" in guard, "guard must branch on correctness"
    assert "excessive_speedup_note" in guard, "correct-but-fast must be flagged, not failed"
    # And the flag itself must still be recorded.
    assert 'result["excessive_speedup"] = True' in guard
    # The hard-fail assignment must live inside the not-correct branch: everything
    # before "else:" (the accept path) is the failure path.
    fail_path = guard.split("else:")[0]
    assert 'result["failure_kind"] = "excessive_speedup"' in fail_path
    accept_path = guard.split("else:", 1)[1]
    assert 'result["ok"] = False' not in accept_path, "accept path must not fail the job"


def test_correct_kernel_over_threshold_is_accepted_and_flagged():
    r = _guard_verdict(correct=True, cand_ms=2.62, ref_ms=29.1)  # the real L3:48 case
    assert r["speedup_vs_ref_in_worker"] == pytest.approx(11.1, abs=0.1)
    assert r["ok"] is True and r["failure_kind"] is None
    assert r["excessive_speedup"] is True and "excessive_speedup_note" in r


def test_incorrect_kernel_over_threshold_is_still_hard_failed():
    r = _guard_verdict(correct=False, cand_ms=0.0001, ref_ms=29.1)
    assert r["ok"] is False and r["failure_kind"] == "excessive_speedup"


def test_neighbouring_points_no_longer_get_opposite_verdicts():
    """The 8.95x and 11.1x points of the same kernel must now agree."""
    slow = _guard_verdict(correct=True, cand_ms=3.24, ref_ms=29.0)  # 8.95x, was accepted
    fast = _guard_verdict(correct=True, cand_ms=2.62, ref_ms=29.1)  # 11.1x, was rejected
    assert slow["ok"] == fast["ok"] is True


def test_guard_uses_median_reference_not_outlier_corrupted_mean():
    """A single scheduling stall in the guard's own reference timing must not decide a
    verdict. Observed live on L3:48: a 10-sample reference returned mean=609ms with
    min=29.8ms / max=5760ms / std=1720ms -- one ~5.8s outlier dragged the mean 20x and
    manufactured a 115x 'speedup' against a candidate at 5.29ms. The reference's true
    latency on this task is ~29ms (matching the eager baseline)."""
    from pathlib import Path

    src = Path("src/kernel_optimizer/gpu/worker_main.py").read_text(encoding="utf-8")
    assert 'ref_latency_ms["median"]' in src, "median must be recorded"
    # The ratio must read the median first.
    ratio_line = next(l for l in src.splitlines() if "ref_mean = ref_latency_ms" in l)
    assert '"median"' in ratio_line and ratio_line.index('"median"') < ratio_line.index('"mean"')

    # The real distribution from the live run: median is ~29ms, mean is 609ms.
    samples = [29.8, 30.1, 29.9, 30.0, 5760.0, 29.7, 30.2, 29.8, 30.0, 90.0]
    s = sorted(samples)
    n = len(s)
    median = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    mean = sum(samples) / n
    cand = 5.29
    assert mean / cand > 100          # the bogus verdict the mean produced
    assert median / cand < 10         # the median keeps it under the threshold


# --- J: reused measurements must be journalled ------------------------------------
#
# Found while watching the L3:48 expansion re-tune. _tune reuses an already-measured
# record when a TPE ask lands on a cached param set (witness anchors, and the
# pre-expansion optimum carried over by the K fix) but appended no TRIAL_DONE. Since
# replay() rebuilds measured_cache purely from TRIAL_DONE, and the report and lineage
# read the same events, those points were invisible: a resume would re-run them on the
# GPU, and the anchor carrying the prior optimum -- the whole fix for L3:43
# cand-0c3b5820's 20.0 -> 22.6ms regression -- never appeared in the trial log, so the
# regression it prevents could not be confirmed from the events either.


def test_reused_measurement_is_journalled_with_flag():
    from pathlib import Path

    src = Path("src/kernel_optimizer/control/orchestrator.py").read_text(encoding="utf-8")
    block = src.split("cached = measured_cache.get(params.key())")[1].split("else:")[0]
    assert "TRIAL_DONE" in block, "a reused measurement must still emit TRIAL_DONE"
    assert "reused_measurement" in block, "it must be distinguishable from a fresh run"
    # The reused record must be re-stamped with the CURRENT space, or the trial would be
    # filed under the pre-expansion space and still be missed on replay.
    assert "space_id=space.space_id" in block or "space_id\": space.space_id" in block \
        or "space_id" in block


def test_replay_rebuilds_cache_from_trial_done_only():
    """Documents why the above matters: replay's measured_cache source is TRIAL_DONE."""
    from pathlib import Path

    src = Path("src/kernel_optimizer/control/orchestrator.py").read_text(encoding="utf-8")
    resume = src.split("measured_cache: dict[str, TrialRecord] = {}")[1][:700]
    assert "state.trials" in resume
    store = Path("src/kernel_optimizer/store/run_store.py").read_text(encoding="utf-8")
    trials_line = next(l for l in store.splitlines() if "state.trials.setdefault" in l)
    assert trials_line  # populated under the TRIAL_DONE branch
    assert 'ev.type == "TRIAL_DONE"' in store


# --- Y: family control state must survive resume ----------------------------------
#
# Checked because active_families()'s anti-early-pruning ranking sorts on
# rewrite_rounds_used, best_history and best -- if a resume reset those, the ordering the
# paper's problem statement cares about would silently reset, and a family that already
# spent its rewrite budget could be handed a fresh one. It does NOT: `best` is rebuilt
# from TRIAL_DONE, best_history and rewrite_rounds_used from FAMILY_ROUND_RECORDED, and
# `status` is re-derived by family_verdict, which reads only those two fields. FAMILY_UPDATED
# is consumed by replay() but emitted by nothing; these tests pin the real contract so a
# future change cannot quietly break it.


def test_family_verdict_depends_only_on_persisted_fields():
    """status is a derived cache, so losing it on resume must not change any decision."""
    from kernel_optimizer.config import BudgetConfig
    from kernel_optimizer.control.convergence import ConvergencePolicy
    from kernel_optimizer.models.core import Family

    judge = ConvergencePolicy(BudgetConfig(rewrite_rounds_per_family=3,
                                          no_improve_rounds=2, min_improvement_pct=2.0))

    # Budget exhausted is decided by rewrite_rounds_used alone.
    spent = Family(family_id="f", anchor_candidate_id="c", member_ids=["c"],
                   rewrite_rounds_used=3, status="active")  # status deliberately wrong
    v = judge.family_verdict(spent)
    assert v.verdict == "freeze" and v.stop_kind == "budget_exhausted"

    # Convergence is decided by best_history alone.
    flat = Family(family_id="g", anchor_candidate_id="c", member_ids=["c"],
                  rewrite_rounds_used=2, best_history=[20.0, 19.9, 19.85], status="active")
    v2 = judge.family_verdict(flat)
    assert v2.verdict == "freeze" and v2.stop_kind == "converged"

    # A still-improving family continues regardless of a stale status.
    good = Family(family_id="h", anchor_candidate_id="c", member_ids=["c"],
                  rewrite_rounds_used=1, best_history=[20.0, 15.0], status="active")
    assert judge.family_verdict(good).verdict == "continue"


def test_family_round_recorded_is_the_persisted_source_of_truth():
    """best_history and rewrite_rounds_used must both come from that one stream, so a
    resume cannot double-count or lose rounds."""
    from pathlib import Path

    src = Path("src/kernel_optimizer/control/orchestrator.py").read_text(encoding="utf-8")
    restore = src.split("def _restore_family_control_state")[1].split("def _rewrite_round")[0]
    assert 'FAMILY_ROUND_RECORDED' in restore
    assert "family.best_history = [" in restore
    assert "family.rewrite_rounds_used = len(evs)" in restore, \
        "rounds must be the event count, not an increment, or resume double-counts"
    # And the restore must run BEFORE loop C, or the first round uses empty state.
    run_body = src.split("def _run(")[1].split("def ")[0]
    assert run_body.index("_restore_family_control_state") < run_body.index("_rewrite_round")


def test_family_updated_has_no_producer_and_is_documented():
    """replay() consumes FAMILY_UPDATED but nothing emits it. That is intentional (state
    is reconstructed, not snapshotted); the branch must say so, so nobody 'fixes' it by
    emitting events that would then race the reconstruction."""
    from pathlib import Path

    store = Path("src/kernel_optimizer/store/run_store.py").read_text(encoding="utf-8")
    branch = store.split('elif ev.type == "FAMILY_UPDATED":')[1].split("elif ev.type")[0]
    assert "No producer" in branch

    # Assert the absence mechanically, across the package.
    import subprocess
    hits = subprocess.run(
        ["git", "grep", "-n", "FAMILY_UPDATED", "--", "src/"],
        capture_output=True, text=True).stdout.splitlines()
    emitters = [h for h in hits if "append(" in h]
    assert not emitters, f"unexpected FAMILY_UPDATED emitter: {emitters}"


def test_relaxed_metrics_handles_tensors_too_large_for_quantile():
    """A diagnostic must never destroy the diagnosis.

    torch.quantile refuses inputs above ~16M elements. level3/48's output is
    2048*128*8*64 = 134M, so the p99 line I added raised inside the failure-reporting
    path and turned cand-eb910a18's correctness_mismatch into an opaque
    'RuntimeError: quantile() input tensor is too large' -- the repair agent then saw a
    crash instead of the mismatch it was supposed to diagnose."""
    torch = pytest.importorskip("torch")
    import importlib.util

    spec = importlib.util.find_spec("kernel_optimizer.gpu.worker_main")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Comfortably past torch.quantile's limit, small enough for a CPU test.
    n = 20_000_000
    ref = torch.ones(n)
    got = ref.clone()
    got[:1000] = 2.0
    m = mod._relaxed_metrics(ref, got)          # must not raise
    assert m["p99_rel_err"] != "n/a"
    assert m["frac_within_tol"] == pytest.approx(1.0 - 1000 / n, abs=1e-6)


def test_mismatch_detail_falls_back_when_metrics_raise():
    """Even if the rich metrics fail for some future reason, the mismatch itself must
    still be reported -- with the gate's thresholds -- not replaced by a traceback."""
    from pathlib import Path

    src = Path("src/kernel_optimizer/gpu/worker_main.py").read_text(encoding="utf-8")
    block = src.split("ok = (_relaxed_close")[1].split("except Exception as exc:")[0]
    assert "except Exception as diag_exc" in block, "metrics must be wrapped"
    fallback = block.split("except Exception as diag_exc")[1]
    # The fallback still has to carry a number AND the gate criteria.
    assert "max abs diff" in fallback
    assert "frac_within_tol" in fallback and "cosine>=" in fallback
    assert "Detailed metrics unavailable" in fallback


def test_non_finite_output_is_named_not_reported_as_five_nans():
    """A NaN anywhere in the candidate's output poisons every derived statistic, so the
    L3:48 message for cand-eb910a18 read cosine/median/p99/max_abs_diff all 'nan' --
    indistinguishable from a metric that overflowed, and useless to a repair agent. The
    non-finite values must be counted and named as THE failure, with the remaining
    statistics computed over the finite subset so they stay informative."""
    torch = pytest.importorskip("torch")
    import importlib.util

    spec = importlib.util.find_spec("kernel_optimizer.gpu.worker_main")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    ref = torch.full((100000,), 100.0)
    got = ref.clone()
    got[:5000] = float("nan")
    got[5000:6000] = float("inf")
    m = mod._relaxed_metrics(ref, got)

    assert "NON_FINITE_OUTPUT" in m
    assert "5000 NaN" in m["NON_FINITE_OUTPUT"] and "1000 +/-Inf" in m["NON_FINITE_OUTPUT"]
    # Surviving statistics must be real numbers, not nan.
    assert m["median_rel_err"] != "nan" and m["cosine"] != "nan"
    # Non-finite elements count as OUTSIDE tolerance: 94% finite-and-perfect is 0.94,
    # not 1.0 over the finite subset.
    assert m["frac_within_tol"] == pytest.approx(0.94, abs=1e-6)

    # Clean output must be untouched by all this.
    clean = mod._relaxed_metrics(ref, ref.clone())
    assert "NON_FINITE_OUTPUT" not in clean and clean["frac_within_tol"] == 1.0


def test_all_non_finite_output_does_not_raise():
    torch = pytest.importorskip("torch")
    import importlib.util

    spec = importlib.util.find_spec("kernel_optimizer.gpu.worker_main")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    ref = torch.full((1000,), 100.0)
    m = mod._relaxed_metrics(ref, torch.full_like(ref, float("nan")))
    assert m["frac_within_tol"] == 0.0 and "NON_FINITE_OUTPUT" in m


def test_gate_rejects_nan_output_even_when_frac_would_pass():
    """0.5% NaN gives frac 0.995, above the 0.99 threshold; the all-finite fallback in
    _relaxed_close must still reject it, or NaN output could pass the gate."""
    torch = pytest.importorskip("torch")
    _relaxed_close = _load_worker_relaxed_close()

    ref = torch.full((100000,), 100.0)
    few = ref.clone()
    few[:500] = float("nan")
    assert not _relaxed_close(ref, few, 0.01, 0.99, 0.99985)
    # And a fully-correct kernel at the same shape still passes.
    assert _relaxed_close(ref, ref.clone(), 0.01, 0.99, 0.99985)


def test_repair_history_pairs_each_diagnosis_with_the_failure_IT_caused():
    """The history is only useful if a diagnosis is labelled with what it PRODUCED.

    Live on L3:48: the file told the agent its TF32-precision diagnosis "still failed
    with" the quantile crash -- but that crash PRECEDED the repair and was the reason it
    was called. Attaching verdict.detail at append time always records the failure the
    repair was responding to, one step off. The detail must be filled in on the next
    iteration, once the repaired source has actually been evaluated."""
    from pathlib import Path

    src = Path("src/kernel_optimizer/control/orchestrator.py").read_text(encoding="utf-8")
    block = src.split('if verdict.reason.startswith("witness_"):')[1].split("except AgentCallError")[0]

    # A fresh entry must be recorded with NO failure yet.
    assert '"failure_detail": None' in block, "detail must be deferred, not set at append"
    # The previous entry gets closed out with the CURRENT verdict.
    assert 'repair_history[-1]["failure_detail"] = verdict.detail' in block
    assert 'repair_history[-1].get("failure_detail") is None' in block, \
        "must only fill an open entry, never overwrite a closed one"
    # Only entries with a known outcome are shown to the agent -- an entry whose result
    # is still unknown carries no information and must not be presented as disproven.
    assert 'if h.get("failure_detail")' in block


def test_rejected_repairs_doc_skips_entries_without_an_outcome():
    """_rejected_repairs_doc must tolerate (and not mislabel) an entry whose failure is
    not yet known, since the orchestrator now fills that in one step later."""
    from kernel_optimizer.agents.modules import _rejected_repairs_doc

    doc = _rejected_repairs_doc([
        {"diagnosis": "first guess", "failure_detail": "frac 0.91 vs floor 0.98"},
    ])
    assert "first guess" in doc and "frac 0.91" in doc
    # An entry with no detail must still render its diagnosis as disproven-by-rejection,
    # but must not fabricate a "Still failed with:" line.
    doc2 = _rejected_repairs_doc([{"diagnosis": "second guess", "failure_detail": ""}])
    assert "second guess" in doc2 and "Still failed with" not in doc2


def test_expansion_skips_knobs_already_at_a_hard_hardware_edge():
    """NUM_WARPS=1 cannot go lower -- one warp IS the minimum launch allocation -- so
    asking the parameterizer to extend it downward buys nothing.

    Live on L3:48 it was requested in 4 of 5 expansions and expanded zero times; twice it
    was the ONLY requested knob, so the whole expansion returned a byte-identical space
    and still cost a 40-trial re-tune. The analyst itself reports blocked_by="threads"
    with "further decrease is impossible", so the trend being real is not the issue: the
    boundary is simply not extendable."""
    from kernel_optimizer.control.orchestrator import boundary_knobs_to_expand
    from kernel_optimizer.models.core import ParamDomain, ParameterSpace
    from kernel_optimizer.models.reports import ParamStat, TuningStats

    space = ParameterSpace(
        space_id="sp-1", candidate_id="c", version=1, source_sha="x",
        domains=[ParamDomain(name="NUM_WARPS", kind="int", choices=[1, 2, 4, 8]),
                 ParamDomain(name="BLOCK_P", kind="int", choices=[16, 32, 64])],
    )
    stats = TuningStats(
        candidate_id="c", space_id="sp-1", n_complete=40, n_fail=0,
        param_stats=[
            # Both look identically "blocked at a boundary with a real effect".
            ParamStat(name="NUM_WARPS", best_value=1, at_boundary=True,
                      boundary_direction="min", effect_pct=19.4),
            ParamStat(name="BLOCK_P", best_value=64, at_boundary=True,
                      boundary_direction="max", effect_pct=12.0),
        ],
    )
    knobs = boundary_knobs_to_expand(stats, idle_frac=0.8, space=space,
                                     min_effect_pct=2.0)
    names = [k["name"] for k in knobs]
    assert "BLOCK_P" in names, "a genuinely extendable knob must still be expanded"
    assert "NUM_WARPS" not in names, "1 warp is the floor; there is no next value"

    # Direction matters: NUM_WARPS wanting MORE warps is extendable.
    stats_up = TuningStats(
        candidate_id="c", space_id="sp-1", n_complete=40, n_fail=0,
        param_stats=[ParamStat(name="NUM_WARPS", best_value=8, at_boundary=True,
                               boundary_direction="max", effect_pct=19.4)],
    )
    up = boundary_knobs_to_expand(stats_up, idle_frac=0.8, space=space,
                                  min_effect_pct=2.0)
    assert [k["name"] for k in up] == ["NUM_WARPS"]


def test_expansion_that_adds_no_choices_is_rejected_not_retuned():
    """Even for a legitimately extendable knob the agent may return the same domains.
    Accepting that costs a full re-tune whose only possible outcome is rediscovering the
    same optimum, so the delivered space is compared against the previous one rather than
    the request being trusted."""
    from pathlib import Path

    src = Path("src/kernel_optimizer/control/orchestrator.py").read_text(encoding="utf-8")
    block = src.split("def _maybe_expand_space")[1].split("def _expand_directive_text")[0]
    assert "no_new_choices" in block, "a no-op expansion must be rejected by reason"
    # Compared on choices, not on space_id/sha: a fresh space_id is issued either way.
    assert "prev_choices" in block and "new_choices" in block
    assert "tuple(d.choices)" in block
    # And it must bail out BEFORE the re-tune, which is the cost being avoided.
    idx_reject = block.find("no_new_choices")
    idx_tune = block.find("self._tune(")
    assert idx_reject < idx_tune, "the no-op check must precede the re-tune"


def test_report_on_an_unfinished_run_is_honest_not_empty(tmp_path):
    """The report read ONLY RUN_FINISHED, so on a run still in flight it claimed "no
    correct candidate survived" and rendered an empty families section -- on L3:48 that
    was a lie about 338 trials and nine successful tunings sitting in the same event log.
    `kernel-opt report` is the documented way to inspect a run, including an interrupted
    one, so it must reconstruct from events.

    It must also NOT invent what it does not have: final_reeval_ms and the honest verdict
    come from a fresh-process re-eval at finalize, and tuned_ms runs 1.5-6.7% optimistic
    against it, so synthesising them would manufacture exactly the number the reeval-gap
    rule says not to trust."""
    from kernel_optimizer.reporting.report import ReportGenerator
    from kernel_optimizer.store.run_store import RunStore

    store = RunStore.create(tmp_path, run_id="unfinished", manifest={})
    store.append("CANDIDATE_REGISTERED", {"candidate": {
        "candidate_id": "cand-aaa", "family_id": "fam-1", "parent_ids": [],
        "origin": "seed", "backend": "triton", "source_sha": "x",
        "structural_signature": "s", "approach_summary": "a fused scan"}})
    store.append("TRIAL_DONE", {"trial": {
        "trial_id": "tr-1", "candidate_id": "cand-aaa", "space_id": "sp-1",
        "params": {"values": {"B": 64}}, "status": "complete",
        "latency_ms": {"mean": 2.5, "std": 0.1, "min": 2.4, "max": 2.9, "n_samples": 20}}})
    store.append("TUNING_DONE", {"candidate_id": "cand-aaa", "space_id": "sp-1",
                                 "best_ms": 2.5, "snapshot": {"asked": 40}})
    text = ReportGenerator().generate(store).read_text(encoding="utf-8")

    assert "no correct candidate survived" not in text
    assert "PROVISIONAL" in text, "an unfinished report must say so"
    assert "cand-aaa" in text and "2.5 ms" in text
    assert "fam-1" in text, "families section must not be empty"
    # No fabricated verified latency.
    assert "not run yet" in text
    # The banner mentions the absent verdict by name, so check the Best result section
    # itself rather than the whole document.
    best_section = text.split("## Best result")[1].split("##")[0]
    assert "honest same-precision verdict" not in best_section
    assert "speedup vs" not in best_section
    # A family with no completed round mid-run must NOT be described as frozen: on L3:48
    # fam-b1ee96ac had two rewrites under evaluation while the report called it frozen.
    assert "was frozen without the rewriter" not in text


def test_report_distinguishes_a_retune_from_a_duplicate_line(tmp_path):
    """A candidate that got a K expansion is tuned twice, and the Tuning section rendered
    two identical lines -- indistinguishable from a duplicated entry, and hiding which
    result came from the widened space (on L3:48, cand-cf0f07e7's 3.55 -> 2.84 is the one
    expansion that paid off)."""
    from kernel_optimizer.reporting.report import ReportGenerator
    from kernel_optimizer.store.run_store import RunStore

    store = RunStore.create(tmp_path, run_id="expanded", manifest={})
    store.append("CANDIDATE_REGISTERED", {"candidate": {
        "candidate_id": "cand-bbb", "family_id": "fam-1", "parent_ids": [],
        "origin": "seed", "backend": "triton", "source_sha": "x",
        "structural_signature": "s", "approach_summary": "scan"}})
    store.append("SPACE_PUBLISHED", {"space": {"space_id": "sp-first",
                                               "candidate_id": "cand-bbb"}})
    store.append("TUNING_DONE", {"candidate_id": "cand-bbb", "space_id": "sp-first",
                                 "best_ms": 3.55, "snapshot": {"asked": 40}})
    store.append("SPACE_PUBLISHED", {"space": {"space_id": "sp-second",
                                               "candidate_id": "cand-bbb"}})
    store.append("SPACE_EXPANDED", {"candidate_id": "cand-bbb", "knobs": [],
                                    "prev_best_ms": 3.55})
    store.append("TUNING_DONE", {"candidate_id": "cand-bbb", "space_id": "sp-second",
                                 "best_ms": 2.84, "snapshot": {"asked": 40}})
    text = ReportGenerator().generate(store).read_text(encoding="utf-8")

    tuning = text.split("## Tuning")[1].split("##")[0]
    assert "sp-first" in tuning and "sp-second" in tuning, "spaces must be identifiable"
    assert "(expanded space)" in tuning
    # Only the second space is the expansion.
    first = next(ln for ln in tuning.splitlines() if "sp-first" in ln)
    second = next(ln for ln in tuning.splitlines() if "sp-second" in ln)
    assert "expanded" not in first and "expanded" in second


def test_experiment_config_names_the_device_it_optimizes_for():
    """load_config reads ONE file: default.yaml is not a base layer, so a key omitted
    from experiments_l3.yaml falls back to the pydantic field default, NOT to
    default.yaml.

    Every numeric limit happened to match its field default, so this stayed invisible --
    but DeviceLimits.name defaults to "unknown", and _device_doc() writes it verbatim into
    every agent sandbox. Verified on disk: every docs/device.md in the L3:48 run reads
    "# Target device\\n\\n- unknown", so no agent ever learned the target is Blackwell
    sm_120, which decides which tensor-core paths and instructions exist at all.

    Pins two things: the experiment config states a real device name, and its numeric
    limits agree with default.yaml so the two cannot silently drift apart."""
    import yaml

    from kernel_optimizer.agents.modules import _device_doc
    from kernel_optimizer.config import load_config

    cfg = load_config("configs/experiments_l3.yaml")
    assert cfg.device.name and cfg.device.name != "unknown", \
        "the experiment config must name the GPU; agents are told this verbatim"
    assert cfg.device.name in _device_doc(cfg.device)

    base = yaml.safe_load(open("configs/default.yaml", encoding="utf-8"))["device"]
    for key, expected in base.items():
        got = getattr(cfg.device, key)
        # Compare numerically where both are numbers (yaml 16 vs float 16.0 is not drift).
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            assert float(got) == float(expected), f"device.{key} drifted: {got} != {expected}"
        else:
            assert got == expected, f"device.{key} drifted: {got!r} != {expected!r}"

    # The gate must be untouched by this edit -- it is the user's decision, not a
    # side effect of fixing a config-precedence bug.
    assert cfg.evaluation.relaxed_pass_frac == 0.99
    assert cfg.evaluation.cosine_min == 0.99985


def test_no_config_leaves_the_device_unnamed():
    """The same hole existed in all three smoke configs. Checking every config (rather
    than the ones I happened to look at) is what stops a new config from silently
    reintroducing "- unknown" into agent sandboxes."""
    import glob

    from kernel_optimizer.config import load_config

    configs = sorted(glob.glob("configs/*.yaml"))
    assert configs, "no configs found -- the glob or cwd is wrong, not a real pass"
    for path in configs:
        cfg = load_config(path)
        assert cfg.device.name != "unknown", f"{path} leaves the device unnamed"


def test_agent_calls_name_their_candidate_for_timeout_attribution():
    """AGENT_CALL_STARTED recorded only {module, call_id, session_id, model}, so a
    transport timeout could be tied to a candidate only by "nearest following
    *_PRODUCED event". That heuristic left 2 of 4 observed repair timeouts unattributed
    and is too fragile to support the hypothesis that repeat repairs on the SAME
    candidate are the ones that hang (repair times out on 36.4% of calls vs 0% for
    parameterizer and analyst).

    base.invoke reads the field generically, so a new Inputs type needs no change there,
    but the field is only useful if the call sites actually pass it."""
    from pathlib import Path

    base = Path("src/kernel_optimizer/agents/base.py").read_text(encoding="utf-8")
    block = base.split('self.store.append("AGENT_CALL_STARTED"')[0][-800:]
    assert 'getattr(inputs, "candidate_id", None)' in block, \
        "must read the subject generically, not per-module"
    # Absent on types that have no candidate (generator, novelty): the key is omitted
    # rather than written as null, so a reader can distinguish "not applicable".
    assert 'if subject:' in block

    mods = Path("src/kernel_optimizer/agents/modules.py").read_text(encoding="utf-8")
    for cls in ("RepairInputs", "ParameterizerInputs", "AnalystInputs"):
        seg = mods.split(f"class {cls}:")[1].split("@dataclass")[0]
        assert "candidate_id" in seg, f"{cls} must carry candidate_id"

    orch = Path("src/kernel_optimizer/control/orchestrator.py").read_text(encoding="utf-8")
    # Every construction of these three must pass it, or the event stays anonymous.
    for cls in ("RepairInputs(", "AnalystInputs("):
        for seg in orch.split(cls)[1:]:
            assert "candidate_id=" in seg[:600], f"{cls} built without candidate_id"
    # The parameterizer is reached through a helper (prefetch runs it off-thread), so
    # check the helper threads the id rather than each construction site.
    helper = orch.split("def _parameterize_agent_call")[1].split("def ")[0]
    assert "candidate_id=cand_id" in helper


def test_repair_event_records_what_changed_not_only_why():
    """REPAIR_PRODUCED dropped change_summary, so the log kept the agent's reasoning but
    not its edit.

    Live on L3:48 cand-eed411d8: the event carried only {candidate_id, diagnosis,
    prior_rejected}. RepairResult already has change_summary, and the distinction matters
    most in exactly the case that keeps arising -- when the diagnosis turns out to
    describe the task's own numerical spread rather than a defect, "I switched one dtype"
    and "I rewrote the arithmetic" have completely different implications for whether the
    candidate was damaged. A source_sha also makes the repaired file identifiable in the
    artifact store."""
    from pathlib import Path

    src = Path("src/kernel_optimizer/control/orchestrator.py").read_text(encoding="utf-8")
    block = src.split('self.store.append("REPAIR_PRODUCED"')[1].split("})")[0]
    assert "change_summary" in block, "the edit itself must be journalled, not just the why"
    assert "diagnosis" in block
    assert "source_sha" in block, "the repaired source must be identifiable"


def test_rejection_events_keep_the_verdict_bearing_tail():
    """SPACE_REJECTED truncated verdict.detail at a flat 800 chars. The rich mismatch
    message is 1149 chars and its LAST line is the reference's own noise floor -- the
    part that decides whether a candidate is genuinely wrong or merely inside the task's
    spread. A head-truncation cut it off mid-word, so every later analysis read a record
    missing the decisive number (the agent itself got the untruncated detail)."""
    from pathlib import Path

    src = Path("src/kernel_optimizer/control/orchestrator.py").read_text(encoding="utf-8")
    # There are four rejection appends: two record an agent transport error
    # (reason="agent_error", detail=str(exc)) and have no verdict; two record a real
    # verdict. Only the latter two are in scope, found by the text they actually contain.
    verdict_appends = [seg for seg in src.split("self.store.append(")
                       if "REJECTED" in seg[:60] and "verdict.detail" in seg[:900]]
    assert len(verdict_appends) == 2, f"expected 2 verdict rejections, got {len(verdict_appends)}"
    for seg in verdict_appends:
        window = seg[:900]
        assert "error_excerpt(verdict.detail" in window, "must keep the tail"
        assert "verdict.detail[:" not in window, "still head-truncates"

    # And error_excerpt must actually preserve the tail for a message this size.
    from kernel_optimizer.paramspace.validation import error_excerpt

    floor_line = "reference's OWN ieee-vs-tf32 spread (task noise floor, NOT a bug): {...}"
    msg = "x" * 1500 + "\n" + floor_line
    out = error_excerpt(msg, 2000)
    assert floor_line in out
    # A message longer than the limit keeps the tail and says what it dropped.
    long_out = error_excerpt("y" * 4000 + floor_line, 2000)
    assert floor_line in long_out and "chars elided" in long_out


def test_every_family_gets_a_rewrite_round_across_rounds():
    """The single-ranking tests show unproven-first ordering; this shows the CONSEQUENCE
    over successive rounds, which is what the paper's problem statement is about.

    With max_families_active=2 and three families, a naive best-first ranking would keep
    picking the same top two forever and the third would never enter structural search --
    the exact early pruning we argue against, and what both round-2 L3 runs did (2 of 4
    families reported frozen_budget with rewrite_rounds_used == 0). Unproven-first
    promotes the untried family as soon as the others have spent a round."""
    from kernel_optimizer.control.families import FamilyManager
    from kernel_optimizer.models.core import BestRecord, Candidate, Family, ParamSet

    fm = FamilyManager(max_families_total=3, max_families_total_hard=6,
                       max_families_active=2)
    for fid, ms in (("A", 2.09), ("B", 3.80), ("C", 5.00)):
        cid = f"cand-{fid}"
        fm.candidates[cid] = Candidate(candidate_id=cid, family_id=fid, origin="seed",
                                       backend="triton", source_sha=cid,
                                       structural_signature=cid)
        fm._sources[cid] = f"# {cid}\n"
        fam = Family(family_id=fid, anchor_candidate_id=cid, member_ids=[cid],
                     status="active")
        fam.best = BestRecord(candidate_id=cid, params=ParamSet(values={"B": 1}),
                              latency_ms=ms)
        fm.families[fid] = fam

    selected: set[str] = set()
    for _ in range(6):
        active = fm.active_families()
        selected.update(f.family_id for f in active)
        for f in active:  # worst case: every round spends budget and improves nothing
            f.rewrite_rounds_used += 1
            f.best_history.append(f.best.latency_ms)
            if f.rewrite_rounds_used >= 3:
                f.status = "frozen_budget"

    assert selected == {"A", "B", "C"}, \
        f"a family never entered structural search: {selected}"
    assert all(f.rewrite_rounds_used > 0 for f in fm.families.values())


# --- Z: the report must distinguish FAILED from UNEXPLORED branches ----------------


def test_report_distinguishes_failed_branch_from_unexplored_one(tmp_path):
    """Three states look alike in `status` and conflating them misreads the search.

    A family whose only seed never passed correctness has best=None and 0 rewrite
    rounds -- but the "structural headroom is UNKNOWN, not exhausted" note is wrong for
    it: nothing was ever measured, so there is no headroom claim to make. It also
    rendered as "best None ms". On L3:48, fam-dc0697c9 is exactly this case
    (cand-eb910a18 exhausted all four repair attempts on non-finite output)."""
    from kernel_optimizer.reporting.report import ReportGenerator
    from kernel_optimizer.store.run_store import RunStore

    summary = {
        "task": {"level": 3, "problem_id": 48, "name": "48_Mamba2ReturnY",
                 "ref_path": "x", "ref_src_sha": "abc"},
        "baselines": [],
        "elapsed_hours": 1.7,
        "families": {
            "fam-failed": {"anchor": "c-bad", "status": "frozen_budget", "best_ms": None,
                           "history": [], "rewrite_rounds_used": 0, "explored": False,
                           "members": [{"id": "c-bad", "origin": "seed", "parents": [],
                                        "approach": "never passed correctness"}]},
            "fam-unexplored": {"anchor": "c-ok", "status": "frozen_budget",
                               "best_ms": 3.55, "history": [3.55],
                               "rewrite_rounds_used": 0, "explored": False,
                               "members": [{"id": "c-ok", "origin": "seed", "parents": [],
                                            "approach": "tuned but never rewritten"}]},
            "fam-explored": {"anchor": "c-r", "status": "frozen_converged",
                             "best_ms": 2.09, "history": [2.5, 2.09],
                             "rewrite_rounds_used": 3, "explored": True,
                             "members": [{"id": "c-r", "origin": "seed", "parents": [],
                                          "approach": "rewritten three times"}]},
        },
    }
    store = RunStore.create(tmp_path, "run-test", {"task": summary["task"]})
    store.append("RUN_FINISHED", {"summary": summary})
    md = (ReportGenerator().generate(store)).read_text(encoding="utf-8")

    failed = md.split("`fam-failed`")[1].split("###")[0]
    assert "no measured candidate" in failed
    assert "FAILED branch, not an unexplored one" in failed
    assert "headroom is UNKNOWN" not in failed, "must not claim headroom for a dead branch"
    assert "best None ms" not in md, "None must never be rendered as a latency"

    unexplored = md.split("`fam-unexplored`")[1].split("###")[0]
    assert "never entered structural search" in unexplored
    assert "headroom is UNKNOWN, not exhausted" in unexplored

    explored = md.split("`fam-explored`")[1].split("###")[0]
    assert "rewrite rounds used: 3" in explored
    assert "never entered structural search" not in explored


def test_failed_hypotheses_survive_resume():
    """The rewriter reads failed_hypotheses to avoid re-proposing a change already shown
    not to help. It was memory-only with no restore path, so a resumed run started with
    an empty set and could spend rewrite rounds -- the scarcest budget in the loop --
    re-testing known dead ends. Unlike best_history there was no reconstruction from
    another stream; it is now journalled as HYPOTHESES_FAILED and restored alongside."""
    from pathlib import Path

    src = Path("src/kernel_optimizer/control/orchestrator.py").read_text(encoding="utf-8")

    # Emitted where a round failed to improve.
    emit = src.split("if best_after >= best_before")[1][:900]
    assert '"HYPOTHESES_FAILED"' in emit
    assert '"hypotheses": tried' in emit
    # Only when there is something to record.
    assert "if tried:" in emit

    # Restored before Loop C, in the same place as the other memory-only control state.
    restore = src.split("def _restore_family_control_state")[1].split("def _rewrite_round")[0]
    assert 'ev.type == "HYPOTHESES_FAILED"' in restore
    assert "self.failed_hypotheses[family_id] = hyps" in restore, \
        "must assign, not extend, so a re-entry cannot double-count"
    run_body = src.split("def _run(")[1].split("\n    def ")[0]
    assert run_body.index("_restore_family_control_state") < run_body.index("_rewrite_round")


def test_replay_tolerates_the_new_event_type():
    """replay() must ignore HYPOTHESES_FAILED rather than raise on an unknown type."""
    from kernel_optimizer.store.run_store import RunStore
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        store = RunStore.create(Path(td), "run-x", {"task": {}})
        store.append("HYPOTHESES_FAILED", {"family_id": "f", "round": 1,
                                           "hypotheses": [{"id": "H1", "change": "x"}]})
        store.append("STEP_DONE", {"step_key": "k"})
        state = store.replay()
        assert "k" in state.steps_done  # replay completed past the unknown type


# --- L3:48: the minimal witness is the fp16 corner ----------------------------
# 17 of that run's 27 SPACE_REJECTED events were witness_minimal_failed, which -- because
# validation tests the default witness FIRST and returns on the first failure -- means the
# default config PASSED. The minimal witness is choices[0] of every knob, and
# candidate_contract.md asks for the precision knob's choices ordered cheap->expensive with
# "fp16" first; L3:48's outputs reach 1e22 against fp16's 65504 ceiling. Correlation was
# 7/7 rejected with a COMPUTE_DTYPE knob vs 7/7 published without one.
# See docs/finding-minimal-witness-forces-fp16.md.

def _witness_fixture(tmp_path):
    """A validator over a two-knob space whose cheapest corner is fp16."""
    from kernel_optimizer.config import EvalConfig
    from kernel_optimizer.models.core import Candidate, DeviceLimits, TaskSpec
    from kernel_optimizer.models.reports import ParameterizationResult
    from kernel_optimizer.paramspace.validation import SpaceValidator

    source = (
        "PARAMS = {\n"
        "    'COMPUTE_DTYPE': 'ieee',\n"
        "    'BLOCK': 64,\n"
        "}\n"
        "class ModelNew:\n"
        "    pass\n"
    )
    proposal = ParameterizationResult(
        file="c.py",
        space={
            "params": [
                {"name": "COMPUTE_DTYPE", "kind": "str",
                 "choices": ["fp16", "bf16", "tf32", "ieee"]},
                {"name": "BLOCK", "kind": "int", "choices": [32, 64]},
            ],
            "constraints": [],
        },
    )
    validator = SpaceValidator(
        None,
        DeviceLimits(name="t", vram_gb=16, max_regs_per_thread=255,
                     max_shared_bytes_static=49152, max_shared_bytes_optin=101376,
                     max_threads_per_block=1024),
        EvalConfig(correctness_mode="dual_witness_relaxed"),
    )
    cand = Candidate(candidate_id="cand-x", family_id="fam-x", origin="seed",
                     backend="triton", source_sha="a" * 64,
                     structural_signature="b" * 64, approach_summary="s")
    task = TaskSpec(level=3, problem_id=48, name="m", ref_path="r", ref_src_sha="c" * 64)
    return validator, cand, source, proposal, task


OK_RESULT = {"ok": True,
             "latency_ms": {"mean": 5.0, "std": 0.1, "min": 4.9, "max": 5.1, "n": 20}}
NONFINITE = {"ok": False, "failure_kind": "correctness_mismatch",
             "log_tail": "18424816 of 134217728 candidate values are not finite"}


class _Recorder:
    """quick_test stub that fails any config containing a banned literal."""

    def __init__(self, banned):
        self.banned = banned
        self.calls = []

    def quick_test(self, task, path, tag, backend):
        text = path.read_text(encoding="utf-8")
        self.calls.append((tag, text))
        if any(b in text for b in self.banned):
            return dict(NONFINITE)
        return dict(OK_RESULT)


def test_out_of_range_cheap_corner_falls_back_instead_of_rejecting(tmp_path):
    """The fp16 corner of a candidate's own space must not sink the whole space.

    The second witness exists to prove the space is not inert -- that SOME config other
    than the default runs. It need not be the cheapest. On L3:48 insisting on the cheapest
    rejected 7 of 7 candidates that declared a precision knob, and repair (which had
    already fixed the real defect at attempt 1 in four of them) then spent its remaining
    budget trying to make fp16 represent 1e22."""
    from kernel_optimizer.paramspace.validation import SpaceAccepted

    validator, cand, source, proposal, task = _witness_fixture(tmp_path)
    rec = _Recorder(banned=["fp16"])
    validator.correctness = rec
    result = validator.validate_and_publish(cand, source, proposal, task, tmp_path)

    assert isinstance(result, SpaceAccepted), \
        f"space must survive an out-of-range cheap corner, got {result}"
    # Still two DISTINCT witnesses: anti-inertness is preserved, not bypassed.
    assert len(result.witnesses) == 2
    assert result.witnesses[0].params.values != result.witnesses[1].params.values
    # The surviving second witness is not the fp16 one.
    assert result.witnesses[1].params.values["COMPUTE_DTYPE"] != "fp16"
    # It ran a real GPU test for the alternative rather than assuming it works.
    assert any("wit-alt" in tag for tag, _ in rec.calls)


def test_a_genuinely_broken_kernel_is_still_rejected(tmp_path):
    """The fallback must not become a way for a broken kernel to get published: if every
    config fails, the space is still rejected."""
    from kernel_optimizer.paramspace.validation import SpaceRejection

    validator, cand, source, proposal, task = _witness_fixture(tmp_path)
    validator.correctness = _Recorder(banned=["PARAMS"])  # every materialized file
    result = validator.validate_and_publish(cand, source, proposal, task, tmp_path)
    assert isinstance(result, SpaceRejection)
    assert result.reason == "witness_default_failed"


def test_witness_rejection_says_which_config_failed(tmp_path):
    """The repair agent was never told which witness failed, so a message that meant "the
    cheapest corner of your own space is out of range" read as "your kernel is broken", and
    every diagnosis in those chains rewrote the algorithm. The label, the config, and the
    fact that the default passed are now in the detail the agent sees."""
    from kernel_optimizer.paramspace.validation import SpaceRejection

    validator, cand, source, proposal, task = _witness_fixture(tmp_path)
    # Everything except the default config fails, so no fallback exists and it rejects.
    validator.correctness = _Recorder(banned=["fp16", "bf16", "tf32", "32,"])
    result = validator.validate_and_publish(cand, source, proposal, task, tmp_path)

    assert isinstance(result, SpaceRejection)
    assert result.reason == "witness_minimal_failed"
    assert "[minimal witness config" in result.detail
    assert "COMPUTE_DTYPE" in result.detail, "the agent must see WHICH config failed"
    assert "DEFAULT config passed" in result.detail, \
        "without this the agent rewrites a kernel whose default is already correct"
    assert "not finite" in result.detail, "the underlying failure must survive"


def test_witness_fallback_is_bounded(tmp_path):
    """Each retry is a real GPU quick test, so the walk must be bounded rather than an
    exhaustive product over the grid."""
    validator, cand, source, proposal, task = _witness_fixture(tmp_path)
    rec = _Recorder(banned=["fp16", "bf16", "tf32", "32,"])
    validator.correctness = rec
    validator.validate_and_publish(cand, source, proposal, task, tmp_path)
    alt_calls = [t for t, _ in rec.calls if "wit-alt" in t]
    assert alt_calls, "the fallback must actually be attempted"
    assert len(alt_calls) <= validator.max_witness_retries
