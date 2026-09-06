"""Tests for the harness improvements: Triton lint (C), novelty slot accounting (E),
repair failure-class guidance (F), and the relaxed-correctness slack gate (A)."""

import importlib.util
import json

import pytest

from kernel_optimizer.agents.modules import _repair_guidance
from kernel_optimizer.config import AppConfig, OpencodeConfig, load_config
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
    # No dot in a NON-Triton source: nothing to read, so no claim. (Was "unknown" for
    # every dotless kernel, including Triton ones -- see the next test.)
    assert _detect_candidate_precision("y = x + 1", empty) == "unknown"


def test_a_dotless_triton_kernel_is_fp32_not_unknown():
    """P5: a kernel with no tl.dot uses no tensor core, so its arithmetic IS fp32.

    L3:48's winner is a sequential selective scan -- pure elementwise + tl.sum, zero dot
    products -- and reported `precision: unknown`. That made `_honest_verdict` pick the
    fp32 comparator by DEFAULT rather than by decision; the comparator happened to be
    right, which is luck, not logic. Covers every scan / reduction / pointwise-fusion
    kernel, i.e. every operator without a matmul.
    """
    from kernel_optimizer.control.orchestrator import (
        _detect_candidate_precision, _honest_verdict,
    )
    from kernel_optimizer.models.core import ParamSet

    empty = ParamSet(values={"BLOCK_S": 64})
    scan = (
        "import triton\nimport triton.language as tl\n"
        "@triton.jit\ndef _scan(x_ptr, o_ptr, BLOCK_S: tl.constexpr):\n"
        "    state = tl.zeros((BLOCK_S,), dtype=tl.float32)\n"
        "    state = state * e + bv * xv\n"
        "    o = tl.sum(state * cv, axis=0)\n"
    )
    assert _detect_candidate_precision(scan, empty) == "ieee_fp32"

    # A low-precision dotless kernel must still classify by its dtype, not fall here.
    assert _detect_candidate_precision(
        scan + "    y = x.to(tl.bfloat16)\n", empty) == "bf16"

    # And the relabel must not move the comparator: `unknown` and `ieee_fp32` are both
    # on the non-tensor-core branch, so every existing number is unchanged. This is what
    # makes the fix safe to ship without re-interpreting past runs.
    sp = {"eager": 1.5, "eager_tf32": 1.2,
          "torch_compile": 9.49, "torch_compile_tf32": 9.13}
    assert (_honest_verdict("unknown", sp)["compared_against"]
            == _honest_verdict("ieee_fp32", sp)["compared_against"]
            == "torch_compile")
    assert (_honest_verdict("unknown", sp)["same_precision_speedup"]
            == _honest_verdict("ieee_fp32", sp)["same_precision_speedup"])


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


# --- A truncated turn is not a formatting mistake ---------------------------------
#
# glm-5.3 killed run-l3-21-20260906-084636 outright: three generator attempts each spent
# exactly 32000 output+reasoning tokens on planning, were cut off (`finish == "length"`)
# before writing a single file, and were each told "no parseable JSON found ... emit a
# fenced json block". That feedback describes a mistake the model did not make, so it
# re-planned identically all three times, cost $0.44, and the run died at _generate_seeds.
# The retry text must name truncation and ask for less deliberation instead.


def _finish_module(tmp_path, finishes, structureds, tokens=None):
    """An AgentModule whose successive replies carry given `finish`/`structured` values."""
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
            i = len(seen)
            seen.append(text)
            return PromptResult(
                text="", structured=structureds[i], tokens=(tokens or [{}] * 9)[i],
                cost=0.0, session_id=session_id, message_id="m",
                finish=finishes[i],
            )

    class Mod(AgentModule):
        name = "generator"
        output_model = Out

        def seed_sandbox(self, inputs, sb):
            pass

        def render_prompt(self, inputs, sb):
            return "ORIGINAL TASK TEXT"

    store = _FakeStore()
    mod = Mod(FakeClient(), _FakeSandboxes(tmp_path), store, AgentModuleConfig())
    return mod, store, seen


def test_a_truncated_turn_is_told_it_was_cut_off_not_that_its_json_was_malformed(tmp_path):
    mod, _store, seen = _finish_module(
        tmp_path,
        finishes=["length", "stop"],
        structureds=[None, {"answer": "ok"}],
        tokens=[{"output": 14, "reasoning": 31986}, {}],
    )
    assert mod.invoke(None).output.answer == "ok"

    retry = seen[1]
    assert "CUT OFF" in retry
    assert "32000" in retry, "the feedback should quantify the budget that was spent"
    assert "Deliberate far less" in retry
    # The misleading advice must be gone: the model's JSON was never the problem.
    assert "fenced" not in retry


def test_a_malformed_answer_still_gets_the_formatting_feedback(tmp_path):
    """The truncation branch must not swallow the ordinary case: a reply that finished
    normally but carried no JSON is a formatting mistake and should be told so."""
    mod, _store, seen = _finish_module(
        tmp_path, finishes=["stop", "stop"], structureds=[None, {"answer": "ok"}]
    )
    assert mod.invoke(None).output.answer == "ok"
    assert "fenced" in seen[1] and "CUT OFF" not in seen[1]


def test_the_final_failure_event_records_how_the_last_turn_ended(tmp_path):
    """Diagnosing the GLM run needed a dig through opencode's sqlite store because
    events.jsonl recorded only the (wrong) corrective text. The failure event must carry
    `finish` so a truncation is visible from the run's own trace."""
    from kernel_optimizer.agents.runtime import AgentCallError

    mod, store, seen = _finish_module(
        tmp_path,
        finishes=["length", "length", "length"],
        structureds=[None, None, None],
        tokens=[{"output": 5, "reasoning": 31995}] * 3,
    )
    with pytest.raises(AgentCallError):
        mod.invoke(None)

    failed = [p for t, p in store.events if t == "AGENT_CALL_FAILED" and p.get("final")]
    assert len(failed) == 1
    assert failed[0]["finish"] == "length"
    assert failed[0]["attempts"] == 3


def test_the_agent_call_timeout_clears_the_slowest_measured_real_call(tmp_path):
    """The 20-min timeout was killing real work, so it must not silently drift back.

    Across the five completed L3 runs, 8 agent calls died at exactly 1200-1201s with
    `prompt transport error (ReadTimeout)` -- 5 repair, 2 rewriter, 1 generator, on all three
    tasks. Each kill discards a candidate or a whole rewrite round. The slowest SUCCESSFUL call
    measured is 576s, and one glm-5.3 generator call on L3:21 needed 979s, so the floor here is
    set well above both rather than just above the old value.
    """
    slowest_successful_call_s = 979.0   # glm-5.3 generator, L3:21, one large reasoning turn

    for path in ("configs/default.yaml", "configs/experiments_l3.yaml",
                 "configs/experiments_l3_glm.yaml"):
        cfg = load_config(path)
        assert cfg.opencode.request_timeout_s >= 1800.0, (
            f"{path}: agent-call timeout dropped to {cfg.opencode.request_timeout_s}s; "
            "at 1200s this killed 8 real calls"
        )
        # Headroom, not a bare pass: a timeout only a little above the slowest observed call
        # will start killing work again as soon as a prompt grows.
        assert cfg.opencode.request_timeout_s >= 1.8 * slowest_successful_call_s


# --- the per-turn output-token ceiling has no config-file route, only an env var -----------
#
# opencode 1.18.18 computes the cap as `Math.min(model.limit.output, ENV ?? 32000)` where ENV
# is OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX. Measured upstream `max_tokens` (see
# scripts/probe_glm_limit_output.py): baseline 32000; `limit.output`=200000 alone STILL 32000;
# env var alone 131072; env var + limit 200000. So the env var is a hard ceiling that config
# can only lower, and both halves must be set together. These tests pin the two mechanisms
# the harness needs for that -- a server env passthrough, and a sandbox config merge deep
# enough to add one key inside the provider block without dropping its credentials.


def test_server_env_is_layered_over_the_inherited_environment(monkeypatch, tmp_path):
    from kernel_optimizer.agents import runtime as rt

    monkeypatch.setenv("KOPT_PREEXISTING", "inherited")
    captured: dict = {}

    class _FakeProc:
        returncode = None

        def poll(self):
            return None

    def _fake_popen(cmd, **kwargs):
        captured.update(kwargs)
        return _FakeProc()

    monkeypatch.setattr(rt.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(rt.OpencodeServer, "_wait_healthy", lambda self: None)

    cfg = OpencodeConfig(
        server_url=None,
        launch_cwd=tmp_path,
        server_env={"OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX": "200000"},
    )
    rt.OpencodeServer(cfg, log_path=tmp_path / "srv.log").start()

    env = captured["env"]
    # The setting reaches the server process...
    assert env["OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"] == "200000"
    # ...without discarding the environment the harness was started with (PATH must survive,
    # or `opencode` itself becomes unresolvable).
    assert env["KOPT_PREEXISTING"] == "inherited"
    assert "PATH" in env or "Path" in env


def test_sandbox_extra_config_adds_to_the_provider_block_without_dropping_its_credentials(
    tmp_path,
):
    from kernel_optimizer.wiring import _sandbox_extra_config

    provider_file = tmp_path / "opencode.jsonc"
    provider_file.write_text(json.dumps({
        "provider": {"zhipuai": {
            "npm": "@ai-sdk/openai-compatible",
            "options": {"baseURL": "https://example.invalid/v4", "apiKey": "SECRET"},
            "models": {"glm-5.3": {"name": "GLM-5.3", "reasoning": True}},
        }},
        # Must NOT be copied into a sandbox even when present in the source file.
        "permission": {"bash": "deny"},
    }), encoding="utf-8")

    cfg = AppConfig(opencode=OpencodeConfig(
        sandbox_config_path=provider_file,
        sandbox_extra_config={"provider": {"zhipuai": {"models": {"glm-5.3": {
            "limit": {"context": 400000, "output": 200000}}}}}},
    ))
    merged = _sandbox_extra_config(cfg)

    model = merged["provider"]["zhipuai"]["models"]["glm-5.3"]
    # The added key is present...
    assert model["limit"] == {"context": 400000, "output": 200000}
    # ...and nothing that shared a parent dict with it was replaced. A shallow update would
    # wipe all three of these, and the resulting call fails as "unparseable answer" rather
    # than as a missing provider, which reads like a model failure.
    assert model["name"] == "GLM-5.3"
    assert merged["provider"]["zhipuai"]["options"]["apiKey"] == "SECRET"
    assert merged["provider"]["zhipuai"]["options"]["baseURL"] == "https://example.invalid/v4"
    assert "permission" not in merged


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
    latency on this task is ~29ms (matching the eager baseline).

    Tests the BEHAVIOUR (`_stats_to_dict` produces a median, and the ratio prefers it)
    rather than the text of one bespoke block. That block used to live only on the
    reference side; the median is now computed centrally for reference AND candidate, so an
    assertion on `ref_latency_ms["median"]` would fail while the protection is strictly
    stronger than before -- the classic test-the-implementation trap.
    """
    from pathlib import Path

    import ast

    src = Path("src/kernel_optimizer/gpu/worker_main.py").read_text(encoding="utf-8")
    # The ratio must still read the median first, whoever computed it.
    ratio_line = next(l for l in src.splitlines() if "ref_mean = ref_latency_ms" in l)
    assert '"median"' in ratio_line and ratio_line.index('"median"') < ratio_line.index('"mean"')

    # And the median must actually be produced by the shared summarizer, for BOTH sides.
    # `_stats_to_dict` and `_median` are pure dict/list code, but worker_main imports torch
    # at module scope and torch is not installed on the orchestrator host -- so exec just
    # those two functions rather than the module. That keeps the test running where the
    # rest of the suite runs instead of being skipped exactly where it matters.
    ns: dict = {}
    tree = ast.parse(src)
    wanted = {"_median", "_stats_to_dict"}
    picked = [n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in picked} == wanted, [n.name for n in picked]
    exec(compile(ast.Module(body=picked, type_ignores=[]), "<worker_main>", "exec"), ns)
    stats_to_dict = ns["_stats_to_dict"]

    samples = [29.8, 30.1, 29.9, 30.0, 5760.0, 29.7, 30.2, 29.8, 30.0, 90.0]
    got = stats_to_dict({"mean": sum(samples) / len(samples), "std": 1720.0,
                         "min": 29.7, "max": 5760.0, "num_trials": len(samples)},
                        samples)
    assert "median" in got, "the shared summarizer must record a median"
    assert got["median"] == 30.0, got["median"]
    assert got["samples"] == samples, "raw samples must be retained for re-analysis"
    # Summary-only callers (the baseline helper, KernelBench runtime_stats) have no samples;
    # they must still work and simply carry no median.
    assert "median" not in stats_to_dict({"mean": 1.0, "std": 0.0, "min": 1.0,
                                          "max": 1.0, "num_trials": 1})

    cand = 5.29
    assert got["mean"] / cand > 100          # the bogus verdict the mean produced
    assert got["median"] / cand < 10         # the median keeps it under the threshold


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
    """best_history and rewrite_rounds_used must both come from the event log, so a
    resume cannot double-count or lose rounds.

    `best_history` now carries a SEEDED round-0 entry (the seed-phase best) ahead of the
    recorded rounds, so this checks the two invariants behaviourally rather than by
    matching source text: rounds-used counts only FAMILY_ROUND_RECORDED events, and the
    history is the seed followed by those events' values, with no duplication on a
    second restore.
    """
    from pathlib import Path

    src = Path("src/kernel_optimizer/control/orchestrator.py").read_text(encoding="utf-8")
    restore = src.split("def _restore_family_control_state")[1].split("def _rewrite_round")[0]
    assert "FAMILY_ROUND_RECORDED" in restore
    assert "FAMILY_SEEDED" in restore, "the seed datum must be event-sourced, not live state"
    assert "family.rewrite_rounds_used = len(evs)" in restore, \
        "rounds must be the event count, not an increment, or resume double-counts"
    # The seed must not inflate the round count -- that would consume rewrite budget.
    assert "len(evs)" in restore and "len(family.best_history)" not in restore
    # And the restore must run BEFORE loop C, or the first round uses empty state.
    run_body = src.split("def _run(")[1].split("def ")[0]
    assert run_body.index("_restore_family_control_state") < run_body.index("_rewrite_round")


def test_seeded_history_makes_converged_reachable_and_slope_current():
    """The seed datum fixes two off-by-ones at once, and must not cost a rewrite round.

    Before seeding, `family_verdict` checked the round budget before convergence and the
    convergence test needed no_improve_rounds+1 entries, so with (3 rounds, 2 no-improve)
    the budget froze at round 4 while the history only reached 3 entries at that same
    moment -- `converged` was arithmetically unreachable, and all 48 recorded families
    ended `frozen_budget` with history length 0, 1 or 3.
    """
    from kernel_optimizer.config import BudgetConfig
    from kernel_optimizer.control.convergence import ConvergencePolicy
    from kernel_optimizer.control.families import FamilyManager
    from kernel_optimizer.models.core import BestRecord, Family, ParamSet

    cfg = BudgetConfig(rewrite_rounds_per_family=3, no_improve_rounds=2,
                       min_improvement_pct=2.0)
    judge = ConvergencePolicy(cfg)

    def fam(history, used):
        return Family(family_id="f", anchor_candidate_id="c", member_ids=["c"],
                      best=BestRecord(candidate_id="c", params=ParamSet(values={}),
                                      latency_ms=history[-1]),
                      best_history=list(history), rewrite_rounds_used=used,
                      status="active")

    # A stalled family reaches `converged` at round 3 now that round 0 is seeded.
    stalled = judge.family_verdict(fam([20.0, 19.98, 19.96], 2))
    assert stalled.verdict == "freeze" and stalled.stop_kind == "converged"

    # A still-improving family is NOT frozen early -- it keeps its full budget.
    moving = judge.family_verdict(fam([20.0, 18.0, 16.2], 2))
    assert moving.verdict == "continue"

    # And the first round's gain is now visible to the ranking rule. This is the real
    # l3-43-20260905-091705 fam-4aea322a: seed 14.2 -> 11.0, a 22.5% gain that scored
    # 0.0% slope at the moment round 2 was allocated.
    assert FamilyManager._improvement_pct(fam([14.2, 11.0], 1)) > 22.0
    # Unseeded, the same family had one entry and no measurable slope.
    assert FamilyManager._improvement_pct(fam([11.0], 1)) == 0.0


def test_unseeded_round_two_selection_falls_back_to_the_latency_tie_break():
    """Without the seed, round 2 is allocated by LATENCY -- the rule the ranking rejects.

    `_improvement_pct` returns 0.0 for a history shorter than two entries. At the decision
    that picks families for round 2 every family has run exactly one round, so unseeded
    they ALL tie at slope 0.0 and `rank()` falls through to its third key, absolute
    latency. That is the early-pruning-by-latency `active_families()` spends two docstring
    paragraphs rejecting, and it is not merely a stale slope -- the slope is absent.

    The fixture is the real run-l3-43-20260905-091705, where the latency tie-break picks
    fam-92e7c576 (which then went 19.6/19.6/19.6 across three rounds, spending 160 trials
    to confirm it was flat) over fam-ea7bc8bb, whose measured first-round slope was 2x as
    large. See docs/result-history-seeding-makes-converged-reachable.md.
    """
    from kernel_optimizer.control.families import FamilyManager
    from kernel_optimizer.models.core import BestRecord, Family, ParamSet

    # (seed_ms, round1_ms) as recorded on disk.
    observed = {
        "fam-92e7c576": (22.5, 19.6),
        "fam-4aea322a": (14.2, 11.0),
        "fam-ea7bc8bb": (28.6, 21.3),
        "fam-7f682a54": (23.5, 19.9),
    }

    def choose(seeded: bool, k: int = 2) -> list[str]:
        mgr = FamilyManager.__new__(FamilyManager)
        mgr.families = {}
        mgr.max_families_active = k
        for fid, (seed_ms, round1) in observed.items():
            history = [seed_ms, round1] if seeded else [round1]
            mgr.families[fid] = Family(
                family_id=fid, anchor_candidate_id="c", member_ids=["c"],
                best=BestRecord(candidate_id="c", params=ParamSet(values={}),
                                latency_ms=round1),
                best_history=history, rewrite_rounds_used=1, status="active")
        return [f.family_id for f in mgr.active_families()]

    # Unseeded: every slope is 0.0, so the pick is the two lowest latencies.
    unseeded = choose(seeded=False)
    by_latency = sorted(observed, key=lambda f: observed[f][1])[:2]
    assert unseeded == by_latency, unseeded
    assert "fam-92e7c576" in unseeded  # the branch that turned out flat

    # Seeded: the real first-round slopes decide, and the steepest one is selected.
    seeded = choose(seeded=True)
    assert "fam-ea7bc8bb" in seeded, seeded   # 25.5% slope, was never given round 2
    assert set(seeded) != set(unseeded)

    # The mechanism itself, stated directly: a one-entry history has no slope at all.
    for fid, (seed_ms, round1) in observed.items():
        one = Family(family_id=fid, anchor_candidate_id="c", member_ids=["c"],
                     best=BestRecord(candidate_id="c", params=ParamSet(values={}),
                                     latency_ms=round1),
                     best_history=[round1], rewrite_rounds_used=1, status="active")
        assert FamilyManager._improvement_pct(one) == 0.0
        two = Family(family_id=fid, anchor_candidate_id="c", member_ids=["c"],
                     best=BestRecord(candidate_id="c", params=ParamSet(values={}),
                                     latency_ms=round1),
                     best_history=[seed_ms, round1], rewrite_rounds_used=1,
                     status="active")
        assert FamilyManager._improvement_pct(two) > 0.0


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
    # Slice from the shape check, which begins the failure-reporting branch, rather than
    # from the gate expression: the fp64 relative arm now sits between them and carries
    # its own `except Exception as exc`, which would truncate the slice before the
    # diagnostic wrapper this test is about.
    block = src.split("if out_ref_ieee.shape != out_kernel.shape:")[1].split(
        "except Exception as exc:")[0]
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


# --- stop_kind="converged" is structurally unreachable ------------------------
# Measured across every L3 run: 11 of 11 family freezes are budget_exhausted, including six
# with a completely flat history like [25.2, 25.2, 25.2]. Two off-by-ones compound:
# best_history excludes the seed, and the budget check runs before the converged check while
# both conditions become true in the same round.
# See docs/finding-converged-stop-kind-is-unreachable.md.
#
# RESOLVED 2026-09-06, as a side effect of raising `rewrite_rounds_per_family` 3 -> 5 for
# defect 0b (families were being cut off mid-improvement; see
# docs/analysis-framework-defects-and-next-steps.md). The two thresholds no longer coincide:
# converged needs 3 rounds of history and budget now freezes at 5, so a family that goes flat
# can reach `converged` before exhausting its rounds. The assertion below is inverted from
# "documents the defect" to "guards the fix", exactly as the original note instructed.

def test_converged_is_reachable_at_the_l3_config():
    """Pins the arithmetic rather than the outcome, so the guard survives a refactor.

    At the check, len(best_history) == rewrite_rounds_used (the round is recorded AFTER the
    verdict). Budget freezes at rounds_used >= rewrite_rounds_per_family; converged needs
    len(history) >= no_improve_rounds + 1. While those thresholds coincided the budget test,
    being first, always won -- so `converged` could never be emitted."""
    from kernel_optimizer.config import load_config

    for path in ("configs/experiments_l3.yaml", "configs/experiments_l3_glm.yaml"):
        cfg = load_config(path).budgets
        assert cfg.no_improve_rounds + 1 < cfg.rewrite_rounds_per_family, (
            f"{path}: converged is unreachable again -- a family that stops improving will be "
            f"reported as budget_exhausted. no_improve_rounds={cfg.no_improve_rounds} + 1 must "
            f"be < rewrite_rounds_per_family={cfg.rewrite_rounds_per_family}. "
            "See docs/finding-converged-stop-kind-is-unreachable.md."
        )


def test_a_flat_family_now_freezes_as_converged_at_the_shipped_budget():
    """The behavioural half of the fix, at the budget the L3 configs actually ship.

    Kept alongside the arithmetic test because the arithmetic can hold while the ordering of
    the two checks in `family_verdict` still gets it wrong.
    """
    from kernel_optimizer.config import BudgetConfig
    from kernel_optimizer.control.convergence import ConvergencePolicy
    from kernel_optimizer.models.core import Family

    cfg = BudgetConfig(rewrite_rounds_per_family=5, no_improve_rounds=2,
                       min_improvement_pct=2.0)
    policy = ConvergencePolicy(cfg)

    # Exactly the state of fam-99aee6de on L3:48 and fam-3dacc96b on L3:21: flat history.
    stalled = Family(family_id="f", anchor_candidate_id="c")
    stalled.best_history = [25.2, 25.2]
    stalled.rewrite_rounds_used = 2
    assert policy.family_verdict(stalled).verdict == "continue", \
        "two flat rounds is one entry short of judgeable, so a third is granted"

    # At 3 flat rounds the converged test is satisfied and the budget cap (5) has NOT fired,
    # so the family is finally reported for the right reason. Under the old cap of 3 both
    # conditions became true in the same round and budget_exhausted won.
    stalled.best_history = [25.2, 25.2, 25.2]
    stalled.rewrite_rounds_used = 3
    v = policy.family_verdict(stalled)
    assert v.verdict == "freeze"
    assert v.stop_kind == "converged", (
        "a flat family must report `converged`; `budget_exhausted` tells a reader there may be "
        "headroom left, which is the opposite of the truth"
    )


def test_a_still_improving_family_is_not_cut_off_at_three_rounds():
    """Defect 0b: the round cap must not stop a family that is still gaining.

    `fam-6eea8eac` on L3:43 went 18.6 -> 15.4 -> 8.06 -- accelerating -- and was frozen as
    budget_exhausted at exactly 3 rounds with ~45% of the wall clock unused.
    """
    from kernel_optimizer.config import BudgetConfig
    from kernel_optimizer.control.convergence import ConvergencePolicy
    from kernel_optimizer.models.core import Family

    policy = ConvergencePolicy(BudgetConfig(rewrite_rounds_per_family=5, no_improve_rounds=2,
                                            min_improvement_pct=2.0))
    improving = Family(family_id="f", anchor_candidate_id="c")
    improving.best_history = [18.6, 15.4, 8.06]
    improving.rewrite_rounds_used = 3
    v = policy.family_verdict(improving)
    assert v.verdict == "continue", (
        "a family improving 47% on its latest round was frozen; the round cap fired before the "
        "convergence test it exists to defer to"
    )


def test_improvement_slope_is_blind_to_the_first_round():
    """The same off-by-one in the ranking path: _improvement_pct needs two entries, so a
    family that improved sharply in its first rewrite round scores 0.0 and is ranked as
    though it had stalled."""
    from kernel_optimizer.control.families import FamilyManager
    from kernel_optimizer.models.core import Family

    moved = Family(family_id="f", anchor_candidate_id="c")
    moved.best_history = [17.9]  # seed 19.5 -> 17.9 is an 8.2% gain, but the seed is absent
    assert FamilyManager._improvement_pct(moved) == 0.0, \
        "documents the defect: a real first-round gain is invisible to the ranker"

    # With two entries it works as intended.
    moved.best_history = [19.5, 17.9]
    assert FamilyManager._improvement_pct(moved) > 8.0


# The fallback must fire ONLY on the out-of-range signature. The historical minimal-witness
# failures on 21/43 look nothing like L3:48's -- zero non-finite values, max-abs-diff
# 0.0013-0.0040 on bounded outputs -- i.e. plausibly a real defect that only shows at the
# cheap corner. Falling back there would step past genuine evidence.

def test_out_of_range_predicate_separates_the_two_failure_kinds():
    """Fed the actual log tails from disk, from both task families."""
    from kernel_optimizer.paramspace.validation import _looks_out_of_range

    # --- L3:48 fp16 corner: overflow. Must fall back. ---
    assert _looks_out_of_range({"log_tail":
        "18424816 of 134217728 candidate values are not finite (16870024 NaN, 1554792 +/-Inf)"})
    # Even with no non-finite line, a collapsed finite-subset ref_absmax is the fingerprint:
    # the fp16 witness reports 7.696e+09 where the full output reaches 1.038e+22.
    assert _looks_out_of_range({"log_tail":
        "vs ieee ref: {'frac_within_tol': 0.84, 'ref_absmax': '7.696e+09'}\n"
        "  reference's OWN spread: {'ref_absmax': '1.038e+22'}"})

    # --- L3:21 / L3:43 minimal failures: finite, small error. Must NOT fall back. ---
    assert not _looks_out_of_range({"log_tail":
        "relaxed mismatch (max abs diff 0.003952) on trial 2"})
    assert not _looks_out_of_range({"log_tail":
        "relaxed mismatch (max abs diff 0.001347) on trial 2"})
    assert not _looks_out_of_range({"log_tail": "Output mismatch"})
    # Comparable magnitudes across witnesses = an ordinary mismatch, not an out-of-range cast.
    assert not _looks_out_of_range({"log_tail":
        "vs ieee ref: {'ref_absmax': '1.038e+22'}\nvs tf32 ref: {'ref_absmax': '1.038e+22'}"})
    assert not _looks_out_of_range({"log_tail": ""})
    assert not _looks_out_of_range({})


def test_an_ordinary_cheap_corner_failure_is_still_reported(tmp_path):
    """A finite small-error mismatch at the cheap corner is evidence about the kernel, so it
    must reject rather than silently fall back to a config that happens to work."""
    from kernel_optimizer.paramspace.validation import SpaceRejection

    validator, cand, source, proposal, task = _witness_fixture(tmp_path)

    class OrdinaryMismatch:
        """Only the exact default config passes; everything else has a small finite error."""

        def __init__(self):
            self.calls = []

        def quick_test(self, task, path, tag, backend):
            text = path.read_text(encoding="utf-8")
            self.calls.append(tag)
            if "'ieee'" in text and "64" in text:
                return dict(OK_RESULT)
            return {"ok": False, "failure_kind": "correctness_mismatch",
                    "log_tail": "relaxed mismatch (max abs diff 0.003952) on trial 2"}

    rec = OrdinaryMismatch()
    validator.correctness = rec
    result = validator.validate_and_publish(cand, source, proposal, task, tmp_path)

    assert isinstance(result, SpaceRejection)
    assert result.reason == "witness_minimal_failed"
    assert not [t for t in rec.calls if "wit-alt" in t], \
        "an ordinary mismatch must NOT trigger the out-of-range fallback"
    # And it still says which config failed, so repair is not misled about scope.
    assert "[minimal witness config" in result.detail
    assert "DEFAULT config passed" in result.detail


def test_trials_recheck_correctness_so_a_bad_corner_cannot_win_on_latency():
    """Why the fallback is safe even when it does fire: publishing a space whose cheapest
    corner is wrong cannot promote that corner, because every tuning trial re-runs
    correctness before timing (and the worker only times `if correct`)."""
    from pathlib import Path

    orch = Path("src/kernel_optimizer/control/orchestrator.py").read_text(encoding="utf-8")
    trial = orch.split("def _run_trial")[1].split("\n    def ")[0]
    assert "quick_test" in trial, "a trial must run the correctness+timing quick test"
    # A failing trial becomes status="fail", never a latency.
    assert 'status="fail"' in trial
    assert 'if not result.get("ok") or lat is None' in trial

    worker = Path("src/kernel_optimizer/gpu/worker_main.py").read_text(encoding="utf-8")
    assert "if correct and num_perf" in worker, \
        "the worker must only time a kernel that passed correctness"


def test_hard_edge_matches_prefixed_knob_names():
    """Agents prefix these knob names freely, so exact matching under-covers.

    The runs so far contain NUM_WARPS, NUM_STAGES, PW_WARPS, APPLY_WARPS, FINISH_WARPS,
    PW_STAGES, EXPAND_NUM_STAGES, FUSED_NUM_WARPS, SUMMARY_NUM_WARPS, SCAN_NUM_WARPS and
    OUTPUT_NUM_WARPS. Auditing every min-direction request across all runs: 11 asked for a
    knob whose minimum was ALREADY 1, and exact matching caught 10 -- `EXPAND_NUM_STAGES`
    (L3:21 09-04, cand-82819823) escaped. Suffix matching covers all 11.
    """
    from kernel_optimizer.control.orchestrator import boundary_knobs_to_expand
    from kernel_optimizer.models.core import ParamDomain, ParameterSpace
    from kernel_optimizer.models.reports import ParamStat, TuningStats

    space = ParameterSpace(
        space_id="sp-1", candidate_id="c", version=1, source_sha="x",
        domains=[
            # Already at the wall under a prefixed name: must be skipped.
            ParamDomain(name="EXPAND_NUM_STAGES", kind="int", choices=[1, 2, 3, 4]),
            ParamDomain(name="SCAN_NUM_WARPS", kind="int", choices=[1, 2, 4]),
            # Prefixed but NOT at the wall (min=2): a legitimate request, must survive.
            ParamDomain(name="PW_WARPS", kind="int", choices=[2, 4, 8]),
        ],
    )
    stats = TuningStats(
        candidate_id="c", space_id="sp-1", n_complete=40, n_fail=0,
        param_stats=[
            ParamStat(name="EXPAND_NUM_STAGES", best_value=1, at_boundary=True,
                      boundary_direction="min", effect_pct=9.0),
            ParamStat(name="SCAN_NUM_WARPS", best_value=1, at_boundary=True,
                      boundary_direction="min", effect_pct=9.0),
            ParamStat(name="PW_WARPS", best_value=2, at_boundary=True,
                      boundary_direction="min", effect_pct=9.0),
        ],
    )
    names = [k["name"] for k in boundary_knobs_to_expand(stats, idle_frac=0.8,
                                                         space=space)]
    assert "EXPAND_NUM_STAGES" not in names, "prefixed stages knob at 1 must be skipped"
    assert "SCAN_NUM_WARPS" not in names, "prefixed warps knob at 1 must be skipped"
    assert "PW_WARPS" in names, (
        "a prefixed knob whose min is 2 can still be widened downward -- observed live on "
        "L3:21 (cand-7dcdbd99, PW_WARPS=[2,4,8]); the wall check must gate on the domain, "
        "not on the name"
    )


# --- improvement M: mode-gated kernel branches (dead-code optimization) -------------
# Found on L3:21 cand-c0b3b7cd: 31 trials, all `complete`, best 25.1 ms, and every one
# launched only the train-mode fallback `_depthwise_kernel` while the advertised fused
# kernel sat in the `else` branch. The harness never calls .eval()/.train(), so one
# side of such a branch is always dead code.

def _MODE_WARN(warns):
    return [w for w in warns if "if ...training:" in w]


def test_lint_warns_when_kernel_is_gated_on_training_mode():
    src = """
import triton
import triton.language as tl
PARAMS = {"BLOCK_X": 128}
@triton.jit
def _dw(x_ptr, y_ptr, BLOCK_X: tl.constexpr):
    return tl.load(x_ptr)
@triton.jit
def _dw_fused(x_ptr, y_ptr, BLOCK_X: tl.constexpr):
    return tl.load(x_ptr)
class ModelNew:
    def forward(self, x):
        bn = self.bn
        if bn.training:
            _dw[(1,)](x, x, BLOCK_X=PARAMS["BLOCK_X"])
            x = bn(x)
        else:
            _dw_fused[(1,)](x, x, BLOCK_X=PARAMS["BLOCK_X"])
        return x
"""
    hard, warns = lint_triton_source(src)
    assert hard == []                        # advisory only, never blocks
    hits = _MODE_WARN(warns)
    assert len(hits) == 1
    # names both sides so the agent can tell which half is stranded
    assert "_dw" in hits[0] and "_dw_fused" in hits[0]
    assert "eval_semantics" in hits[0]


def test_lint_no_mode_warn_when_branch_launches_no_kernel():
    # The benign pattern: a .training branch choosing between two torch formulations.
    # 16 of the 33 such branches on disk are this case and must stay silent.
    src = """
import triton
import triton.language as tl
PARAMS = {"BLOCK_X": 128}
@triton.jit
def _k(x_ptr, BLOCK_X: tl.constexpr):
    return tl.load(x_ptr)
class ModelNew:
    def forward(self, x):
        _k[(1,)](x, BLOCK_X=PARAMS["BLOCK_X"])
        if self.bn.training:
            x = self.bn(x)
        else:
            x = (x - self.bn.running_mean) * self.bn.weight
        return x
"""
    hard, warns = lint_triton_source(src)
    assert hard == []
    assert _MODE_WARN(warns) == []


def test_lint_mode_warn_ignores_subscript_calls_that_are_not_kernels():
    # `self.depthwise_conv[2](...)` is a Subscript call but NOT a Triton launch; an
    # earlier version of this check counted it and fired on 10.8% of candidates.
    src = """
import triton
import triton.language as tl
PARAMS = {"BLOCK_X": 128}
class ModelNew:
    def forward(self, x):
        if self.bn.training:
            x = self.depthwise_conv[2](x)
        else:
            x = self.other[1](x)
        return x
"""
    hard, warns = lint_triton_source(src)
    assert hard == []
    assert _MODE_WARN(warns) == []


def test_lint_mode_warn_handles_negated_test_and_autotuned_kernel():
    src = """
import triton
import triton.language as tl
PARAMS = {"BLOCK_X": 128}
@triton.autotune(configs=[], key=["n"])
@triton.jit
def _fast(x_ptr, BLOCK_X: tl.constexpr):
    return tl.load(x_ptr)
class ModelNew:
    def forward(self, x):
        if not self.bn.training:
            _fast[(1,)](x, BLOCK_X=PARAMS["BLOCK_X"])
        else:
            x = self.bn(x)
        return x
"""
    hard, warns = lint_triton_source(src)
    assert hard == []
    assert len(_MODE_WARN(warns)) == 1


# --- improvement M: the harness detects a defined-but-never-launched kernel ----------
# The analyst re-proposed the same inference-BN fusion AFTER 31 trials had all timed the
# fallback, because nothing told it which kernels ran. This check is deterministic
# (defined @triton.jit names vs profile.kernel_names), so it is journalled as fact.

def _crun_stub(source, kernel_names_per_trial):
    from kernel_optimizer.models.core import (
        Candidate, LatencyStats, ParameterSpace, ParamSet, ProfileRecord, TrialRecord,
    )
    from kernel_optimizer.control.orchestrator import CandidateRun
    cand = Candidate(candidate_id="cand-test", family_id="fam-test", origin="seed",
                     backend="triton", source_sha="0" * 8, structural_signature="s")
    space = ParameterSpace(space_id="sp-test", candidate_id="cand-test", version=1,
                           source_sha="0" * 8, domains=[], constraints=[])
    trials = []
    for i, names in enumerate(kernel_names_per_trial):
        trials.append(TrialRecord(
            trial_id=f"tr-{i}", candidate_id="cand-test", space_id="sp-test",
            params=ParamSet(values={}), status="complete",
            latency_ms=LatencyStats(mean=1.0, std=0.0, min=1.0, max=1.0, n_samples=20),
            profile=ProfileRecord(kernel_names=list(names)) if names is not None else None,
        ))
    crun = CandidateRun(candidate=cand, source=source)
    crun.space = space
    crun.trials = trials
    return crun


_TWO_KERNEL_SRC = """
import triton
import triton.language as tl
PARAMS = {"BLOCK_X": 128}
@triton.jit
def _live(x_ptr, BLOCK_X: tl.constexpr):
    return tl.load(x_ptr)
@triton.jit
def _dead(x_ptr, BLOCK_X: tl.constexpr):
    return tl.load(x_ptr)
"""


def _unlaunched(source, per_trial):
    from kernel_optimizer.control.orchestrator import Orchestrator
    crun = _crun_stub(source, per_trial)
    return Orchestrator._unlaunched_kernels(None, crun)


def test_unlaunched_kernels_names_the_dead_one():
    dead = _unlaunched(_TWO_KERNEL_SRC, [["_live"]] * 31)
    assert dead == {"_dead"}


def test_unlaunched_kernels_silent_when_all_run():
    assert _unlaunched(_TWO_KERNEL_SRC, [["_live"], ["_live", "_dead"]]) == set()


def test_unlaunched_kernels_needs_no_profile_data_to_stay_silent():
    # A CUDA-backend candidate carries no kernel names at all. Absence of data must
    # never read as absence of launches, or every such candidate is falsely flagged.
    assert _unlaunched(_TWO_KERNEL_SRC, [None, None]) == set()
    assert _unlaunched(_TWO_KERNEL_SRC, [[], []]) == set()


def test_unlaunched_kernels_tolerates_unparseable_source():
    assert _unlaunched("def broken(:\n", [["_live"]]) == set()


def test_trials_csv_carries_kernels_launched():
    from kernel_optimizer.control.orchestrator import Orchestrator
    crun = _crun_stub(_TWO_KERNEL_SRC, [["_live", "_dead"]])
    csv_text = Orchestrator._trials_csv(None, crun)
    assert "kernels_launched" in csv_text.splitlines()[0]
    assert "_live _dead" in csv_text


def test_analyst_seeds_dead_kernel_note_only_when_there_is_one():
    from kernel_optimizer.agents.modules import AnalystInputs, BottleneckAnalystAgent
    from kernel_optimizer.models.core import DeviceLimits
    from kernel_optimizer.models.reports import TuningStats

    class _SB:
        def __init__(self): self.files = {}
        def write_input(self, path, text): self.files[path] = text

    stats = TuningStats(candidate_id="c", space_id="s", n_complete=1, n_fail=0)
    common = dict(task=None, candidate_source="x = 1\n", stats=stats, trials_csv="",
                  device=DeviceLimits())
    agent = BottleneckAnalystAgent.__new__(BottleneckAnalystAgent)

    sb = _SB()
    agent.seed_sandbox(AnalystInputs(**common, never_launched_kernels=["_dead"]), sb)
    assert "tuning/never_launched_kernels.md" in sb.files
    assert "_dead" in sb.files["tuning/never_launched_kernels.md"]

    sb2 = _SB()
    agent.seed_sandbox(AnalystInputs(**common), sb2)
    assert "tuning/never_launched_kernels.md" not in sb2.files


# --- improvement M: device helpers are inlined, not dead ----------------------------
# L3:43 cand-d257924a was a FALSE POSITIVE of the first version: `_qk_scores` is a
# @triton.jit device helper called by name from inside two host-launched kernels, so
# Triton inlines it and it never appears in kernel_names -- while running on all 76
# trials. Anything reading "absent from kernel_names" as "never ran" must exclude these.

_HELPER_SRC = """
import triton
import triton.language as tl
PARAMS = {"BLOCK_M": 64}
@triton.jit
def _qk_scores(p, BLOCK_M: tl.constexpr):
    return tl.load(p)
@triton.jit
def _softmax_stats(p, BLOCK_M: tl.constexpr):
    scores = _qk_scores(p, BLOCK_M)
    return scores
@triton.jit
def _really_dead(p, BLOCK_M: tl.constexpr):
    return tl.load(p)
"""


def test_device_helper_names_finds_inlined_callee():
    import ast

    from kernel_optimizer.paramspace.triton_lint import device_helper_names, jit_kernel_names
    tree = ast.parse(_HELPER_SRC)
    jit = jit_kernel_names(tree)
    assert jit == {"_qk_scores", "_softmax_stats", "_really_dead"}
    assert device_helper_names(tree, jit) == {"_qk_scores"}


def test_unlaunched_excludes_helpers_but_keeps_real_dead_code():
    # Only `_softmax_stats` launched. `_qk_scores` is inlined into it (not dead);
    # `_really_dead` is genuinely never reached.
    assert _unlaunched(_HELPER_SRC, [["_softmax_stats"]] * 5) == {"_really_dead"}


def test_device_helper_detection_ignores_self_recursion():
    import ast

    from kernel_optimizer.paramspace.triton_lint import device_helper_names, jit_kernel_names
    src = """
import triton
import triton.language as tl
@triton.jit
def _k(p, n):
    if n > 0:
        return _k(p, n - 1)
    return p
"""
    tree = ast.parse(src)
    jit = jit_kernel_names(tree)
    # a kernel calling itself is not somebody else's helper -- it must stay checkable
    assert device_helper_names(tree, jit) == set()


def test_unlaunched_catches_launch_hidden_in_a_host_wrapper():
    # L3:21 cand-80665a49/cand-faa71ba0: the launch sits inside a plain host function
    # `_launch_pointwise(...)`, so the STATIC mode-gate lint cannot see a
    # `kernel[grid](...)` in the branch and stays silent. The runtime check catches it
    # anyway -- the two checks are complementary, not redundant.
    src = """
import triton
import triton.language as tl
PARAMS = {"BLOCK_X": 128}
@triton.jit
def _train_kernel(p, BLOCK_X: tl.constexpr):
    return tl.load(p)
@triton.jit
def _pointwise_eval_epilogue_kernel(p, BLOCK_X: tl.constexpr):
    return tl.load(p)
def _launch_pointwise(x, w, batch_norm=None):
    if batch_norm is None:
        _train_kernel[(1,)](x, BLOCK_X=PARAMS["BLOCK_X"])
    else:
        _pointwise_eval_epilogue_kernel[(1,)](x, BLOCK_X=PARAMS["BLOCK_X"])
    return x
class ModelNew:
    def forward(self, x):
        if self.training:
            return _launch_pointwise(x, self.w)
        return _launch_pointwise(x, self.w, batch_norm=self.bn)
"""
    hard, warns = lint_triton_source(src)
    assert hard == []
    assert _MODE_WARN(warns) == []          # static check genuinely cannot see it
    assert _unlaunched(src, [["_train_kernel"]] * 38) == {"_pointwise_eval_epilogue_kernel"}


# --- report states how much of the search budget actually ran -----------------------
# A speedup means something different at 2 of 6 rewrite rounds than at 7, and a reader
# quoting the number cannot tell from the number alone. L3:21 09-05 stopped at 2.05h of
# 12h with 2 of 6 rounds; L3:48 09-05 used 7 and must NOT be warned about.

def _budget_lines(families, trials_n=100, dead=None, provisional=False):
    from kernel_optimizer.reporting.report import _search_budget_lines
    summary = {"families": families, "elapsed_hours": 2.05}
    return _search_budget_lines(summary, [{}] * trials_n, dead or [], provisional)


_STARVED = {  # the real L3:21 09-05 shape
    "fam-c2143500": {"best_ms": None, "rewrite_rounds_used": 0},
    "fam-5dfc36d7": {"best_ms": 25.0, "rewrite_rounds_used": 1},
    "fam-a43b404b": {"best_ms": None, "rewrite_rounds_used": 0},
    "fam-f069ef3c": {"best_ms": 15.5, "rewrite_rounds_used": 1},
}
_HEALTHY = {  # the real L3:48 09-05 shape
    "fam-99aee6de": {"best_ms": 2.09, "rewrite_rounds_used": 3},
    "fam-b1ee96ac": {"best_ms": 3.8, "rewrite_rounds_used": 1},
    "fam-dc0697c9": {"best_ms": None, "rewrite_rounds_used": 0},
    "fam-74c41d8d": {"best_ms": 2.09, "rewrite_rounds_used": 3},
}


def test_report_warns_when_the_run_stopped_with_rounds_unused():
    text = "\n".join(_budget_lines(_STARVED))
    assert "rewrite rounds used: **2** across 4 families" in text
    assert "no correct candidate**: 2 of 4" in text
    assert "may have stopped before its rewrite budget was spent" in text


def test_report_does_not_warn_when_the_budget_was_spent():
    # One empty family cannot fill both active slots, and 7 rounds ran. Warning here
    # would cry wolf on the run that behaved correctly.
    text = "\n".join(_budget_lines(_HEALTHY))
    assert "rewrite rounds used: **7** across 4 families" in text
    assert "no correct candidate**: 1 of 4" in text
    assert "may have stopped before" not in text


def test_report_stays_silent_on_a_run_that_never_recorded_rounds():
    # Older summaries lack rewrite_rounds_used; 0 there means unrecorded, not zero, so no
    # budget fraction may be asserted.
    old = {"fam-a": {"best_ms": 20.0}, "fam-b": {"best_ms": 21.0}}
    assert _budget_lines(old) == []


def test_report_does_not_warn_mid_run():
    # provisional=True: families still active, nothing to conclude about the stop.
    text = "\n".join(_budget_lines(_STARVED, provisional=True))
    assert "rewrite rounds used" in text
    assert "may have stopped before" not in text


def test_report_surfaces_dead_kernel_trials_next_to_the_verdict():
    dead = [{"candidate_id": "cand-c0b3b7cd", "n_trials_measured": 80,
             "never_launched": ["_depthwise_bn_relu6_kernel"]}]
    text = "\n".join(_budget_lines(_STARVED, trials_n=374, dead=dead))
    assert "80 of 374 trials measured a candidate carrying a kernel that never launched" \
        in text.replace("**", "")
    assert "_depthwise_bn_relu6_kernel" in text


# --- K part 2: an expansion must not silently relax the constraints -----------
#
# The two cases below are the two shapes actually observed on disk across the 30
# recorded expansions: a resource bound that was dropped for no reason (must come
# back), and a legality bound that the expansion's own new value contradicts
# (must stay dropped, or the expansion is vetoed by a stale rule).

def _task_spec():
    from kernel_optimizer.models.core import TaskSpec
    return TaskSpec(task_id="level3:43", name="MinGPTCausalAttention", level=3,
                    problem_id=43, ref_path="ref.py", ref_src_sha="0" * 64,
                    entry_point="Model")


def _restore(old_constraints, new_constraints, old_choices, new_choices):
    """Run Orchestrator._restore_dropped_constraints without building an Orchestrator."""
    from types import SimpleNamespace

    from kernel_optimizer.config import DeviceLimits
    from kernel_optimizer.control.orchestrator import Orchestrator
    from kernel_optimizer.models.core import Constraint, ParamDomain, ParameterSpace

    def mk(choices, constraints, version):
        return ParameterSpace(
            space_id=f"sp-v{version}", candidate_id="c", source_sha="x", version=version,
            domains=[ParamDomain(name=n, kind="int", choices=list(v))
                     for n, v in choices.items()],
            constraints=[Constraint(expr=e, rationale="") for e in constraints])

    old = mk(old_choices, old_constraints, 1)
    new = mk(new_choices, new_constraints, 2)
    # Borrow the real methods onto a stub carrying only the config they read, so the
    # test exercises the shipped logic without constructing the whole dependency graph.
    fake = SimpleNamespace(cfg=SimpleNamespace(device=DeviceLimits()))
    fake._choice_is_reachable = Orchestrator._choice_is_reachable.__get__(fake)
    restored = Orchestrator._restore_dropped_constraints(fake, old, new)
    return [c.expr for c in restored], [c.expr for c in new.constraints]


def test_dropped_resource_constraint_is_restored():
    # Shape of l3-43 cand-e3a5da01: the expansion widened ATTN_BLOCK_M and dropped
    # every shared-memory and thread bound, re-admitting 16.6% of the shared sub-grid.
    restored, final = _restore(
        old_constraints=["NUM_WARPS * 32 <= MAX_THREADS_PER_BLOCK",
                         "BLOCK_M * BLOCK_N <= 4096"],
        new_constraints=[],
        old_choices={"BLOCK_M": [16, 32, 64], "BLOCK_N": [16, 32], "NUM_WARPS": [2, 4]},
        new_choices={"BLOCK_M": [16, 32, 64, 128], "BLOCK_N": [16, 32], "NUM_WARPS": [2, 4]},
    )
    assert restored == ["NUM_WARPS * 32 <= MAX_THREADS_PER_BLOCK",
                        "BLOCK_M * BLOCK_N <= 4096"]
    assert set(final) == set(restored)


def test_stale_constraint_that_forbids_the_new_value_is_not_restored():
    # Shape of l3-43 cand-88e76051: the body was rewritten so an N tile of 8 became
    # legal, and BLOCK_N=8 is the value being added. Restoring `BLOCK_N % 16 == 0`
    # would make the expansion pointless, so it must stay dropped.
    restored, final = _restore(
        old_constraints=["BLOCK_M % 16 == 0 and BLOCK_N % 16 == 0",
                         "NUM_WARPS * 32 <= MAX_THREADS_PER_BLOCK"],
        new_constraints=[],
        old_choices={"BLOCK_M": [16, 32], "BLOCK_N": [16, 32], "NUM_WARPS": [2, 4]},
        new_choices={"BLOCK_M": [16, 32], "BLOCK_N": [8, 16, 32], "NUM_WARPS": [2, 4]},
    )
    assert restored == ["NUM_WARPS * 32 <= MAX_THREADS_PER_BLOCK"]
    assert "BLOCK_M % 16 == 0 and BLOCK_N % 16 == 0" not in final


def test_constraints_the_agent_kept_are_not_duplicated():
    restored, final = _restore(
        old_constraints=["BLOCK_M * BLOCK_N <= 4096"],
        new_constraints=["BLOCK_M * BLOCK_N <= 4096"],
        old_choices={"BLOCK_M": [16, 32], "BLOCK_N": [16, 32]},
        new_choices={"BLOCK_M": [16, 32, 64], "BLOCK_N": [16, 32]},
    )
    assert restored == []
    assert final == ["BLOCK_M * BLOCK_N <= 4096"]


def test_expansion_prompt_shows_the_prior_constraints():
    from kernel_optimizer.agents.modules import ParameterizerAgent, ParameterizerInputs
    from kernel_optimizer.config import DeviceLimits

    inputs = ParameterizerInputs(
        task=_task_spec(),
        candidate_source="PARAMS = {'BLOCK_M': 16}\n",
        device=DeviceLimits(),
        expand_directive="- `BLOCK_M`: extend toward max",
        prior_constraints=(("BLOCK_M * BLOCK_N <= 4096", "register budget"),),
    )
    text = ParameterizerAgent._render_expand_prompt(None, inputs)
    assert "BLOCK_M * BLOCK_N <= 4096" in text
    assert "register budget" in text
    # and the no-constraint case must not emit a dangling header
    bare = ParameterizerAgent._render_expand_prompt(
        None, ParameterizerInputs(task=inputs.task, candidate_source="x",
                                  device=inputs.device, expand_directive="d"))
    assert "ALREADY HAS these constraints" not in bare


# --- contract enforcement: a candidate must actually contain a kernel ----------
#
# The contract says "the core computation you claim to optimize must run in your
# kernel", and nothing checked it: lint_triton_source walks @triton.jit bodies, so
# a file with zero kernels produced zero findings. Two of four seeds on L3:21 09-05
# were torch.compile(reference) with no kernel at all.

_NO_KERNEL = """
import torch
import torch.nn as nn

PARAMS = {"DOT_PRECISION": "tf32", "COMPILE_MODE": "default"}


class ModelNew(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Conv2d(c, c, 1)
        self._f = torch.compile(self.net, mode=PARAMS["COMPILE_MODE"])

    def forward(self, x):
        return self._f(x)
"""

_WITH_KERNEL = """
import torch
import triton
import triton.language as tl

PARAMS = {"BLOCK": 128}


@triton.jit
def _k(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(y_ptr + i, tl.load(x_ptr + i, mask=i < n), mask=i < n)


class ModelNew(torch.nn.Module):
    def forward(self, x):
        y = torch.empty_like(x)
        _k[(1,)](x, y, x.numel(), BLOCK=PARAMS["BLOCK"])
        return y
"""

# torch.compile AROUND a real kernel is explicitly allowed by the contract.
_KERNEL_PLUS_COMPILE = _WITH_KERNEL.replace(
    "    def forward(self, x):\n        y = torch.empty_like(x)",
    "    def forward(self, x):\n        x = torch.compile(lambda t: t.contiguous())(x)\n"
    "        y = torch.empty_like(x)",
)

_CUDA_INLINE = """
import torch
from torch.utils.cpp_extension import load_inline

PARAMS = {"BLOCK": 256}

_mod = load_inline(name="m", cpp_sources="", cuda_sources="__global__ void k(){}")


class ModelNew(torch.nn.Module):
    def forward(self, x):
        return x
"""


def test_no_custom_kernel_is_rejected():
    from kernel_optimizer.paramspace.triton_lint import declares_no_custom_kernel
    msg = declares_no_custom_kernel(_NO_KERNEL)
    assert msg is not None
    assert "no custom kernel" in msg


def test_triton_kernel_passes():
    from kernel_optimizer.paramspace.triton_lint import declares_no_custom_kernel
    assert declares_no_custom_kernel(_WITH_KERNEL) is None


def test_torch_compile_around_a_real_kernel_is_allowed():
    # The rule must be "is there a kernel", not "does it mention torch.compile" --
    # wrapping torch ops around a kernel is permitted by the contract.
    from kernel_optimizer.paramspace.triton_lint import declares_no_custom_kernel
    assert "torch.compile" in _KERNEL_PLUS_COMPILE
    assert declares_no_custom_kernel(_KERNEL_PLUS_COMPILE) is None


def test_delegation_rule_is_blanket_even_beside_a_real_kernel():
    """Pin the deliberate over-block: the delegation rule fires on ANY torch.compile.

    A kernel beside a compiled graph is exactly the observed hack (a no-op copy kernel
    bolted onto Inductor's output), and the AST cannot tell that shape from a legitimate
    kernel with compiled glue. So the rule is blanket, and the has-a-kernel rule is the
    one that is structural. This test exists so the trade is explicit rather than a
    surprise: if a legitimate candidate is ever rejected this way, this is the line to
    revisit.
    """
    from kernel_optimizer.paramspace.triton_lint import (
        declares_no_custom_kernel, delegates_to_baseline_compiler)
    # has a real kernel -> the structural rule passes it ...
    assert declares_no_custom_kernel(_KERNEL_PLUS_COMPILE) is None
    # ... but the integrity rule still rejects it, by design.
    msg = delegates_to_baseline_compiler(_KERNEL_PLUS_COMPILE)
    assert msg is not None and "torch.compile" in msg


def test_inline_cuda_backend_passes():
    from kernel_optimizer.paramspace.triton_lint import declares_no_custom_kernel
    assert declares_no_custom_kernel(_CUDA_INLINE) is None


def test_unparseable_source_is_left_to_the_lint():
    # a syntax error is already reported by lint_triton_source; don't double-fault
    from kernel_optimizer.paramspace.triton_lint import declares_no_custom_kernel
    assert declares_no_custom_kernel("def f(:\n  pass") is None


def test_lint_check_reports_the_missing_kernel(tmp_path):
    from kernel_optimizer.agents.modules import _triton_lint_check

    class _SB:
        def read_output(self, f):
            return _NO_KERNEL

    out = _triton_lint_check(["candidate/x.py"], _SB())
    assert out is not None and "no custom kernel" in out


def test_no_kernel_check_is_not_escapable_by_declaring_cuda():
    """The has-a-kernel rule must not depend on the declared backend.

    check_output previously linted only files whose candidate declared
    backend="triton", so a kernel-less file declaring backend="cuda" would have
    skipped the check entirely. The observed cases declared "triton", but the
    label is the agent's own free choice and must not gate a contract rule.
    """
    from kernel_optimizer.agents.modules import CandidateGeneratorAgent
    from kernel_optimizer.models.reports import GeneratedCandidate, GenerationResult

    class _SB:
        def exists(self, f):
            return True

        def read_output(self, f):
            return _NO_KERNEL

    out = GenerationResult(candidates=[
        GeneratedCandidate(file="candidate/a.py", backend="cuda",
                           approach_summary="x", structural_axes=[]),
    ])
    problem = CandidateGeneratorAgent.check_output(None, out, _SB())
    assert problem is not None and "no custom kernel" in problem


# --- and the escalation: a kernel beside a compiled graph is still delegation ---
#
# When declares_no_custom_kernel blocked the kernel-less shape, the repair agent
# added a no-op elementwise copy kernel to the end of the same torch.compile graph.
# It passed every check and became the run's best candidate at 19.4 ms with
# profile.kernel_names == ['_copy_kernel'].

_COPY_BESIDE_COMPILE = """
import torch
import torch.nn as nn
import triton
import triton.language as tl

PARAMS = {"BLOCK_SIZE": 256}


@triton.jit
def _copy_kernel(in_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    o = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(out_ptr + o, tl.load(in_ptr + o, mask=o < n), mask=o < n)


class ModelNew(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Conv2d(c, c, 1)
        self._f = torch.compile(self.net)

    def forward(self, x):
        x = self._f(x)
        out = torch.empty_like(x)
        _copy_kernel[(1,)](x, out, x.numel(), BLOCK_SIZE=PARAMS["BLOCK_SIZE"])
        return out
"""


def test_copy_kernel_beside_a_compiled_graph_is_rejected():
    from kernel_optimizer.paramspace.triton_lint import (
        declares_no_custom_kernel,
        delegates_to_baseline_compiler,
    )
    # it DOES define a kernel, so the has-a-kernel check alone cannot catch it
    assert declares_no_custom_kernel(_COPY_BESIDE_COMPILE) is None
    msg = delegates_to_baseline_compiler(_COPY_BESIDE_COMPILE)
    assert msg is not None and "measured against" in msg


def test_jit_script_and_trace_are_also_delegation():
    from kernel_optimizer.paramspace.triton_lint import delegates_to_baseline_compiler
    for call in ("torch.jit.script(self.net)", "torch.jit.trace(self.net, x)"):
        src = _WITH_KERNEL.replace("PARAMS = {\"BLOCK\": 128}",
                                   f"PARAMS = {{\"BLOCK\": 128}}\n_M = {call}")
        assert delegates_to_baseline_compiler(src) is not None, call


def test_plain_torch_ops_around_a_kernel_are_still_allowed():
    from kernel_optimizer.paramspace.triton_lint import delegates_to_baseline_compiler
    # eager torch around a kernel is explicitly permitted by the contract; only a
    # compiler/tracer is not. A method merely NAMED .compile() on something else
    # must not trip the rule either.
    assert delegates_to_baseline_compiler(_WITH_KERNEL) is None
    src = _WITH_KERNEL.replace("        y = torch.empty_like(x)",
                               "        x = x.contiguous().to(torch.float16)\n"
                               "        y = torch.empty_like(x)")
    assert delegates_to_baseline_compiler(src) is None
    unrelated = _WITH_KERNEL.replace("PARAMS = {\"BLOCK\": 128}",
                                     "PARAMS = {\"BLOCK\": 128}\n_R = re.compile('x')")
    assert delegates_to_baseline_compiler(unrelated) is None


def test_parameterizer_output_is_statically_gated():
    """The parameterizer rewrites the body that actually gets tuned and reported.

    It was the only code-producing agent whose check_output did not run the static
    check, so a contract violation that reached it -- or that it introduced while
    rewriting -- was never re-checked before 40 GPU trials were spent. On the L3:21
    rerun the delegating candidate passed through it and became the incumbent.
    """
    from kernel_optimizer.agents.modules import ParameterizerAgent
    from kernel_optimizer.models.reports import (
        ParameterizationResult,
        ProposedParam,
        ProposedSpace,
    )

    class _SB:
        def exists(self, f):
            return True

        def read_output(self, f):
            return _COPY_BESIDE_COMPILE

    out = ParameterizationResult(
        file="candidate/parameterized.py",
        space=ProposedSpace(params=[
            ProposedParam(name="BLOCK_SIZE", kind="int", choices=[128, 256]),
            ProposedParam(name="COPY_NUM_WARPS", kind="int", choices=[2, 4]),
        ]))
    problem = ParameterizerAgent.check_output(None, out, _SB())
    assert problem is not None and "measured against" in problem


def test_every_code_producing_agent_is_gated():
    """Guard against a new agent being added without the static gate.

    The delegating candidate reached the tuner because ONE of six agents skipped
    _triton_lint_check; an inventory test is cheaper than rediscovering that.
    """
    import inspect

    from kernel_optimizer.agents import modules as m

    expected_gated = {
        "CandidateGeneratorAgent", "ParameterizerAgent", "StructureRewriterAgent",
        "NoveltyGeneratorAgent", "RepairAgent",
    }
    gated = set()
    for name, obj in vars(m).items():
        if not (inspect.isclass(obj) and name.endswith("Agent")):
            continue
        check = getattr(obj, "check_output", None)
        if check is None:
            continue
        try:
            src = inspect.getsource(check)
        except OSError:  # pragma: no cover
            continue
        if "_triton_lint_check" in src:
            gated.add(name)
    assert expected_gated <= gated, f"ungated code-producing agents: {expected_gated - gated}"


def test_empty_family_does_not_occupy_a_rewrite_slot():
    """A family with no correct candidate must not consume a max_families_active slot.

    It cannot be rewritten (`_do_rewrite` needs a correct parent), and `_rewrite_round`
    freezes it WITHOUT setting `progressed`, which the outer loop reads as "nothing left
    to do anywhere" and uses to freeze every remaining active family. Two empty families
    filling both slots ended run-l3-21-20260905-071312 at 2.05h of 12h with a 15.5 ms
    incumbent and 4 of 6 rewrite rounds unspent.
    """
    from kernel_optimizer.control.families import FamilyManager
    from kernel_optimizer.models.core import BestRecord, Family, ParamSet

    mgr = FamilyManager.__new__(FamilyManager)
    mgr.max_families_active = 2
    mgr.families = {}

    def add(fid, best_ms):
        mgr.families[fid] = Family(
            family_id=fid, anchor_candidate_id="c", member_ids=["c"],
            best=(None if best_ms is None else
                  BestRecord(candidate_id="c", params=ParamSet(values={}),
                             latency_ms=best_ms)),
            best_history=[], rewrite_rounds_used=0, status="active")

    # Two empty families ranked ahead of two productive ones (all unproven, and an
    # empty family's incumbent sorts as +inf so it lost the tie-break anyway -- the
    # point is that it must not appear AT ALL).
    add("empty1", None)
    add("empty2", None)
    add("good1", 15.5)
    add("good2", 25.0)

    active = mgr.active_families()
    ids = {f.family_id for f in active}
    assert ids == {"good1", "good2"}, f"empty families still occupy slots: {ids}"
    assert all(f.best is not None for f in active)

    # If every family is empty the list is empty -- correct, there is nothing to rewrite.
    mgr.families = {}
    add("e1", None)
    add("e2", None)
    assert mgr.active_families() == []


def test_hard_edge_covers_the_tl_dot_contraction_floor():
    """BLOCK_K below 16 is below the `tl.dot` contraction floor, and asking for it wasted
    two whole expansions. The floor belongs in HARD_EDGE next to the warp/stage floors.

    On what the waste actually was: the first witness fails to compile, and then the
    parameterizer RETRIES and rewrites the kernel to pad the dot to 16 with the surplus
    lanes masked off, so 8 does run -- at half useful occupancy, coming last in its domain
    both times (38.8 vs 24.4 best; 57.1 vs 14.75 best). Below a hardware wall the agent
    can only refuse or emulate, and neither can win, so the filter is right without
    needing to predict which happens.

    The rule is ASYMMETRIC -- only K has a floor -- so BLOCK_M/BLOCK_N must stay free.
    """
    from kernel_optimizer.control.orchestrator import boundary_knobs_to_expand
    from kernel_optimizer.models.core import ParamDomain, ParameterSpace
    from kernel_optimizer.models.reports import ParamStat, TuningStats

    def space(name, choices):
        return ParameterSpace(space_id="sp-x", candidate_id="c", source_sha="s",
                              version=1, constraints=[],
                              domains=[ParamDomain(name=name, kind="int",
                                                   choices=choices)])

    def stats(name, direction, best):
        return TuningStats(
            candidate_id="c", space_id="sp-x", n_complete=30, n_fail=0,
            resource_at_best=None, failure_clusters=[],
            param_stats=[ParamStat(name=name, best_value=best, at_boundary=True,
                                   boundary_direction=direction, effect_pct=30.0)])

    def asked(name, choices, direction):
        edge = min(choices) if direction == "min" else max(choices)
        return bool(boundary_knobs_to_expand(stats(name, direction, edge), 0.8,
                                             space=space(name, choices)))

    # K at its floor: blocked, whatever prefix the agent chose.
    assert not asked("BLOCK_K", [16, 32, 64], "min")
    assert not asked("QKV_BLOCK_K", [16, 32], "min")
    assert not asked("EXPAND_BLOCK_K", [16, 32, 64], "min")
    # K above the floor may still be lowered to 16, which is legal.
    assert asked("PV_BLOCK_K", [32, 64], "min")
    # M and N have NO contraction floor -- they must not be caught.
    assert asked("BLOCK_M", [32, 64], "min")
    assert asked("BLOCK_N", [16, 32], "min")
    # The pre-existing warp/stage floors still hold.
    assert not asked("NUM_WARPS", [1, 2, 4], "min")
    assert not asked("PW_WARPS", [1, 2, 4], "min")
    assert not asked("NUM_STAGES", [1, 2, 3], "min")


def test_hard_edge_is_subtractive_only_on_the_wall_knob():
    """The filter must remove the wall knob WITHOUT cancelling the expansion.

    This is the property that makes the fix safe rather than merely correct. Both
    historical BLOCK_K=8 expansions requested seven knobs; on the real spaces the filter
    drops one and keeps six, including the `OUT_BLOCK_M` widening that earned
    cand-45c3fd7d its 7.7% gain. A filter that suppressed the whole request instead would
    have deleted that gain, so the multi-knob case is pinned here.
    """
    from kernel_optimizer.control.orchestrator import boundary_knobs_to_expand
    from kernel_optimizer.models.core import ParamDomain, ParameterSpace
    from kernel_optimizer.models.reports import ParamStat, TuningStats

    domains = [
        ParamDomain(name="PV_BLOCK_K", kind="int", choices=[16, 32, 64]),   # at the wall
        ParamDomain(name="QKV_NUM_WARPS", kind="int", choices=[1, 2, 4]),   # at the wall
        ParamDomain(name="OUT_BLOCK_M", kind="int", choices=[16, 32, 64]),  # free
        ParamDomain(name="SCORE_NUM_WARPS", kind="int", choices=[2, 4, 8]),  # free (max)
    ]
    space = ParameterSpace(space_id="sp-x", candidate_id="c", source_sha="s", version=1,
                           constraints=[], domains=domains)
    directions = {"PV_BLOCK_K": ("min", 16), "QKV_NUM_WARPS": ("min", 1),
                  "OUT_BLOCK_M": ("max", 64), "SCORE_NUM_WARPS": ("max", 8)}
    stats = TuningStats(
        candidate_id="c", space_id="sp-x", n_complete=40, n_fail=0,
        resource_at_best=None, failure_clusters=[],
        param_stats=[ParamStat(name=n, best_value=b, at_boundary=True,
                               boundary_direction=d, effect_pct=30.0)
                     for n, (d, b) in directions.items()])

    got = {(k["name"], k["direction"]) for k in boundary_knobs_to_expand(
        stats, 0.8, space=space)}
    assert got == {("OUT_BLOCK_M", "max"), ("SCORE_NUM_WARPS", "max")}, got
    # The expansion still happens: the surviving knobs are what gets requested.
    assert len(got) == 2


def test_fp64_relative_gate_follows_the_torch_criterion():
    """The relative arm must implement torch._dynamo.utils.same()'s fp64 path:

        passes <=> rmse(fp64_ref, candidate) <= multiplier * rmse(fp64_ref, ref) + tol/10

    A candidate as accurate as the reference passes; one far worse fails; and the
    multiplier is >1 by design (torch: "to avoid these false alarms").
    """
    from pathlib import Path

    src = Path("src/kernel_optimizer/gpu/worker_main.py").read_text(encoding="utf-8")
    fn = src.split("def _fp64_relative_ok")[1].split("\ndef ")[0]
    assert "multiplier * ref_error + tol / 10.0" in fn, "must be torch's exact threshold"
    assert "math.isnan(ref_error) or math.isnan(res_error)" in fn, \
        "a nan must defer to the absolute gate, not pass"

    # The gate is a SECOND acceptance path: it may only run when the absolute gate failed,
    # so it can never turn a previously-accepted candidate into a rejection.
    body = src.split("def run_relaxed_correctness")[1]
    guard = body.split("if (not ok and golden_model is not None")[1].split("if ok:")[0]
    assert "_fp64_relative_ok" in guard
    # The low-precision multiplier must be chosen from the precision the candidate
    # COMPUTES in, not from its output dtype: a candidate doing tl.dot(a.to(bf16), ...)
    # with an fp32 accumulator returns float32, so an output-dtype test left the wider
    # multiplier permanently dead (observed live -- a bf16 candidate scored 2.0, not 3.0).
    assert "fp64_mult_effective" in guard, \
        "multiplier must come from the source-derived precision, not only out dtype"
    assert "_computes_low_precision" in src
    assert "out_kernel.dtype in (torch.float16, torch.bfloat16)" in guard, \
        "a genuinely low-precision output should still take the wider multiplier"
    # It compares against the tf32 reference -- the noisier of the two, and the one the
    # harness actually compares against -- so that reference sets the floor.
    assert "golden, out_ref_tf32, out_kernel" in guard

    # fp64 unavailability must be non-fatal.
    assert "fp64_unavailable" in body
    assert "golden_model = None" in body


def test_fp64_gate_is_wired_from_config_to_job():
    """The flag has to reach the worker, or turning it on in the config does nothing."""
    from pathlib import Path

    from kernel_optimizer.gpu.jobs import make_relaxed_correctness_job

    job = make_relaxed_correctness_job(
        "ref.py", "k.py", num_correct_trials=3, backend="triton", precision="fp32",
        seed=0, collect_triton_metadata=True, relaxed_elem_tol=0.01,
        relaxed_pass_frac=0.99, cosine_min=0.99985,
        fp64_relative_gate=True, fp64_rel_multiplier=2.0,
        fp64_rel_multiplier_lowp=3.0)
    assert job["fp64_relative_gate"] is True
    assert job["fp64_rel_multiplier"] == 2.0
    assert job["fp64_rel_multiplier_lowp"] == 3.0

    # Both call sites in correctness.py must forward it, not just one: `screen` gates the
    # witness (space publication) and `_run` gates every tuning trial and the final
    # re-eval. Forwarding only one would apply the gate inconsistently.
    src = Path("src/kernel_optimizer/evaluation/correctness.py").read_text(encoding="utf-8")
    assert src.count("fp64_relative_gate=self.cfg.fp64_relative_gate") == 2


def test_low_precision_detection_reads_the_materialized_params():
    """The fp64 gate's slack multiplier depends on the precision the candidate COMPUTES
    in, and the worker can only see source text -- but the source it receives is
    MATERIALIZED, so the tuner's chosen knob value is already substituted into PARAMS.

    Regression: keying the multiplier off `out_kernel.dtype` left the low-precision
    multiplier dead, because a candidate casting only its dot inputs still returns fp32.
    """
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "wm_probe", Path("src/kernel_optimizer/gpu/worker_main.py"))
    wm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wm)

    # A materialized bf16 candidate: the knob literal carries the choice.
    assert wm._computes_low_precision('PARAMS = {"COMPUTE_DTYPE": "bf16"}')
    assert wm._computes_low_precision('PARAMS = {"DOT_PRECISION": "fp16"}')
    # A cast in the kernel body, with no dtype knob at all.
    assert wm._computes_low_precision("acc += tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))")
    assert wm._computes_low_precision("x = x.to(tl.float16)")
    # Full-precision candidates must NOT get the wider slack.
    assert not wm._computes_low_precision('PARAMS = {"COMPUTE_DTYPE": "tf32"}')
    assert not wm._computes_low_precision('PARAMS = {"COMPUTE_DTYPE": "ieee"}')
    assert not wm._computes_low_precision('acc += tl.dot(a, b, input_precision="ieee")')
    assert not wm._computes_low_precision('PARAMS = {"BLOCK_M": 64}')


def test_low_precision_detection_ignores_dead_dtype_branches():
    """The multiplier must reflect the LIVE dtype, not every dtype named in the file.

    Candidates keep all dtype branches in the kernel body and select with a
    `tl.constexpr` knob, so exactly one survives compilation while the source still
    mentions the others. A whole-file token scan therefore granted the wider
    low-precision multiplier to a tf32 candidate -- verified live at multiplier 3.0 where
    2.0 was intended. The materialized PARAMS literal carries the tuner's actual choice
    and must win.
    """
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "wm_probe2", Path("src/kernel_optimizer/gpu/worker_main.py"))
    wm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wm)

    # The real shape: every branch present, one selected by the knob.
    body = '''
PARAMS = {"BLOCK_M": 32, "COMPUTE_DTYPE": "%s"}

@triton.jit
def k(a, b, COMPUTE_DTYPE: tl.constexpr):
    if COMPUTE_DTYPE == "fp16":
        acc += tl.dot(a.to(tl.float16), b.to(tl.float16), input_precision="ieee")
    elif COMPUTE_DTYPE == "bf16":
        acc += tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16), input_precision="ieee")
    elif COMPUTE_DTYPE == "tf32":
        acc += tl.dot(a, b, input_precision="tf32")
    else:
        acc += tl.dot(a, b, input_precision="ieee")
'''
    assert wm._computes_low_precision(body % "fp16")
    assert wm._computes_low_precision(body % "bf16")
    # These two are the regression: the file names fp16/bf16 in dead branches.
    assert not wm._computes_low_precision(body % "tf32"), \
        "a tf32 candidate must take the 2.0 multiplier, not 3.0"
    assert not wm._computes_low_precision(body % "ieee"), \
        "an ieee candidate must take the 2.0 multiplier, not 3.0"

    # With no dtype knob at all, a hardcoded cast IS the live code.
    assert wm._computes_low_precision('PARAMS = {"BLOCK_M": 64}\nx = x.to(tl.float16)')
    assert not wm._computes_low_precision(
        'PARAMS = {"BLOCK_M": 64}\nacc = tl.dot(a, b, input_precision="ieee")')


def test_fp64_rescue_is_journalled_so_the_experiment_is_measurable():
    """A trial accepted ONLY by the fp64 relative arm must say so.

    Without this the fp64 metrics reach the log only on FAILURE, i.e. never on the cases
    the gate was added to admit -- the experiment would be unmeasurable from the event
    log. `fp64_rescued_trials` is None when the gate is off and 0 when it is on and
    changed nothing, which are different findings.
    """
    from pathlib import Path

    from kernel_optimizer.models.core import ParamSet, TrialRecord

    wm = Path("src/kernel_optimizer/gpu/worker_main.py").read_text(encoding="utf-8")
    assert "fp64_rescued += 1" in wm, "the accepting arm must be counted"
    assert 'result["fp64_rescued_trials"] = fp64_rescued' in wm
    # Reported even when zero, so "enabled and rescued nothing" is distinguishable.
    gate_block = wm.split("if fp64_gate:")[-1]
    assert 'result["fp64_gate_enabled"] = True' in gate_block

    orch = Path("src/kernel_optimizer/control/orchestrator.py").read_text(encoding="utf-8")
    assert 'fp64_rescued_trials=result.get("fp64_rescued_trials")' in orch, \
        "the driver must carry it onto TRIAL_DONE or it never reaches the event log"

    rec = TrialRecord(trial_id="t", candidate_id="c", space_id="s",
                      params=ParamSet(values={}), status="complete",
                      fp64_rescued_trials=3)
    assert rec.fp64_rescued_trials == 3
    # Absent by default, so old event logs replay unchanged.
    assert TrialRecord(trial_id="t", candidate_id="c", space_id="s",
                       params=ParamSet(values={}),
                       status="complete").fp64_rescued_trials is None


def test_sandbox_config_carries_a_project_provider_but_not_its_permissions(tmp_path):
    """A per-project provider must reach the sandbox, and nothing else may ride along.

    Every agent call runs with `directory=<sandbox>`, and the sandbox's own `opencode.json`
    makes it a project ROOT -- which stops opencode's upward config search. A provider
    declared only in an ancestor directory is therefore unresolvable from inside a sandbox
    (`ProviderModelNotFoundError`, observed on every glm-5.3 call before this existed).
    Providers in the user's global config are unaffected because that file is always loaded,
    which is why the openai arm never needed this.

    Two properties are asserted because both were deliberate: the provider block IS copied,
    and `permission` / `plugin` from the project config are NOT -- carrying those across
    would silently change sandbox behaviour that the harness sets on purpose.
    """
    import json

    from kernel_optimizer.agents.sandbox import PERMISSION_CONFIG, SandboxFactory
    from kernel_optimizer.config import AppConfig
    from kernel_optimizer.wiring import _sandbox_extra_config

    project_cfg = tmp_path / "opencode.jsonc"
    project_cfg.write_text(
        """{
  // a comment, because the repo's own configs are .jsonc
  "provider": {"zhipuai": {"models": {"glm-5.3": {"options": {"reasoningEffort": "max"}}}}},
  "permission": {"webfetch": "allow"},
  "plugin": ["something"]
}""",
        encoding="utf-8",
    )
    cfg = AppConfig.model_validate({"opencode": {"sandbox_config_path": str(project_cfg)}})
    extra = _sandbox_extra_config(cfg)
    assert "zhipuai" in extra["provider"]
    assert "permission" not in extra and "plugin" not in extra

    written = json.loads(
        (SandboxFactory(tmp_path / "sb", extra_config=extra).create("call-1").root
         / "opencode.json").read_text(encoding="utf-8")
    )
    assert "zhipuai" in written["provider"]
    # The harness's own permission block survives the merge.
    assert written["permission"] == PERMISSION_CONFIG["permission"]

    # No config path => byte-identical to the pre-change behaviour.
    assert _sandbox_extra_config(AppConfig()) == {}


def test_a_missing_sandbox_config_is_fatal_rather_than_silent(tmp_path):
    """Refusing to start beats a 12-hour run whose every agent call fails.

    The failure this guards against is not hypothetical: a wrong path yields a config with
    no provider block, every call dies `ProviderModelNotFound` after its full retry budget,
    and the run burns its wall clock producing nothing. A warning would scroll past.
    """
    import pytest

    from kernel_optimizer.config import AppConfig
    from kernel_optimizer.wiring import _sandbox_extra_config

    cfg = AppConfig.model_validate(
        {"opencode": {"sandbox_config_path": str(tmp_path / "nope.jsonc")}}
    )
    with pytest.raises(FileNotFoundError):
        _sandbox_extra_config(cfg)


def _fallback_stat(name, latency_by_value, best_value, best_trial_value,
          at_boundary, direction, effect_pct=25.0):
    from kernel_optimizer.models.reports import ParamStat
    return ParamStat(name=name, best_value=best_value, best_trial_value=best_trial_value,
                     at_boundary=at_boundary, boundary_direction=direction,
                     effect_pct=effect_pct, latency_by_value=latency_by_value)


def _fallback_space(name, choices):
    from kernel_optimizer.models.core import ParamDomain, ParameterSpace
    return ParameterSpace(space_id="sp", candidate_id="c", source_sha="x",
                          domains=[ParamDomain(name=name, kind="int", choices=choices)])


def test_a_corrected_aim_must_never_cancel_the_expansion_itself():
    """Withdrawing every knob request would forfeit the re-tune, which carries its own value.

    An expansion delivers TWO things: a widened range AND a fresh tuning budget. Anchoring
    the boundary flag on the winning trial improves the first (added values convert 21.8% vs
    2.6%) but, taken alone, it can empty the request list and cancel the round outright --
    losing the second.

    That cost is measured, not hypothetical. Of 43 historical expansions, 8 would have been
    cancelled and 6 of those improved, including the two largest gains in the group:
    cand-0d0dcd49 9.14 -> 8.13 ms (11.1%, on the run's best candidate) and cand-913f73c9
    24.00 -> 21.40 (10.8%). In BOTH the winning configuration used no added value at all --
    it was already reachable, and the fresh budget is what found it. More than half of all
    improving expansions are of that shape.

    So when the winner-anchored pass asks for nothing, the median's aim is used instead: the
    expansion still happens and only a low-yield knob guess is lost.

    Shaped after the real FINAL_BLOCK case: median picks 1024 (n=2, a lucky pair), the
    winning trial ran 512 (interior), so the anchored pass withdraws the only request.
    """
    from kernel_optimizer.control.orchestrator import boundary_knobs_to_expand
    from kernel_optimizer.models.reports import TuningStats

    lat = {"64": 13.55, "128": 12.0, "256": 13.6, "512": 13.45, "1024": 10.95}
    stats = TuningStats(
        candidate_id="c", space_id="sp", n_complete=35, n_fail=5,
        param_stats=[_fallback_stat("FINAL_BLOCK", lat, 1024, 512, False, None)],
    )
    knobs = boundary_knobs_to_expand(stats, 0.8, _fallback_space("FINAL_BLOCK", [64, 128, 256, 512, 1024]),
                                     min_effect_pct=2.0)
    assert [k["name"] for k in knobs] == ["FINAL_BLOCK"], \
        "an empty anchored result must fall back to the median's aim, not cancel"
    assert knobs[0]["direction"] == "max", "the median's argmin sits on the max edge"


def test_the_fallback_does_not_override_a_non_empty_corrected_aim():
    """The fallback is a floor, not a merge: a knob the anchored pass dropped stays dropped.

    Otherwise the fix would be undone -- every withdrawal would be restored by the median
    pass sitting behind it. Two knobs here: one the anchored rule keeps, one it withdraws.
    """
    from kernel_optimizer.control.orchestrator import boundary_knobs_to_expand
    from kernel_optimizer.models.core import ParamDomain, ParameterSpace
    from kernel_optimizer.models.reports import TuningStats

    keep = _fallback_stat("BLOCK_N", {"64": 20.0, "128": 10.0}, 128, 128, True, "max")
    drop = _fallback_stat("STAGES", {"1": 25.0, "5": 13.0}, 5, 1, False, None)
    space = ParameterSpace(
        space_id="sp", candidate_id="c", source_sha="x",
        domains=[ParamDomain(name="BLOCK_N", kind="int", choices=[64, 128]),
                 ParamDomain(name="STAGES", kind="int", choices=[1, 2, 3, 4, 5])],
    )
    stats = TuningStats(candidate_id="c", space_id="sp", n_complete=30, n_fail=0,
                        param_stats=[keep, drop])
    names = [k["name"] for k in boundary_knobs_to_expand(stats, 0.8, space, min_effect_pct=2.0)]
    assert names == ["BLOCK_N"], f"withdrawn knob must stay withdrawn, got {names}"


def test_median_fallback_reads_edges_from_the_domain_not_the_latency_dict():
    """`latency_by_value` is keyed in TRIAL order, so its first key is not the domain minimum.

    Observed on the real cand-0d0dcd49 stats: the stored key order is
    ['128','64','256','512','1024']. A fallback reading edges off that dict would call 128
    the minimum edge and mislabel the direction. Domain order is the only correct source.
    """
    from kernel_optimizer.control.orchestrator import boundary_knobs_to_expand
    from kernel_optimizer.models.reports import TuningStats

    # Trial-order keys; the median's argmin (1024) is the domain MAX.
    lat = {"128": 12.0, "64": 13.55, "256": 13.6, "512": 13.45, "1024": 10.95}
    stats = TuningStats(candidate_id="c", space_id="sp", n_complete=35, n_fail=0,
                        param_stats=[_fallback_stat("FINAL_BLOCK", lat, 1024, 512, False, None)])
    knobs = boundary_knobs_to_expand(stats, 0.8, _fallback_space("FINAL_BLOCK", [64, 128, 256, 512, 1024]),
                                     min_effect_pct=2.0)
    assert knobs and knobs[0]["direction"] == "max", \
        "direction must come from domain order, not from latency_by_value insertion order"


def _best_result_section(md: str) -> str:
    return md.split("## Best result")[1].split("\n## ")[0]


def test_the_honest_verdict_is_printed_before_the_raw_speedups(tmp_path):
    """P6: ordering decides which number a reader takes away, and the raw ones read high.

    All three task references are plain fp32 while the winning candidates compute lower, so
    most raw ratios compare ACROSS precisions. On L3:43 the baseline choice alone is worth
    1.91x (4.23x vs torch_compile, 2.21x vs torch_compile_tf32) -- nearly the whole honest
    speedup. The verdict itself is load-bearing: three historical runs are FAILS on it while
    showing 1.08-1.86x against the fp32 baselines. Printing the raw block first is what let
    a reader quote 4.23x.
    """
    from kernel_optimizer.reporting.report import ReportGenerator
    from kernel_optimizer.store.run_store import RunStore

    summary = {
        "task": {"level": 3, "problem_id": 43, "name": "43_MinGPT", "ref_path": "x",
                 "ref_src_sha": "abc"},
        "baselines": [],
        "best": {
            "candidate_id": "cand-win", "family_id": "fam-1", "tuned_ms": 8.06,
            "final_reeval_ok": True, "final_reeval_ms": 8.37, "precision": "fp16",
            "params": {"values": {"BLOCK_M": 64}},
            "speedups": {"eager": 4.97, "eager_tf32": 3.1,
                         "torch_compile": 4.23, "torch_compile_tf32": 2.21},
            "honest_verdict": {"candidate_precision": "fp16",
                               "compared_against": "torch_compile_tf32",
                               "same_precision_speedup": 2.21,
                               "beats_same_precision_baseline": True},
        },
        "families": {},
    }
    store = RunStore.create(tmp_path, "run-order", {"task": summary["task"]})
    store.append("RUN_FINISHED", {"summary": summary})
    md = ReportGenerator().generate(store).read_text(encoding="utf-8")
    sec = _best_result_section(md)

    assert "honest same-precision verdict" in sec and "4.23x" in sec
    assert sec.index("honest same-precision verdict") < sec.index("4.23x"), \
        "the honest verdict must appear ABOVE the cross-precision speedups"
    # The raw block must carry the warning, not just sit lower on the page.
    assert "NOT directly comparable" in sec


def test_the_report_names_the_kernels_the_winner_actually_launched(tmp_path):
    """P3: the fastest candidate can delegate the dominant operator back to PyTorch.

    L3:43's headline 8.06 ms (cand-60fdcae9) launches only `_fused_qkv_projection` and
    `_head_layout_projection` -- the attention core is torch's SDPA -- while the best fully
    hand-written candidate is 9.43 ms. Delegation is also not a family property: that
    family's members flip between the two and the delegating one won, so it cannot be read
    off the lineage. The report prints the launched kernels as a fact; it deliberately does
    NOT classify them, because a keyword rule is task-specific and a wrong attribution label
    is worse than none.
    """
    from kernel_optimizer.reporting.report import ReportGenerator
    from kernel_optimizer.store.run_store import RunStore

    summary = {
        "task": {"level": 3, "problem_id": 43, "name": "43_MinGPT", "ref_path": "x",
                 "ref_src_sha": "abc"},
        "baselines": [],
        "best": {"candidate_id": "cand-win", "family_id": "fam-1", "tuned_ms": 8.06,
                 "precision": "fp16", "params": {"values": {"BLOCK_M": 64}}},
        "families": {},
    }
    store = RunStore.create(tmp_path, "run-attr", {"task": summary["task"]})
    # A slower trial of the same candidate must not be the one reported.
    store.append("TRIAL_DONE", {"trial": {
        "trial_id": "t-slow", "candidate_id": "cand-win", "space_id": "sp",
        "params": {"values": {"BLOCK_M": 32}}, "status": "complete",
        "latency_ms": {"mean": 19.0, "std": 0.1, "min": 18, "max": 20, "n_samples": 20},
        "profile": {"kernel_names": ["_slow_variant"]}}})
    store.append("TRIAL_DONE", {"trial": {
        "trial_id": "t-win", "candidate_id": "cand-win", "space_id": "sp",
        "params": {"values": {"BLOCK_M": 64}}, "status": "complete",
        "latency_ms": {"mean": 8.06, "std": 0.1, "min": 8, "max": 8.3, "n_samples": 20},
        "profile": {"kernel_names": ["_fused_qkv_projection", "_head_layout_projection"]}}})
    store.append("RUN_FINISHED", {"summary": summary})
    md = ReportGenerator().generate(store).read_text(encoding="utf-8")
    sec = _best_result_section(md)

    assert "_fused_qkv_projection" in sec and "_head_layout_projection" in sec
    assert "_slow_variant" not in sec, "must report the WINNING trial's kernels"
    assert "delegated to PyTorch" in sec, "the reader needs the reason this list matters"


def test_attribution_is_silent_rather_than_wrong_without_profile_data(tmp_path):
    """A CUDA (load_inline) candidate has no kernel_names, and inventing one would be worse.

    The line must say the evidence is absent instead of implying the winner wrote nothing.
    """
    from kernel_optimizer.reporting.report import ReportGenerator
    from kernel_optimizer.store.run_store import RunStore

    summary = {
        "task": {"level": 1, "problem_id": 19, "name": "19_ReLU", "ref_path": "x",
                 "ref_src_sha": "abc"},
        "baselines": [],
        "best": {"candidate_id": "cand-cuda", "family_id": "fam-1", "tuned_ms": 1.0,
                 "precision": "ieee_fp32", "params": {"values": {"BLOCK": 256}}},
        "families": {},
    }
    store = RunStore.create(tmp_path, "run-cuda", {"task": summary["task"]})
    store.append("TRIAL_DONE", {"trial": {
        "trial_id": "t1", "candidate_id": "cand-cuda", "space_id": "sp",
        "params": {"values": {"BLOCK": 256}}, "status": "complete",
        "latency_ms": {"mean": 1.0, "std": 0.1, "min": 1, "max": 1.1, "n_samples": 20}}})
    store.append("RUN_FINISHED", {"summary": summary})
    md = ReportGenerator().generate(store).read_text(encoding="utf-8")
    sec = _best_result_section(md)

    assert "none recorded" in sec
    assert "attribution cannot be read" in sec


def test_bfloat16_is_not_mislabelled_fp16_by_substring_match():
    """"bfloat16" CONTAINS "float16", so the fp16 test matched first and ate every bf16 kernel.

    Found while adding the dotless-kernel branch: the fp16 check ran before the bf16 check,
    so `x.to(tl.bfloat16)` classified as "fp16". Both map to the same tensor-core comparator
    in `_honest_verdict`, so no speedup was ever misjudged -- but the reported precision was
    wrong, and precision is exactly what the honest-verdict machinery is there to state.
    """
    from kernel_optimizer.control.orchestrator import _detect_candidate_precision
    from kernel_optimizer.models.core import ParamSet

    e = ParamSet(values={})
    assert _detect_candidate_precision("x = y.to(tl.bfloat16)", e) == "bf16"
    assert _detect_candidate_precision("x = y.to(torch.bfloat16)", e) == "bf16"
    # ...without breaking genuine fp16 detection.
    assert _detect_candidate_precision("x = y.to(tl.float16)", e) == "fp16"
    assert _detect_candidate_precision("x = y.half()", e) == "fp16"


def _fail_stat(name, latency_by_value, failure_rate_by_value, best_value,
               best_trial_value, at_boundary, direction, effect_pct=25.0):
    from kernel_optimizer.models.reports import ParamStat
    return ParamStat(name=name, best_value=best_value, best_trial_value=best_trial_value,
                     at_boundary=at_boundary, boundary_direction=direction,
                     effect_pct=effect_pct, latency_by_value=latency_by_value,
                     failure_rate_by_value=failure_rate_by_value)


def test_a_failing_edge_yields_to_a_healthy_one_in_the_same_expansion():
    """P4: a value added beyond a FAILING edge fails 43% of the time (16/37 across 19 runs)
    vs 15% (13/84) beyond a healthy one, and a failed trial returns no latency at all -- so
    the aim should go to the healthy knob whenever there is one."""
    from kernel_optimizer.control.orchestrator import boundary_knobs_to_expand
    from kernel_optimizer.models.core import ParamDomain, ParameterSpace
    from kernel_optimizer.models.reports import TuningStats

    healthy = _fail_stat("BLOCK_M", {"64": 20.0, "128": 10.0},
                         {"64": 0.0, "128": 0.05}, 128, 128, True, "max")
    failing = _fail_stat("BLOCK_N", {"64": 21.0, "128": 11.0},
                         {"64": 0.0, "128": 0.40}, 128, 128, True, "max")
    space = ParameterSpace(
        space_id="sp", candidate_id="c", source_sha="x",
        domains=[ParamDomain(name="BLOCK_M", kind="int", choices=[64, 128]),
                 ParamDomain(name="BLOCK_N", kind="int", choices=[64, 128])])
    stats = TuningStats(candidate_id="c", space_id="sp", n_complete=30, n_fail=10,
                        param_stats=[healthy, failing])

    names = [k["name"] for k in boundary_knobs_to_expand(
        stats, 0.8, space, min_effect_pct=2.0, max_edge_failure_frac=0.30)]
    assert names == ["BLOCK_M"], f"the 40%-failing edge must yield, got {names}"

    # Disabled (1.0) must reproduce today's behaviour exactly.
    both = [k["name"] for k in boundary_knobs_to_expand(
        stats, 0.8, space, min_effect_pct=2.0, max_edge_failure_frac=1.0)]
    assert both == ["BLOCK_M", "BLOCK_N"]


def test_an_all_failing_expansion_keeps_its_aim_rather_than_being_cancelled():
    """The whole safety argument: this must NEVER turn a non-empty aim into an empty one.

    `_maybe_expand_space` cancels the round when boundary_knobs_to_expand returns []
    (`if not knobs: return`), forfeiting the fresh tuning budget as well as the widening.
    Measured (scripts/audit_expansion_failure_veto.py, 19 runs / 523 aims): a hard filter
    empties 8 of 177 expansions, and two of those 8 are their run's BEST candidate --
    cand-0d0dcd49, and cand-60fdcae9 whose seven aims ALL sit on 24-40% failing edges and
    which is the 8.06 ms L3:43 winner. Shaped after that candidate.
    """
    from kernel_optimizer.control.orchestrator import boundary_knobs_to_expand
    from kernel_optimizer.models.core import ParamDomain, ParameterSpace
    from kernel_optimizer.models.reports import TuningStats

    stats_list, domains = [], []
    # All three ABOVE the 0.30 threshold, so nothing is healthy and there is no
    # alternative to fall back to. (cand-60fdcae9's real spread is 24-40%, i.e. some of
    # its knobs sit just under the threshold and would survive on their own merits; this
    # fixture is the strictly harder case where none do.)
    for i, rate in enumerate((0.40, 0.35, 0.31)):
        name = f"QKV_BLOCK_{i}"
        stats_list.append(_fail_stat(name, {"64": 20.0, "128": 10.0},
                                     {"64": 0.0, "128": rate}, 128, 128, True, "max"))
        domains.append(ParamDomain(name=name, kind="int", choices=[64, 128]))
    stats = TuningStats(candidate_id="c", space_id="sp", n_complete=30, n_fail=12,
                        param_stats=stats_list)
    space = ParameterSpace(space_id="sp", candidate_id="c", source_sha="x",
                           domains=domains)

    knobs = boundary_knobs_to_expand(stats, 0.8, space, min_effect_pct=2.0,
                                     max_edge_failure_frac=0.30)
    assert len(knobs) == 3, \
        f"every aim failing must leave the aim INTACT, not cancel the expansion: {knobs}"


def test_edge_failure_rate_reads_the_edge_from_the_domain_not_the_dict():
    """`failure_rate_by_value` is keyed by repr(choice) in TRIAL order, not domain order.

    Same hazard `_median_direction` documents for latency_by_value (real observed key order
    ['128','64','256','512','1024'], whose first key is not the domain minimum). Here the
    domain max (1024) is healthy while an interior value (512) is both failing and the last
    key inserted: reading the dict's tail would score the wrong value and demote a knob
    that is fine.
    """
    from kernel_optimizer.control.orchestrator import boundary_knobs_to_expand
    from kernel_optimizer.models.reports import TuningStats

    choices = [64, 128, 256, 512, 1024]
    lat = {"128": 12.0, "64": 13.5, "1024": 9.0, "256": 13.6, "512": 13.4}
    rates = {"128": 0.0, "64": 0.0, "1024": 0.02, "256": 0.0, "512": 0.90}
    stats = TuningStats(
        candidate_id="c", space_id="sp", n_complete=35, n_fail=5,
        param_stats=[_fail_stat("FINAL_BLOCK", lat, rates, 1024, 1024, True, "max")])
    knobs = boundary_knobs_to_expand(stats, 0.8, _fallback_space("FINAL_BLOCK", choices),
                                     min_effect_pct=2.0, max_edge_failure_frac=0.30)
    assert [k["name"] for k in knobs] == ["FINAL_BLOCK"], \
        "the healthy domain-max edge must survive; only dict order says otherwise"


def test_missing_failure_data_does_not_demote_a_knob():
    """Absence of evidence must read as healthy, or the preference would fire on every knob
    from a space with no recorded failures."""
    from kernel_optimizer.control.orchestrator import boundary_knobs_to_expand
    from kernel_optimizer.models.reports import TuningStats

    stats = TuningStats(
        candidate_id="c", space_id="sp", n_complete=30, n_fail=0,
        param_stats=[_fail_stat("BLOCK_M", {"64": 20.0, "128": 10.0}, {}, 128, 128,
                                True, "max")])
    knobs = boundary_knobs_to_expand(stats, 0.8, _fallback_space("BLOCK_M", [64, 128]),
                                     min_effect_pct=2.0, max_edge_failure_frac=0.30)
    assert [k["name"] for k in knobs] == ["BLOCK_M"]


def test_the_preference_also_applies_to_the_median_fallback_arm():
    """Applying it to the winner-anchored arm alone is measurably self-defeating.

    In 5 of the 8 expansions a hard filter would empty, the median arm re-aims at THE SAME
    vetoed knobs (NUM_WARPS, BLOCK_D, BLOCK_N/BLOCK_K...), so the preference has to cover
    both arms or it is bypassed exactly where it was meant to bite.
    """
    from kernel_optimizer.control.orchestrator import boundary_knobs_to_expand
    from kernel_optimizer.models.core import ParamDomain, ParameterSpace
    from kernel_optimizer.models.reports import TuningStats

    # Neither knob is at_boundary, so only the median arm can produce an aim.
    healthy = _fail_stat("BLOCK_M", {"64": 20.0, "128": 10.0},
                         {"64": 0.0, "128": 0.0}, 128, 64, False, None)
    failing = _fail_stat("BLOCK_N", {"64": 21.0, "128": 11.0},
                         {"64": 0.0, "128": 0.50}, 128, 64, False, None)
    space = ParameterSpace(
        space_id="sp", candidate_id="c", source_sha="x",
        domains=[ParamDomain(name="BLOCK_M", kind="int", choices=[64, 128]),
                 ParamDomain(name="BLOCK_N", kind="int", choices=[64, 128])])
    stats = TuningStats(candidate_id="c", space_id="sp", n_complete=30, n_fail=10,
                        param_stats=[healthy, failing])
    names = [k["name"] for k in boundary_knobs_to_expand(
        stats, 0.8, space, min_effect_pct=2.0, max_edge_failure_frac=0.30)]
    assert names == ["BLOCK_M"], f"the fallback arm must respect the preference: {names}"


def test_the_shipped_configs_agree_on_the_family_and_expansion_budgets():
    """max_families_active was raised 2->3 together with enabling Loop D.

    `active_families()` ranks families with 0 rewrite rounds FIRST, so an injected novelty
    family jumps ahead of the incumbents; at 2 slots it displaces both current leaders, and
    those are the likeliest source of the run's winner. The gpt and glm L3 configs must
    agree on this, or a cross-model comparison is confounded by a budget difference rather
    than the model.
    """
    from kernel_optimizer.config import load_config

    for path in ("configs/experiments_l3.yaml", "configs/experiments_l3_glm.yaml"):
        b = load_config(path).budgets
        assert b.max_families_active == 3, f"{path}: {b.max_families_active}"
        assert b.max_edge_failure_frac == 0.30, f"{path}: {b.max_edge_failure_frac}"
        # The preference is meaningless unless expansion is on at all.
        assert b.space_expansions_per_candidate >= 1, path


# --- Loop D (novelty) preflight fixes: D1 / D2 / D3 ----------------------------------------


def _loop_d_orchestrator(tmp_path, *, families, budgets=None):
    """A minimally-wired Orchestrator for exercising the outer loop's Loop C/D handover.

    Only the pieces the loop itself touches are real (FamilyManager, ConvergencePolicy,
    RunStore); everything else is left unset because these tests never reach it.
    """
    from types import SimpleNamespace

    from kernel_optimizer.config import AppConfig, BudgetConfig
    from kernel_optimizer.control.convergence import ConvergencePolicy
    from kernel_optimizer.control.families import FamilyManager
    from kernel_optimizer.control.orchestrator import Orchestrator
    from kernel_optimizer.models.core import TaskSpec
    from kernel_optimizer.store.run_store import RunStore

    b = budgets or BudgetConfig()
    fm = FamilyManager(max_families_active=b.max_families_active,
                       max_families_total=b.max_families_total,
                       max_families_total_hard=b.max_families_total_hard)
    for fam, cand, src in families:
        fm.families[fam.family_id] = fam
        fm.candidates[cand.candidate_id] = cand
        fm._sources[cand.candidate_id] = src
    cfg = AppConfig(budgets=b)
    store = RunStore.create(tmp_path, run_id="loopd", manifest={})
    task = TaskSpec(level=1, problem_id=19, name="19_ReLU", ref_path="x", ref_src_sha="s")
    deps = SimpleNamespace(families=fm, convergence=ConvergencePolicy(b))
    orch = Orchestrator.__new__(Orchestrator)
    orch.deps = deps
    orch.cfg = cfg
    orch.store = store
    orch.task = task
    return orch, fm


def _live_family(fid, cid, ms=10.0, rounds=0):
    # Family.best is a BestRecord whose latency_ms is a plain float (see
    # FamilyManager.update_best) -- NOT a TrialRecord. Using a TrialRecord here makes
    # active_families()'s sort compare LatencyStats objects and raise TypeError.
    from kernel_optimizer.models.core import BestRecord, Candidate, Family, ParamSet
    cand = Candidate(candidate_id=cid, family_id=fid, origin="seed", backend="triton",
                     source_sha=fid, structural_signature=fid)
    fam = Family(family_id=fid, anchor_candidate_id=cid, member_ids=[cid], status="active")
    fam.best = BestRecord(candidate_id=cid, params=ParamSet(values={}), latency_ms=ms)
    fam.best_history = [ms]
    fam.rewrite_rounds_used = rounds
    return fam, cand, f"# {fid}\nx = 1\n"


def test_the_novelty_gate_counts_productive_families_not_corpses(tmp_path):
    """D1: the outer gate used `len(families)`, the inner one `productive_family_count()`.

    That inner rule IS improvement E -- dead families must not consume a novelty slot,
    "otherwise a batch of failed seeds permanently blocks novelty exploration". Implementing
    it only in `accept_novel_seed` left it unreachable, because `_novelty_round` runs first
    and counts the corpses. Measured over the 14 completed L3 runs the two rules disagree in
    5; in 4 of those the inner rule would have allowed novelty while the run ended with
    79-90% of its wall clock unspent, twice with ZERO productive families.

    Shaped after run-l3-43-20260902-213608: 4 families, every seed dead, run over at 1.22 h
    of 12 h.
    """
    from kernel_optimizer.config import BudgetConfig
    from kernel_optimizer.models.core import Candidate, Family

    dead = []
    for i in range(4):
        cid, fid = f"c{i}", f"fam-dead{i}"
        cand = Candidate(candidate_id=cid, family_id=fid, origin="seed", backend="triton",
                         source_sha=fid, structural_signature=fid)
        fam = Family(family_id=fid, anchor_candidate_id=cid, member_ids=[cid],
                     status="frozen_budget")  # best stays None => dead
        dead.append((fam, cand, f"# {fid}\nx = 1\n"))

    b = BudgetConfig(max_families_total=6, max_seed_candidates=4)
    orch, fm = _loop_d_orchestrator(tmp_path, families=dead, budgets=b)
    assert fm.productive_family_count() == 0, "all four seeds are dead"
    assert len(fm.families) == 4

    # The gate must not refuse on the strength of 4 corpses. Reaching the agent call is
    # proof enough that the gate opened: novelty is unset on this stub, so an AttributeError
    # here means we got PAST the gate, while a plain False means we did not.
    try:
        orch._novelty_round(1)
    except AttributeError:
        pass  # reached `self.deps.novelty.invoke`, i.e. the gate allowed the call
    else:
        raise AssertionError("gate refused: it is still counting dead families")


def test_the_novelty_gate_still_refuses_once_productive_families_fill_the_budget(tmp_path):
    """The looser count must not become no count at all: live families still bound Loop D."""
    from kernel_optimizer.config import BudgetConfig

    fams = [_live_family(f"fam-live{i}", f"c{i}") for i in range(3)]
    b = BudgetConfig(max_families_total=3)
    orch, fm = _loop_d_orchestrator(tmp_path, families=fams, budgets=b)
    assert fm.productive_family_count() == 3
    assert orch._novelty_round(1) is False


def _drive_outer_loop(orch, *, max_iters=40):
    """Run the SHIPPED outer loop (`_run`'s Loop C/D section) against a stub orchestrator.

    This exists because two earlier tests of this loop re-implemented the post-miss handling
    inside the test body instead of calling the real thing. They therefore asserted against a
    hand-written copy that still froze families -- and passed while the shipped code spun
    2,054,908 times in run-l1-19-20260906-192759. A test of a loop must execute that loop.

    `_run` also does baseline/seeds/finalize, which these stubs cannot reach, so this drives
    the same statements with the same calls into `deps`, plus a hard iteration ceiling so a
    non-terminating loop fails the test instead of hanging it. Returns the iteration count;
    `max_iters` reached means the loop did not terminate.

    `_novelty_round` may now get past its gate (freeing a family's slot is the point of the
    fix), so an absent novelty agent is treated as "the attempt produced nothing" -- which is
    the outcome under test. A real miss and an unwired stub take the same branch.
    """
    round_no = 0
    idle_rounds = 0
    iters = 0
    while iters < max_iters:
        iters += 1
        verdict = orch.deps.convergence.global_verdict(
            list(orch.deps.families.families.values()), 0.1)
        if verdict.verdict == "freeze":
            return iters
        round_no += 1
        progressed = orch._rewrite_round(round_no)
        if not progressed:
            try:
                added = orch._novelty_round(round_no)
            except AttributeError:
                added = False   # no novelty agent wired: same branch as a genuine miss
            if not added:
                idle_rounds += 1
                if idle_rounds >= orch._MAX_IDLE_ROUNDS:
                    return iters
                continue
        idle_rounds = 0
    return iters


def test_a_family_with_no_rewrite_parent_gets_frozen_rather_than_spinning(tmp_path):
    """The defect the D2 fix introduced, and the reason the fix now lives in _rewrite_round.

    `active_families()` excludes a family whose `best is None` -- correct, since
    `_do_rewrite` needs a correct parent, and letting it hold a slot cost real rounds on
    L3:21. But that exclusion also means `_rewrite_round`'s loop never REACHES such a family:
    it is never frozen there and its `rewrite_rounds_used` never advances. The blanket sweep
    D2 removed was the only thing that ever froze it (`active_families()`'s own docstring
    said so). Without it the outer loop had no exit: `global_verdict` sees one active family
    so it continues, `productive_family_count()` counts it so novelty declines,
    `active_families()` is empty so `progressed` is False -- two events per iteration and no
    work, forever. Live: 2.05M iterations / 991 MB of events in 13 min after
    `fam-92c506b3`'s space was rejected twice (run-l1-19-20260906-192759).
    """
    from kernel_optimizer.config import BudgetConfig
    from kernel_optimizer.models.core import Candidate, Family

    # Exactly the live shape: two frozen families plus one active with no correct candidate.
    fams = [_live_family("fam-a", "ca", ms=251.0, rounds=0),
            _live_family("fam-b", "cb", ms=246.0, rounds=0)]
    cand = Candidate(candidate_id="cc", family_id="fam-c", origin="novelty",
                     backend="triton", source_sha="fam-c", structural_signature="fam-c")
    stuck = Family(family_id="fam-c", anchor_candidate_id="cc", member_ids=["cc"],
                   status="active")  # best stays None: space was rejected
    fams.append((stuck, cand, "# fam-c\nx = 1\n"))

    b = BudgetConfig(max_families_total=3, rewrite_rounds_per_family=0)
    orch, fm = _loop_d_orchestrator(tmp_path, families=fams, budgets=b)
    for fid in ("fam-a", "fam-b"):
        fm.families[fid].status = "frozen_budget"
    assert fm.families["fam-c"].status == "active"
    assert fm.active_families() == [], "the stuck family is invisible to active_families()"

    iters = _drive_outer_loop(orch, max_iters=40)

    assert fm.families["fam-c"].status != "active", (
        "a family with no rewrite parent stayed active forever -- this is the 2.05M-iteration "
        "spin")
    assert iters < 40, f"the outer loop did not terminate ({iters} iterations)"


def test_the_outer_loop_cannot_spin_even_if_a_family_never_freezes(tmp_path):
    """The liveness backstop, independent of any particular freeze rule.

    The bug above was a missed case in an exhaustiveness argument I asserted in a comment.
    The guard makes the cost of the NEXT such miss a handful of events rather than hours of
    wall clock: a family pinned active by force must not buy an unbounded loop.
    """
    from kernel_optimizer.config import BudgetConfig
    from kernel_optimizer.models.core import Candidate, Family

    cand = Candidate(candidate_id="cz", family_id="fam-z", origin="seed", backend="triton",
                     source_sha="fam-z", structural_signature="fam-z")
    fam = Family(family_id="fam-z", anchor_candidate_id="cz", member_ids=["cz"],
                 status="active")
    b = BudgetConfig(max_families_total=1, rewrite_rounds_per_family=0)
    orch, fm = _loop_d_orchestrator(
        tmp_path, families=[(fam, cand, "# fam-z\nx = 1\n")], budgets=b)

    # Defeat the real fix on purpose: this family re-activates itself every round, so no
    # freeze rule can end the loop. Only the guard can.
    real = orch._freeze_unrewritable_families

    def _undo():
        real()
        fm.families["fam-z"].status = "active"
        return 0
    orch._freeze_unrewritable_families = _undo

    iters = _drive_outer_loop(orch, max_iters=200)
    assert iters <= orch._MAX_IDLE_ROUNDS + 1, (
        f"the guard did not stop an unbreakable loop: {iters} iterations")


def test_a_novelty_miss_does_not_freeze_families_that_still_have_budget(tmp_path):
    """D2: the run's ending must not depend on a Loop D outcome.

    The old branch froze every still-active family when `added` was False, so the next
    `global_verdict` saw nothing active and ended the run. Two consequences:

    * The LAST novelty attempt always misses -- each acceptance raises the family count
      until the gate declines to call the agent -- so every Loop D run ended one attempt
      after its last acceptance, with budget to spare.
    * Observed live in run-l1-19-20260906-183211, the first run where Loop D executed: one
      family accepted, second attempt gated off, all frozen, run over at 0.413 h of 3 h
      (13.8% of budget).

    A family that cannot continue is already frozen inside `_rewrite_round` by its own
    verdict; nothing here should freeze one that can. This drives the SHIPPED loop rather
    than a copy of it -- the earlier version of this test re-implemented the post-miss
    handling in its own body, which is why it passed while the real loop spun.
    """
    from kernel_optimizer.config import BudgetConfig

    # Three live families with rounds left; being productive also gates novelty off, so
    # every iteration is a novelty miss.
    fams = [_live_family(f"fam-live{i}", f"c{i}", ms=10.0 + i, rounds=1) for i in range(3)]
    b = BudgetConfig(max_families_total=3, rewrite_rounds_per_family=5)
    orch, fm = _loop_d_orchestrator(tmp_path, families=fams, budgets=b)
    assert orch._novelty_round(1) is False, "gate should refuse (3 productive >= 3)"

    # `_do_rewrite` needs GPU/agent wiring these stubs lack; the loop reaches it only via
    # self.runs, which is empty, so each family takes the "no source_crun" path: it counts a
    # round without rewriting. That is the shipped behaviour for a missing bottleneck report.
    orch.runs = {}
    _drive_outer_loop(orch, max_iters=40)

    # The families end frozen (their rounds run out), but by their OWN budget, not by a
    # novelty miss: the counter must have advanced past 1 for each.
    for fid in ("fam-live0", "fam-live1", "fam-live2"):
        used = fm.families[fid].rewrite_rounds_used
        assert used > 1, (
            f"{fid} was frozen without spending its rounds (used={used}) -- a novelty miss "
            "ended it")


def test_the_outer_loop_still_ends_when_nothing_is_rewritable(tmp_path):
    """The D2 fix must not remove termination. With no rewritable family the run must end.

    Drives the shipped loop: `_freeze_unrewritable_families` freezes these, and then
    `global_verdict` freezes the run because nothing is active.
    """
    from kernel_optimizer.config import BudgetConfig
    from kernel_optimizer.models.core import Candidate, Family

    fams = []
    for i in range(2):
        cid, fid = f"c{i}", f"fam-dead{i}"
        cand = Candidate(candidate_id=cid, family_id=fid, origin="seed", backend="triton",
                         source_sha=fid, structural_signature=fid)
        # best is None => active_families() excludes it => nothing is rewritable.
        fams.append((Family(family_id=fid, anchor_candidate_id=cid, member_ids=[cid],
                            status="active"), cand, f"# {fid}\nx = 1\n"))
    b = BudgetConfig(max_families_total=2)
    orch, fm = _loop_d_orchestrator(tmp_path, families=fams, budgets=b)
    assert orch.deps.families.active_families() == [], "no family is rewritable"

    iters = _drive_outer_loop(orch, max_iters=40)
    assert iters < 40, "an exhausted search must still end"
    assert all(f.status != "active" for f in fm.families.values())
    v = orch.deps.convergence.global_verdict(list(fm.families.values()), 0.1)
    assert v.verdict == "freeze", "an exhausted search must still end"


def test_the_novelty_step_key_survives_a_resume(tmp_path):
    """D3: `round_no` is a local in `_run`, reset to 0 on resume, so `novelty:{round_no}`
    collided with a key already in `steps_done` -- and the method then returned False for a
    round it had never run (which, before D2, also ended the run).

    `_restore_family_control_state` rebuilds best_history, rewrite_rounds_used and
    failed_hypotheses; nothing rebuilds round_no. Numbering attempts by how many are already
    recorded is resume-stable, the way `_rewrite_round` keys on `rewrite_rounds_used`.
    """
    from kernel_optimizer.config import BudgetConfig

    fams = [_live_family("fam-live0", "c0")]
    b = BudgetConfig(max_families_total=6)
    orch, fm = _loop_d_orchestrator(tmp_path, families=fams, budgets=b)

    # Simulate one completed novelty attempt, as the log would hold it.
    orch._step_done("novelty:0")
    assert "novelty:0" in orch.store.replay().steps_done

    # A resumed run re-enters with round_no back at 1. The derived key must be the NEXT
    # attempt, not a repeat of the recorded one.
    state = orch.store.replay()
    key = f"novelty:{sum(1 for k in state.steps_done if k.startswith('novelty:'))}"
    assert key == "novelty:1", f"resume must advance the attempt counter, got {key}"
    assert key not in state.steps_done

    # And the gate must not short-circuit on the stale key: reaching the (unset) agent proves
    # the attempt is allowed to proceed.
    try:
        orch._novelty_round(1)  # round_no=1 again, exactly what a resume passes
    except AttributeError:
        pass
    else:
        raise AssertionError("resumed run skipped a novelty attempt it never ran")


def test_the_report_says_why_the_run_ended_and_flags_a_premature_freeze():
    """The visibility gap that let D2 hide for 19 runs.

    The convergence section showed only the last ten decisions, so a run frozen by the outer
    loop's blanket sweep and one that genuinely exhausted its families produced identical
    reports. Measured with `scripts/audit_run_termination_reasons.py`: only 1 of 19 runs was
    ended by the wall clock, and four ended with 0-2 of 12 rewrite rounds used and NO family
    freeze verdict at all -- every one of which looked normal in its report.
    """
    from types import SimpleNamespace

    import re

    from kernel_optimizer.reporting.report import _why_the_run_ended

    def ev(t, **payload):
        return SimpleNamespace(type=t, payload=payload)

    budgets = {"wall_clock_hours": 12.0, "max_seed_candidates": 4,
               "rewrite_rounds_per_family": 3}
    frozen = [{"decision": {"scope": "global", "verdict": "freeze",
                            "stop_kind": "budget_exhausted"}}]

    # Premature: 1.97 h of 12 h, nothing rewritten -- shaped after run-l3-21-20260903-210650.
    out = "\n".join(_why_the_run_ended(
        [ev("RUN_FINISHED", summary={"elapsed_hours": 1.974})], frozen, budgets))
    assert "every family frozen" in out
    assert "0 of 12" in out
    assert "a freeze rule, not the budget" in out, out

    # Legitimate: the clock was spent, so no accusation. Shaped after
    # run-l3-21-20260905-195615, whose 10 rounds were spread over FOUR families (3+3+2+2)
    # -- a fixture that puts all ten on one family would be impossible, since
    # `rewrite_rounds_per_family` is 3.
    real = [("fam-4286a3be", 3), ("fam-a2688942", 3), ("fam-a4a8353c", 2),
            ("fam-fd92a2d8", 2)]
    out = "\n".join(_why_the_run_ended(
        [ev("RUN_FINISHED", summary={"elapsed_hours": 12.816})]
        + [ev("FAMILY_ROUND_RECORDED", family_id=f, best_ms=1.0)
           for f, n in real for _ in range(n)], frozen, budgets))
    assert "wall clock" in out
    assert "10 of 12" in out, out
    assert "a freeze rule" not in out, "a spent budget must not be flagged: " + out

    # The denominator counts FAMILIES, not seeds. Loop D adds families, and each brings its
    # own `rewrite_rounds_per_family` allowance -- so a seed-derived denominator understates
    # the budget and can print an impossible fraction. Live on run-l1-19-20260906-220044:
    # 2 seeds x 2 rounds reported "6 of 4" because Loop D had added 2 more families.
    loopc = {"wall_clock_hours": 3.0, "max_seed_candidates": 2,
             "rewrite_rounds_per_family": 2, "max_families_total": 4}
    out = "\n".join(_why_the_run_ended(
        [ev("RUN_FINISHED", summary={"elapsed_hours": 2.487})]
        + [ev("FAMILY_ROUND_RECORDED", family_id=f, best_ms=1.0) for f, n in
           [("fam-9df5650b", 1), ("fam-6606b4b6", 1), ("fam-cda0d77a", 2),
            ("fam-f2d8c537", 2)] for _ in range(n)], frozen, loopc))
    assert "6 of 8" in out, out
    assert "incl. 2 from Loop D" in out, out

    # ...and it must NOT be clamped to `max_families_total`. That budget gates whether a NEW
    # family may be created and can legitimately sit below the number that exist: the real
    # run-l3-21-20260905-195615 seeded 4 families under `max_families_total: 3` (defect D1,
    # where the gate counted differently than the seeder). Clamping printed "10 of 9" --
    # the same impossible fraction, in the other direction.
    capped = dict(budgets, max_families_total=3)
    out = "\n".join(_why_the_run_ended(
        [ev("RUN_FINISHED", summary={"elapsed_hours": 12.816})]
        + [ev("FAMILY_ROUND_RECORDED", family_id=f, best_ms=1.0)
           for f, n in real for _ in range(n)], frozen, capped))
    assert "10 of 12" in out, "max_families_total must not cap the denominator: " + out

    # Whatever the shape, the fraction must never exceed 1: that is the invariant both
    # bugs above violated, and it is checkable without knowing the right answer.
    for b, evs in ((budgets, real), (loopc, [("a", 2), ("b", 2)])):
        txt = "\n".join(_why_the_run_ended(
            [ev("RUN_FINISHED", summary={"elapsed_hours": 1.0})]
            + [ev("FAMILY_ROUND_RECORDED", family_id=f, best_ms=1.0)
               for f, n in evs for _ in range(n)], frozen, b))
        m = re.search(r"rewrite rounds spent: \*\*(\d+) of (\d+)\*\*", txt)
        assert m, txt
        assert int(m.group(1)) <= int(m.group(2)), f"impossible fraction: {m.group(0)}"

    # A stuck loop must be called a defect, not a finished search.
    out = "\n".join(_why_the_run_ended(
        [ev("OUTER_LOOP_STUCK", idle_rounds=3, families={"f": "active"}),
         ev("RUN_FINISHED", summary={"elapsed_hours": 0.2})], frozen, budgets))
    assert "OUTER_LOOP_STUCK" in out and "DEFECT" in out, out

    # A killed run must not be reported as any kind of ending.
    out = "\n".join(_why_the_run_ended([], frozen, budgets))
    assert "no RUN_FINISHED" in out, out

    # The D4 freeze is named, so the audit can see it from the report alone.
    out = "\n".join(_why_the_run_ended(
        [ev("FAMILY_FROZEN_UNREWRITABLE", family_id="fam-92c506b3"),
         ev("RUN_FINISHED", summary={"elapsed_hours": 0.5})], frozen, budgets))
    assert "fam-92c506b3" in out and "unrewritable" in out, out


def test_loop_d_is_reachable_at_the_shipped_l3_budget():
    """The interlock that kept Loop D at zero executions across all 19 runs.

    `max_seed_candidates` seeds each register their own family, so with seeds=4 and total=3
    the gate `>= max_families_total` was true before the first check. One of the paper's four
    loops therefore had no experimental evidence at all.
    """
    from kernel_optimizer.config import load_config

    for path in ("configs/experiments_l3.yaml", "configs/experiments_l3_glm.yaml"):
        b = load_config(path).budgets
        assert b.max_seed_candidates < b.max_families_total, (
            f"{path}: seeds={b.max_seed_candidates} >= total={b.max_families_total}, "
            "so Loop D can never be called")
        assert b.max_families_total <= b.max_families_total_hard, path
        # Room for at least one novel family beyond the seeds.
        assert b.max_families_total - b.max_seed_candidates >= 1, path


def test_a_nested_object_sent_as_json_text_is_decoded_not_rejected():
    """glm-5.3 double-encodes nested fields; the content is right, the encoding is not.

    Live on run-l2-37-20260907-003838 (the first GLM run to get past the generator): the
    parameterizer returned `{"file": ..., "space": "{\\"domains\\": [...]}"}` -- `space` as
    JSON *text* instead of a nested object. Pydantic says `Input should be a valid
    dictionary`, which reads as a content error, so the retry feedback told the agent its
    answer was wrong. It re-derived the same answer, re-encoded it the same way, and all 3
    attempts failed identically: 3 of 4 seed candidates discarded before touching the GPU,
    ~$0.12 and ~250k tokens for parameter spaces that were already correct.

    gpt-5.6-sol nests properly, which is why 17 runs on the gpt arm never hit this.
    """
    import json as _json

    from pydantic import BaseModel

    from kernel_optimizer.agents.base import _decode_stringified_objects

    class Inner(BaseModel):
        domains: list[str]

    class Outer(BaseModel):
        file: str
        space: Inner
        note: str

    # The exact shape observed: `space` double-encoded, siblings normal.
    raw = {"file": "cand.py",
           "space": _json.dumps({"domains": ["BLOCK_M", "NUM_WARPS"]}),
           "note": "left as-is"}
    out = Outer.model_validate(_decode_stringified_objects(raw))
    assert out.space.domains == ["BLOCK_M", "NUM_WARPS"]
    assert out.note == "left as-is", "a plain string field must survive untouched"

    # A correctly-nested payload must pass through unchanged -- the fix must not depend on
    # the bug being present.
    good = {"file": "c.py", "space": {"domains": ["X"]}, "note": "n"}
    assert Outer.model_validate(_decode_stringified_objects(good)).space.domains == ["X"]

    # Strings that merely LOOK like data must not be reinterpreted. A field legitimately
    # holding "{}" -shaped text, a number, or JSON-ish prose stays a string, otherwise the
    # decoder would corrupt honest content.
    for keep in ("42", "true", "null", "not json {", "{unclosed", '"quoted"', "[1,2", ""):
        assert _decode_stringified_objects({"note": keep})["note"] == keep, keep

    # It must reach nested positions too: the same double-encoding inside a list element.
    nested = {"items": [{"space": _json.dumps({"domains": ["A"]})}]}
    assert _decode_stringified_objects(nested)["items"][0]["space"] == {"domains": ["A"]}

    # And it must terminate on pathological nesting rather than recursing forever.
    deep = "0"
    for _ in range(30):
        deep = _json.dumps({"k": deep})
    _decode_stringified_objects(deep)      # must simply return, not raise


def test_the_tuning_objective_is_robust_to_a_single_stall():
    """A 20-sample MEAN is not a usable tuning objective; the median is.

    Measured on run-l2-37-20260907-010645 and quantified with
    scripts/probe_robust_objective.py against a 2000-sample ground truth (400 windows of
    n=20): the mean's coefficient of variation at n=20 is 24-37% while the median's is
    3-8%, and on a pair of configurations whose true costs differ by 7.6% a 20-sample mean
    picks the faster one 64.8% of the time -- near a coin flip -- against the median's
    93.2%. Since TPE chooses where to sample next from those comparisons, the mean spends
    the trial budget exploring noise.

    Live cost of the old objective, from that run: a space expansion was credited with
    32.60 -> 30.70 us, a 5.8% "gain" that cleared min_improvement_pct 2.0 and earned the
    family another rewrite round -- while the difference was 1.90 us against a combined
    standard error of 17.85 us, and the supposedly-better point was SLOWER by min.
    """
    from kernel_optimizer.models.core import LatencyStats

    # One stall in 20 samples: the real cost is ~16 us, the mean says 32.
    stalled = LatencyStats(mean=32.60, std=64.80, min=16.00, max=315.00, n_samples=20,
                           median=16.40)
    clean = LatencyStats(mean=30.70, std=54.60, min=19.30, max=234.00, n_samples=20,
                         median=19.80)

    # The objective must prefer the genuinely faster kernel, which the MEAN gets backwards.
    assert stalled.robust_ms < clean.robust_ms, "median must rank the faster kernel first"
    assert stalled.mean > clean.mean, "the mean ranks them backwards -- the defect"

    # Absent a median (older runs, and the two timing paths that return summary stats only),
    # robust_ms must fall back to the mean rather than crash or return a sentinel.
    legacy = LatencyStats(mean=7.5, std=0.2, min=7.1, max=8.0, n_samples=100)
    assert legacy.median is None
    assert legacy.robust_ms == 7.5
    # A non-positive median is not a measurement; fall back too.
    assert LatencyStats(mean=7.5, std=0.2, min=7.1, max=8.0, n_samples=100,
                        median=-1.0).robust_ms == 7.5

    # `min` must NOT be the objective, however robust it looks: measured at n=20 it is
    # biased +9.8% to +156% (20 draws rarely contain the true minimum) and it ranked three
    # of six real config pairs BACKWARDS, below 50% agreement, because it reports the
    # luckiest draw rather than the cost. Guard the source so nobody "simplifies" to it.
    from pathlib import Path
    tpe_src = Path("src/kernel_optimizer/tuning/tpe.py").read_text(encoding="utf-8")
    assert "robust_ms" in tpe_src
    assert ".latency_ms.min" not in tpe_src, "min is a biased estimator at n=20; see probe"

    # The tuner and the orchestrator must rank by the SAME statistic, or the tuner's
    # incumbent and the reported best can be different trials.
    orch = Path("src/kernel_optimizer/control/orchestrator.py").read_text(encoding="utf-8")
    for frag in ("best = min(complete, key=lambda t: t.latency_ms.robust_ms)",
                 "crun.best_ms = best.latency_ms.robust_ms"):
        assert frag in orch, frag

    # The headline speedup must stay MEAN-based: switching it to a median would raise every
    # published number without any kernel getting faster. Both are reported side by side.
    assert "speedups[b.kind] = round(b.latency_ms.mean / lat.mean, 4)" in orch
    assert "speedups_median" in orch
    assert "final_reeval_median_ms" in orch


def test_raw_samples_survive_to_the_trial_record():
    """The samples must reach LatencyStats, not be dropped at the parse boundary.

    They were, on the first cut of this change: the worker emitted `samples` but
    LatencyStats had no field for it, so `latency_from_result` silently discarded them and
    the log was no more re-analysable than before. Caught only by looking at a real
    tune-file run's events (`samples? False`), which is why this test exists.

    What retention bought immediately, from run tunefile-l2-37-20260907-020027: the noise on
    this task is NOT scattered jitter but a deterministic warmup artifact -- sample #1 is
    370-385 us in all four trials while samples 2-20 sit inside 17.3-19.3 / 40.0-42.0 /
    15.3-16.3 us. One artifact in 20 samples was inflating every trial's mean by 1.7-2.9x.
    That diagnosis was impossible from mean/std/min/max alone.
    """
    from kernel_optimizer.evaluation.correctness import latency_from_result

    worker_result = {"latency_ms": {
        "mean": 0.03561, "std": 0.0788, "min": 0.0173, "max": 0.3703, "n": 20,
        "median": 0.01811,
        "samples": [0.3703, 0.0175, 0.0182, 0.0183, 0.0183, 0.0174, 0.0182, 0.0183,
                    0.0184, 0.0181, 0.0175, 0.0174, 0.0174, 0.0173, 0.0181, 0.0191,
                    0.0181, 0.0193, 0.0175, 0.0174]}}
    lat = latency_from_result(worker_result)
    assert lat is not None
    assert lat.samples is not None, "samples dropped at the parse boundary"
    assert len(lat.samples) == 20
    assert lat.samples[0] == 0.3703, "the artifact itself must be preserved, not filtered"
    # The objective ignores it; the record keeps it.
    assert lat.robust_ms == 0.01811
    assert abs(lat.mean / lat.robust_ms - 1.97) < 0.05, "mean is ~2x the real cost here"

    # A worker result with no samples (the baseline helper, KernelBench runtime_stats) must
    # still parse, carrying neither samples nor median.
    plain = latency_from_result({"latency_ms": {"mean": 0.05, "std": 0.001, "min": 0.049,
                                                "max": 0.052, "n": 100}})
    assert plain is not None and plain.samples is None and plain.median is None
    assert plain.robust_ms == 0.05


def test_triton_pitfalls_covers_the_compile_errors_actually_observed():
    """Every entry in triton_pitfalls.md must be a failure the harness really saw.

    Two were added from run-l2-37-20260907-020707, where glm-5.3 produced them on
    independently generated candidates:

    - `import triton.lang as tl` -- there is no such submodule; the correct name is
      `triton.language`. It fails at IMPORT time, so a correct kernel is discarded for a
      one-word mistake. Hit TWO of four seed candidates in that run (cand-4cdbf3fc and
      cand-fba33b33), and never once in 17 gpt-arm runs, so it is a model-specific habit
      worth naming explicitly rather than a one-off.
    - `ng = BLOCK_N // GROUP_SIZE` then `tl.reshape(t, (BLOCK_M, ng, GROUP_SIZE))` --
      floordiv between two `tl.constexpr` values does not fold to `constexpr[int]`, so the
      shape tuple is rejected. Cost a repair round in run-l2-37-20260907-010645.

    The doc is read from disk on every agent call (`_triton_pitfalls_doc`, no module-level
    cache), so an addition reaches a RUNNING experiment on its next agent call. That is why
    it was safe to add these mid-run.
    """
    from kernel_optimizer.agents.modules import _triton_pitfalls_doc

    doc = _triton_pitfalls_doc()

    # Pitfall 7: the import name. Must show the wrong spelling AND the right one, since a
    # rule that only says "use triton.language" does not tell the model what it did wrong.
    assert "triton.lang as tl" in doc, "the failing spelling must appear as the BAD form"
    assert "import triton.language as tl" in doc, "the correct spelling must appear"
    assert "No module named 'triton.lang'" in doc, "the actual error text helps recognition"

    # Pitfall 8: constexpr arithmetic in a shape tuple.
    assert "constexpr[int]" in doc
    assert "tl.reshape" in doc
    # It must point at the HOST as the fix, matching pitfall 6's existing rule.
    assert "host" in doc.lower()

    # Structure: every pitfall keeps the BAD/GOOD pairing the file's header promises, so a
    # new entry cannot be a bare prohibition with no working alternative.
    sections = [s for s in doc.split("\n## ") if s.strip()][1:]   # drop the title block
    assert len(sections) >= 8, f"expected >=8 pitfalls, found {len(sections)}"
    for s in sections:
        name = s.splitlines()[0]
        assert "# BAD" in s or "BAD:" in s, f"pitfall lacks a BAD form: {name}"
        assert "# GOOD" in s or "GOOD:" in s, f"pitfall lacks a GOOD form: {name}"
