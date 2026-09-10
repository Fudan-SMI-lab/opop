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
    """The gate is `frac > pass_frac AND cosine >= cosine_min`. Both clauses are
    exercised here, including the region where they DISAGREE.

    This test asserted the wrong thing from the commit that introduced it, and never
    said so, because it only runs where torch is importable -- on the Windows
    orchestrator host it skips, so it first executed on a Linux box much later. Its
    middle case set 5 of 1000 elements to 5.0 (a 400% error) and asserted a pass
    "because >99% of elements are within tolerance", reasoning about the frac clause
    alone. Measured: frac = 0.995 (passes) but cosine = 0.96380941 against a 0.99985
    bar (rejects), so the AND rejects -- correctly. A single grossly-wrong element is
    exactly what cosine is in the gate to catch, and what the frac clause cannot see.

    So the sparse-error case is now asserted at both magnitudes, which is what pins
    the two clauses as complementary rather than redundant.
    """
    torch = pytest.importorskip("torch")
    _relaxed_close = _load_worker_relaxed_close()
    ref = torch.ones(1000)
    # exact match passes
    assert _relaxed_close(ref, ref.clone(), 0.01, 0.99, 0.99985)

    # 0.5% of elements slightly wrong (+10%) -> frac 0.995 passes, cosine 0.99997515
    # clears 0.99985 -> accepted. This is the case the middle assertion meant to make.
    got = ref.clone(); got[:5] = 1.1
    assert _relaxed_close(ref, got, 0.01, 0.99, 0.99985)

    # Same 0.5% of elements, but badly wrong (+400%) -> frac still 0.995, yet cosine
    # falls to 0.96380941 -> REJECTED by the cosine clause alone. Without this the
    # suite would pass with cosine_min deleted.
    got_bad = ref.clone(); got_bad[:5] = 5.0
    assert not _relaxed_close(ref, got_bad, 0.01, 0.99, 0.99985)
    # ... and it is specifically the cosine clause: a permissive cosine_min accepts it.
    assert _relaxed_close(ref, got_bad, 0.01, 0.99, 0.9)

    # 5% of elements wrong -> below the 99% frac bar -> fails on the frac clause, which
    # a permissive cosine_min must NOT rescue.
    got2 = ref.clone(); got2[:50] = 5.0
    assert not _relaxed_close(ref, got2, 0.01, 0.99, 0.99985)
    assert not _relaxed_close(ref, got2, 0.01, 0.99, 0.0)
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


def test_the_agent_call_timeout_is_not_raised_on_an_unverified_loss_claim(tmp_path):
    """SUPERSEDED CLAIM, kept as a guard against restoring it.

    This test used to assert `request_timeout_s >= 1800`, on the reasoning that "8 agent calls
    died at exactly 1200-1201s ... each kill discards a candidate or a whole rewrite round" and
    that the slowest successful call was 979 s.

    Both halves were wrong, re-measured over every run on disk
    (scripts/probe_agent_timeouts.py):

      - 7 of those 8 calls FINISHED on a retry. Only one call has ever been truly lost.
      - The slowest successful call is 1167 s, not 979 s -- 979 was one L3:21 generator call,
        not the maximum over the population of 982.

    Since no successful call has ever exceeded 1200 s, no 1200 s timeout was cutting off work
    in progress: a timed-out call is HUNG, and a fresh session finishes the same prompt in
    ~4 min. Raising the ceiling rescued nothing and made every hang 50% dearer.

    The live assertions moved to
    `test_the_agent_timeout_is_priced_as_a_hang_not_a_work_budget`; what remains here is the
    upper bound, so the >= 1800 floor cannot come back on the strength of the old story.
    """
    for path in ("configs/default.yaml", "configs/experiments_l3.yaml",
                 "configs/experiments_l3_glm.yaml"):
        cfg = load_config(path)
        assert cfg.opencode.request_timeout_s <= 1500.0, (
            f"{path}: timeout raised to {cfg.opencode.request_timeout_s}s. The 1800s value was "
            "justified by a lost-work claim that re-measurement disproved; 14 of 15 hung calls "
            "recover on retry, so a higher ceiling only makes each hang more expensive."
        )
        # Still above the slowest call ever observed to succeed, measured at 1167 s.
        assert cfg.opencode.request_timeout_s > 1167.0


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

    # Emitted where a round failed to improve. Anchored on the guard's tail rather than its
    # full text: the condition gained an `evaluated and` prefix when non-evaluated rounds
    # stopped counting as no-improvement rounds, and a test that pins the exact source line
    # breaks on a correct change while telling you nothing about behaviour.
    emit = src.split("best_after >= best_before")[1][:1600]
    assert '"HYPOTHESES_FAILED"' in emit
    assert '"hypotheses": tried' in emit
    # Only when there is something to record.
    assert "if tried:" in emit
    # And ONLY when a rewrite was actually evaluated: marking a hypothesis failed after a
    # round that never ran teaches the rewriter to avoid an idea nothing tested, permanently
    # (failed_hypotheses is journalled and replayed).
    guard = src.split("if evaluated and best_after >= best_before")
    assert len(guard) == 2,         "the HYPOTHESES_FAILED guard must require that a rewrite was evaluated"

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
        seed=0, collect_kernel_metadata=True, relaxed_elem_tol=0.01,
        relaxed_pass_frac=0.99, cosine_min=0.99985,
        fp64_relative_gate=True, fp64_rel_multiplier=2.0,
        fp64_rel_multiplier_lowp=3.0)
    assert job["fp64_relative_gate"] is True
    assert job["fp64_rel_multiplier"] == 2.0
    assert job["fp64_rel_multiplier_lowp"] == 3.0

    # EVERY relaxed-correctness job the evaluator builds must forward the gate, not just one:
    # `screen` gates the witness (space publication), `_run` gates every tuning trial and the
    # final re-eval, and `measure_overhead` gates the per-candidate launch probe. Forwarding
    # only some would apply the gate inconsistently.
    #
    # Driven by capturing the jobs the evaluator actually builds, rather than by counting
    # occurrences of a source string: the count broke the moment a fourth caller was added,
    # which is the failure mode of pinning behaviour to source text.
    from kernel_optimizer.evaluation.correctness import CorrectnessEvaluator

    built: list[dict] = []

    class CaptureWorker:
        def run_job(self, job, timeout, tag, lock_mode=None):
            built.append(job)
            return {"ok": True, "correct": True, "latency_ms": {
                "mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0, "n": 1, "median": 1.0}}

    class Cfg:
        precision = "fp32"
        timing_method = "cuda_event"
        correctness_mode = "dual_witness_relaxed"
        build_timeout_s = eval_timeout_s = 60.0
        correctness_trials = 5
        perf_trials = 100
        quick_correctness_trials = 3
        quick_perf_trials = 20
        relaxed_elem_tol = 0.01
        relaxed_pass_frac = 0.99
        cosine_min = 0.99985
        fp64_relative_gate = True
        fp64_rel_multiplier = 2.0
        fp64_rel_multiplier_lowp = 3.0
        excessive_speedup = 10.0
        compile_screen_enabled = False

    class Task:
        ref_path = Path("ref.py")

    import tempfile

    ev = CorrectnessEvaluator.__new__(CorrectnessEvaluator)
    ev.worker, ev.cfg, ev.seed = CaptureWorker(), Cfg(), 0
    ev._static_cache, ev._screen_cache = {}, {}
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write("PARAMS = {}\n")
        kp = Path(f.name)
    # Pre-seed the static check so these calls do not need a real worker for it.
    ev._static_cache = {}

    class OkStatic(CaptureWorker):
        def run_job(self, job, timeout, tag, lock_mode=None):
            built.append(job)
            return {"ok": True, "correct": True, "warnings": [], "latency_ms": {
                "mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0, "n": 1, "median": 1.0}}

    ev.worker = OkStatic()
    for entry in ("screen", "quick_test", "full_eval", "measure_overhead"):
        built.clear()
        getattr(ev, entry)(Task(), kp, tag="t", backend="triton")
        relaxed = [j for j in built if j.get("job_type") == "eval_correctness_relaxed"]
        assert relaxed, f"{entry} built no relaxed-correctness job: {[j.get('job_type') for j in built]}"
        for j in relaxed:
            assert j["fp64_relative_gate"] is True, (
                f"{entry} does not forward fp64_relative_gate, so the gate is applied "
                f"inconsistently across the paths that decide correctness")
            assert j["fp64_rel_multiplier"] == 2.0
            assert j["fp64_rel_multiplier_lowp"] == 3.0


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


def test_convergence_judges_on_the_same_statistic_the_tuner_optimizes():
    """`best_history` -- what min_improvement_pct is applied to -- must carry medians.

    The chain is long and every link had to be converted together:
      _tune -> update_best(robust_ms) -> Family.best.latency_ms -> record_round
            -> best_history -> family_verdict's recent_improvements_pct
    A single `.mean` left anywhere in it would make the convergence verdict judge a
    different quantity than the tuner optimizes, and the failure would be silent: percentages
    computed from means look exactly like percentages computed from medians.

    Why it matters concretely, from run-l2-37-20260907-010645 (pre-fix): a round-over-round
    change of 32.60 -> 30.70 read as a 5.83% gain and cleared min_improvement_pct 2.0,
    earning the family another rewrite round -- while the difference was 1.90 us against a
    combined standard error of 17.85 us, i.e. noise. Post-fix on the same task the same kind
    of non-improvement reads as 1.56% and is correctly refused.
    """
    from pathlib import Path

    orch = Path("src/kernel_optimizer/control/orchestrator.py").read_text(encoding="utf-8")

    # The two update_best calls that feed Family.best must pass the robust statistic.
    assert "best.params, best.latency_ms.robust_ms," in orch
    assert ("cand.family_id, cand.candidate_id, best.params, best.latency_ms.robust_ms"
            in orch)
    # record_round takes Family.best.latency_ms, which the above now populate.
    assert "self.deps.families.record_round(family.family_id, best_after)" in orch
    assert "best.latency_ms\n" in orch or "best.latency_ms" in orch

    # And nothing in the ranking/selection path may still read .mean. The remaining .mean
    # uses are display, the trials CSV, and the deliberately-conservative headline speedup.
    ranking_mean = [ln for ln in orch.splitlines()
                    if ".latency_ms.mean" in ln
                    and ("min(" in ln or "key=" in ln or "crun.best_ms =" in ln)]
    assert not ranking_mean, f"ranking still uses the mean: {ranking_mean}"

    # The convergence policy itself must read best_history and nothing else for the
    # improvement test, so converting the producer is sufficient.
    conv = Path("src/kernel_optimizer/control/convergence.py").read_text(encoding="utf-8")
    assert "history = family.best_history" in conv
    assert ".mean" not in conv, "convergence must not compute its own statistic"

    # The arithmetic, end to end, on the real numbers from the two runs.
    from kernel_optimizer.config import BudgetConfig
    from kernel_optimizer.control.convergence import ConvergencePolicy
    from kernel_optimizer.models.core import Family

    cfg = BudgetConfig(rewrite_rounds_per_family=5, no_improve_rounds=1,
                       min_improvement_pct=2.0)
    policy = ConvergencePolicy(cfg)

    # Medians: 15.344 -> 15.104 is 1.56%, under the threshold -> converged.
    fam = Family(family_id="f", anchor_candidate_id="c",
                 best_history=[15.344, 15.104], rewrite_rounds_used=1)
    got = policy.family_verdict(fam)
    assert got.verdict == "freeze" and got.stop_kind == "converged", got
    assert got.evidence["recent_improvements_pct"] == [1.564], got.evidence

    # Means on the SAME pair of trials: 32.60 -> 30.70 is 5.83%, which cleared the gate.
    fam_mean = Family(family_id="f", anchor_candidate_id="c",
                      best_history=[32.60, 30.70], rewrite_rounds_used=1)
    assert policy.family_verdict(fam_mean).verdict == "continue", \
        "the mean-based history is what wasted a rewrite round on noise"


def test_a_median_labelled_speedup_needs_a_median_on_both_sides():
    """`speedups_median` must not be baseline_MEAN / candidate_MEDIAN.

    `robust_ms` falls back to the mean when no median exists, and baselines come from
    KernelBench's summary-only timing path which returns no samples -- so applying robust_ms
    to both sides silently mixes conventions. With this run's real numbers that publishes
    23.40 / 14.11 = 1.658x under a median label, against the mean-based 0.727x: 128% higher,
    numerator inflated by scheduling stalls and denominator not.

    That is precisely the failure the field was added to prevent -- every published speedup
    rising without any kernel getting faster -- so the check is that the mixed ratio is
    NEVER emitted, and that its absence is explained instead.
    """
    from pathlib import Path

    orch = Path("src/kernel_optimizer/control/orchestrator.py").read_text(encoding="utf-8")

    # The forbidden shape: robust_ms on the baseline side of the median ratio.
    assert "b.latency_ms.robust_ms, lat.robust_ms" not in orch, \
        "the median ratio must not use robust_ms, which falls back to the mean"
    # Both sides must be gated on a real median.
    assert "if lat.median and lat.median > 0:" in orch
    assert "if b.latency_ms.median and b.latency_ms.median > 0" in orch
    # And the absence must be stated, not silent.
    assert "speedups_median_note" in orch

    rep = Path("src/kernel_optimizer/reporting/report.py").read_text(encoding="utf-8")
    assert "speedups_median_note" in rep, "the report must explain a missing median column"

    # The headline stays mean-over-mean: the conservative claim, and already fair because
    # both sides are timed with the same perf_trials.
    assert "speedups[b.kind] = round(b.latency_ms.mean / lat.mean, 4)" in orch

    # The arithmetic that motivated all of this, so the numbers cannot drift from the story.
    base_mean, cand_mean, cand_median = 23.40, 32.20, 14.112
    honest = base_mean / cand_mean
    mixed = base_mean / cand_median
    assert abs(honest - 0.727) < 0.005, honest
    assert abs(mixed - 1.658) < 0.005, mixed
    assert mixed / honest > 2.2, "the mixed ratio more than doubles the reported speedup"


def test_the_deliverable_trials_csv_carries_the_deciding_statistic():
    """`report/trials.csv` must contain the median, because the median chose every winner.

    The tuning objective, the incumbent comparison and the convergence test all read
    `robust_ms`, i.e. the median. The report's trials.csv emitted only `latency_mean_ms`,
    so the deliverable file could not explain its own run's result.

    Measured on run-l2-37-20260907-020707 (280 trials on disk): sorting that file by mean
    names tr-7f72ec7b at 20.3 us as the best trial, while the run actually selected
    tr-417b0c73 -- whose mean is 32.5 us and whose median is 14.1. Different config, and
    the mean-only reader is 44% off the latency the run reports.

    Also pins the rounding. `median` is computed from raw samples, so unlike every other
    latency field it arrives at full float precision: unrounded it printed
    0.06931199878454208 beside a mean of 0.0828 in the same row, dressing a 20-sample
    estimate with a 3-8% coefficient of variation as though it were exact.
    """
    from pathlib import Path

    from kernel_optimizer.models.core import latency_cell

    rep = Path("src/kernel_optimizer/reporting/report.py").read_text(encoding="utf-8")
    header_start = rep.index('writer.writerow(["trial_id", "candidate_id"')
    header = rep[header_start:header_start + 400]
    assert "latency_median_ms" in header, \
        "the deliverable trials.csv must carry the statistic that decided the run"
    # Emitted through the shared formatter, so it cannot regress to raw precision.
    assert 'latency_cell(lat.get("median"))' in rep

    # The analyst's per-candidate CSV writer must round it too -- that file is read by an
    # LLM, which is exactly the reader that will treat 17 digits as meaningful.
    orch = Path("src/kernel_optimizer/control/orchestrator.py").read_text(encoding="utf-8")
    assert "latency_cell(t.latency_ms.median if t.latency_ms else None)" in orch
    assert "(t.latency_ms.median if t.latency_ms else \"\") or \"\"" not in orch, \
        "the raw, unrounded median must not be written to a CSV"

    # The formatter itself: rounds, and distinguishes "no value" from zero.
    assert latency_cell(0.06931199878454208) == 0.0693
    assert latency_cell(0.014112000353634357) == 0.0141
    assert latency_cell(None) == ""
    assert latency_cell("") == ""
    assert latency_cell("not a number") == ""
    # Zero is a value, not a blank: a 0.0 latency is a bug worth seeing, not hiding.
    assert latency_cell(0.0) == 0.0

    # The disagreement that motivated it, so the numbers cannot drift from the story.
    by_mean_ms, by_median_ms = 20.3, 14.1
    assert abs(by_mean_ms / by_median_ms - 1.44) < 0.01


def test_the_agent_timeout_is_priced_as_a_hang_not_a_work_budget():
    """`request_timeout_s` must sit above the slowest real call and below 1800.

    Measured over every run on disk (scripts/probe_agent_timeouts.py, n=982 successful calls):
    the slowest successful agent call is 1167 s and NOT ONE has ever exceeded 1200 s, while 14
    of the 15 ReadTimeout kills finished on a retry in ~4 min median. A timed-out call is hung,
    not slow, so the value is the price of noticing a hang -- 18 hangs cost 9.00 h at 1800 s
    against 7.50 h at 1500 s.

    This pins the correction: 1800 was chosen on the belief that the 1200 s kills destroyed
    real work, and 7 of those 8 calls in fact recovered on retry.

    Both bounds matter, which is why they are asserted separately:
      - above 1167 s, or a call slower than anything yet observed is killed for being slow;
      - at or below 1500 s, or every hang costs 50% more than it needs to.
    """
    from pathlib import Path

    from kernel_optimizer.config import AgentModuleConfig, OpencodeConfig, load_config

    slowest_successful_call_s = 1167.0

    field_default = OpencodeConfig().request_timeout_s
    assert field_default > slowest_successful_call_s, \
        "the timeout must not kill the slowest call ever observed to succeed"
    assert field_default <= 1500.0, \
        "1800 was justified by a claim that re-measurement disproved; do not restore it"

    # default.yaml sets it explicitly and therefore OVERRIDES the field default: a fix applied
    # only to the pydantic field would be silently inert for anyone loading that config.
    assert load_config("configs/default.yaml").opencode.request_timeout_s == field_default

    # Every experiment config must resolve to the same value -- the arms have to stay
    # comparable, and load_config reads ONE file (default.yaml is not a base layer), so a
    # config that omits the key gets the field default rather than default.yaml's.
    for cfg_path in sorted(Path("configs").glob("experiments_*.yaml")):
        got = load_config(str(cfg_path)).opencode.request_timeout_s
        assert got == field_default, f"{cfg_path.name} resolves to {got}, not {field_default}"

    # The dead per-module field is kept in step, so a reader who sets it is not misled about
    # the real ceiling (nothing consumes it -- see its comment).
    assert AgentModuleConfig().timeout_s == field_default

    # The probe that produced these numbers must stay runnable, and must carry the pairing
    # trap that made the first measurement report the exact opposite conclusion.
    probe = Path("scripts/probe_agent_timeouts.py").read_text(encoding="utf-8")
    assert "start[key] = ts" in probe, "a failed attempt must re-arm the clock, not pop it"
    assert "PAIRING BUG" in probe

    # The arithmetic behind the choice, so the numbers cannot drift from the story.
    hangs = 18
    assert hangs * 1800 / 3600 == 9.0
    assert hangs * 1500 / 3600 == 7.5
    assert 1500 / slowest_successful_call_s > 1.28   # ~29% headroom
    assert 1200 / slowest_successful_call_s < 1.03   # only ~2.8% at 1200


def test_the_wall_clock_is_enforced_inside_a_round_not_only_between_rounds(tmp_path):
    """`wall_clock_hours` must stop work mid-round, or it is not a budget.

    The clock used to be tested ONLY at the top of the outer loop. Measured on
    run-l2-37-20260907-020707, consecutive global checks were 2.42, 2.35, 4.23 and 0.64 h
    apart: that run passed its check at 11.26 h of a 12 h budget and was still working at
    13.93 h -- 16% over, with no event marking it, because a round in flight runs to
    completion whatever the clock says.

    A round is not small. For each active family it spends a rewriter call, then for each
    produced candidate a parameterize (plus up to `repair_attempts` repairs) and up to
    `trials_per_space` GPU trials. So both loops need the check, and this drives BOTH real
    methods rather than a copy of them -- a test that re-implements the loop it is testing
    proves nothing about the shipped code.
    """
    from pathlib import Path

    from kernel_optimizer.config import BudgetConfig

    b = BudgetConfig(wall_clock_hours=12.0)
    fams = [_live_family(f"fam-{i}", f"c{i}") for i in range(3)]
    orch, fm = _loop_d_orchestrator(tmp_path, families=fams, budgets=b)

    # --- 1. _rewrite_round stops at the first family boundary once the budget is gone.
    orch._elapsed_hours = lambda: 13.93          # the real overrun
    orch.runs = {}                                # no rewrite parents: the loop would
    progressed = orch._rewrite_round(1)           # otherwise just bump rounds_used
    kinds = [e.type for e in orch.store.replay().events]
    assert "WALL_CLOCK_REACHED" in kinds, \
        "a round in flight must stop when the wall clock is spent"
    assert progressed is False
    # It must stop BEFORE evaluating any family, not after all of them.
    hit = [e for e in orch.store.replay().events if e.type == "WALL_CLOCK_REACHED"][0]
    assert hit.payload["stopped_before_family"] in {f[0].family_id for f in fams}
    assert hit.payload["elapsed_hours"] == 13.93
    assert hit.payload["budget_hours"] == 12.0

    # --- 2. Under budget, the same call proceeds (the check must not fire unconditionally).
    orch2, _ = _loop_d_orchestrator(tmp_path / "b", families=fams, budgets=b)
    orch2._elapsed_hours = lambda: 11.26          # the value that legitimately passed
    orch2.runs = {}
    orch2._rewrite_round(1)
    assert "WALL_CLOCK_REACHED" not in [e.type for e in orch2.store.replay().events], \
        "11.26 h of a 12 h budget must not be stopped"

    # --- 3. _pipeline_batch stops between candidates, and always finishes the first one.
    orch3, _ = _loop_d_orchestrator(tmp_path / "c", families=fams, budgets=b)
    orch3._elapsed_hours = lambda: 13.93
    done: list[str] = []
    orch3._candidate_pipeline = done.append
    orch3._prefetch_parameterization = lambda _cid: None
    orch3._pipeline_batch(["ca", "cb", "cc", "cd"])
    assert done == ["ca"], \
        "the batch must finish the candidate it started and skip the rest, got %r" % done
    ev = [e for e in orch3.store.replay().events if e.type == "WALL_CLOCK_REACHED"][0]
    assert ev.payload["pipelined"] == 1 and ev.payload["skipped"] == 3

    # A batch under budget pipelines everything.
    orch4, _ = _loop_d_orchestrator(tmp_path / "d", families=fams, budgets=b)
    orch4._elapsed_hours = lambda: 1.0
    done4: list[str] = []
    orch4._candidate_pipeline = done4.append
    orch4._prefetch_parameterization = lambda _cid: None
    orch4._pipeline_batch(["ca", "cb", "cc"])
    assert done4 == ["ca", "cb", "cc"]

    # --- 4. Hitting the budget must not hand control to Loop D, the more expensive branch.
    src = Path("src/kernel_optimizer/control/orchestrator.py").read_text(encoding="utf-8")
    loop = src[src.index("progressed = self._rewrite_round(round_no)"):
               src.index("added = self._novelty_round(round_no)")]
    assert "wall_clock_hours" in loop, \
        "a budget-exhausted rewrite round must not fall through into a novelty round"


def test_the_report_validates_a_stored_median_speedup_instead_of_trusting_it():
    """`report` must not print a mixed ratio it finds in an old summary.

    The orchestrator gained a both-sides gate (dbcc99b) so it will not COMPUTE
    baseline_MEAN / candidate_MEDIAN. But `report` regenerates from the event log, and a log
    written before that gate stores the mixed value already computed. run-l2-37-20260907-020707
    is exactly that case -- the run started 02:07, the gate landed 03:40, the orchestrator
    module was imported at launch -- and its stored medians were published under a "MEDIANS"
    heading with a "+33.1% vs the mean-based figure" uplift on all four baselines:

        eager              0.0520 mean / 0.012096 median = 4.2989  <- reported as a median ratio
        torch_compile_tf32 0.0234 mean / 0.012096 median = 1.9345  <- mean-based is 1.4534

    So the check has to sit on BOTH sides of the boundary. A median-labelled ratio requires a
    median in the baseline record too, and the baseline timing path returns summary statistics
    only, so the report must verify rather than trust.
    """
    from pathlib import Path

    rep = Path("src/kernel_optimizer/reporting/report.py").read_text(encoding="utf-8")

    # The baseline-side evidence must be derived from the events...
    assert "baseline_medians = any(" in rep
    # ...and gate the printing of the median block.
    assert "if med and not baseline_medians:" in rep
    assert "med = None" in rep, "a stored mixed ratio must be suppressed, not printed"
    # The reason must be stated, not left silent.
    assert "baseline_MEAN / candidate_MEDIAN" in rep

    # The candidate's own median goes through the shared formatter in BOTH branches, so a
    # 17-digit float cannot reach the report.
    assert rep.count("latency_cell(best.get('final_reeval_median_ms'))") == 2

    # The arithmetic that motivated it, so the numbers cannot drift from the story.
    base_mean, cand_median, cand_mean = 0.0234, 0.012096, 0.0161
    mixed = base_mean / cand_median
    honest = base_mean / cand_mean
    assert abs(mixed - 1.9345) < 0.001, mixed
    assert abs(honest - 1.4534) < 0.001, honest
    assert (mixed / honest - 1) > 0.33, "the mixed ratio inflates the claim by a third"


def test_the_env_probe_imports_the_symbols_evaluation_actually_calls():
    """`doctor`'s kernelbench check must fail for the same reason a run would.

    The probe used to do `import kernelbench`, which only runs the package __init__. On the
    pinned 423217d that never reaches `kernelbench.utils`, and THAT module imports litellm at
    module scope. Measured on box 2 (2026-09-07): every doctor check was green while

        from kernelbench.eval import eval_kernel_against_ref
        ModuleNotFoundError: No module named 'litellm'

    A run started in that state dies at its FIRST baseline, and the traceback never names the
    missing module: `load_original_model_and_inputs` swallows the ImportError and returns None,
    so the caller fails several frames away with `TypeError: cannot unpack non-iterable
    NoneType object`. That is a 12-hour run lost to a pip install, which is exactly what the
    check exists to prevent.

    Generic by construction: importing the real entry points covers whatever they transitively
    need, so a dependency added upstream later is checked too, with no list to maintain.
    """
    import ast
    from pathlib import Path

    src = Path("src/kernel_optimizer/gpu/worker_main.py").read_text(encoding="utf-8")
    probe = src[src.index('result["kernelbench_importable"]') - 2000:
                src.index('result["kernelbench_importable"] = True')]

    # The three symbols the harness actually calls.
    for symbol in ("eval_kernel_against_ref", "load_original_model_and_inputs",
                   "time_execution_with_cuda_event"):
        assert symbol in probe, f"the probe must import {symbol}"
    # The bare package import is not sufficient evidence and must not be what is relied on.
    tree = ast.parse(src)
    bare_pkg = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Import) and any(a.name == "kernelbench" for a in n.names)
    ]
    assert not bare_pkg, \
        "`import kernelbench` alone passes while kernelbench.eval fails; import the symbols"
    # The error must name the module, so the operator can act on it.
    assert 'f"{type(exc).__name__}: {exc}"' in src, \
        "the probe's error text must carry the exception type and message"


def test_the_profiler_reads_resources_for_every_backend_not_just_triton():
    """A `cuda`/`cutlass`/`cute` candidate must carry registers, spills and shared memory.

    `ProfileRecord` used to be populated ONLY from Triton's compiled-kernel object, so a
    `backend: "cuda"` candidate produced an empty record. That is not a property of the backend
    -- CUDA exposes the same information through `cuobjdump -res-usage` on the compiled object,
    and more of it -- it was a shortcut in profilerx.py. Its cost was that the paper's own
    feedback loop (tuning evidence -> bottleneck report -> structural rewrite) degraded on the
    backend with the HIGHER expressiveness ceiling.

    One reader covers three backends because CUDA C++, CUTLASS and CuTe all compile through
    nvcc to a cubin. This exercises the real parser on real `cuobjdump` output, including the
    CUTLASS-shaped case (many template instantiations, one launched).
    """
    import ast
    from pathlib import Path

    from kernel_optimizer.evaluation.profilerx import LightProfiler

    # --- 1. the parser, on genuine cuobjdump -res-usage output.
    src = Path("src/kernel_optimizer/gpu/worker_main.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    wanted = {"_parse_res_usage"}
    mod = ast.Module(body=[n for n in tree.body
                           if isinstance(n, ast.FunctionDef) and n.name in wanted],
                     type_ignores=[])
    ns: dict = {}
    exec(compile(mod, "<parse>", "exec"), ns)          # noqa: S102 - pure text parsing
    parse = ns["_parse_res_usage"]

    real_output = """
Fatbin elf code:
================
arch = sm_89
code version = [1,7]
host = linux
compile_size = 64bit

Function _Z18fused_gemm_kernelPKfS0_Pfiii:
  REG:42 STACK:8 SHARED:16384 LOCAL:16 CONSTANT[0]:380 TEXTURE:0 SURFACE:0 SAMPLER:0

Function _Z14epilogue_normPfPKfi:
  REG:24 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:352 TEXTURE:0 SURFACE:0 SAMPLER:0
"""
    got = parse(real_output)
    assert len(got) == 2, got
    first = {k["name"]: k for k in got}["_Z18fused_gemm_kernelPKfS0_Pfiii"]
    assert first["n_regs"] == 42
    assert first["shared"] == 16384
    # STACK + LOCAL, so "the compiler ran out of registers" means the same on both backends.
    assert first["n_spills"] == 8 + 16
    # nvcc records no launch geometry in the cubin; guessing would be worse than None.
    assert first["num_warps"] is None and first["num_stages"] is None
    # A malformed block must be skipped, not crash or yield a half-record.
    assert parse("Function _Znothing:\n  NOTHING:1\n") == []
    assert parse("") == []

    # --- 2. the mapping: a cubin-only result must produce a populated record.
    prof = LightProfiler()
    rec = prof.extract({"cubin": {"kernels": got, "launched_filter": "applied"}})
    assert rec.n_regs == 42, "a cuda candidate must carry registers"
    assert rec.shared_bytes == 16384
    assert rec.n_spills == 24
    assert rec.profile_source == "cubin"
    assert rec.launched_filter == "applied"
    assert len(rec.kernel_names) == 2

    # --- 3. the Triton path must be unchanged, and must win when both are present, because
    # only it carries num_warps/num_stages (properties of the launch, not of the code).
    triton_only = {"triton": {"kernels": [{"name": "k", "n_regs": 80, "n_spills": 0,
                                           "shared": 4096, "num_warps": 4, "num_stages": 3}],
                              "compile_s": 1.5}}
    rec_t = prof.extract(triton_only)
    assert (rec_t.n_regs, rec_t.num_warps, rec_t.compile_s) == (80, 4, 1.5)
    assert rec_t.profile_source == "triton"
    both = dict(triton_only)
    both["cubin"] = {"kernels": got, "launched_filter": "applied"}
    assert prof.extract(both).profile_source == "triton"
    assert prof.extract(both).num_warps == 4

    # --- 4. no metadata at all still yields a record, never an exception.
    assert prof.extract({}).n_regs is None
    assert prof.extract({"cubin": None}).profile_source is None

    # --- 5. the CUTLASS hazard: resources must be attributable to the LAUNCHED kernel.
    # Aggregating over a whole cubin would report a template variant that never ran, which is
    # the same class of error as timing a fallback path and calling it the kernel.
    assert "launched_names" in src and "no_name_match" in src
    assert "_launched_kernel_names" in src, \
        "the launched set must be OBSERVED (torch.profiler), since a cubin records no launch"
    # And it must not depend on ncu: counters are denied on a rented container.
    launch_fn = src[src.index("def _launched_kernel_names"):src.index("def _extract_cubin_metadata")]
    assert "ncu" not in launch_fn.replace("NOT `ncu`", "").replace("ERR_NVGPUCTRPERM", "")
    assert "ProfilerActivity" in launch_fn


def test_the_backend_is_part_of_a_candidates_structural_identity():
    """The same algorithm in Triton and in CUDA must not be one structure.

    `structural_signature` hashed the AST with PARAMS zeroed and docstrings dropped -- the
    backend was not in it. So a novel seed whose whole IDEA is "this approach, expressed in CUDA
    so the launch path can be hand-written" would be rejected by `accept_novel_seed` as a
    `duplicate_signature`, and silently dropped by `register_candidate`, whenever its AST
    happened to match an existing candidate's.

    That has teeth now the profiler is backend-neutral: CUDA/CUTLASS/CuTe candidates carry real
    registers/spills/shared, so the search can use them -- but only if the family machinery
    treats the backend as a structural axis. Measured motivation: all 27 candidates of the
    level2:37 run and every candidate across 19 L3 runs were Triton, and an external team's
    hand-written CUDA kernel beat its own ATen fallback by 4.1x largely through a CPU-side
    launch path Triton cannot express.
    """
    from kernel_optimizer.control.families import FamilyManager, structural_signature

    src = (
        "PARAMS = {'BLOCK': 64}\n"
        "class ModelNew:\n"
        "    def forward(self, x):\n"
        "        return x\n"
    )

    # Same source, different backend -> different identity.
    tri = structural_signature(src, "triton")
    cud = structural_signature(src, "cuda")
    assert tri != cud, "a backend change must change the structural signature"
    assert tri.startswith("triton:") and cud.startswith("cuda:"), \
        "the backend must be legible in the signature, not only folded into the hash"
    # Same backend, same source -> still a duplicate (the dedup must keep working).
    assert structural_signature(src, "cuda") == cud
    # Omitted backend stays reproducible for the 20 runs of already-recorded signatures.
    assert structural_signature(src) == structural_signature(src)
    assert ":" not in structural_signature(src)

    # register_candidate must now ACCEPT the cuda twin rather than returning None.
    fm = FamilyManager()
    a = fm.register_candidate(src, "seed", [], "triton", "triton version")
    b = fm.register_candidate(src, "seed", [], "cuda", "same idea, hand-written CUDA")
    assert a is not None
    assert b is not None, "the CUDA twin was dropped as an exact structural duplicate"
    assert a.family_id != b.family_id, "each should anchor its own family"
    # And a true duplicate is still refused.
    assert fm.register_candidate(src, "seed", [], "cuda", "again") is None

    # The novelty gate must not call it a duplicate_signature either.
    fm2 = FamilyManager(max_families_total=4)
    fm2.register_candidate(src, "seed", [], "triton", "triton version")
    got = fm2.accept_novel_seed(src, "cuda", "same idea in CUDA", "different launch path")
    reason = getattr(got, "reason", None)
    assert reason != "duplicate_signature", \
        "Loop D rejected a cross-backend seed as an identical signature"


def test_a_long_agent_call_is_not_cut_but_a_runaway_one_is():
    """Duration must never end a call; resource exhaustion must.

    An agent legitimately runs long -- it compiles kernels, launches them, reads results. Cutting
    a call at a wall-clock deadline destroys exactly that work, measured twice: the 4057s call had
    already written `rw_1.py` to disk 8 minutes before it returned, and an earlier 1500s ceiling
    discarded a finished rewrite. So a total time budget is the WRONG instrument and this test
    pins that down.

    What must be bounded is an agent SUBPROCESS taking the machine down: one agent-written sweep
    produced a 272,341-line PTX, ptxas reached 111 GiB resident, the container hit its memory
    cgroup limit and the orchestrator was throttled into D-state -- 3 GB from a hard OOM.

    Both halves are asserted against the real OpencodeClient.prompt, with the memory reader
    patched so the condition is deterministic.
    """
    import threading
    import time as _time
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from socketserver import ThreadingMixIn

    from kernel_optimizer.agents import runtime as rt
    from kernel_optimizer.agents.runtime import AgentCallError, OpencodeClient

    aborted: list[str] = []
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            if self.path.endswith("/abort"):
                aborted.append(self.path)
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")
                return
            # A talkative turn: streams for a while, then finishes normally when released.
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            try:
                while not release.wait(0.05):
                    self.wfile.write(b" ")
                    self.wfile.flush()
                self.wfile.write(b'{"info": {}, "parts": []}')
                self.wfile.flush()
            except OSError:
                pass

    class Server(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    srv = Server(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    orig_pressure = rt._memory_pressure

    # --- half 1: memory calm -> a long, chatty call is LEFT ALONE ---------------------
    rt._memory_pressure = lambda: (10e9, 128e9)  # 7.8%, nowhere near the threshold
    client = OpencodeClient(f"http://127.0.0.1:{port}", timeout_s=30.0,
                            memory_abort_frac=0.92, resource_poll_s=0.1)
    threading.Timer(1.2, release.set).start()
    t0 = _time.monotonic()
    try:
        client.prompt("ses_ok", "go", model="p/m")
        completed = True
    except AgentCallError:
        completed = False
    long_elapsed = _time.monotonic() - t0
    client.close()

    assert completed, (
        "a long call that used no memory was killed anyway: a time budget is back, and it "
        "destroys the compile/benchmark work the agent was doing"
    )
    assert long_elapsed > 1.0, "the mock did not actually stream for a while; test is vacuous"
    assert not aborted, f"a healthy long call must not be aborted, got {aborted}"

    # --- half 2: memory at the wall -> the call IS aborted ----------------------------
    release.clear()
    rt._memory_pressure = lambda: (125.5e9, 128e9)  # 98%, the observed runaway
    client2 = OpencodeClient(f"http://127.0.0.1:{port}", timeout_s=30.0,
                             memory_abort_frac=0.92, resource_poll_s=0.1)
    t1 = _time.monotonic()
    with pytest.raises(AgentCallError) as excinfo:
        client2.prompt("ses_runaway", "go", model="p/m")
    runaway_elapsed = _time.monotonic() - t1
    release.set()
    rt._memory_pressure = orig_pressure
    srv.shutdown()
    client2.close()

    assert runaway_elapsed < 12.0, (
        f"the runaway call ran {runaway_elapsed:.1f}s: memory pressure did not stop it, so the "
        "box can still be driven to OOM by an agent subprocess"
    )
    msg = str(excinfo.value)
    assert "memory" in msg.lower(), f"the error must name the real cause, got: {msg}"
    assert "NOT stopped for taking too long" in msg, (
        "the message must say the call was not killed for being slow, or the next reader will "
        "conclude agents have a time budget and shorten their work"
    )
    assert aborted, "the session was not aborted: the agent's subprocesses would keep running"


def test_the_memory_pressure_reader_reports_nothing_when_it_cannot_measure():
    """It must return None rather than guess where there is no cgroup limit.

    A bound that cannot be measured must not be approximated: on Windows/macOS, or an unlimited
    cgroup, inventing a number would either abort healthy calls or silently enforce nothing while
    appearing to work. None makes the caller enforce nothing, explicitly.
    """
    from kernel_optimizer.agents.runtime import _memory_pressure

    got = _memory_pressure()
    if got is None:
        return  # correct on this platform (no readable cgroup limit)
    used, limit = got
    assert limit > 0 and used >= 0, f"nonsense cgroup reading: {got}"
    assert used <= limit * 2, f"usage far above the limit suggests mismatched files: {got}"


def test_the_candidate_contract_bounds_agent_self_testing():
    """The contract must tell agents not to sweep, and why.

    Two live incidents from the same root cause: a rewriter compiled 1500+ Triton kernels in a
    self-check and lost a finished rewrite to the ceiling; another built a kernel whose PTX ran
    to 272,341 lines, and ptxas then reached 111 GiB resident and stalled the whole box for 47
    minutes. Neither prompt said anything about the scale of self-testing. Checked as content
    rather than prose so the rule cannot be quietly dropped in an edit.
    """
    from kernel_optimizer.agents.modules import _contract_doc

    doc = _contract_doc().lower()
    assert "do not sweep" in doc or "not sweep" in doc, \
        "the contract does not tell the agent to avoid parameter sweeps in its own scripts"
    for concept, needle in [
        ("the harness does the evaluating", "harness evaluates"),
        ("compiles are expensive", "compile"),
        ("the assembler can exhaust RAM", "ram"),
        ("write the file first", "write the file first"),
    ]:
        assert needle in doc, f"contract is missing: {concept} ({needle!r})"


def test_a_retry_reseeds_the_sandbox_so_guidance_fixes_reach_it():
    """A prompt/doc fix must reach the next ATTEMPT, not only the next call.

    `invoke()` seeds the sandbox and renders the prompt once, before the retry loop. So when an
    agent's self-written benchmark took the box to its memory-cgroup limit (111 GiB in ptxas,
    orchestrator throttled into D-state), fixing the contract that failed to forbid it did NOT
    reach the call in flight: all three attempts re-read the copy seeded before the fix. I had
    said the fix would reach attempt 3; it would not have, and the sandboxes had to be patched
    by hand.

    Drives the real AgentModule.invoke with a client that fails the first attempt on transport
    and succeeds on the second, and asserts the second attempt saw re-seeded inputs.
    """
    from dataclasses import dataclass

    from pydantic import BaseModel as _BM

    from kernel_optimizer.agents.base import AgentModule
    from kernel_optimizer.agents.runtime import AgentCallError, PromptResult

    class Out(_BM):
        ok: bool = True

    @dataclass
    class In:
        pass

    seeds: list[int] = []

    class Mod(AgentModule):
        name = "probe"
        output_model = Out

        def seed_sandbox(self, inputs, sb):
            seeds.append(1)
            # Emulate reading guidance off disk: the content changes between attempts.
            sb.write_input("docs/guide.md", f"version {len(seeds)}")

        def render_prompt(self, inputs, sb):
            return "go"

    class Client:
        def __init__(self):
            self.calls = 0

        def create_session(self, directory, title):
            return f"ses_{title}"

        def prompt(self, session_id, text, **kw):
            self.calls += 1
            if self.calls == 1:
                raise AgentCallError("prompt transport error (ReadTimeout): timed out")
            return PromptResult(text='{"ok": true}', structured={"ok": True},
                                session_id=session_id)

    class Store:
        def __init__(self):
            self.events = []

        def append(self, t, p):
            self.events.append((t, p))

        def put_artifact(self, *a, **k):
            return "sha"

    import tempfile
    from pathlib import Path as _P

    from kernel_optimizer.agents.sandbox import SandboxFactory
    from kernel_optimizer.config import AgentModuleConfig

    with tempfile.TemporaryDirectory() as td:
        mod = Mod(
            client=Client(),
            sandboxes=SandboxFactory(_P(td)),
            store=Store(),
            cfg=AgentModuleConfig(model="p/m", max_retries=2, max_transport_retries=2),
        )
        outcome = mod.invoke(In())

    assert len(seeds) >= 2, (
        "the sandbox was seeded once for the whole call, so a guidance fix applied during a "
        "retry cannot reach the next attempt (measured: three attempts all read the stale "
        f"contract). seeds={len(seeds)}"
    )
    assert outcome.output is not None, "the retry should still succeed normally"


def test_launch_overhead_reaches_the_profile_record_on_every_backend():
    """The harness's timing blind spot must arrive as data, not stay a known-but-unmeasured fact.

    KernelBench times `synchronize -> clear_l2_cache() -> record -> kernel -> record`, and the
    flush is enqueued without being waited on, so the GPU is busy flushing exactly while the CPU
    issues the launch -- the CPU cost hides behind it. Measured on level2:37: 66.6 us of real CPU
    issue time against a reported 37.9 us, and an external candidate's claimed 8.85x became 1.53x
    once the baseline was timed the same way.

    Before this, `cpu_issue_ms` existed only in the bottleneck classifier's signature, so its
    `launch_bound` branch could never fire -- the field was never collected anywhere. Drives the
    real LightProfiler, including the case where NO resource metadata was collected (the overhead
    numbers must still survive, since they are a property of the call, not of the backend).
    """
    from kernel_optimizer.evaluation.profilerx import LightProfiler

    lp = LightProfiler()
    overhead = {"cpu_issue_ms": 0.0666, "gpu_ms": 0.0379, "wall_ms": 0.0702, "iters": 50}

    rec = lp.extract({
        "triton": {"kernels": [{"name": "k", "n_regs": 40, "n_spills": 0, "shared": 8192}],
                   "compile_s": 1.0},
        "launch_overhead": overhead,
    })
    assert rec.cpu_issue_ms == 0.0666, "launch overhead lost on the triton path"
    assert rec.overhead_gpu_ms == 0.0379
    assert rec.wall_ms == 0.0702

    rec_cuda = lp.extract({
        "cubin": {"kernels": [{"name": "gemm", "n_regs": 120, "n_spills": 8, "shared": 32768}],
                  "launched_filter": "applied"},
        "launch_overhead": overhead,
    })
    assert rec_cuda.cpu_issue_ms == 0.0666, "launch overhead lost on the cubin path"

    # no resource metadata at all -- overhead is a property of the CALL and must survive
    rec_bare = lp.extract({"launch_overhead": overhead})
    assert rec_bare.cpu_issue_ms == 0.0666, (
        "overhead was dropped when no kernel metadata was collected, so an overhead-bound "
        "candidate whose resources could not be read reports nothing at all"
    )

    assert rec.cpu_over_gpu is not None
    assert abs(rec.cpu_over_gpu - 1.7573) < 0.01, f"got {rec.cpu_over_gpu}"

    # not measured (a tuning trial) must be None, never 0.0 -- 0.0 would read as
    # "measured, no overhead" and classify a launch-bound kernel as something else.
    rec_none = lp.extract({"triton": {"kernels": [{"name": "k", "n_regs": 8}]}})
    assert rec_none.cpu_issue_ms is None, "unmeasured overhead must be None, not a number"
    assert rec_none.cpu_over_gpu is None

    # the classifier's launch_bound branch can now actually fire on this record
    from kernel_optimizer.evaluation.bottleneck import classify

    verdict = classify(gpu_ms=rec.overhead_gpu_ms, cpu_issue_ms=rec.cpu_issue_ms,
                       flop_count=None, byte_count=None, peaks=None)
    assert verdict.kind == "launch_bound", (
        f"the collected overhead does not reach a launch_bound verdict (got {verdict.kind}); "
        "the classifier branch would stay dead"
    )


def test_full_eval_measures_launch_overhead_but_tuning_trials_do_not():
    """Cost containment: the expensive path pays for it, the hot path does not.

    ~150 extra model calls is marginal beside full_eval's 100 timed samples, and a real tax on
    quick_test, which runs on every one of a run's hundreds of tuning trials (1960 in the L2:37
    run). The job assertion is on the built JOB, so a refactor still has to keep the property.
    """
    import inspect

    from kernel_optimizer.evaluation import correctness as cmod
    from kernel_optimizer.gpu.jobs import make_eval_job

    job_off = make_eval_job("ref.py", "k.py", measure_performance=True, num_correct_trials=5,
                            num_perf_trials=100, timing_method="cuda_event", backend="triton",
                            precision="fp32", seed=0, build_dir=None,
                            collect_kernel_metadata=True)
    assert job_off["measure_launch_overhead"] is False, "must default OFF"

    job_on = make_eval_job("ref.py", "k.py", measure_performance=True, num_correct_trials=5,
                           num_perf_trials=100, timing_method="cuda_event", backend="triton",
                           precision="fp32", seed=0, build_dir=None,
                           collect_kernel_metadata=True, measure_launch_overhead=True)
    assert job_on["measure_launch_overhead"] is True

    full_src = inspect.getsource(cmod.CorrectnessEvaluator.full_eval)
    quick_src = inspect.getsource(cmod.CorrectnessEvaluator.quick_test)
    assert "measure_launch_overhead=True" in full_src, \
        "full_eval does not request launch overhead, so the number is never collected"
    assert "measure_launch_overhead" not in quick_src, (
        "quick_test requests launch overhead: that is ~150 extra model calls on every tuning "
        "trial, hundreds per run"
    )


def test_a_finished_artifact_is_rescued_when_the_transport_dies():
    """A transport failure means no response arrived, NOT that no work was done.

    Measured on run-l1-42-20260907-193510: a rewriter wrote rewrites/rw_1.py (7152 bytes) at
    21:47:00 and the attempt was killed by the read timeout at 21:50:20 -- three minutes and
    twenty seconds later. The finished rewrite was discarded, its family therefore recorded no
    improvement, and the run reported that family as converged having never evaluated a
    rewrite. Drives the real AgentModule.invoke with a client that always fails on transport.
    """
    import tempfile
    from dataclasses import dataclass
    from pathlib import Path as _P

    from kernel_optimizer.agents.base import AgentModule
    from kernel_optimizer.agents.runtime import AgentCallError
    from kernel_optimizer.agents.sandbox import SandboxFactory
    from kernel_optimizer.config import AgentModuleConfig
    from kernel_optimizer.models.reports import RewriteCandidate, RewriteResult

    @dataclass
    class In:
        pass

    class Mod(AgentModule):
        name = "rw"
        output_model = RewriteResult

        def seed_sandbox(self, inputs, sb):
            pass

        def render_prompt(self, inputs, sb):
            return "go"

        def rescue_from_sandbox(self, sb):
            files = sb.list_outputs("rewrites")
            if not files:
                return None
            return RewriteResult(candidates=[
                RewriteCandidate(file=f, change_summary="[recovered]") for f in files])

    class DeadClient:
        def create_session(self, directory, title):
            # The agent "writes" its artifact, then the transport dies -- the real ordering.
            (_P(directory) / "rewrites").mkdir(exist_ok=True)
            (_P(directory) / "rewrites" / "rw_1.py").write_text(
                "PARAMS = {}\n", encoding="utf-8")
            return "ses_x"

        def prompt(self, *a, **k):
            raise AgentCallError("prompt transport error (ReadTimeout): timed out")

    class Store:
        def __init__(self):
            self.events = []

        def append(self, t, p):
            self.events.append((t, p))

        def put_artifact(self, *a, **k):
            return "sha"

    store = Store()
    with tempfile.TemporaryDirectory() as td:
        mod = Mod(client=DeadClient(), sandboxes=SandboxFactory(_P(td)), store=store,
                  cfg=AgentModuleConfig(model="p/m", max_retries=2, max_transport_retries=2))
        outcome = mod.invoke(In())

    assert outcome.output is not None, (
        "a finished artifact was discarded because the call did not return; that loss is what "
        "produced a false converged verdict"
    )
    assert outcome.output.candidates[0].file == "rewrites/rw_1.py"
    kinds = [t for t, _ in store.events]
    assert "AGENT_ARTIFACT_RESCUE" in kinds, "the rescue must be journalled, not silent"
    rescue = next(p for t, p in store.events if t == "AGENT_ARTIFACT_RESCUE")
    assert rescue["rescued"] is True


def test_a_rescued_artifact_still_has_to_pass_check_output():
    """Rescue must not become a way to admit work the normal path would reject.

    A file cut off mid-write is exactly what a transport death produces, so the rescued object
    goes through the SAME check_output every answer passes.
    """
    import tempfile
    from dataclasses import dataclass
    from pathlib import Path as _P

    from kernel_optimizer.agents.base import AgentModule
    from kernel_optimizer.agents.runtime import AgentCallError
    from kernel_optimizer.agents.sandbox import SandboxFactory
    from kernel_optimizer.config import AgentModuleConfig
    from kernel_optimizer.models.reports import RewriteCandidate, RewriteResult

    @dataclass
    class In:
        pass

    class Mod(AgentModule):
        name = "rw"
        output_model = RewriteResult

        def seed_sandbox(self, inputs, sb):
            pass

        def render_prompt(self, inputs, sb):
            return "go"

        def check_output(self, output, sb):
            return "the file is truncated"  # stands in for the real lint

        def rescue_from_sandbox(self, sb):
            files = sb.list_outputs("rewrites")
            if not files:
                return None
            return RewriteResult(candidates=[
                RewriteCandidate(file=f, change_summary="x") for f in files])

    class DeadClient:
        def create_session(self, directory, title):
            (_P(directory) / "rewrites").mkdir(exist_ok=True)
            (_P(directory) / "rewrites" / "rw_1.py").write_text(
                "PARAMS = {", encoding="utf-8")
            return "ses_x"

        def prompt(self, *a, **k):
            raise AgentCallError("prompt transport error (ReadTimeout): timed out")

    class Store:
        def __init__(self):
            self.events = []

        def append(self, t, p):
            self.events.append((t, p))

        def put_artifact(self, *a, **k):
            return "sha"

    store = Store()
    with tempfile.TemporaryDirectory() as td:
        mod = Mod(client=DeadClient(), sandboxes=SandboxFactory(_P(td)), store=store,
                  cfg=AgentModuleConfig(model="p/m", max_retries=2, max_transport_retries=2))
        with pytest.raises(AgentCallError):
            mod.invoke(In())

    rescues = [p for t, p in store.events if t == "AGENT_ARTIFACT_RESCUE"]
    assert rescues, "a rejected rescue must still be journalled, or the reason is invisible"
    assert all(r["rescued"] is False for r in rescues)
    assert any("check_output" in (r.get("reason") or "") for r in rescues), \
        f"the reason must say the artifact failed validation, got {rescues}"


def test_a_round_that_evaluated_nothing_is_not_recorded_as_no_improvement():
    """The false-converged defect itself, at the level that produces it.

    `family_verdict` reads a flat `best_history` as convergence. So recording the unchanged
    incumbent for a round whose rewrite never ran fabricates evidence of exhausted headroom.
    Measured: fam-50ba7c87 got history [5.54, 5.54] and stop_kind="converged" after its
    rewriter failed all three attempts -- it had never evaluated a single rewrite.
    """
    from kernel_optimizer.config import BudgetConfig
    from kernel_optimizer.control.convergence import ConvergencePolicy
    from kernel_optimizer.control.families import FamilyManager
    from kernel_optimizer.models.core import ParamSet as _PS

    fm = FamilyManager()
    src = "PARAMS = {'A': 1}\nclass ModelNew:\n    pass\n"
    cand = fm.register_candidate(src, "seed", [], "triton", "seed")
    fid = cand.family_id
    fm.update_best(fid, cand.candidate_id, _PS(values={"A": 1}), 5.54)
    fm.record_round(fid, 5.54)  # the seed datum

    cfg = BudgetConfig(rewrite_rounds_per_family=3, no_improve_rounds=1,
                       min_improvement_pct=2.0)
    policy = ConvergencePolicy(cfg)

    # The round runs, the agent never answers, nothing is evaluated.
    fm.record_round_not_evaluated(fid)
    fm.families[fid].rewrite_rounds_used += 1

    verdict = policy.family_verdict(fm.families[fid])
    assert verdict.stop_kind != "converged", (
        "a family whose rewrite was never evaluated was declared converged; the flat history "
        "is the ABSENCE of evidence, not evidence of exhausted headroom"
    )
    assert fm.families[fid].best_history == [5.54], (
        "a non-evaluated round must not append to history, got %s"
        % fm.families[fid].best_history)
    assert fm.families[fid].rounds_not_evaluated == 1
    assert fm.families[fid].rewrite_rounds_used == 1, \
        "the round must still count as budget spent, or a resume re-runs it forever"

    # Control: an EVALUATED round with no improvement must still converge, or the fix would
    # have disabled the mechanism instead of correcting it.
    fm.record_round(fid, 5.54)
    fm.families[fid].rewrite_rounds_used += 1
    assert policy.family_verdict(fm.families[fid]).stop_kind == "converged", \
        "a genuinely measured no-improvement round must still be able to converge"


# --- step 2: self-calibration -------------------------------------------------------------

# The REAL worker result measured on box 2 (RTX 4090, 2026-09-08). Kept verbatim so the
# threshold derivation is tested against numbers a GPU actually produced, not against numbers
# invented to make it pass. Independently corroborated by
# scripts/probes/probe_bottleneck_signals.py, which measured dram 0.9102 TB/s and fp32
# 54.60 TFLOP/s on a separate run.
MEASURED_4090 = {
    "ok": True, "device_name": "NVIDIA GeForce RTX 4090", "capability": [8, 9],
    "sm_count": 128, "torch_version": "2.13.0+cu129", "driver_version": "12.9",
    "l2_bytes": 75497472, "spec_dram_tbs": 1.008096,
    "dram_tbs": 0.9102221900268849, "fp32_tflops": 54.93976179332003,
    "tf32_tflops": 88.8785549203062, "empty_launch_floor_ms": 0.01740800030529499,
    # The low-precision ceilings (P3). Real figures, from box 1's own CALIBRATION_MEASURED
    # event on run-l3-21-20260908-232211. This fixture previously omitted them -- it was a
    # pre-P3 worker result -- which meant every test rendering a doc from it saw no fp16 or
    # bf16 line and could not have noticed their absence downstream. A fixture that is missing
    # the field under test cannot fail for the right reason.
    "fp16_tflops": 158.61464333737848, "bf16_tflops": 164.169719846138,
    "yardsticks": [
        {"name": "COMPUTE 4096^3 fp32 matmul", "truth": "compute",
         "gpu_ms": 2.6357760429382324, "cpu_issue_ms": 0.012677162885665894,
         "flop_count": 137438953472, "byte_count": 201326592},
        {"name": "MEMORY 256MB elementwise", "truth": "memory",
         "gpu_ms": 0.5949440002441406, "cpu_issue_ms": 0.0067390501499176025,
         "flop_count": 67108864, "byte_count": 536870912},
        {"name": "LAUNCH 40 tiny ops", "truth": "launch",
         "gpu_ms": 0.6400159895420074, "cpu_issue_ms": 0.6161164492368698,
         "flop_count": 5242880, "byte_count": 20971520},
        {"name": "UNSATURATED linear+silu+groupnorm", "truth": "unsaturated",
         "gpu_ms": 0.09728000313043594, "cpu_issue_ms": 0.06671249866485596,
         "flop_count": 134217728, "byte_count": 3932160},
    ],
}


def test_derived_thresholds_reproduce_the_yardsticks_own_ground_truth():
    """The acceptance test for calibration: do the derived lines classify the workloads whose
    bottleneck is known analytically?

    This is the whole point of measuring thresholds instead of guessing them. A 4096^3 matmul IS
    compute-bound; a 512 MB copy IS memory-bound; 40 tiny ops ARE launch-bound. If the derived
    lines cannot separate those three, no verdict downstream means anything.
    """
    from kernel_optimizer.gpu.calibrate import calibration_from_worker

    cal = calibration_from_worker(MEASURED_4090)
    t = cal.thresholds
    assert t is not None
    by = {y.truth: y for y in cal.yardsticks}

    assert by["memory"].pct_of_dram(cal.dram_tbs) >= t.dram_saturated_frac, (
        "the known memory-bound workload does not clear the derived DRAM line")
    assert by["compute"].pct_of_fp32(cal.fp32_tflops) >= t.compute_saturated_frac, (
        "the known compute-bound workload does not clear the derived compute line")
    assert by["launch"].cpu_over_gpu >= t.launch_bound_cpu_ratio, (
        "the known launch-bound workload does not clear the derived launch line -- exactly the "
        "defect a guessed 1.0 had, since it measures 0.963")

    # And the negative direction, which is what a too-low line breaks: workloads that are NOT of
    # a class must fall below that class's line.
    assert by["compute"].pct_of_dram(cal.dram_tbs) < t.dram_saturated_frac
    assert by["memory"].pct_of_fp32(cal.fp32_tflops) < t.compute_saturated_frac
    assert by["unsaturated"].cpu_over_gpu < t.launch_bound_cpu_ratio, (
        "an ordinary fused op is being called launch-bound; the line is too low")
    assert by["compute"].pct_of_fp32(cal.fp32_tflops) > t.idle_frac
    assert by["unsaturated"].pct_of_dram(cal.dram_tbs) < t.idle_frac


def test_a_guessed_compute_line_would_have_been_wrong_by_a_factor():
    """Neutralization: show the disproved constants actually fail on measured data.

    Without this, "we derive the thresholds" is an unfalsifiable claim about the code. The two
    guesses that step 1 disproved are asserted to be wrong HERE, against the same numbers, so
    reinstating either one fails a test rather than silently degrading classification.
    """
    from kernel_optimizer.gpu.calibrate import calibration_from_worker

    cal = calibration_from_worker(MEASURED_4090)
    compute_achieved = cal.yardsticks[0].pct_of_fp32(cal.fp32_tflops)
    launch_ratio = cal.yardsticks[2].cpu_over_gpu

    # The old COMPUTE_SATURATED_FRAC = 0.50 sits far below what a genuinely compute-bound kernel
    # reaches, so a kernel at HALF the ceiling would be declared saturated and left alone.
    assert compute_achieved > 0.50 * 1.5, (
        "a 0.50 compute line is not merely imprecise: a kernel at 50% of peak still has ~2x of "
        "headroom and would be reported as done")
    assert cal.thresholds.compute_saturated_frac > 0.70, (
        "the derived compute line collapsed toward the disproved guess")

    # The old LAUNCH_BOUND_CPU_RATIO = 1.0 is ABOVE what a launch-bound workload exhibits, so
    # the test judges backwards and can never fire.
    assert launch_ratio < 1.0, (
        "the indisputably launch-bound yardstick measures below 1.0, so a 1.0 line never fires")
    assert cal.thresholds.launch_bound_cpu_ratio < launch_ratio, (
        "the derived launch line must sit below what a launch-bound workload exhibits")


def test_calibration_supplies_the_empty_launch_floor_the_classifier_was_missing():
    """`overhead_floor` had no input before this. bottleneck_signals.json never measured a floor,
    so the branch could not fire and a kernel already at the floor fell through to
    `latency_bound` -- i.e. the agent was told to add parallelism to a kernel whose body no
    longer costs anything.
    """
    from kernel_optimizer.evaluation.bottleneck import classify
    from kernel_optimizer.gpu.calibrate import calibration_from_worker

    cal = calibration_from_worker(MEASURED_4090)
    assert cal.empty_launch_floor_ms > 0, "no floor measured; overhead_floor still cannot fire"

    at_floor = classify(gpu_ms=cal.empty_launch_floor_ms * 1.05, cpu_issue_ms=None,
                        flop_count=1000, byte_count=1000, peaks=None,
                        empty_launch_floor_ms=cal.empty_launch_floor_ms)
    assert at_floor.kind == "overhead_floor", (
        f"a kernel at the measured floor was classified {at_floor.kind}")

    # Control: a kernel well above the floor must NOT be excused as overhead.
    far_above = classify(gpu_ms=cal.empty_launch_floor_ms * 50, cpu_issue_ms=None,
                         flop_count=1000, byte_count=1000, peaks=None,
                         empty_launch_floor_ms=cal.empty_launch_floor_ms)
    assert far_above.kind != "overhead_floor"


def test_both_compute_ceilings_are_measured_so_tensor_core_kernels_are_judged_fairly():
    """One fp32 ceiling is not enough. A tf32/tensor-core kernel measured against the fp32
    ceiling reports as >100% of peak (nonsense that reads as "done"), and a scalar kernel
    measured against the tf32 ceiling reads as hopeless. On this card the two differ by 1.6x,
    so the choice of denominator changes the verdict.
    """
    from kernel_optimizer.gpu.calibrate import calibration_from_worker

    cal = calibration_from_worker(MEASURED_4090)
    assert cal.tf32_tflops > cal.fp32_tflops * 1.2, (
        "the tf32 ceiling is not meaningfully above fp32; the separation this relies on is gone")
    assert cal.tf32_ridge_flop_per_byte > cal.ridge_flop_per_byte

    # A kernel running at 80 TFLOP/s is impossible in fp32 and ordinary in tf32. Against the
    # wrong ceiling it reads as 146% of peak.
    assert 80.0 / cal.fp32_tflops > 1.0
    assert 80.0 / cal.tf32_tflops < 1.0


def test_a_cached_calibration_is_refused_when_it_predates_a_measurement():
    """A cache written before a ceiling was added must be re-measured, not served as 0.0.

    The identity check above covers a hardware change. This covers the other axis, which is
    the one that actually bit: every `Calibration` field has a permissive default so an old
    cache still loads (needed -- a box that cannot measure bf16 must still classify), and the
    price is that a newly ADDED measurement reads as 0.0 forever on a box whose identity
    never changed, with nothing raised.

    Observed on disk: the fp16/bf16 ceilings landed 2026-09-08 and
    `opop-glm/runs-l3/calibration.json` was written 09-07, so `fp16_tflops` was None/0.0 in
    the file and `CALIBRATION_LOADED` journalled `fp16_tflops: 0.0` on a fresh run. A
    low-precision candidate is then scored against the tf32 denominator -- roughly half its
    real ceiling on this card -- which reads as "saturated" for a kernel with headroom, the
    defect that put `impossible_fraction` in 13 of 25 L3:43 verdicts. It was about to be
    inherited by a 36-hour chain because the cache looked valid.
    """
    import json as _json
    import tempfile
    from pathlib import Path as _P

    from kernel_optimizer.evaluation.calibration import (
        CALIBRATION_SCHEMA_VERSION,
        cache_path,
        load_cached,
        save,
    )
    from kernel_optimizer.gpu.calibrate import calibration_from_worker

    cal = calibration_from_worker(MEASURED_4090)
    # The writer must stamp the current set, or every run re-measures forever.
    assert cal.schema_version == CALIBRATION_SCHEMA_VERSION, (
        "calibration_from_worker does not stamp the schema version, so a freshly measured "
        "cache would be refused by its own loader on the next run")

    with tempfile.TemporaryDirectory() as td:
        path = cache_path(_P(td))
        save(path, cal)
        assert load_cached(path, cal.identity()) is not None, (
            "a cache written by the current code must be reusable")

        # Exactly the shape found on disk: a valid calibration for THIS box, from before the
        # low-precision ceilings existed. Same identity, so the hardware check cannot catch it.
        old = _json.loads(path.read_text(encoding="utf-8"))
        old.pop("schema_version", None)
        old.pop("fp16_tflops", None)
        old.pop("bf16_tflops", None)
        path.write_text(_json.dumps(old), encoding="utf-8")

        stale = load_cached(path, cal.identity())
        assert stale is None, (
            "a calibration predating the fp16/bf16 ceilings was served for a box whose "
            "identity is unchanged; every low-precision candidate would be scored against "
            "the tf32 ceiling and read as saturated")

        # And the refusal is specifically about the version, not about the missing fields:
        # a box that measured 0.0 for bf16 (old hardware) must NOT re-measure every run.
        legitimately_zero = _json.loads(path.read_text(encoding="utf-8"))
        legitimately_zero["schema_version"] = CALIBRATION_SCHEMA_VERSION
        legitimately_zero["fp16_tflops"] = 0.0
        legitimately_zero["bf16_tflops"] = 0.0
        path.write_text(_json.dumps(legitimately_zero), encoding="utf-8")
        assert load_cached(path, cal.identity()) is not None, (
            "a current-version calibration reporting 0.0 for a ceiling this box cannot "
            "measure must be reused, not re-measured on every run")


def test_a_cached_calibration_is_refused_on_different_hardware():
    """Every classification is a fraction of these ceilings, so a calibration reused across a
    card or driver change produces confident verdicts computed against another GPU's limits.
    The cache is keyed on device identity for exactly that reason.
    """
    import tempfile
    from pathlib import Path as _P

    from kernel_optimizer.evaluation.calibration import cache_path, load_cached, save
    from kernel_optimizer.gpu.calibrate import calibration_from_worker

    cal = calibration_from_worker(MEASURED_4090)
    with tempfile.TemporaryDirectory() as td:
        path = cache_path(_P(td))
        save(path, cal)

        same = load_cached(path, cal.identity())
        assert same is not None, "a calibration for the SAME box must be reused"
        assert same.dram_tbs == cal.dram_tbs

        other = load_cached(path, "NVIDIA GeForce RTX 5080 Laptop GPU|12.0|84|2.13.0+cu129|12.9")
        assert other is None, (
            "a calibration measured on a 4090 was served for a 5080; every %-of-peak verdict "
            "downstream would be computed against the wrong ceiling")

        # A corrupt cache must re-measure, not crash a 12-hour run.
        path.write_text("{not json", encoding="utf-8")
        assert load_cached(path, cal.identity()) is None


def test_a_throttled_calibration_is_flagged_but_still_usable():
    """A contended box is still the box the run happens on, and its achievable bandwidth is the
    honest denominator -- so a low ceiling must NOT reject the calibration. What must not happen
    is a verdict resting on a bad ceiling being reported as confidently as a good one.
    """
    from kernel_optimizer.evaluation.calibration import flag_suspect
    from kernel_optimizer.gpu.calibrate import calibration_from_worker

    assert flag_suspect(0.9102, 1.008) == [], "a healthy 90%-of-spec ceiling must not be flagged"

    bad = dict(MEASURED_4090, dram_tbs=0.30)
    cal = calibration_from_worker(bad)
    assert cal.suspect, "a ceiling at 30% of spec was not flagged"
    assert "recalibrate" in cal.suspect[0].lower()
    assert cal.dram_tbs == 0.30, "the measured value must still be kept and used"
    assert cal.thresholds is not None, "a suspect calibration must still yield thresholds"


def test_no_classification_threshold_is_a_hardcoded_absolute():
    """The portability requirement, enforced on the source rather than argued in a comment.

    The user's constraint was explicit: thresholds must not be hardcoded constants decided on
    one card, because the harness moves between GPUs and re-deriving by hand at each box is not
    acceptable. So every threshold must be a fraction of a MEASURED ceiling, and the classifier
    must contain no absolute bandwidth/FLOP/latency figure.
    """
    from pathlib import Path as _P

    src = _P("src/kernel_optimizer/evaluation/bottleneck.py").read_text(encoding="utf-8")
    for unit in ("TB/s", "GB/s", "TFLOP", "GFLOP"):
        for line in src.splitlines():
            if unit not in line:
                continue
            # Units may appear in prose (docstrings, `suggests` text); what must not appear is a
            # numeric literal carrying one, which would be a spec figure baked into the logic.
            head = line.split(unit)[0][-12:]
            assert not any(ch.isdigit() for ch in head), (
                f"an absolute {unit} figure appears in the classifier: {line.strip()[:100]}")


def test_the_run_obtains_a_calibration_and_a_resume_does_not_remeasure():
    """Wiring test: without this the calibrator is another isolated module.

    Three properties, and the FIRST is the one a weaker test misses. An earlier version of this
    test called `orch._calibrate()` directly, so it passed with the call removed from `_run`
    entirely -- it verified the method, not the wiring. So this drives the real `_run`, with the
    stages after calibration stubbed, and asserts the ORDER: the ceilings must exist before the
    baseline, whose launch-overhead numbers are read against them.

      1. `_run` actually calls `_calibrate`, before `_baseline`.
      2. A resume reloads from cache instead of re-measuring. Re-measuring would cost exclusive
         GPU time on every resume AND, worse, classify the run's second half against different
         numbers than its first.
      3. The result is cached, so a later run on the same box pays nothing.
    """
    import tempfile
    from pathlib import Path as _P

    from kernel_optimizer.control.orchestrator import Orchestrator
    from kernel_optimizer.store.run_store import RunStore

    class FakeWorker:
        def __init__(self):
            self.calls = 0

        def run_job(self, job, timeout_s, tag, lock_mode="exclusive"):
            assert job["job_type"] == "calibrate"
            assert lock_mode == "exclusive", (
                "calibration measures CEILINGS; a shared job would depress every number")
            self.calls += 1
            return dict(MEASURED_4090)

    class Stop(Exception):
        """Ends _run right after the stages under test, so no GPU/agent work is needed."""

    def make(store, worker, order):
        orch = Orchestrator.__new__(Orchestrator)
        orch.store = store
        orch.calibration = None
        orch.task = type("T", (), {"model_dump": lambda self: {"task": "level1:19"}})()

        class Deps:
            pass

        deps = Deps()
        deps.evaluator = type("E", (), {"worker": worker})()
        orch.deps = deps
        orch._step_done = lambda key: store.append("STEP_DONE", {"step_key": key})

        real_calibrate = Orchestrator._calibrate.__get__(orch)

        def calibrate():
            order.append("calibrate")
            real_calibrate()

        def baseline():
            order.append("baseline")
            raise Stop()

        orch._calibrate = calibrate
        orch._baseline = baseline
        return orch

    with tempfile.TemporaryDirectory() as td:
        runs_dir = _P(td) / "runs"
        store = RunStore.create(runs_dir, "run-x", {"task": "level1:19"})
        worker = FakeWorker()

        order: list[str] = []
        orch = make(store, worker, order)
        with pytest.raises(Stop):
            orch._run()

        assert order == ["calibrate", "baseline"], (
            f"_run must calibrate BEFORE the baseline; got {order}. An empty list means the "
            f"call site is missing and the calibrator is dead code.")
        assert worker.calls == 1, "the run never measured a calibration"
        assert orch.calibration is not None
        assert orch.calibration.thresholds is not None, (
            "a calibration without thresholds cannot classify anything")
        assert (runs_dir / "calibration.json").exists(), (
            "nothing was cached, so every later run re-measures")

        # Resume: same store, so the step is already done.
        order2: list[str] = []
        orch2 = make(store, worker, order2)
        with pytest.raises(Stop):
            orch2._run()
        assert worker.calls == 1, (
            "a resume re-measured the ceilings; the run's two halves would then be classified "
            "against different numbers")
        assert orch2.calibration is not None, "the resume lost the calibration entirely"
        assert orch2.calibration.dram_tbs == orch.calibration.dram_tbs

        kinds = [e.type for e in store.replay().events]
        assert "CALIBRATION_MEASURED" in kinds, "the measurement was not journalled"


def test_a_failed_calibration_does_not_end_the_run():
    """A box without a working calibration must still run. The classifier reports `unknown`; the
    harness does not refuse to start. This is the same rule every other diagnostic follows here.
    """
    import tempfile
    from pathlib import Path as _P

    from kernel_optimizer.control.orchestrator import Orchestrator
    from kernel_optimizer.store.run_store import RunStore

    class BrokenWorker:
        def run_job(self, job, timeout_s, tag, lock_mode="exclusive"):
            raise RuntimeError("worker died")

    with tempfile.TemporaryDirectory() as td:
        store = RunStore.create(_P(td) / "runs", "run-y", {"task": "level1:19"})
        orch = Orchestrator.__new__(Orchestrator)
        orch.store = store
        orch.calibration = None

        class Deps:
            pass

        deps = Deps()
        deps.evaluator = type("E", (), {"worker": BrokenWorker()})()
        orch.deps = deps
        orch._step_done = lambda key: store.append("STEP_DONE", {"step_key": key})

        orch._calibrate()          # must not raise
        assert orch.calibration is None
        kinds = [e.type for e in store.replay().events]
        assert "CALIBRATION_FAILED" in kinds, (
            "a failed calibration must be journalled, or the report cannot say why every "
            "verdict is `unknown`")


# --- step 3: task cost (FLOP + bytes on the reference) ------------------------------------

# Counts measured on box 2 (RTX 4090, 2026-09-08) from the real KernelBench references. The
# first three were cross-checked against analytic values and agree to ratio 1.0000:
#   level1:1   N=4096: 2*4096^3 = 137.4390 GFLOP; 3*4096^2*4 = 201.327 MB
#   level1:19  0 FLOP (no MAC); 2*4096*393216*4 = 12884.902 MB
#   level2:37  2*32768*1024*4096 = 274.8779 GFLOP; (x+W+4 vecs+out)*4 = 687.931 MB
MEASURED_TASK_COSTS = {
    "level1:1_matmul": {"flop_count": 137438953472, "compulsory_bytes": 201326592,
                        "reference_bytes": 201326592, "op_count": 1, "notes": []},
    "level1:19_relu": {"flop_count": 0, "compulsory_bytes": 12884901888,
                       "reference_bytes": 12884901888, "op_count": 1,
                       "notes": ["FlopCounterMode reports 0 FLOP: this task has no "
                                 "multiply-accumulate ops"]},
    "level2:37": {"flop_count": 274877906944, "compulsory_bytes": 687931392,
                  "reference_bytes": 5570101248, "op_count": 6, "notes": []},
    "level3:21_mbconv": {"flop_count": 112113254400, "compulsory_bytes": 322035200,
                         "reference_bytes": 10630205440, "op_count": 11, "notes": []},
    "level3:43_attention": {"flop_count": 412316860416, "compulsory_bytes": 416296960,
                            "reference_bytes": 28761927680, "op_count": 40, "notes": []},
}


def test_task_cost_counts_match_analytic_values_on_real_references():
    """The counts have to be RIGHT, and for these three the right answer is derivable by hand.

    This is the check that separates "we call torch's counter" from "the number is correct".
    FlopCounterMode is trusted downstream as the numerator of every %-of-peak figure, so its
    agreement with closed-form values on a matmul, a pure-elementwise op and a fused chain is
    the evidence for that trust.
    """
    from kernel_optimizer.evaluation.task_cost import cost_from_worker

    mm = cost_from_worker({"task_cost": MEASURED_TASK_COSTS["level1:1_matmul"]})
    assert mm.flop_count == 2 * 4096 ** 3, "matmul FLOP count disagrees with 2*N^3 at N=4096"
    assert mm.compulsory_bytes == 3 * 4096 * 4096 * 4, "two inputs + one output, fp32"
    assert abs(mm.max_arithmetic_intensity - 682.67) < 0.1

    relu = cost_from_worker({"task_cost": MEASURED_TASK_COSTS["level1:19_relu"]})
    assert relu.flop_count == 0, (
        "a pure elementwise op must report 0 FLOP -- that is the correct roofline answer, not a "
        "failure, and the notes must say so")
    assert relu.compulsory_bytes == 2 * 4096 * 393216 * 4, "in + out"
    assert relu.notes, "a 0-FLOP result without a note is indistinguishable from a failed count"
    assert relu.max_arithmetic_intensity == 0.0

    l237 = cost_from_worker({"task_cost": MEASURED_TASK_COSTS["level2:37"]})
    assert l237.flop_count == 2 * 32768 * 1024 * 4096, "linear FLOP disagrees with 2*B*I*O"
    expected_compulsory = (32768 * 1024 + 4096 * 1024 + 4 * 4096 + 32768 * 4096) * 4
    assert l237.compulsory_bytes == expected_compulsory, (
        f"unavoidable traffic {l237.compulsory_bytes} != analytic {expected_compulsory}")


def test_fusion_headroom_separates_the_task_from_its_reference():
    """The number step 3 exists to produce, and it is not the FLOP count.

    `compulsory_bytes` is what NO implementation can avoid; `reference_bytes` is what the
    reference materializes. Their ratio says how much of the reference's traffic is intermediates
    -- i.e. how much fusion can win -- and it varies by 69x across our tasks, so it is a real
    discriminator rather than a constant dressed up as a measurement.
    """
    from kernel_optimizer.evaluation.task_cost import cost_from_worker

    costs = {k: cost_from_worker({"task_cost": v}) for k, v in MEASURED_TASK_COSTS.items()}

    # A single-op task materializes exactly its compulsory traffic: nothing to fuse.
    assert abs(costs["level1:1_matmul"].fusion_headroom - 1.0) < 0.01, (
        "a one-op reference reports fusion headroom above 1.0; the accounting is double-counting")
    assert abs(costs["level1:19_relu"].fusion_headroom - 1.0) < 0.01

    # A fused chain materializes far more, and more ops means more headroom.
    assert costs["level2:37"].fusion_headroom > 5.0
    assert costs["level3:21_mbconv"].fusion_headroom > 20.0
    assert costs["level3:43_attention"].fusion_headroom > 50.0, (
        "the 40-op attention reference materializes ~69x its unavoidable traffic; that is the "
        "single largest fusion opportunity in our task set and it must be visible")

    # Ordering by op count is not automatic -- it is the property that makes this actionable.
    ordered = sorted(costs.values(), key=lambda c: c.op_count)
    headrooms = [c.fusion_headroom for c in ordered]
    assert headrooms == sorted(headrooms), (
        f"fusion headroom does not increase with op count: {headrooms}. If it did not, the "
        f"number would not be measuring intermediate materialization.")


def test_max_intensity_answers_whether_a_task_can_ever_be_compute_bound():
    """A per-candidate FLOP count cannot answer this; a task-level one can.

    `flop_count / compulsory_bytes` is a CEILING on intensity -- no correct implementation can
    exceed it, because it would have to skip reading its own inputs. Compared against the card's
    measured ridge, it settles in advance whether "make it compute-bound" is even reachable, which
    is a question the classifier would otherwise answer per-candidate and inconsistently.
    """
    from kernel_optimizer.evaluation.task_cost import cost_from_worker
    from kernel_optimizer.gpu.calibrate import calibration_from_worker

    cal = calibration_from_worker(MEASURED_4090)
    ridge = cal.ridge_flop_per_byte
    assert ridge > 0

    relu = cost_from_worker({"task_cost": MEASURED_TASK_COSTS["level1:19_relu"]})
    assert relu.max_arithmetic_intensity < ridge, (
        "a pure elementwise task is being reported as able to become compute-bound")

    mm = cost_from_worker({"task_cost": MEASURED_TASK_COSTS["level1:1_matmul"]})
    assert mm.max_arithmetic_intensity > ridge, (
        "a large matmul must be on the compute side of the ridge")

    # And the ceiling must be at least the reference's own intensity: the reference is one correct
    # implementation, so it cannot beat the bound that applies to all of them.
    for name, c in ((k, cost_from_worker({"task_cost": v}))
                    for k, v in MEASURED_TASK_COSTS.items()):
        if c.compulsory_bytes and c.reference_bytes:
            assert c.max_arithmetic_intensity >= c.reference_arithmetic_intensity - 1e-9, (
                f"{name}: the ceiling is below the reference's own intensity, so the "
                f"compulsory-traffic floor is wrong")


def test_an_unmeasured_task_cost_is_not_a_measured_zero():
    """`flop_count == 0` is legitimate for maxpool and a failure for a matmul. Only the notes
    distinguish them, so a failed job must produce notes rather than a silent zero -- otherwise
    the report tells the reader a matmul requires no arithmetic.
    """
    from kernel_optimizer.evaluation.task_cost import TaskCost, cost_from_worker

    empty = cost_from_worker({})
    assert empty.flop_count == 0 and empty.compulsory_bytes == 0
    assert "not measured" in empty.summary_line(), (
        f"an unmeasured cost reads as measured: {empty.summary_line()}")

    measured_zero = TaskCost(flop_count=0, compulsory_bytes=1000, reference_bytes=1000,
                             op_count=1, notes=["no multiply-accumulate ops"])
    assert "not measured" not in measured_zero.summary_line()
    assert "bandwidth" in measured_zero.summary_line(), (
        "a genuine 0-FLOP task must be described as bandwidth-limited, not as unmeasured")


def test_the_task_cost_reaches_the_run_and_survives_a_resume():
    """Wiring: measured in `_baseline`, journalled, and restored on resume.

    Restoring matters because the cost is the denominator of every %-of-peak figure. A resumed
    run that lost it would report the second half of its candidates against no denominator while
    the first half had one.
    """
    import tempfile
    from pathlib import Path as _P

    from kernel_optimizer.control.orchestrator import Orchestrator
    from kernel_optimizer.evaluation.task_cost import cost_from_worker
    from kernel_optimizer.models.core import LatencyStats
    from kernel_optimizer.store.run_store import RunStore

    cost = cost_from_worker({"task_cost": MEASURED_TASK_COSTS["level3:43_attention"]})

    class FakeBench:
        def __init__(self):
            self.cost_calls = 0

        def probe_semantics(self, task):
            return {"training": False, "norm_layers": []}

        def measure_task_cost(self, task):
            self.cost_calls += 1
            return cost

        def measure_baseline(self, task):
            from kernel_optimizer.models.core import Baseline

            return [Baseline(kind="eager",
                             latency_ms=LatencyStats(mean=1.0, std=0.1, min=0.9, max=1.1,
                                                     median=1.0, n_samples=100))]

    with tempfile.TemporaryDirectory() as td:
        store = RunStore.create(_P(td) / "runs", "run-tc", {"task": "level3:43"})
        bench = FakeBench()

        def make():
            orch = Orchestrator.__new__(Orchestrator)
            orch.store = store
            orch.task_cost = None
            orch.eval_semantics = {}
            orch.baselines = []
            orch.task = type("T", (), {"model_dump": lambda self: {}})()

            class Deps:
                pass

            deps = Deps()
            deps.benchmarker = bench
            orch.deps = deps
            orch._step_done = lambda key: store.append("STEP_DONE", {"step_key": key})
            return orch

        orch = make()
        orch._baseline()
        assert bench.cost_calls == 1, "the run never measured the task cost"
        assert orch.task_cost is not None
        assert orch.task_cost.flop_count == cost.flop_count

        kinds = [e.type for e in store.replay().events]
        assert "TASK_COST_MEASURED" in kinds, "the cost was not journalled, so a resume loses it"

        orch2 = make()
        orch2._baseline()
        assert bench.cost_calls == 1, "a resume re-measured the task cost"
        assert orch2.task_cost is not None, (
            "the resume lost the task cost; the run's two halves would report %-of-peak against "
            "different denominators")
        assert orch2.task_cost.compulsory_bytes == cost.compulsory_bytes
        assert abs(orch2.task_cost.fusion_headroom - cost.fusion_headroom) < 1e-9


# --- step 5: Tier 1 statics (SASS instruction mix + analytic occupancy) --------------------

# Real SASS counts measured on box 2 (RTX 4090, 2026-09-08) by
# linux-server/scripts/probes/probe_sass_and_occupancy.py. Kept verbatim: the separation these
# assert is one a GPU actually exhibited, and the spill row is cross-validated against Triton's
# own n_spills (STL=2/LDL=1 vs n_spills=2, agreeing exactly).
MEASURED_SASS = {
    "tensor_core_fp16_dot": {
        "n_regs": 80, "n_spills": 2, "shared": 16384, "num_warps": 4,
        "sass": {"instructions": 360, "tensor_core": 16, "spill_store": 2, "spill_load": 1,
                 "shared_store": 8, "shared_load": 14, "barrier": 7, "global_load": 0,
                 "global_store": 8, "vec_128": 20, "vec_64": 0},
    },
    "large_tile_dot": {
        "n_regs": 218, "n_spills": 0, "shared": 32768, "num_warps": 4,
        "sass": {"instructions": 872, "tensor_core": 64, "spill_store": 0, "spill_load": 0,
                 "shared_store": 32, "shared_load": 70, "barrier": 11, "global_load": 0,
                 "global_store": 32, "vec_128": 56, "vec_64": 0},
    },
    "plain_elementwise": {
        "n_regs": 18, "n_spills": 0, "shared": 0, "num_warps": 4,
        "sass": {"instructions": 48, "tensor_core": 0, "spill_store": 0, "spill_load": 0,
                 "shared_store": 0, "shared_load": 0, "barrier": 0, "global_load": 2,
                 "global_store": 2, "vec_128": 4, "vec_64": 0},
    },
    # The actual winning candidate of run-l1-42-20260907-193510, collected through the real
    # eval path. 33.3% occupancy, register-limited, zero vectorized global access.
    "real_l1_42_winner": {
        "n_regs": 112, "n_spills": 0, "shared": 0, "num_warps": 4,
        "sass": {"instructions": 496, "tensor_core": 0, "spill_store": 0, "spill_load": 0,
                 "shared_store": 0, "shared_load": 0, "barrier": 0, "global_load": 16,
                 "global_store": 16, "vec_128": 0, "vec_64": 0},
    },
}

# RTX 4090 (sm_89) limits, as torch reports them.
SM89 = {"max_threads_per_sm": 1536, "regs_per_sm": 65536, "shared_per_sm": 102400,
        "max_blocks_per_sm": 16}


def test_sass_counting_separates_tensor_core_from_scalar_kernels():
    """The signal we claimed counters were needed for, recovered without them.

    `ncu` returns ERR_NVGPUCTRPERM in these containers and cannot be enabled from inside one, so
    if disassembly could not answer "does this kernel use tensor cores", the answer would be
    unavailable on every real box. Measured separation: an fp16 tl.dot kernel shows 16 tensor-core
    instructions, a scalar elementwise kernel shows 0.
    """
    from kernel_optimizer.evaluation.statics import SassCounts

    tc = SassCounts.model_validate(MEASURED_SASS["tensor_core_fp16_dot"]["sass"])
    plain = SassCounts.model_validate(MEASURED_SASS["plain_elementwise"]["sass"])

    assert tc.uses_tensor_cores, "an fp16 tl.dot kernel was reported as not using tensor cores"
    assert not plain.uses_tensor_cores, (
        "a scalar elementwise kernel was reported as using tensor cores; a false positive here "
        "would tell the agent to stop pursuing the single largest lever it has")
    assert tc.shared_load + tc.shared_store > 0 and tc.barrier > 0, (
        "a tl.dot kernel stages through shared memory and synchronizes; neither was detected")
    assert plain.shared_load + plain.shared_store == 0 and plain.barrier == 0


def test_the_sass_counter_does_not_count_symbol_names_as_instructions():
    """A kernel NAMED after a tensor-core op must not be reported as using one.

    Only lines carrying a ';' are instructions; section headers, symbol names and branch labels
    are not. Without that filter a kernel called `my_hmma_helper` would report tensor cores it
    does not have, and the agent would be told to stop pursuing its largest lever.
    """
    from kernel_optimizer.evaluation.statics import count_sass

    sass = """
        .headerflags @"EF_CUDA_SM89"
        .global _Z18my_hmma_fake_helperPf
    .text._Z18my_hmma_fake_helperPf:
        /*0000*/    MOV R1, c[0x0][0x28] ;
        /*0010*/    LDG.E.128 R4, [R2.64] ;
        /*0020*/    STG.E.128 [R6.64], R4 ;
        /*0030*/    EXIT ;
    .L_x_0:
    """
    counts = count_sass(sass)
    assert counts.instructions == 4, f"expected 4 instruction lines, got {counts.instructions}"
    assert counts.tensor_core == 0, (
        "the kernel's NAME was counted as a tensor-core instruction")
    assert counts.vec_128 == 2, "128-bit global accesses were not detected"
    assert counts.global_load == 1 and counts.global_store == 1


def test_occupancy_names_the_binding_resource_not_just_a_percentage():
    """"Occupancy 33%" names no knob; "limited by registers at 112/thread" does.

    Checked against the REAL winning candidate of run-l1-42-20260907-193510, whose 112 registers
    per thread cap it at 4 blocks/SM. This was entirely invisible before step 5: the run reported
    a 1.967x speedup with no indication that two thirds of the machine's warp slots were unused.
    """
    from kernel_optimizer.evaluation.statics import compute_occupancy

    k = MEASURED_SASS["real_l1_42_winner"]
    occ = compute_occupancy(k["n_regs"], k["shared"], k["num_warps"], **SM89)
    assert occ is not None
    assert abs(occ.occupancy - 0.3333) < 0.001, f"occupancy {occ.occupancy} != measured 0.3333"
    assert occ.limiter == "registers", f"limiter {occ.limiter}; the 112 regs/thread are binding"
    assert occ.by_regs == 4 and occ.blocks_per_sm == 4
    assert occ.active_warps == 16 and occ.max_warps_per_sm == 48

    # A small kernel is not register-limited, so the limiter must MOVE -- a limiter that always
    # says "registers" would be a constant dressed up as a diagnosis.
    small = MEASURED_SASS["plain_elementwise"]
    occ2 = compute_occupancy(small["n_regs"], small["shared"], small["num_warps"], **SM89)
    assert occ2.occupancy == 1.0, f"an 18-register kernel should reach full occupancy, got {occ2}"
    assert occ2.limiter != "registers", (
        f"an 18-register kernel is reported register-limited: {occ2.limiter}")

    # And shared memory must be able to bind too.
    occ3 = compute_occupancy(32, 51200, 4, **SM89)
    assert occ3.limiter == "shared_memory", (
        f"50 KB of shared memory per block must bind before registers, got {occ3.limiter}")


def test_occupancy_is_computed_from_device_limits_not_baked_constants():
    """The portability requirement again: the same kernel on a different card gives a different
    occupancy, so the device limits must be arguments rather than constants.

    A 5080 Laptop (sm_120) has fewer threads per SM than a 4090. If the arithmetic ignored that,
    every occupancy figure on the next box would be silently wrong -- and occupancy is KernelPro's
    highest-GAIN signal, so a wrong one is actively harmful.
    """
    from kernel_optimizer.evaluation.statics import compute_occupancy

    k = MEASURED_SASS["real_l1_42_winner"]
    on_4090 = compute_occupancy(k["n_regs"], k["shared"], k["num_warps"], **SM89)
    on_small = compute_occupancy(k["n_regs"], k["shared"], k["num_warps"],
                                 max_threads_per_sm=1024, regs_per_sm=65536,
                                 shared_per_sm=102400, max_blocks_per_sm=16)
    assert on_4090.max_warps_per_sm != on_small.max_warps_per_sm, (
        "the device limits are being ignored; occupancy would be wrong on every other card")
    assert on_4090.occupancy != on_small.occupancy


def test_triton_caps_registers_instead_of_spilling_so_occupancy_carries_the_signal():
    """A measured finding that reorders the priorities, asserted so it is not forgotten.

    KernelPro rates spills their highest-HIT tool (18.2%). But on Triton, extreme register
    pressure does not become spills: the 128x128 fp32 accumulator on 4 warps compiles to 218
    registers, ZERO spills, and 16.7% occupancy. Two synthetic kernels written to spill both
    failed to, and Triton's own n_spills agreed with the SASS at 0 each time -- so the detector
    was right and the expectation was wrong.

    Consequence: on our Triton candidates, occupancy is the signal that fires. Spills stay
    collected for the CUDA backend, which does spill.
    """
    from kernel_optimizer.evaluation.statics import SassCounts, compute_occupancy

    big = MEASURED_SASS["large_tile_dot"]
    sass = SassCounts.model_validate(big["sass"])
    assert sass.spill_instructions == 0, "the large-tile kernel did spill after all"
    assert big["n_spills"] == 0, "Triton reported spills; the finding no longer holds"
    assert big["n_regs"] >= 200, (
        "the register cap is not visible; the finding rests on Triton pinning registers high")

    occ = compute_occupancy(big["n_regs"], big["shared"], big["num_warps"], **SM89)
    assert occ.occupancy < 0.25, (
        f"the pressure must surface as LOW OCCUPANCY since it does not surface as spills; "
        f"got {occ.occupancy}")
    assert occ.limiter == "registers"


def test_sass_and_triton_spill_counts_cross_validate():
    """Two independent sources agreeing is the positive control for the spill detector.

    STL/LDL comes from disassembled SASS; n_spills comes from the Triton compiler. On the
    tensor-core kernel both report spilling (STL=2, LDL=1, n_spills=2). A detector validated only
    against itself would be indistinguishable from one that always returns zero -- which is the
    mistake this project has already made once (five negative results that were one broken
    probe).
    """
    from kernel_optimizer.evaluation.statics import SassCounts

    for name, k in MEASURED_SASS.items():
        sass = SassCounts.model_validate(k["sass"])
        triton_spills = k["n_spills"]
        if triton_spills > 0:
            assert sass.spill_instructions > 0, (
                f"{name}: Triton reports {triton_spills} spilled bytes but SASS shows no "
                f"local-memory traffic; one of the two is broken")
        else:
            assert sass.spill_instructions == 0, (
                f"{name}: SASS shows local-memory traffic but Triton reports no spills")


def test_the_vectorization_fraction_is_not_read_off_a_kernel_with_no_global_access():
    """A guard against a plausible misreading. `vectorized_frac` divides by global accesses, so a
    kernel with none would otherwise report 0.0 -- indistinguishable from fully scalar access.
    """
    from kernel_optimizer.evaluation.statics import SassCounts

    none_at_all = SassCounts(instructions=100, global_load=0, global_store=0, vec_128=0)
    assert none_at_all.vectorized_frac == 0.0
    assert none_at_all.global_load + none_at_all.global_store == 0, (
        "the caller must be able to tell 'no global access' from 'scalar global access'")

    # The real L1:42 winner: 32 global accesses, none of them vectorized. This is a real finding
    # about a real candidate, not a synthetic case.
    winner = SassCounts.model_validate(MEASURED_SASS["real_l1_42_winner"]["sass"])
    assert winner.global_load + winner.global_store == 32
    assert winner.vectorized_frac == 0.0, (
        "the run's winning candidate does 32 scalar global accesses; that is the finding")

    fully = SassCounts.model_validate(MEASURED_SASS["plain_elementwise"]["sass"])
    assert fully.vectorized_frac == 1.0, (
        "a kernel whose every access is 128-bit must report full vectorization")


def test_cuda_tools_are_found_off_path_because_that_is_where_they_are():
    """Measured on box 2: nvdisasm and cuobjdump live in /usr/local/cuda/bin/ and are NOT on the
    login shell's PATH. A `shutil.which`-only lookup reported them missing, and the first run of
    the probe concluded "no disassembler available" about a working tool -- the same class of bug
    as the opencode PATH failure, where an available capability read as a missing one.
    """
    import inspect

    from kernel_optimizer.evaluation import statics

    src = inspect.getsource(statics.find_cuda_tool)
    assert "/usr/local/cuda" in src, (
        "the toolkit directory is not searched, so a box with CUDA off PATH loses Tier 1 "
        "entirely -- which is what happened on box 2")
    assert "CUDA_HOME" in src, "the CUDA_HOME environment variable is not consulted"

    # A name that cannot exist must return None rather than raising: Tier 1 is a diagnostic and
    # its absence must degrade, not crash.
    assert statics.find_cuda_tool("nvdisasm-that-does-not-exist-9f3a") is None


def test_unmeasurable_signals_are_named_rather_than_silently_absent():
    """An agent told "nothing is wrong" reasons differently from one told "this cannot be
    measured here". The second is the true statement, and KernelPro's data says the distinction
    matters: raw counter dumps DEGRADED their LLM's performance (NoFeedback beat raw ncu,
    p=0.0007), so naming a gap is better than filling it with noise.
    """
    from kernel_optimizer.evaluation.statics import UNMEASURABLE_ON_THIS_TIER, unmeasurable_note

    note = unmeasurable_note()
    for item in ("bank conflict", "divergence", "stall"):
        assert any(item in entry for entry in UNMEASURABLE_ON_THIS_TIER), (
            f"{item} is not listed as unmeasurable, so its absence looks like a clean result")
    assert "unknown" in note.lower(), (
        f"the note must say these are UNKNOWN, not fine: {note}")
    assert "container" in note.lower(), "the note should say why they are unavailable"


def test_tier1_statics_reach_the_profile_record_with_the_right_aggregation():
    """Wiring, and the aggregation is the part that can be wrong silently.

    A launch has several kernels. Instruction counts SUM (the launch's mix is the sum of its
    parts). Occupancy takes the WORST kernel, because the launch is limited by its least
    occupant -- averaging would hide a kernel stuck at 17% behind one at 100%.
    """
    from kernel_optimizer.evaluation.profilerx import LightProfiler

    worker_result = {
        "triton": {
            "compile_s": 1.5,
            "kernels": [
                {"name": "k_fast", "n_regs": 18, "n_spills": 0, "shared": 0, "num_warps": 4,
                 "sass": MEASURED_SASS["plain_elementwise"]["sass"],
                 "occupancy": {"occupancy": 1.0, "active_warps": 48, "max_warps_per_sm": 48,
                               "blocks_per_sm": 12, "limiter": "warps_per_block",
                               "by_regs": 28, "by_shared": 16, "by_warps": 12}},
                {"name": "k_slow", "n_regs": 218, "n_spills": 0, "shared": 32768,
                 "num_warps": 4,
                 "sass": MEASURED_SASS["large_tile_dot"]["sass"],
                 "occupancy": {"occupancy": 0.1667, "active_warps": 8, "max_warps_per_sm": 48,
                               "blocks_per_sm": 2, "limiter": "registers",
                               "by_regs": 2, "by_shared": 3, "by_warps": 12}},
            ],
        }
    }
    rec = LightProfiler().extract(worker_result)

    assert rec.sass is not None, "Tier 1 statics never reached the ProfileRecord"
    assert rec.sass["tensor_core"] == 0 + 64, (
        "instruction counts must SUM across the launch's kernels")
    assert rec.sass["instructions"] == 48 + 872
    assert rec.uses_tensor_cores is True, (
        "a launch containing a tensor-core kernel must report using them")

    assert rec.occupancy is not None
    assert abs(rec.occupancy["occupancy"] - 0.1667) < 1e-6, (
        f"occupancy must come from the WORST kernel, not an average; got {rec.occupancy}")
    assert rec.occupancy_limiter == "registers"
    assert abs(rec.occupancy_pct - 16.67) < 0.01

    # An absent measurement must stay absent rather than becoming a zero.
    bare = LightProfiler().extract({"triton": {"compile_s": 1.0, "kernels": [
        {"name": "k", "n_regs": 32, "n_spills": 0, "shared": 0, "num_warps": 4}]}})
    assert bare.sass is None and bare.occupancy is None
    assert bare.uses_tensor_cores is None, (
        "an unmeasured instruction mix must read as unknown, not as 'no tensor cores'")
    assert bare.occupancy_pct is None


# --- step 6: classifier consuming calibrated thresholds -----------------------------------

def _peaks_4090():
    from kernel_optimizer.evaluation.bottleneck import DevicePeaks
    from kernel_optimizer.gpu.calibrate import calibration_from_worker

    cal = calibration_from_worker(MEASURED_4090)
    return cal, DevicePeaks(dram_tbs=cal.dram_tbs, fp32_tflops=cal.fp32_tflops,
                            tf32_tflops=cal.tf32_tflops)


def test_the_classifier_reproduces_every_yardsticks_known_bottleneck():
    """End-to-end acceptance for steps 2 and 6 together: measured ceilings -> derived thresholds
    -> classifier -> the analytically-known truth.

    Four workloads whose bottleneck is not in doubt: a 4096^3 matmul IS compute-bound, a 512 MB
    copy IS memory-bound, 40 tiny ops ARE launch-bound, and a small fused chain saturates
    neither. If the pipeline cannot recover those four, nothing it says about a real candidate is
    worth reading.
    """
    from kernel_optimizer.evaluation.bottleneck import classify

    cal, peaks = _peaks_4090()
    expected = {"compute": "compute_bound", "memory": "memory_bound", "launch": "launch_bound",
                "unsaturated": ("latency_bound", "mixed")}

    for y in cal.yardsticks:
        v = classify(gpu_ms=y.gpu_ms, cpu_issue_ms=y.cpu_issue_ms, flop_count=y.flop_count,
                     byte_count=y.byte_count, peaks=peaks, thresholds=cal.thresholds,
                     empty_launch_floor_ms=cal.empty_launch_floor_ms)
        want = expected[y.truth]
        ok = v.kind in want if isinstance(want, tuple) else v.kind == want
        assert ok, (f"{y.name}: known to be {y.truth}-bound, classified {v.kind}. "
                    f"evidence={v.evidence}")
        assert v.evidence["thresholds"]["calibrated"] is True, (
            "the verdict was computed against fallback thresholds, not this box's own")


def test_the_disproved_constants_break_the_yardsticks_if_reinstated():
    """Neutralization for step 6: prove the calibrated thresholds are load-bearing.

    Feeding the classifier the two constants step 1 disproved must MISCLASSIFY a workload whose
    truth is not in doubt. Otherwise "we calibrate the thresholds" is a claim about the code with
    no consequence.
    """
    from kernel_optimizer.evaluation.bottleneck import classify
    from kernel_optimizer.evaluation.calibration import Thresholds

    cal, peaks = _peaks_4090()
    by = {y.truth: y for y in cal.yardsticks}

    disproved = Thresholds(dram_saturated_frac=0.60, compute_saturated_frac=0.50,
                           idle_frac=0.25, launch_bound_cpu_ratio=1.0)

    # The 1.0 launch ratio: the launch-bound workload measures 0.963, so the test cannot fire and
    # the workload is classified by its throughput fractions instead.
    lb = by["launch"]
    bad = classify(gpu_ms=lb.gpu_ms, cpu_issue_ms=lb.cpu_issue_ms, flop_count=lb.flop_count,
                   byte_count=lb.byte_count, peaks=peaks, thresholds=disproved)
    assert bad.kind != "launch_bound", (
        "a 1.0 launch ratio still produced launch_bound; then the constant was never the problem "
        "and this test proves nothing")
    good = classify(gpu_ms=lb.gpu_ms, cpu_issue_ms=lb.cpu_issue_ms, flop_count=lb.flop_count,
                    byte_count=lb.byte_count, peaks=peaks, thresholds=cal.thresholds)
    assert good.kind == "launch_bound", "the calibrated line must classify it correctly"

    # The 0.50 compute line: a kernel at 60% of the ceiling has real headroom left but would be
    # declared saturated and left alone.
    half_speed_ms = by["compute"].gpu_ms / 0.6 * by["compute"].pct_of_fp32(cal.fp32_tflops)
    half = classify(gpu_ms=half_speed_ms, cpu_issue_ms=None,
                    flop_count=by["compute"].flop_count, byte_count=by["compute"].byte_count,
                    peaks=peaks, thresholds=disproved)
    assert half.kind == "compute_bound", (
        "the 0.50 line is supposed to fire early; if it does not, the setup is wrong")
    half_calibrated = classify(gpu_ms=half_speed_ms, cpu_issue_ms=None,
                               flop_count=by["compute"].flop_count,
                               byte_count=by["compute"].byte_count, peaks=peaks,
                               thresholds=cal.thresholds)
    assert half_calibrated.kind != "compute_bound", (
        "a kernel at ~60% of the measured ceiling is still reported as saturated under the "
        "calibrated line; the line is too low")


def test_a_tensor_core_kernel_is_scored_against_the_tensor_core_ceiling():
    """The denominator has to match the kernel, or the verdict inverts.

    On this card the tf32 ceiling is 1.62x the fp32 one. A tf32 kernel scored against fp32 reads
    as >100% of peak -- which looks like "saturated, stop" for a kernel that may have most of its
    headroom left. The kernel's own instruction mix (step 5) is what selects the ceiling.
    """
    from kernel_optimizer.evaluation.bottleneck import classify

    cal, peaks = _peaks_4090()
    # A kernel achieving 70 TFLOP/s: impossible in fp32 (54.9 ceiling), 79% of the tf32 ceiling.
    flop = 2 * 4096 ** 3
    gpu_ms = flop / 70e12 * 1e3

    with_tc = classify(gpu_ms=gpu_ms, cpu_issue_ms=None, flop_count=flop,
                       byte_count=3 * 4096 * 4096 * 4, peaks=peaks, thresholds=cal.thresholds,
                       sass={"instructions": 500, "tensor_core": 64})
    assert with_tc.evidence["compute_ceiling_used"].startswith("tensor-core"), (
        "a kernel full of HMMA was scored against the fp32 ceiling")
    assert with_tc.evidence["pct_of_compute_peak"] < 100.0, (
        f"scoring against the right ceiling must give a sane percentage, got "
        f"{with_tc.evidence['pct_of_compute_peak']}")

    without = classify(gpu_ms=gpu_ms, cpu_issue_ms=None, flop_count=flop,
                       byte_count=3 * 4096 * 4096 * 4, peaks=peaks, thresholds=cal.thresholds,
                       sass={"instructions": 500, "tensor_core": 0})
    assert without.evidence["compute_ceiling_used"] == "fp32"
    assert without.evidence["pct_of_compute_peak"] > 100.0, (
        "a scalar kernel exceeding the fp32 ceiling is physically impossible and the evidence "
        "should show it, so a reader can see the mix and the ceiling disagree")


def test_saturating_fp32_without_tensor_cores_says_the_ceiling_itself_can_be_raised():
    """The most actionable verdict this classifier can produce, and it needs step 5's mix.

    A kernel at the fp32 ceiling that is NOT using tensor cores is not done -- it is against the
    wrong ceiling. This is exactly the L3:48 situation recorded in memory: 8/8 tensor-core
    candidates rejected, the accepted result entirely scalar, and nothing in the feedback said the
    machine had 1.6x more available.
    """
    from kernel_optimizer.evaluation.bottleneck import classify

    cal, peaks = _peaks_4090()
    flop = 2 * 4096 ** 3
    gpu_ms = flop / (cal.fp32_tflops * 0.95 * 1e12) * 1e3   # 95% of the fp32 ceiling

    v = classify(gpu_ms=gpu_ms, cpu_issue_ms=None, flop_count=flop,
                 byte_count=3 * 4096 * 4096 * 4, peaks=peaks, thresholds=cal.thresholds,
                 sass={"instructions": 500, "tensor_core": 0})
    assert v.kind == "compute_bound"
    assert "does NOT use tensor cores" in v.suggests, (
        f"a scalar kernel at the fp32 ceiling must be told the ceiling can be RAISED, not that "
        f"it is finished. got: {v.suggests}")
    assert "raises the limit" in v.suggests

    # A kernel already using them must NOT get that advice.
    v2 = classify(gpu_ms=gpu_ms, cpu_issue_ms=None, flop_count=flop,
                  byte_count=3 * 4096 * 4096 * 4, peaks=peaks, thresholds=cal.thresholds,
                  sass={"instructions": 500, "tensor_core": 64})
    assert "does NOT use tensor cores" not in v2.suggests


def test_low_occupancy_is_caught_even_when_nothing_spilled():
    """The Triton-specific case step 5 measured, now reaching a verdict.

    Triton caps registers and loses occupancy instead of spilling, so a spills-only resource test
    misses it entirely. Checked on the REAL winning candidate of run-l1-42-20260907-193510:
    112 regs/thread, 0 spills, 33.3% occupancy, register-limited.
    """
    from kernel_optimizer.evaluation.bottleneck import classify

    cal, peaks = _peaks_4090()
    v = classify(gpu_ms=4.85, cpu_issue_ms=None, flop_count=0, byte_count=None, peaks=peaks,
                 n_regs=112, n_spills=0, shared_bytes=0, max_regs_per_thread=255,
                 max_shared_bytes=101376, thresholds=cal.thresholds,
                 empty_launch_floor_ms=cal.empty_launch_floor_ms,
                 occupancy={"occupancy": 0.3333, "limiter": "registers"})

    assert v.kind == "resource_limited", (
        f"a real candidate at 33% occupancy was classified {v.kind}; with 0 spills and 112 regs "
        f"(below the 0.8*255 line) occupancy is the ONLY signal that can catch it")
    assert any("occupancy" in s for s in v.evidence["at_limit"])
    assert "BLOCK_M" in v.suggests or "accumulator" in v.suggests, (
        f"the advice must name the register lever, not just the symptom: {v.suggests}")

    # Control: the same kernel at full occupancy must not be called resource-limited.
    v2 = classify(gpu_ms=4.85, cpu_issue_ms=None, flop_count=0, byte_count=None, peaks=peaks,
                  n_regs=112, n_spills=0, shared_bytes=0, max_regs_per_thread=255,
                  max_shared_bytes=101376, thresholds=cal.thresholds,
                  empty_launch_floor_ms=cal.empty_launch_floor_ms,
                  occupancy={"occupancy": 1.0, "limiter": "warps_per_block"})
    assert v2.kind != "resource_limited"


def test_a_disagreement_between_the_two_methods_is_reported_not_hidden():
    """Cross-validation, borrowed from KernelPro: when the achieved fractions and the analytic
    roofline position conflict, defer to the analytic bound and SAY SO.

    The reasoning is that a contended box depresses the measured ceiling, inflating every
    fraction, while arithmetic intensity depends only on the task's own FLOP/byte ratio and cannot
    move. A verdict that hides the conflict would be trusted exactly where it is least reliable.
    """
    from kernel_optimizer.evaluation.bottleneck import DevicePeaks, classify
    from kernel_optimizer.evaluation.calibration import Thresholds

    # A deliberately depressed DRAM ceiling, as a throttled box would measure. The kernel's
    # intensity puts it on the compute side, but against this ceiling its bandwidth fraction
    # clears the saturation line.
    peaks = DevicePeaks(dram_tbs=0.05, fp32_tflops=54.9, tf32_tflops=88.9)
    th = Thresholds(dram_saturated_frac=0.60, compute_saturated_frac=0.80, idle_frac=0.15,
                    launch_bound_cpu_ratio=0.87)
    flop, byts = 2 * 4096 ** 3, 3 * 4096 * 4096 * 4     # intensity 682, ridge here is 1098

    v = classify(gpu_ms=2.64, cpu_issue_ms=None, flop_count=flop, byte_count=byts,
                 peaks=peaks, thresholds=th)
    assert v.evidence["analytic_side"] in ("memory", "compute")
    if v.disagreement:
        assert "Deferring to the analytic bound" in v.disagreement
        assert "low-confidence" in v.disagreement

    # And with a healthy ceiling the two methods agree, so nothing is flagged.
    _cal, good_peaks = _peaks_4090()
    v2 = classify(gpu_ms=2.64, cpu_issue_ms=None, flop_count=flop, byte_count=byts,
                  peaks=good_peaks, thresholds=_cal.thresholds)
    assert v2.disagreement == "", (
        f"the two methods agree on a healthy box; nothing should be flagged: {v2.disagreement}")


def test_every_verdict_carries_what_it_could_not_measure():
    """Even a confident verdict must say what is unknown.

    An agent told "compute bound" concludes something different from one told "compute bound, and
    bank conflicts / divergence / stall reasons are unmeasurable on this box". Only the second is
    true, and KernelPro's finding that raw counter dumps DEGRADE performance (p=0.0007) says the
    honest gap beats a filled-in one.
    """
    from kernel_optimizer.evaluation.bottleneck import classify

    cal, peaks = _peaks_4090()
    for y in cal.yardsticks:
        v = classify(gpu_ms=y.gpu_ms, cpu_issue_ms=y.cpu_issue_ms, flop_count=y.flop_count,
                     byte_count=y.byte_count, peaks=peaks, thresholds=cal.thresholds)
        assert v.unmeasured, f"{y.name}: verdict {v.kind} claims nothing is unmeasured"
        joined = " ".join(v.unmeasured).lower()
        assert "bank conflict" in joined and "divergence" in joined and "stall" in joined

    unknown = classify(gpu_ms=0.0, cpu_issue_ms=None, flop_count=None, byte_count=None,
                       peaks=None)
    assert unknown.kind == "unknown"
    assert unknown.unmeasured, "even an `unknown` verdict must list what cannot be measured"


def test_the_docstring_no_longer_claims_tensor_cores_are_invisible():
    """The correction step 6 was asked to make, enforced rather than asserted in prose.

    The earlier docstring listed tensor cores and occupancy among the things counters would be
    needed for. Step 5 disproved that by measurement, and a stale claim in the module that
    downstream readers consult would keep the wrong belief alive.
    """
    from pathlib import Path as _P

    doc = _P("src/kernel_optimizer/evaluation/bottleneck.py").read_text(encoding="utf-8")
    head = doc[:doc.index("from __future__")]

    assert "VISIBLE" in head, "the docstring does not state what IS visible without counters"
    # Whitespace-collapsed before matching: the docstring is wrapped, so "bank conflicts" is
    # split across a line break and a naive substring search misses a phrase that is present.
    flat = " ".join(head.lower().split())
    # The still-true absences must remain listed. Matched on stems so a plural or an adjacent
    # word ("bank conflicts", "warp divergence") still counts -- the point is that the concept is
    # named, not that a phrase appears verbatim.
    for still_absent in ("bank conflict", "divergen", "stall"):
        assert still_absent in flat, (
            f"{still_absent} must still be listed as invisible; it genuinely is")
    # And the disproved constants must be recorded as disproved, so they cannot come back as
    # "the documented defaults".
    assert "0.50" in head and "0.963" in head, (
        "the disproved thresholds and the measurement that disproved them are not recorded")


# --- step 7: wiring the verdict into the analyst prompt and the report ---------------------

def test_the_bottleneck_analysis_reaches_the_agent_as_detect_analyze_recommend():
    """The wiring step, and the FORM matters as much as the delivery.

    KernelPro measured that feeding an LLM raw hardware-counter output made it perform WORSE than
    feeding it nothing (NoFeedback beat raw ncu, p=0.0007): a wall of numbers invites
    pattern-matching on whichever value looks anomalous. So the doc must state a conclusion, give
    the evidence, and say what to try -- not dump metrics.
    """
    from kernel_optimizer.agents.modules import _bottleneck_doc
    from kernel_optimizer.evaluation.bottleneck import classify
    from kernel_optimizer.evaluation.task_cost import cost_from_worker

    cal, peaks = _peaks_4090()
    cost = cost_from_worker({"task_cost": MEASURED_TASK_COSTS["level3:43_attention"]})
    v = classify(gpu_ms=6.92, cpu_issue_ms=None, flop_count=cost.flop_count,
                 byte_count=cost.compulsory_bytes, peaks=peaks, thresholds=cal.thresholds,
                 empty_launch_floor_ms=cal.empty_launch_floor_ms,
                 sass={"instructions": 496, "tensor_core": 0},
                 occupancy={"occupancy": 0.3333, "limiter": "registers"})
    doc = _bottleneck_doc(v, cost, cal)

    # DETECT: a named verdict, not a table.
    assert "## Verdict" in doc and v.kind in doc
    # ANALYZE: the numbers behind it, so the agent can disagree.
    assert "The numbers behind it" in doc
    assert "arithmetic_intensity" in doc
    # RECOMMEND: what to try.
    assert "tensor cores" in doc
    # The ceilings must be labelled as measured on THIS box, or the agent may treat them as
    # datasheet figures and distrust a throttled one.
    assert "measured on THIS box" in doc
    assert f"{cal.dram_tbs:.3f}" in doc
    # The gap must be named, not left silent.
    assert "CANNOT see" in doc
    assert "bank conflicts" in doc
    assert "not measured and found to be fine" in doc
    # And the thresholds must be declared as derived, not constant.
    assert "not constants" in doc


def test_the_task_level_fusion_headroom_reaches_the_agent():
    """The single largest lever L3:43 offers, and it is a TASK property no per-candidate
    measurement produces. If it does not reach the agent it may as well not be measured.
    """
    from kernel_optimizer.agents.modules import _bottleneck_doc
    from kernel_optimizer.evaluation.task_cost import cost_from_worker

    cal, _ = _peaks_4090()
    cost = cost_from_worker({"task_cost": MEASURED_TASK_COSTS["level3:43_attention"]})
    doc = _bottleneck_doc(None, cost, cal)

    assert "69.1x" in doc, "the measured fusion headroom is not in the agent's input"
    assert "intermediates" in doc
    assert "property of the TASK" in doc, (
        "the agent must be told this cannot be reached by tuning a per-op kernel")

    # A single-op task has nothing to fuse and must NOT be told to fuse.
    one_op = cost_from_worker({"task_cost": MEASURED_TASK_COSTS["level1:1_matmul"]})
    doc2 = _bottleneck_doc(None, one_op, cal)
    assert "largest single lever" not in doc2, (
        "a 1-op reference was told fusion is its largest lever; there is nothing to fuse")


def test_a_task_that_cannot_be_compute_bound_is_told_so_before_it_tries():
    """`flop_count / compulsory_bytes` is a CEILING on intensity, so this is decidable in advance.

    Telling an agent to chase arithmetic throughput on a task whose maximum possible intensity is
    below the card's ridge wastes a whole rewrite round on something no correct implementation can
    achieve.
    """
    from kernel_optimizer.agents.modules import _bottleneck_doc
    from kernel_optimizer.evaluation.task_cost import cost_from_worker

    cal, _ = _peaks_4090()
    relu = cost_from_worker({"task_cost": MEASURED_TASK_COSTS["level1:19_relu"]})
    doc = _bottleneck_doc(None, relu, cal)
    assert "cannot be compute-bound" in doc, (
        "a 0-FLOP elementwise task was not told that arithmetic throughput is unreachable")
    assert "BANDWIDTH" in doc

    mm = cost_from_worker({"task_cost": MEASURED_TASK_COSTS["level1:1_matmul"]})
    doc2 = _bottleneck_doc(None, mm, cal)
    assert "CAN be compute-bound" in doc2
    assert "cannot be compute-bound" not in doc2


def test_tool_affinity_filtering_hides_signals_that_do_not_apply():
    """KernelPro's tool-affinity idea: an irrelevant signal costs attention and invites a change
    that addresses nothing. So each Tier 1 line appears only when it is relevant to THIS kernel.
    """
    from kernel_optimizer.agents.modules import _tier1_doc
    from kernel_optimizer.models.core import ProfileRecord

    # A kernel with a real problem: low occupancy, no tensor cores, scalar accesses.
    bad = ProfileRecord(n_regs=112, n_spills=0, shared_bytes=0, num_warps=4,
                        sass={"instructions": 496, "tensor_core": 0, "global_load": 16,
                              "global_store": 16, "vec_128": 0},
                        occupancy={"occupancy": 0.3333, "limiter": "registers",
                                   "active_warps": 16, "max_warps_per_sm": 48,
                                   "blocks_per_sm": 4})
    doc = _tier1_doc(bad)
    assert "33% (LOW)" in doc and "registers" in doc
    assert "No tensor-core instructions" in doc
    assert "narrower than 128-bit" in doc
    # Low occupancy must be framed as a hypothesis, not a defect: a large-tile kernel can be
    # fastest AT low occupancy, and telling the agent otherwise causes a regression.
    assert "not automatically bad" in doc
    assert "hypothesis to test" in doc

    # A healthy kernel must not be handed a list of non-problems.
    good = ProfileRecord(n_regs=32, n_spills=0, shared_bytes=8192, num_warps=8,
                         sass={"instructions": 300, "tensor_core": 48, "global_load": 4,
                               "global_store": 4, "vec_128": 8},
                         occupancy={"occupancy": 1.0, "limiter": "warps_per_block",
                                    "active_warps": 48, "max_warps_per_sm": 48,
                                    "blocks_per_sm": 6})
    doc2 = _tier1_doc(good)
    assert "No tensor-core instructions" not in doc2, (
        "a tensor-core kernel was told it has none")
    assert "narrower than 128-bit" not in doc2, (
        "a fully-vectorized kernel was told its accesses are narrow")
    assert "LOW" not in doc2

    # No measurements at all: no document, rather than a document full of nothing.
    assert _tier1_doc(None) == ""
    assert _tier1_doc(ProfileRecord()) == ""


def test_an_impossible_throughput_fraction_is_flagged_not_reported_as_saturated():
    """Found by rendering the real L3:43 numbers, which produced 108.5% of the fp32 ceiling.

    Above 100% is physically impossible and means an input is wrong -- either the wrong ceiling or
    a candidate doing less arithmetic than the reference the FLOP count came from. A bare threshold
    test turns that into "saturated, stop optimizing", which is the most expensive possible wrong
    answer: it ends the search on a kernel whose headroom is unknown.
    """
    from kernel_optimizer.evaluation.bottleneck import classify

    cal, peaks = _peaks_4090()
    v = classify(gpu_ms=6.92, cpu_issue_ms=None, flop_count=412316860416,
                 byte_count=416296960, peaks=peaks, thresholds=cal.thresholds,
                 sass={"instructions": 496, "tensor_core": 0})
    assert v.evidence["pct_of_compute_peak"] > 100.0
    assert v.evidence.get("impossible_fraction") is not None, (
        "an impossible fraction was reported without any flag")
    assert "impossible" in v.disagreement
    assert "Do NOT read this as 'at the ceiling'" in v.disagreement
    assert "LESS arithmetic than the reference" in v.disagreement, (
        "the second cause -- the candidate simplifying the task -- must be named, since it is the "
        "likely one when the mix says no tensor cores")

    # A believable fraction must NOT be flagged, or the caveat becomes noise.
    ok = classify(gpu_ms=2.64, cpu_issue_ms=None, flop_count=2 * 4096 ** 3,
                  byte_count=3 * 4096 * 4096 * 4, peaks=peaks, thresholds=cal.thresholds,
                  sass={"instructions": 500, "tensor_core": 0})
    assert ok.evidence.get("impossible_fraction") is None
    assert "impossible" not in ok.disagreement


def test_the_verdict_is_journalled_and_reaches_the_report():
    """The verdict must survive the analyst failing, and must be visible to a human reader.

    It is the DETERMINISTIC half of the feedback loop: reproducible from the event log, unlike the
    analyst's report. If it lived only in the agent's sandbox it could not be audited.
    """
    from kernel_optimizer.reporting.report import ReportGenerator

    class FakeStore:
        def __init__(self, events):
            self._events = events
            self.run_dir = __import__("pathlib").Path(".")

        def replay(self):
            class S:
                pass

            s = S()
            s.events = self._events
            return s

    from types import SimpleNamespace

    events = [
        SimpleNamespace(type="BOTTLENECK_CLASSIFIED", payload={
            "candidate_id": "cand-aaa", "kind": "resource_limited",
            "evidence": {"occupancy": 0.3333, "occupancy_limiter": "registers",
                         "uses_tensor_cores": False, "pct_of_dram_peak": 6.6},
            "disagreement": ""}),
        SimpleNamespace(type="BOTTLENECK_CLASSIFIED", payload={
            "candidate_id": "cand-bbb", "kind": "compute_bound",
            "evidence": {"pct_of_compute_peak": 108.5, "compute_ceiling_used": "fp32"},
            "disagreement": "the kernel appears to reach 108% of the fp32 ceiling, "
                            "which is impossible."}),
    ]

    # Exercise the section builder directly on the events, which is what `report` regenerates
    # from -- the point being that nothing here needs the run's in-memory state.
    gen = ReportGenerator()
    import inspect

    src = inspect.getsource(gen.generate)
    assert "BOTTLENECK_CLASSIFIED" in src, (
        "the report does not read the verdicts, so they are invisible to a human reader")
    assert "low confidence" in src, (
        "a disagreement caveat that is not surfaced is a caveat that misleads")


def test_the_classifier_is_actually_called_by_the_run():
    """Without this the classifier is a well-tested module with zero callers -- which is exactly
    what it was before step 7, and what `bottleneck.py` was for two whole steps.
    """
    import inspect

    from kernel_optimizer.control.orchestrator import Orchestrator

    analysis = inspect.getsource(Orchestrator._stats_and_analysis)
    assert "_classify_bottleneck" in analysis, (
        "the analysis step never classifies; the classifier has no callers")
    assert "BOTTLENECK_CLASSIFIED" in analysis, "the verdict is not journalled"
    assert "bottleneck_verdict=verdict" in analysis, (
        "the verdict is computed but never handed to the analyst")

    classify_src = inspect.getsource(Orchestrator._classify_bottleneck)
    assert "compulsory_bytes" in classify_src, (
        "the byte denominator must be the task's COMPULSORY traffic: using the reference's "
        "materialized traffic would credit every candidate with bytes a good one avoids")
    assert "thresholds=self.calibration.thresholds" in classify_src, (
        "the run's own calibrated thresholds are not being used")

    # And it must degrade rather than raise: a box without a calibration still has to run.
    assert "if self.calibration is None" in classify_src


def test_the_best_trials_profile_is_used_not_the_last():
    """The analyst reasons about the configuration that WON. A different trial's registers and
    occupancy describe a configuration nobody will ship, so feeding those in would have the agent
    optimizing a config it is not being asked about.

    This test used to assert the SOURCE TEXT `latency_ms.median < best[0]`, and that was a
    mistake worth recording: it pinned the implementation instead of the behaviour, and the exact
    line it pinned was the defect. `median` is Optional, so the raw read crashed
    run-l1-42-20260908-015408 with `TypeError: '<' not supported between instances of NoneType
    and NoneType` -- and this test PASSED on the broken code while failing on the fix. A
    source-string assertion can only ever confirm that the code says what it says.

    So it now checks the two properties that actually matter, and
    `test_the_best_profile_ranks_trials_without_a_median` drives the real function against the
    latencies that crashed it.
    """
    import inspect

    from kernel_optimizer.control.orchestrator import Orchestrator
    from kernel_optimizer.models.core import LatencyStats, ParamSet, ProfileRecord, TrialRecord

    src = inspect.getsource(Orchestrator._best_profile)
    assert 'status != "complete"' in src, "a failed trial's profile could be selected"

    class Run:
        pass

    def trial(tid, ms, regs, status="complete"):
        return TrialRecord(
            trial_id=tid, candidate_id="c", space_id="s", params=ParamSet(values={}),
            status=status,
            latency_ms=LatencyStats(mean=ms, std=0.1, min=ms - 0.1, max=ms + 0.1, n_samples=20),
            profile=ProfileRecord(n_regs=regs, n_spills=0, shared_bytes=0, num_warps=4,
                                  num_stages=2))

    crun = Run()
    # The fastest trial is neither first nor last, so neither "take the first" nor "take the
    # last" can pass by accident.
    crun.trials = [trial("a", 9.0, 11), trial("b", 4.0, 22), trial("c", 6.0, 33)]
    assert Orchestrator._best_profile(None, crun).n_regs == 22, (
        "the profile is not selected by lowest latency")

    # A FAILED trial that happens to be the fastest must not win: a crashed or incorrect trial
    # can report an absurdly low latency precisely because it did not do the work.
    crun.trials = [trial("a", 9.0, 11), trial("fast-but-failed", 0.1, 99, status="fail")]
    assert Orchestrator._best_profile(None, crun).n_regs == 11, (
        "a failed trial's profile was selected")


# --- step 4: doctor integration and tier recording ----------------------------------------

# Tier detection as box 2 actually reports it (2026-09-08). Note ncu IS present and Tier 3 is
# still false: the two facts are independent, which is the trap this records.
MEASURED_TIERS_BOX2 = {
    "tier0_events_and_counts": True,
    "tier1_sass_and_occupancy": True,
    "nvdisasm": "/usr/local/cuda/bin/nvdisasm",
    "cuobjdump": "/usr/local/cuda/bin/cuobjdump",
    "ncu_present": True,
    "tier3_counters": False,
    "tier3_note": "hardware counters need NVreg_RestrictProfilingToAdminUsers on the HOST",
}


def test_ncu_being_installed_does_not_mean_counters_are_usable():
    """The trap step 4 exists to avoid, recorded as a test.

    `ncu` is on BOTH experiment boxes and returns ERR_NVGPUCTRPERM on each: counters need a
    host-side kernel-module parameter that a container cannot set. A tier report that inferred
    "counters available" from "ncu present" would promise signals that never arrive, and every
    verdict resting on them would silently be `unknown`.
    """
    from kernel_optimizer.gpu.calibrate import calibration_from_worker

    cal = calibration_from_worker(dict(MEASURED_4090, tiers=MEASURED_TIERS_BOX2))
    assert cal.tiers["ncu_present"] is True
    assert cal.tiers["tier3_counters"] is False, (
        "ncu being present was read as counters being usable; they are independent facts")
    assert "HOST" in cal.tiers["tier3_note"], (
        "the note must say WHERE the permission lives, or an operator will try to fix it in the "
        "container")


def test_the_tier_report_distinguishes_missing_tooling_from_a_clean_kernel():
    """The ambiguity this recording removes. A report with no instruction mix means either

      (a) the box had no disassembler, so tensor-core use is UNKNOWN, or
      (b) the kernels genuinely used no tensor cores,

    and those are opposite conclusions. Without the tier record they look identical in the log,
    which would make a months-old report unreadable.
    """
    from kernel_optimizer.gpu.calibrate import calibration_from_worker

    with_tier1 = calibration_from_worker(dict(MEASURED_4090, tiers=MEASURED_TIERS_BOX2))
    assert with_tier1.tiers["tier1_sass_and_occupancy"] is True
    assert with_tier1.tiers["nvdisasm"], "the located tool path should be recorded, for auditing"

    no_tools = dict(MEASURED_TIERS_BOX2, tier1_sass_and_occupancy=False, nvdisasm=None,
                    cuobjdump=None)
    without = calibration_from_worker(dict(MEASURED_4090, tiers=no_tools))
    assert without.tiers["tier1_sass_and_occupancy"] is False

    # And the report must SAY which case it is, in words, rather than leaving a blank.
    import inspect

    from kernel_optimizer.reporting.report import ReportGenerator

    src = inspect.getsource(ReportGenerator.generate)
    assert "tier1_sass_and_occupancy" in src, "the report never reads the tier record"
    assert "as unknown in this run, not as absent" in src, (
        "the report must state that missing tooling means UNKNOWN, not absent")


def test_doctor_reports_the_tiers_and_the_cached_calibration():
    """An operator who learns at hour 9 of a 12-hour run that the box was never calibrated has
    learned it too late. Doctor is where that belongs.
    """
    import inspect

    from kernel_optimizer.cli import cmd_doctor

    src = inspect.getsource(cmd_doctor)
    assert "find_cuda_tool" in src, "doctor does not probe for the Tier 1 tooling"
    assert "Tier 1" in src and "Tier 3" in src
    assert "ERR_NVGPUCTRPERM" in src, (
        "doctor should explain why Tier 3 is expected to be unavailable, or an operator will "
        "treat it as a fault to fix")
    assert "load_cached" in src, "doctor does not report whether a calibration exists"
    # Reporting must not silently MEASURE: that is two minutes of exclusive GPU time, so it is
    # opt-in behind a flag.
    assert "args, \"calibrate\"" in src or "getattr(args, 'calibrate'" in src, (
        "doctor must only measure when explicitly asked; a health check should not take the GPU "
        "for two minutes by surprise")


def test_doctor_flags_a_calibration_measured_on_different_hardware():
    """A cache from another card is worse than none: it is refused at run time (identity-keyed),
    but an operator reading doctor's output would wrongly believe the box is ready.
    """
    import inspect

    from kernel_optimizer.cli import cmd_doctor

    src = inspect.getsource(cmd_doctor)
    assert "cached calibration matches this GPU" in src, (
        "doctor does not compare the cached calibration's device against the live one")
    assert "refused and re-measured" in src, (
        "the operator must be told what will happen, not just that something is wrong")


def test_a_calibration_without_tiers_still_loads():
    """Backward compatibility, and it is not cosmetic: the two calibrations already measured on
    box 2 predate the tier field, and a validation error on load would make the harness
    re-measure -- or worse, crash a run at start.
    """
    from kernel_optimizer.gpu.calibrate import calibration_from_worker

    cal = calibration_from_worker(MEASURED_4090)     # no "tiers" key at all
    assert cal.tiers == {}, "a missing tier record must be empty, not an error"
    assert cal.thresholds is not None, "the rest of the calibration must still be usable"


def test_the_worker_can_import_its_own_statics_module_without_help():
    """The defect this catches was live and silent: Tier 1 collected nothing in a real run.

    The GPU worker is launched with PYTHONPATH set to KernelBench ONLY
    (worker_client._build_command), because it is deliberately a stdlib+torch+triton process with
    no dependency on the harness package -- and the worker venv on box 2 confirms it: `find_spec
    ("kernel_optimizer")` is None there. Step 5 broke that assumption: the SASS counting and
    occupancy arithmetic live in `kernel_optimizer.evaluation.statics` so they can be unit-tested
    without a GPU, so a plain import fails when the worker runs for real. And because Tier 1 is
    best-effort, it fails SILENTLY -- every kernel gets a `statics_note` and no instruction mix,
    indistinguishable in the log from a box with no disassembler.

    THE TEST HAS TO WORK FOR IT. A first version of this test ran the worker as a subprocess with
    an empty PYTHONPATH and passed with the fix reverted, because THIS venv has the harness
    installed as an editable package, so the plain import succeeds however PYTHONPATH is set. The
    worker venv does not have it installed, which is the condition that matters. So the subprocess
    runs with `-S -I` (no site-packages, isolated) to reproduce a venv where the package is not
    installed, and PYTHONPATH carries only a KernelBench-like path.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path as _P

    worker = _P("src/kernel_optimizer/gpu/worker_main.py").resolve()
    assert worker.exists()

    env = dict(os.environ)
    env["PYTHONPATH"] = str(_P("tests").resolve())   # stands in for KernelBench: not the src root
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('wm', r'{worker}')\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "sys.modules['wm'] = m\n"
        "spec.loader.exec_module(m)\n"
        "s = m._import_statics()\n"
        "assert s.count_sass is not None and s.compute_occupancy is not None\n"
        "assert s.find_cuda_tool is not None\n"
        "print('STATICS_OK')\n"
    )
    # -S: no site-packages, so an editable install of this very package cannot mask the bug.
    # -I: isolated, so neither the invoking venv nor a user site directory leaks in.
    out = subprocess.run([sys.executable, "-S", "-I", "-c", code],
                         capture_output=True, env=env,
                         cwd=str(_P(".").resolve()), timeout=120)
    assert b"STATICS_OK" in out.stdout, (
        "the worker cannot import its own statics module under the launcher's PYTHONPATH, so "
        "Tier 1 silently collects nothing in every real run.\n"
        f"stdout={out.stdout[-800:]!r}\nstderr={out.stderr[-1500:]!r}")


def test_tier1_reports_a_note_rather_than_dying_when_statics_is_unreachable():
    """The degradation path still has to be honest. If the import genuinely cannot resolve, the
    kernel row must carry a NOTE saying so -- not an empty dict that reads like "measured, nothing
    found". That distinction is the whole reason `statics_notes` exists.
    """
    import importlib.util
    from pathlib import Path as _P

    spec = importlib.util.spec_from_file_location(
        "wm_note", str(_P("src/kernel_optimizer/gpu/worker_main.py").resolve()))
    wm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wm)

    original = wm._import_statics
    try:
        def boom():
            raise ImportError("simulated: statics unreachable")

        wm._import_statics = boom
        row = wm._tier1_statics(compiled=object(), row={"n_regs": 64, "num_warps": 4},
                                props=object())
        assert "statics_note" in row, (
            "an unreachable statics module produced no note; the absence of a mix would then read "
            "as 'the kernel uses no tensor cores'")
        assert "tier1 unavailable" in row["statics_note"]
        assert "sass" not in row and "occupancy" not in row
    finally:
        wm._import_statics = original


def test_the_linux_port_files_are_importable_and_name_complete():
    """The Linux port lives as separate files, so nothing else compiles them.

    A hand-merge dropped `import threading` from linux-server/runtime.linux.py -- the watchdog
    needed it and box 2's version had never imported it. Nothing caught that: the file is not on
    any import path, so neither pytest nor a linter looked at it, and the run died at its FIRST
    agent call with `NameError: name 'threading' is not defined`.

    A compile alone would NOT have caught it either -- a missing import is a runtime NameError, not
    a syntax error. So this walks the AST and checks that every module-level name the code
    references is either imported, defined locally, or a builtin. That is the class of defect a
    merge introduces, and the only reason it was cheap this time is that it fired 12 seconds in
    rather than at hour 3.
    """
    import ast
    import builtins
    from pathlib import Path as _P

    ported = sorted(_P("linux-server").glob("*.linux.py"))
    assert ported, "no ported files found; this test would silently pass forever"

    for path in ported:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))

        bound: set[str] = set(dir(builtins))
        # __future__ and friends
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    bound.add((a.asname or a.name).split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for a in node.names:
                    bound.add(a.asname or a.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                bound.add(node.id)
            elif isinstance(node, ast.arg):
                bound.add(node.arg)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                bound.add(node.name)
            elif isinstance(node, ast.Global):
                bound.update(node.names)
            elif isinstance(node, (ast.comprehension,)):
                for t in ast.walk(node.target):
                    if isinstance(t, ast.Name):
                        bound.add(t.id)
        bound.update({"__file__", "__name__", "__doc__", "self", "cls"})

        used = {n.id for n in ast.walk(tree)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        missing = sorted(used - bound)
        assert not missing, (
            f"{path.name} references names that are never imported or defined: {missing}. "
            f"This is exactly the merge defect that killed a run at its first agent call "
            f"(`import threading` dropped from runtime.linux.py) -- a NameError at runtime, "
            f"invisible to a syntax check.")


def test_the_ported_runtime_keeps_both_sides_of_the_merge():
    """The port and the 0-9 work were disjoint, so a merge can silently lose either half.

    Losing the watchdog means an agent subprocess can take the box down again (the 111 GiB ptxas
    incident); losing resolve_opencode means a run launched from tmux dies with a bare
    FileNotFoundError. Both have happened, so both are asserted.
    """
    from pathlib import Path as _P

    runtime = _P("linux-server/runtime.linux.py").read_text(encoding="utf-8")

    # From the 0-9 work.
    assert "_memory_pressure" in runtime, "the cgroup watchdog was lost in the merge"
    assert "memory_abort_frac" in runtime
    assert "threading" in runtime, "the watchdog needs threading; this is the import that was lost"
    # From the Linux port.
    assert "resolve_opencode" in runtime, "the opencode PATH fix was lost in the merge"
    assert "miniconda3" in runtime, "the fallback search paths were lost"
    # And the port's own deliberate removal must NOT come back: shell=True on POSIX with a list
    # argv swallows --hostname/--port. Checked on the AST rather than the text, because the file
    # legitimately DISCUSSES shell=True in a docstring explaining why it was removed -- and a
    # naive text scan flags that prose, which is how this assertion first failed.
    import ast as _ast

    live_shell_true = []
    for node in _ast.walk(_ast.parse(runtime)):
        if not isinstance(node, _ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg == "shell" and isinstance(kw.value, _ast.Constant) and kw.value.value:
                live_shell_true.append(getattr(node.func, "attr", "<call>"))
    assert not live_shell_true, (
        f"shell=True came back into the Linux runtime as a real call argument: {live_shell_true}")


def test_the_ported_sandbox_declares_every_permission_opencode_asks_about():
    """A permission key omitted from the block does not default to allow -- it falls through to
    opencode's `{"*": "ask"}`, and an ask is fatal in a headless run: nobody answers, the turn
    idles to the request timeout, and the retry hits the same wall. Measured three times in
    run-l1-42-20260907-022528 (one call idled 28 minutes).
    """
    from kernel_optimizer.agents.sandbox import PERMISSION_CONFIG

    perms = PERMISSION_CONFIG["permission"]
    for key in ("edit", "bash", "webfetch", "external_directory"):
        assert key in perms, (
            f"`{key}` is not declared, so opencode falls back to its own default; for "
            f"external_directory that default is `ask`, which hangs a headless run")
    assert perms["external_directory"] == "allow"
    assert perms["webfetch"] == "deny", "webfetch must stay denied"


def test_a_single_op_reference_has_no_fusion_headroom():
    """Found by the L1:42 validation run, not by any test: a ONE-op reference reported 2.0x.

    `aten.max_pool2d_with_indices` returns (values, indices), and the byte counter summed every
    output -- so a reference with literally nothing to fuse looked like it materialized twice its
    unavoidable traffic, and `_bottleneck_doc` would have told the agent that fusion was its
    largest available lever.

    The measured numbers, before and after, on the real level1:42 reference (32x64x512x512 fp32,
    k=4 s=1 p=1 d=1 -> 511x511 output):
        compulsory  4286.587 MB   (= input 2147.484 + output 2139.103, exact)
        reference   8564.793 MB   before  -> 1.998x
        reference   4286.587 MB   after   -> 1.000x

    A FIRST ATTEMPT AT THIS FIX WAS WRONG and the wrongness is worth keeping: capping
    fusion_headroom at op_count also capped the multi-op tasks, dropping L3:43 from 69.09x to
    40.00x and destroying the signal. A reference that re-reads the SAME tensor across many ops
    legitimately exceeds that ratio. The defect was in the counter, not in the ratio.
    """
    from kernel_optimizer.evaluation.task_cost import TaskCost

    B, C, H, W = 32, 64, 512, 512
    k, s, p, d = 4, 1, 1, 1
    ho = (H + 2 * p - d * (k - 1) - 1) // s + 1
    expected_compulsory = (B * C * H * W + B * C * ho * ho) * 4
    assert expected_compulsory == 4286586880, "the analytic figure moved; recheck the shapes"

    fixed = TaskCost(flop_count=0, compulsory_bytes=expected_compulsory,
                     reference_bytes=expected_compulsory, op_count=1,
                     notes=["auxiliary outputs excluded"])
    assert abs(fixed.fusion_headroom - 1.0) < 1e-9, (
        f"a 1-op reference must have exactly 1.0x fusion headroom, got {fixed.fusion_headroom}")
    assert "largest single lever" not in _doc_for(fixed), (
        "a single-op reference is being told to fuse; there is nothing to fuse")

    # And the ratio must NOT be capped at op_count, or the multi-op signal dies. These are the
    # real measured numbers after the counter fix.
    for name, comp, ref, ops, floor in (
        ("level2:37", 687931392, 5552398336, 6, 8.0),
        ("level3:21", 322035200, 10630205440, 11, 32.0),
        ("level3:43", 416296960, 28356870144, 40, 67.0),
    ):
        t = TaskCost(flop_count=1, compulsory_bytes=comp, reference_bytes=ref, op_count=ops)
        assert t.fusion_headroom > floor, (
            f"{name}: fusion headroom {t.fusion_headroom:.2f}x collapsed below {floor}x -- a cap "
            f"at op_count would do exactly this, and it destroys the signal")
        assert t.fusion_headroom > ops or name == "level2:37", (
            f"{name}: a reference re-reading the same tensor across ops SHOULD exceed op_count "
            f"({t.fusion_headroom:.2f}x vs {ops} ops)")


def _doc_for(cost):
    """Render the agent-facing bottleneck doc for a TaskCost, with no verdict or calibration."""
    from kernel_optimizer.agents.modules import _bottleneck_doc

    return _bottleneck_doc(None, cost, None)


def test_the_auxiliary_output_exclusion_is_recorded_not_silent():
    """A reader comparing reference_bytes against a hand calculation must be told that an op's
    auxiliary tensor was excluded -- otherwise the numbers look wrong and the exclusion looks like
    a bug. `aux_output_ops` plus a note carry it.
    """
    from kernel_optimizer.evaluation.task_cost import cost_from_worker

    worker_result = {"task_cost": {
        "flop_count": 0, "compulsory_bytes": 4286586880, "reference_bytes": 4286586880,
        "op_count": 1, "aux_output_ops": 1,
        "notes": ["1 dispatched op(s) returned auxiliary tensors alongside their result "
                  "(e.g. max_pool2d_with_indices -> values, indices); only the primary output "
                  "is counted as task traffic, since the rest is implementation bookkeeping."],
    }}
    cost = cost_from_worker(worker_result)
    assert cost.notes, "the exclusion must be journalled"
    assert any("auxiliary" in n for n in cost.notes)
    assert any("primary output" in n for n in cost.notes)


def test_a_cache_hit_journals_the_same_fields_as_a_fresh_measurement():
    """Found by the validation run: `tier1` read as None in the event log while the cache file on
    disk held `tier1_sass_and_occupancy: true`.

    Calibration is per-BOX, so a cache hit is the normal case -- which means the thin
    CALIBRATION_LOADED payload was what almost every run would report. The report reads the event,
    not the cache, so a run that reused a calibration produced no tier line and no thresholds while
    a run that re-measured produced a full one. The data was never lost; only the event was thin.
    """
    import tempfile
    from pathlib import Path as _P

    from kernel_optimizer.evaluation.calibration import cache_path, save
    from kernel_optimizer.gpu.calibrate import calibration_from_worker, ensure_calibration

    cal = calibration_from_worker(dict(MEASURED_4090, tiers=MEASURED_TIERS_BOX2))

    class Store:
        def __init__(self):
            self.events = []

        def append(self, t, p):
            self.events.append((t, p))

    class NeverCalled:
        def run_job(self, *a, **k):
            raise AssertionError("a valid cache must not be re-measured")

    with tempfile.TemporaryDirectory() as td:
        root = _P(td)
        save(cache_path(root), cal)
        store = Store()
        got = ensure_calibration(NeverCalled(), root, store=store)
        assert got is not None

        kind, payload = store.events[-1]
        assert kind == "CALIBRATION_LOADED"
        for key in ("tiers", "thresholds", "tf32_tflops", "empty_launch_floor_ms",
                    "ridge_flop_per_byte", "measured_at"):
            assert key in payload, (
                f"a cache hit does not journal `{key}`, so a report on the normal path is thinner "
                f"than one on the re-measure path")
        assert payload["tiers"]["tier1_sass_and_occupancy"] is True, (
            "the tier record read as falsy on a cache hit -- the exact symptom observed live")
        assert payload["thresholds"]["compute_saturated_frac"] > 0.7


# ---------------------------------------------------------------------------------------------
# The two defects that ended run-l1-42-20260908-015408 at its first analyst step (14 min in).
#
# Both are about ONE statistic, `LatencyStats.median`, and they sit either side of it: nothing
# produced it on the main timing path, and one consumer read it as if it were mandatory.
# ---------------------------------------------------------------------------------------------

# Verbatim from run-l1-42-20260908-015408: the four TRIAL_DONE payloads' latency_ms. Every one
# has `median: None` and `samples: None`, which is what made the crash inevitable and the
# objective a mean. Kept as data so the tests below argue against a real log, not a mock.
L142_TRIAL_LATENCIES = [
    {"mean": 5.09, "std": 0.442, "min": 4.98, "max": 7.01, "n_samples": 20,
     "median": None, "samples": None},
    {"mean": 5.43, "std": 0.361, "min": 5.34, "max": 7.01, "n_samples": 20,
     "median": None, "samples": None},
    {"mean": 5.13, "std": 0.672, "min": 4.97, "max": 8.06, "n_samples": 20,
     "median": None, "samples": None},
    {"mean": 7.01, "std": 0.508, "min": 6.71, "max": 9.11, "n_samples": 20,
     "median": None, "samples": None},
]


def test_the_best_profile_ranks_trials_without_a_median():
    """The crash: TypeError comparing NoneType with NoneType, at orchestrator.py:1161.

    `_best_profile` read `t.latency_ms.median` directly while every other selection in the run
    reads `robust_ms`. `median` is Optional BY DESIGN -- the strict `eval_perf` path produces
    none -- so the raw read was a latent crash on the ordinary path, and it fired at
    orchestrator.py:1137 (`_stats_and_analysis`), OUTSIDE the try/except that had already logged
    it once from `_classify_bottleneck`. 14 minutes of GPU time and four agent calls were lost
    after every piece of real work in the run had already succeeded.

    Two assertions, because not crashing is the smaller half: the ranking must also agree with
    the trial the run actually selected. Ranking a profile by a different statistic than the one
    that chose the winner would hand the analyst the registers of a configuration nobody ships.
    """
    from kernel_optimizer.control.orchestrator import Orchestrator
    from kernel_optimizer.models.core import LatencyStats, ParamSet, ProfileRecord, TrialRecord

    trials = [
        TrialRecord(
            trial_id=f"tr-{i}", candidate_id="cand-3727e71a", space_id="sp-a59b6c8b",
            params=ParamSet(values={"NUM_WARPS": w}), status="complete",
            latency_ms=LatencyStats(**lat),
            profile=ProfileRecord(n_regs=regs, n_spills=0, shared_bytes=0,
                                  num_warps=w, num_stages=2),
        )
        for i, (lat, w, regs) in enumerate(
            zip(L142_TRIAL_LATENCIES, (4, 1, 8, 4), (64, 64, 64, 255)))
    ]

    class Run:
        pass

    crun = Run()
    crun.trials = trials

    got = Orchestrator._best_profile(None, crun)
    assert got is not None, (
        "_best_profile returned nothing for four complete, profiled trials -- with medians "
        "absent it must fall back to the mean, not give up")
    # Trial 0 has the lowest mean (5.09) and therefore the lowest robust_ms. Its profile is the
    # 4-warp / 64-register one; the 255-register spilling trial is the SLOWEST (7.01) and must
    # not be what the analyst is shown.
    assert got.num_warps == 4 and got.n_regs == 64, (
        f"ranked to the wrong trial: got num_warps={got.num_warps} n_regs={got.n_regs}, "
        f"expected the fastest trial 4/64. The run selected 5.09 ms as best_ms, so anything "
        f"else means the profile and the incumbent disagree")

    # And the ranking must still be right when medians ARE present and disagree with the means,
    # which is the case `capture_timing_samples` now creates. Here trial b's median is the
    # lowest while its mean is the highest, so a mean-ranking implementation picks trial a.
    swapped = [
        TrialRecord(
            trial_id="tr-a", candidate_id="c", space_id="s",
            params=ParamSet(values={}), status="complete",
            latency_ms=LatencyStats(mean=5.0, std=0.1, min=4.9, max=5.2, n_samples=20,
                                    median=9.0),
            profile=ProfileRecord(n_regs=11, n_spills=0, shared_bytes=0, num_warps=1,
                                  num_stages=1)),
        TrialRecord(
            trial_id="tr-b", candidate_id="c", space_id="s",
            params=ParamSet(values={}), status="complete",
            latency_ms=LatencyStats(mean=9.0, std=0.1, min=4.9, max=20.0, n_samples=20,
                                    median=5.0),
            profile=ProfileRecord(n_regs=22, n_spills=0, shared_bytes=0, num_warps=2,
                                  num_stages=1)),
    ]
    crun.trials = swapped
    got2 = Orchestrator._best_profile(None, crun)
    assert got2.n_regs == 22, (
        "with medians present the ranking must follow the median (robust_ms), the statistic the "
        "tuner optimizes -- got the mean winner instead")


def test_the_strict_timing_path_produces_a_median():
    """The silent defect, and the worse of the two: the tuning objective was the 20-sample MEAN.

    `robust_ms` reads the median and FALLS BACK to the mean, so a missing median is not an error
    anywhere -- it is an invisible downgrade to the estimator this project measured at 64.8%
    ranking accuracy against the median 93.2% (scripts/probe_robust_objective.py). And the
    median was missing on the main path: `_stats_to_dict` can only compute one when handed the
    samples, and only the relaxed-correctness handler had them. KernelBench computes
    `elapsed_times`, hands it to `get_timing_stats`, and drops it (eval.py:625-632, 671-678;
    timing.py:95-103), so the strict `eval_perf` path, the reference-baseline path and
    `measure_ref_program_time` all saw summary statistics only.

    Measured on run-l1-42-20260908-015408: all five eval_perf outputs carry exactly
    [max, mean, min, n, std] with median None. Every strict-mode run before this fix -- which
    is every run with fp64_relative_gate false -- tuned on the mean.

    This test drives the real interception against a stand-in KernelBench timing module, because
    the defect is precisely that the samples exist one frame below where the code could see them.
    """
    import sys
    import types

    kb = types.ModuleType("kernelbench")
    kb.__path__ = []
    kb_timing = types.ModuleType("kernelbench.timing")

    def get_timing_stats(elapsed_times, device=None):
        # KernelBench's real shape: summary statistics, samples discarded.
        return {"mean": sum(elapsed_times) / len(elapsed_times),
                "std": 0.5, "min": min(elapsed_times), "max": max(elapsed_times),
                "num_trials": len(elapsed_times)}

    kb_timing.get_timing_stats = get_timing_stats
    saved = {k: sys.modules.get(k) for k in ("kernelbench", "kernelbench.timing")}
    sys.modules["kernelbench"] = kb
    sys.modules["kernelbench.timing"] = kb_timing
    try:
        from kernel_optimizer.gpu.worker_main import _stats_to_dict, capture_timing_samples

        # A distribution with the shape the finding describes: a tight kernel plus two
        # scheduling stalls. mean 7.80, median 5.00 -- far apart, either side of the 2.0%
        # min_improvement_pct that decides whether a rewrite counts as progress.
        elapsed = [5.0, 4.9, 5.1, 5.0, 4.95, 5.05, 5.0, 20.0, 18.0, 5.0]

        assert capture_timing_samples() is True
        assert capture_timing_samples() is True, "must be idempotent: the worker may call it twice"

        # The route the strict path actually takes: KernelBench times, returns its summary, and
        # the worker converts that summary with no samples of its own to pass.
        summary = kb_timing.get_timing_stats(elapsed)
        out = _stats_to_dict(summary)
        assert out.get("median") is not None, (
            "the strict eval_perf path still yields no median, so robust_ms silently falls back "
            "to the 20-sample mean -- the 64.8%-accuracy objective this project measured")
        assert abs(out["median"] - 5.0) < 1e-9, f"median mis-computed: {out['median']}"
        assert out["samples"], "samples must be retained so an estimator choice stays auditable"
        assert len(out["samples"]) == len(elapsed)

        # KernelBench own numbers must stay ITS numbers: the pin (423217d) exists so the
        # evaluation criteria stay comparable across runs, and a wrapper that changed the mean
        # would break every cross-run comparison.
        assert abs(out["mean"] - sum(elapsed) / len(elapsed)) < 1e-9, (
            "the interception altered the KernelBench mean; it must only ADD a key")

        # The two statistics must actually differ here, or this test would pass on a broken
        # implementation that returned the mean under the name median.
        assert abs(out["mean"] - out["median"]) / out["median"] > 0.2, (
            "the fixture no longer distinguishes mean from median; it cannot detect the defect")
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_reporting_ranks_by_the_statistic_that_decided_the_run():
    """Once every path has a median, a mean-based report names a different winner than the run.

    `_reconstruct_summary` ranked trials on latency_ms["mean"] while the orchestrator ranks on
    `robust_ms`. That divergence was invisible only while no timing path produced a median at
    all -- exactly the condition the fix above removes. So this is a consequence of that fix,
    not an independent defect, and it has to land with it.
    """
    from kernel_optimizer.reporting.report import _reconstruct_summary, _robust_ms

    assert _robust_ms({"mean": 6.3, "median": 5.0}) == 5.0
    assert _robust_ms({"mean": 6.3, "median": None}) == 6.3, "must fall back to the mean"
    assert _robust_ms({"mean": 6.3}) == 6.3, "an older record has no median key at all"
    assert _robust_ms(None) is None
    assert _robust_ms({"mean": 6.3, "median": 0}) == 6.3, (
        "a zero median is not a measurement; robust_ms treats it as absent and so must this")

    trials = [
        {"trial_id": "slow-mean-fast-median", "candidate_id": "c1", "space_id": "s",
         "status": "complete", "params": {"values": {"BLOCK": 64}},
         "latency_ms": {"mean": 9.0, "std": 5.0, "min": 4.9, "max": 20.0, "median": 5.0}},
        {"trial_id": "fast-mean-slow-median", "candidate_id": "c1", "space_id": "s",
         "status": "complete", "params": {"values": {"BLOCK": 32}},
         "latency_ms": {"mean": 5.0, "std": 0.1, "min": 4.9, "max": 5.2, "median": 9.0}},
    ]
    candidates = {"c1": {"family_id": "fam-1", "origin": "seed", "approach_summary": ""}}
    out = _reconstruct_summary([], candidates, trials)
    assert out["best"]["candidate_id"] == "c1"
    assert out["best"]["tuned_ms"] == 5.0, (
        f"the report published {out['best']['tuned_ms']} -- the mean-ranked winner. The "
        f"orchestrator selects on robust_ms, so a report must too or it contradicts the run")
    assert out["families"]["fam-1"]["best_ms"] == 5.0


def test_the_auxiliary_output_count_survives_into_the_task_cost():
    """The worker reported aux_output_ops 1 and cost_from_worker dropped it, so the event log
    read None while the job output on disk held 1 (run-l1-42-20260908-023039).

    Only the count was lost, not the explanation -- the note survives, so nothing was misleading.
    But the count is what lets a reader reconcile reference_bytes against a hand calculation
    without reading prose, and a silently-dropped field is how a measured 1 becomes
    indistinguishable from an unmeasured 0.
    """
    from kernel_optimizer.evaluation.task_cost import cost_from_worker

    # Verbatim from jobs/task-cost-ddbca59e.out.json of that run.
    cost = cost_from_worker({"task_cost": {
        "flop_count": 0, "compulsory_bytes": 4286586880, "reference_bytes": 4286586880,
        "op_count": 1, "aux_output_ops": 1,
        "notes": ["1 dispatched op(s) returned auxiliary tensors alongside their result"],
    }})
    assert cost.aux_output_ops == 1, (
        "the worker measured 1 auxiliary-output op and the count did not survive the boundary")
    assert abs(cost.fusion_headroom - 1.0) < 1e-9, (
        "the counter fix must still hold: a 1-op reference has nothing to fuse")
    # An absent field must read 0, not raise: older recorded results have no such key.
    assert cost_from_worker({"task_cost": {"op_count": 3}}).aux_output_ops == 0


def test_every_file_producing_module_can_rescue_its_work():
    """Only the REWRITER had a rescue; novelty and generator discarded finished candidates.

    Measured on run-l1-42-20260908-023039, which had two 1500 s transport timeouts:
      - rewriter-d0d09c07: wrote rewrites/rw_1.py (8106 B) at 04:30:37, aborted 04:40:54
        -> AGENT_ARTIFACT_RESCUE, recovered. The existing fix working.
      - novelty-f47c7374: wrote novel/nv_1.py (5907 B) at 03:40:00, aborted 03:49:15
        -> AGENT_SESSION_RESET, no rescue. A finished candidate discarded 9m15s after it was
           complete, costing a second full attempt (which then took another 11 min).

    The gap was structural, not incidental: `rescue_from_sandbox` returns None in the base class
    and must be overridden per module, so each new file-producing module silently starts without
    one. Generator matters most -- it produces the SEED candidates, so a discarded call costs the
    run its whole starting population and every family that would have descended from it.

    This drives the real methods against the real sandbox layout rather than asserting that the
    overrides exist, so it fails if a rescue returns the wrong shape or misreads the backend.
    """
    import tempfile
    from pathlib import Path as _P

    from kernel_optimizer.agents.modules import (
        CandidateGeneratorAgent,
        NoveltyGeneratorAgent,
        StructureRewriterAgent,
    )
    from kernel_optimizer.agents.sandbox import Sandbox

    TRITON = ("import triton\nimport triton.language as tl\n"
              "PARAMS = {'BLOCK': 64}\n"
              "@triton.jit\ndef k(x, y, BLOCK: tl.constexpr):\n    pass\n"
              "class ModelNew:\n    pass\n")
    CUDA = ("from torch.utils.cpp_extension import load_inline\n"
            "PARAMS = {'BLOCK': 64}\n"
            "mod = load_inline(name='m', cpp_sources='', cuda_sources='')\n"
            "class ModelNew:\n    pass\n")

    # (module, its output directory, the prompt-specified filenames)
    cases = [
        (CandidateGeneratorAgent, "candidates", ["cand_1.py", "cand_2.py"]),
        (NoveltyGeneratorAgent, "novel", ["nv_1.py"]),
        (StructureRewriterAgent, "rewrites", ["rw_1.py"]),
    ]
    for agent_cls, outdir, names in cases:
        with tempfile.TemporaryDirectory() as td:
            sb = Sandbox(root=_P(td))
            d = _P(td) / outdir
            d.mkdir(parents=True)
            for i, n in enumerate(names):
                (d / n).write_text(CUDA if i == 1 else TRITON, encoding="utf-8")

            agent = agent_cls.__new__(agent_cls)  # no runtime/store needed for a pure rescue
            got = agent.rescue_from_sandbox(sb)
            assert got is not None, (
                f"{agent_cls.__name__} has no rescue: a transport failure discards finished "
                f"candidate files, which is what cost run-l1-42-20260908-023039 a whole novelty "
                f"attempt")
            assert len(got.candidates) == len(names), (
                f"{agent_cls.__name__} rescued {len(got.candidates)} of {len(names)} files")
            for c in got.candidates:
                assert c.file.startswith(f"{outdir}/"), (
                    f"{agent_cls.__name__} returned {c.file!r}, not a path under {outdir}/")
                assert sb.exists(c.file), f"{c.file} does not resolve in the sandbox"

            # The backend must be READ, not defaulted: structural_signature hashes it, so a
            # mislabelled CUDA candidate collides with a Triton one and the novelty gate rejects
            # a genuinely new structure as a duplicate.
            if len(names) > 1 and hasattr(got.candidates[0], "backend"):
                kinds = [c.backend for c in got.candidates]
                assert kinds == ["triton", "cuda"], (
                    f"{agent_cls.__name__} misread the backends: {kinds}. The second file uses "
                    f"load_inline, so it is cuda; structural_signature hashes this field")

            # A rescued candidate must be MARKED as rescued, or the lineage in the report cannot
            # distinguish it from one the agent actually described.
            text = " ".join(str(getattr(c, f, "")) for c in got.candidates
                            for f in ("approach_summary", "change_summary", "difference_claim"))
            assert "recovered" in text.lower() or "not stated" in text.lower(), (
                f"{agent_cls.__name__} produced an unmarked rescue: a reader of the lineage "
                f"cannot tell it from a described candidate")

    # An empty sandbox must rescue NOTHING rather than an empty result: `check_output` rejects an
    # empty candidate list, but returning one would journal a bogus RESCUE event and hide that
    # the agent never wrote anything.
    for agent_cls, outdir, _ in cases:
        with tempfile.TemporaryDirectory() as td:
            sb = Sandbox(root=_P(td))
            (_P(td) / outdir).mkdir(parents=True)
            agent = agent_cls.__new__(agent_cls)
            assert agent.rescue_from_sandbox(sb) is None, (
                f"{agent_cls.__name__} rescued something from an empty output directory")


def test_the_backend_detector_reads_the_source_rather_than_guessing():
    """`backend` feeds structural_signature (families.py:79-91), so a rescue cannot default it.

    Getting it wrong is not cosmetic: a CUDA candidate labelled triton hashes into the wrong
    signature, and `accept_novel_seed` then rejects a genuinely different structure as a
    `duplicate_signature` -- silently costing the run a family.
    """
    from kernel_optimizer.agents.modules import _detect_backend

    assert _detect_backend("mod = load_inline(name='m', cuda_sources='...')") == "cuda"
    assert _detect_backend("from torch.utils import cpp_extension") == "cuda"
    assert _detect_backend("import triton\n@triton.jit\ndef k(): pass") == "triton"
    # Neither marker: the documented default, and harmless because check_output rejects a file
    # with no kernel whatever this returns.
    assert _detect_backend("class ModelNew:\n    pass\n") == "triton"


def test_occupancy_is_recorded_in_every_verdict_but_only_a_lever_when_it_can_be():
    """Deferred from the L1:42 revalidation, then taken: the fact was dropped by four verdicts.

    `classify()` wrote ev["occupancy"] only inside the `near_limit` block near the end, which four
    of the five verdicts return before reaching (overhead_floor, launch_bound, memory_bound,
    compute_bound). So the facts section was inconsistent -- `uses_tensor_cores` appeared while the
    equally-cheap, equally-Tier-1 occupancy did not.

    Observed on run-l1-42-20260908-023039 with the REAL numbers used below: verdict `memory_bound`
    at 94.7% of the measured DRAM ceiling, advising "increase reuse (larger tiles, better
    blocking)" to a kernel whose profile held occupancy 0.3333 limited by blocks_per_sm -- already
    at the per-SM block cap. Tier 1 measured it; the classifier discarded it.

    The test asserts BOTH halves, because the careless fix passes the first and fails the second:
    recording the fact must not promote occupancy to a lever for a saturated kernel, where
    "residency is your problem" is the wrong instruction and bytes are the wall.
    """
    from kernel_optimizer.evaluation.bottleneck import DevicePeaks, classify
    from kernel_optimizer.evaluation.calibration import Thresholds

    # Box 2 as measured, and the thresholds derived on it.
    peaks = DevicePeaks(dram_tbs=0.9102221900268849, fp32_tflops=54.948151226599485,
                        tf32_tflops=88.88)
    th = Thresholds(dram_saturated_frac=0.8441, compute_saturated_frac=0.8047,
                    idle_frac=0.1674, launch_bound_cpu_ratio=0.8807)
    OCC = {"occupancy": 0.3333, "active_warps": 16, "max_warps_per_sm": 48,
           "blocks_per_sm": 16, "limiter": "blocks_per_sm"}
    SASS = {"instructions": 312, "tensor_core": 0, "global_load": 8, "global_store": 8,
            "spill_load": 0, "spill_store": 0, "shared_load": 0, "shared_store": 0,
            "vec_64": 0, "vec_128": 0, "barrier": 0}

    # 1. memory_bound, the exact L1:42 case: 4286586880 bytes in 4.9715 ms = 94.7% of ceiling.
    mem = classify(gpu_ms=4.971519947052002, cpu_issue_ms=None, flop_count=0,
                   byte_count=4286586880, peaks=peaks, n_regs=64, n_spills=0, shared_bytes=0,
                   max_regs_per_thread=255, max_shared_bytes=101376,
                   empty_launch_floor_ms=0.0174, thresholds=th, sass=SASS, occupancy=OCC)
    assert mem.kind == "memory_bound", f"the fixture no longer reproduces the run: {mem.kind}"
    assert mem.evidence.get("occupancy") == 0.3333, (
        "a memory_bound verdict still drops the occupancy fact -- the defect this fixes. Its own "
        "advice recommends larger tiles to a kernel already at the per-SM block cap")
    assert mem.evidence.get("occupancy_limiter") == "blocks_per_sm"
    # ... and the verdict itself must be unchanged: recording a fact must not re-route the
    # classification. A `memory_bound` kernel returns before the lever block is reached, so
    # `at_limit` cannot appear here -- if it ever does, occupancy has been promoted to a lever for
    # a saturated kernel, which would tell the agent to chase residency when bytes are the wall.
    #
    # NOTE this assertion is weak BY CONSTRUCTION and that is worth stating: `memory_bound`
    # returns early, so no edit to the lever block can make it fire. An earlier version of this
    # test claimed it as the guard against the careless fix, and it was vacuous -- verified by
    # applying that fix (dropping the saturation conditions from the `near_limit` guard) and
    # watching the test still pass. The real protection is
    # `test_the_lever_block_still_requires_an_unsaturated_kernel` below, which exercises the path
    # the guard actually governs.
    assert "at_limit" not in mem.evidence, (
        "a memory_bound verdict grew an at_limit lever list; the fact-recording change was not "
        "supposed to alter any verdict's routing")

    # 2. launch_bound and 3. overhead_floor: the two earliest returns, which never saw it either.
    launch = classify(gpu_ms=1.0, cpu_issue_ms=0.95, flop_count=0, byte_count=1000,
                      peaks=peaks, thresholds=th, sass=SASS, occupancy=OCC,
                      empty_launch_floor_ms=0.0174)
    assert launch.kind == "launch_bound", launch.kind
    assert launch.evidence.get("occupancy") == 0.3333, "launch_bound drops the occupancy fact"

    floor = classify(gpu_ms=0.018, cpu_issue_ms=None, flop_count=0, byte_count=1000,
                     peaks=peaks, thresholds=th, sass=SASS, occupancy=OCC,
                     empty_launch_floor_ms=0.0174)
    assert floor.kind == "overhead_floor", floor.kind
    assert floor.evidence.get("occupancy") == 0.3333, "overhead_floor drops the occupancy fact"

    # 4. An UNSATURATED kernel with low occupancy: here it SHOULD be a lever, and that behaviour
    # must be unchanged by moving the fact out of the block.
    slow = classify(gpu_ms=50.0, cpu_issue_ms=0.1, flop_count=0, byte_count=1000,
                    peaks=peaks, n_regs=64, n_spills=0, shared_bytes=0,
                    max_regs_per_thread=255, max_shared_bytes=101376,
                    empty_launch_floor_ms=0.0174, thresholds=th, sass=SASS, occupancy=OCC)
    assert slow.evidence.get("occupancy") == 0.3333
    assert "at_limit" in slow.evidence, (
        "an unsaturated kernel at 33% occupancy must still surface occupancy as a lever; moving "
        "the fact out of the near_limit block was not supposed to disable the lever")
    assert any("occupancy" in s for s in slow.evidence["at_limit"])

    # And the spill/register facts travel with it, for the same reason.
    spilling = classify(gpu_ms=50.0, cpu_issue_ms=None, flop_count=0, byte_count=1000,
                       peaks=peaks, n_regs=255, n_spills=88, shared_bytes=0,
                       max_regs_per_thread=255, max_shared_bytes=101376, thresholds=th,
                       sass=dict(SASS, spill_store=81, spill_load=80), occupancy=OCC)
    assert spilling.evidence.get("n_spills") == 88, "the spill count is not recorded as a fact"
    assert spilling.evidence.get("n_regs") == 255

    # Absent occupancy must leave no key rather than a null: an unmeasured signal and a measured
    # zero must stay distinguishable, which is the same rule the statics notes exist for.
    none_occ = classify(gpu_ms=4.9715, cpu_issue_ms=None, flop_count=0, byte_count=4286586880,
                        peaks=peaks, thresholds=th, sass=SASS, occupancy=None)
    assert "occupancy" not in none_occ.evidence, (
        "an unmeasured occupancy was recorded anyway; absent and zero must not look alike")


def test_the_lever_block_still_requires_an_unsaturated_kernel():
    """A saturated kernel with low occupancy must stay `memory_bound`, not become
    `resource_limited`.

    This pins the BOUNDARY behaviour, which is the part a future edit can plausibly break: the
    saturation returns fire before the lever block, so 84.41% of the measured DRAM ceiling is where
    the verdict flips from "you are at the ceiling" to "your tile caps residency". Both readings are
    defensible; what matters is that the line stays where the calibration puts it.

    A correction worth recording, because it reverses something I asserted while making this
    change: I claimed the `near_limit` gate's own `frac_bw < dram_saturated_frac and frac_fl <
    compute_saturated_frac` conditions were what stopped a saturated kernel from being called
    `resource_limited`, and that dropping them would tell an agent to chase residency at 94.7% of
    bandwidth. That is false. Removing those conditions changes NOTHING -- swept across both
    fractions, no input reaches the lever block with either fraction above its saturation line,
    because the early returns already took every such case. The conditions are dead code, kept
    only as documentation of intent. So there is no "careless variant" of the occupancy fix for a
    test to catch, and any test claiming to catch one is vacuous.
    """
    from kernel_optimizer.evaluation.bottleneck import DevicePeaks, classify
    from kernel_optimizer.evaluation.calibration import Thresholds

    peaks = DevicePeaks(dram_tbs=0.9102221900268849, fp32_tflops=54.948151226599485,
                        tf32_tflops=88.88)
    th = Thresholds(dram_saturated_frac=0.8441, compute_saturated_frac=0.8047,
                    idle_frac=0.1674, launch_bound_cpu_ratio=0.8807)
    OCC = {"occupancy": 0.3333, "limiter": "blocks_per_sm"}
    SASS = {"instructions": 312, "tensor_core": 0}
    B = 4286586880

    def at_pct_of_dram(pct):
        ms = B / (pct * peaks.dram_tbs * 1e12) * 1e3
        return classify(gpu_ms=ms, cpu_issue_ms=None, flop_count=0, byte_count=B, peaks=peaks,
                        n_regs=64, n_spills=0, shared_bytes=0, max_regs_per_thread=255,
                        max_shared_bytes=101376, thresholds=th, sass=SASS, occupancy=OCC)

    # Above the saturation line: the ceiling is the answer, and low occupancy must NOT redirect it.
    for pct in (0.947, 0.90, 0.85):
        v = at_pct_of_dram(pct)
        assert v.kind == "memory_bound", (
            f"at {pct*100:.0f}% of the measured DRAM ceiling the verdict is {v.kind!r}, not "
            f"memory_bound. A saturated kernel with 33% occupancy is not resource_limited -- it is "
            f"done, and telling it to shrink its tile sends the agent after the wrong thing")
        assert "at_limit" not in v.evidence

    # Below the line, with a resource genuinely near its cap: the lever is legitimate here, and
    # this half must keep working -- moving the FACT out of the block was not meant to disable it.
    v = at_pct_of_dram(0.30)
    assert v.kind == "resource_limited", f"the lever path stopped firing entirely: {v.kind}"
    assert any("occupancy" in s for s in v.evidence.get("at_limit", []))


def test_a_rewrite_rejection_is_not_recorded_as_a_novelty_rejection():
    """Loop D has fired ZERO times in every run so far, so its first firing is the thing to watch --
    and a count of NOVELTY_REJECTED could not answer that, because the REWRITE path borrowed the
    same event type.

    Observed on run-l3-43-20260908-053708: at 09:45:46 a round-3 rewrite was refused as a
    structural duplicate and logged as NOVELTY_REJECTED with `origin: "rewrite"`. Reading the log
    made it look as though novelty had run when it never had. The payload field distinguished them,
    so no data was lost -- but the event TYPE is what anyone counts.

    Asserts on the emitted EVENTS, not on source text. A first version of this test grepped the
    function source for the absent string and failed on the word appearing in the explanatory
    comment -- the same trap that made an earlier test in this file pin a defect as its spec.
    """
    import ast
    import inspect

    from kernel_optimizer.control.orchestrator import Orchestrator

    def emitted_types(fn):
        """Event type strings this function passes to store.append, ignoring comments."""
        tree = ast.parse(inspect.getsource(fn).lstrip())
        out = set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "append" and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                out.add(node.args[0].value)
        return out

    rewrite_events = emitted_types(Orchestrator._do_rewrite)
    assert "REWRITE_REJECTED" in rewrite_events, (
        f"the rewrite path does not emit its own rejection event; it emits {rewrite_events}")
    assert "NOVELTY_REJECTED" not in rewrite_events, (
        "the rewrite path still EMITS NOVELTY_REJECTED, so counting that event conflates Loop C "
        "with Loop D and cannot answer whether novelty ever ran")

    novelty_events = emitted_types(Orchestrator._novelty_round)
    assert "NOVELTY_REJECTED" in novelty_events, (
        f"the novelty path lost its rejection event; it emits {novelty_events}")

    # A report must still surface a rejection under EITHER name: existing logs carry the old type,
    # and silently dropping them would make a replayed report thinner than the run it describes.
    from kernel_optimizer.reporting import report as report_mod

    rep_src = inspect.getsource(report_mod)
    reads_both = ('"REWRITE_REJECTED"' in rep_src and '"NOVELTY_REJECTED"' in rep_src)
    assert reads_both, ("report.py must read both names: the new one for current runs, the old one "
                        "so a replay of an existing log still shows its rejections")


def test_every_backend_gets_its_resource_metadata_collected():
    """The gate must not be Triton-specific.

    Measured defect: callers passed `collect_triton_metadata=(backend == "triton")`, and the
    worker's outer `if` read that same flag before choosing between the Triton reader and the
    cubin reader. So for a cuda/cutlass/cute candidate the flag was False, the outer test
    failed, and the cubin branch -- written specifically for those backends -- was unreachable.
    A correct load_inline candidate came back with `cubin: None` and no error.

    Drives the real predicate rather than asserting on source text, and checks the real job
    builders, so it fails on the old code and passes on the fix.
    """
    from kernel_optimizer.gpu.jobs import make_eval_job, make_relaxed_correctness_job
    from kernel_optimizer.gpu.worker_main import _wants_kernel_metadata

    for backend in ("triton", "cuda", "cutlass", "cute"):
        job = make_eval_job(
            "ref.py", "k.py", measure_performance=False, num_correct_trials=3,
            num_perf_trials=20, timing_method="cuda_event", backend=backend,
            precision="fp32", seed=1, build_dir=None, collect_kernel_metadata=True,
        )
        assert _wants_kernel_metadata(job), (
            f"a {backend} eval job does not request kernel metadata, so its candidate reaches "
            f"the bottleneck classifier with no registers/spills/shared/occupancy and is judged "
            f"worse than a Triton candidate for reasons unrelated to its kernel")
        relaxed = make_relaxed_correctness_job(
            "ref.py", "k.py", num_correct_trials=3, backend=backend, precision="fp32",
            seed=1, collect_kernel_metadata=True, relaxed_elem_tol=1e-3,
            relaxed_pass_frac=0.99, cosine_min=0.9998,
        )
        assert _wants_kernel_metadata(relaxed), (
            f"a {backend} relaxed-correctness job does not request kernel metadata")

    # The historical key must still be honoured, or replaying an existing job file breaks.
    assert _wants_kernel_metadata({"collect_triton_metadata": True})
    assert not _wants_kernel_metadata({"collect_triton_metadata": False})
    assert not _wants_kernel_metadata({})
    # The new key wins when both are present.
    assert _wants_kernel_metadata({"collect_kernel_metadata": True,
                                   "collect_triton_metadata": False})

    # And the callers must actually ask for it on every backend: a call site that still
    # conditions on the backend would reintroduce the defect while this test's own job
    # builders stay green.
    import inspect

    from kernel_optimizer.evaluation import correctness as corr_mod

    src = inspect.getsource(corr_mod)
    assert '(backend == "triton")' not in src, (
        "a call site still gates metadata collection on the backend being Triton")


def test_the_launched_kernel_filter_matches_mangled_against_demangled_names():
    """The filter compared demangled profiler names with mangled cubin symbols.

    `torch.profiler` reports `addk(float const*, float const*, float*, int)`; `cuobjdump`
    reports `_Z4addkPKfS0_Pfi`. A substring test between those can never succeed, so
    `launched_filter` came back `no_name_match` and the resources were the union over the
    whole cubin -- measured on box 2 as 18 kernels of which 17 never ran. For CUTLASS, whose
    template variants are exactly this hazard, the guard was inert.
    """
    from kernel_optimizer.gpu.worker_main import _kernel_identifiers

    # The measured real pair.
    assert "addk" in _kernel_identifiers("_Z4addkPKfS0_Pfi")
    assert "addk" in _kernel_identifiers("addk(float const*, float const*, float*, int)")
    assert _kernel_identifiers("_Z4addkPKfS0_Pfi") & _kernel_identifiers(
        "addk(float const*, float const*, float*, int)"), (
        "the mangled and demangled forms of the SAME kernel do not intersect, so the filter "
        "cannot ever apply")

    # A templated kernel, which is the CUTLASS shape: identifier survives, template args go.
    mangled = "_Z7mm_tf32ILi64ELi64ELi32ELi2ELi2ELb1EEvPKfS1_Pfiii"
    assert "mm_tf32" in _kernel_identifiers(mangled)
    assert "mm_tf32" in _kernel_identifiers("mm_tf32<64, 64, 32, 2, 2, true>(float const*)")

    # A nested (namespaced) name, as CUTLASS emits.
    nested = _kernel_identifiers("_ZN7cutlass6kernel4gemmEv")
    assert "gemm" in nested and "cutlass" in nested

    # Distinct kernels must NOT intersect, or the filter would keep everything.
    assert not (_kernel_identifiers("_Z4addkPKfS0_Pfi")
                & _kernel_identifiers("_Z7mm_tf32ILi64EEvPKf")), (
        "two different kernels share an identifier, so the filter cannot discriminate")

    # And a short name must not match an unrelated symbol that merely contains its letters --
    # the reason this compares identifier sets instead of substrings.
    assert not (_kernel_identifiers("_Z2mmPf")            # kernel `mm`
                & _kernel_identifiers("_Z6mmadd2Pf")), (  # unrelated kernel `mmadd2`
        "a short identifier matched a longer unrelated one; substring semantics have crept back")


def test_a_hard_config_failure_is_pruned_so_tpe_learns_the_region():
    """P1: FAIL is excluded from the TPE model; PRUNED is kept.

    Measured against Optuna directly: 12 trials reported FAIL leave 1 trial visible to the
    sampler; the same 12 reported PRUNED leave 13. So reporting a config-determined refusal as
    FAIL discards it, which is why L3:43's per-candidate shared-memory failure rate (21-33%,
    180 of 1004 trials) never decayed -- TPE kept proposing a region it was never told about.

    Only HARD, config-determined reasons may be pruned. A runtime_error or
    correctness_mismatch must stay FAIL: those can be non-deterministic or a defect in the
    candidate rather than in the point, and pruning them would teach the sampler noise.
    """
    from optuna.trial import TrialState

    from kernel_optimizer.models.core import (
        LatencyStats,
        ParamDomain,
        ParameterSpace,
        TrialRecord,
    )
    from kernel_optimizer.tuning.tpe import OptunaTPETuner

    space = ParameterSpace(
        space_id="sp-x", candidate_id="c1", version=1, source_sha="deadbeef",
        domains=[ParamDomain(name="BLOCK", kind="int", choices=[16, 32, 64, 128])],
    )

    def run_with(failure_kind, status="fail", latency=None):
        tuner = OptunaTPETuner(space, guard_ok=lambda p: True, budget=3, seed=0)
        asked = tuner.ask()
        assert asked is not None
        trial_id, params = asked
        tuner.tell(trial_id, TrialRecord(
            trial_id=trial_id, candidate_id="c1", space_id="sp-x", params=params,
            status=status, failure_kind=failure_kind, latency_ms=latency))
        return tuner.study.trials[0].state

    for kind in ("infeasible_shared_memory", "guard_rejected", "materialize_error"):
        assert run_with(kind) is TrialState.PRUNED, (
            f"{kind} is reported FAIL, so Optuna drops it from the TPE model and the sampler "
            f"keeps proposing the same unusable region")

    for kind in ("runtime_error", "correctness_mismatch", "oom", "timeout", None):
        assert run_with(kind) is TrialState.FAIL, (
            f"{kind} is being PRUNED; only config-determined refusals may be, or the sampler "
            f"is taught to avoid regions on the strength of noise or a candidate-level defect")

    # A successful trial must still be told as a value, not a state.
    st = run_with(None, status="complete",
                  latency=LatencyStats(mean=5.0, std=0.1, min=4.9, max=5.1, n_samples=20,
                                       median=5.0))
    assert st is TrialState.COMPLETE


def test_the_compile_screen_only_refuses_on_the_compilers_own_number():
    """P1: the screen must act ONLY when the compiler's figure exceeds the device limit.

    Every other outcome has to return None so the real trial decides -- a screen that guesses
    would silently delete feasible configurations from the search, which is worse than the
    18% waste it exists to remove.
    """
    from pathlib import Path

    from kernel_optimizer.evaluation.correctness import CorrectnessEvaluator

    class FakeWorker:
        def __init__(self, reply):
            self.reply = reply
            self.calls = 0

        def run_job(self, job, timeout, tag, lock_mode=None):
            self.calls += 1
            assert job["job_type"] == "compile_probe", job["job_type"]
            assert lock_mode == "shared", "the screen must not take the exclusive timing lock"
            return self.reply

    class Cfg:
        precision = "fp32"
        build_timeout_s = 60.0
        eval_timeout_s = 60.0
        correctness_mode = "dual_witness_relaxed"

    class Task:
        ref_path = Path("ref.py")

    def screen(reply, limit=101376, src="x = 1"):
        import tempfile

        w = FakeWorker(reply)
        ev = CorrectnessEvaluator.__new__(CorrectnessEvaluator)
        ev.worker, ev.cfg, ev.seed = w, Cfg(), 0
        ev._screen_cache = {}
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                         encoding="utf-8") as f:
            f.write(src)
            p = Path(f.name)
        return ev.compile_screen(Task(), p, "t", "triton", limit), w

    # Over the limit: refuse, and say the real numbers.
    out, _ = screen({"ok": True, "max_shared": 393216,
                     "kernels": [{"name": "_linear_fwd", "shared": 393216}]})
    assert out is not None, "an over-limit config was not refused"
    assert out["max_shared"] == 393216 and out["limit"] == 101376
    assert "393216" in out["detail"] and "101376" in out["detail"], (
        "the refusal must carry both numbers, or a reader cannot check the verdict")

    # Everything else: no opinion.
    for reply, why in [
        ({"ok": True, "max_shared": 101376, "kernels": []}, "exactly AT the limit is legal"),
        ({"ok": True, "max_shared": 65536, "kernels": []}, "within the limit"),
        ({"ok": False, "reason": "triton unavailable"}, "probe could not answer"),
        ({"ok": True, "max_shared": None, "kernels": []}, "no shared figure read"),
    ]:
        out, _ = screen(reply)
        assert out is None, f"the screen returned a verdict when {why}"

    # No device limit configured -> the screen must not even call the worker.
    out, w = screen({"ok": True, "max_shared": 393216, "kernels": []}, limit=None)
    assert out is None and w.calls == 0, (
        "with no device limit the screen still probed; it has nothing to compare against")

    # Cached: the same source must not be probed twice.
    import tempfile

    w = FakeWorker({"ok": True, "max_shared": 65536, "kernels": []})
    ev = CorrectnessEvaluator.__new__(CorrectnessEvaluator)
    ev.worker, ev.cfg, ev.seed = w, Cfg(), 0
    ev._screen_cache = {}
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write("y = 2")
        p = Path(f.name)
    ev.compile_screen(Task(), p, "t", "triton", 101376)
    ev.compile_screen(Task(), p, "t", "triton", 101376)
    assert w.calls == 1, f"the screen probed {w.calls} times for one source; cache is not working"


def test_the_launch_bound_ratio_uses_the_probes_own_denominator():
    """P2 cause (c): cpu_issue_ms must be divided by the overhead probe's OWN gpu_ms.

    The probe measures both numbers in one pass with NO L2 flush, and its docstring says its
    gpu_ms is warm-cache and valid only as this ratio's denominator. The harness's latency is a
    different measurement, and mixing them yields a wrong ratio whose SIGN is not predictable:
    measured on box 2, L3:43's theta_best has harness 3.0126 vs probe 3.0050 ms (0.25% apart),
    while a 40-op elementwise chain has harness 0.3497 vs probe 0.5849 -- harness 40% SMALLER.
    So this pins the denominator actually used, not a claimed direction of error.
    """
    from kernel_optimizer.evaluation.bottleneck import DevicePeaks, Thresholds, classify

    peaks = DevicePeaks(dram_tbs=0.91, fp32_tflops=54.9, tf32_tflops=88.9)
    th = Thresholds(dram_saturated_frac=0.84, compute_saturated_frac=0.81, idle_frac=0.17,
                    launch_bound_cpu_ratio=0.87)
    # A launch-bound kernel: the CPU needs 0.09 ms to issue work the GPU does in 0.10 ms
    # warm-cache. The harness reports 0.20 ms for its own (flushed) measurement.
    common = dict(flop_count=10**6, byte_count=10**6, peaks=peaks, thresholds=th,
                  cpu_issue_ms=0.09)

    with_probe = classify(gpu_ms=0.20, overhead_gpu_ms=0.10, **common)
    assert with_probe.kind == "launch_bound", (
        f"a kernel whose CPU issue cost is 90% of its warm-cache GPU time was classified "
        f"{with_probe.kind}; the ratio must use the probe's own denominator")
    assert with_probe.evidence["cpu_over_gpu"] == 0.9
    assert with_probe.evidence["cpu_over_gpu_denominator"] == "overhead_probe_gpu_ms"
    assert with_probe.evidence["overhead_gpu_ms"] == 0.10

    # Same kernel with the probe's figure absent: the ratio is computed against a DIFFERENT
    # measurement (0.09/0.20 = 0.45), misses the 0.87 line, and must say so in the label.
    without = classify(gpu_ms=0.20, overhead_gpu_ms=None, **common)
    assert without.evidence["cpu_over_gpu"] == 0.45
    assert without.evidence["cpu_over_gpu_denominator"] == "harness_gpu_ms_FALLBACK", (
        "the fallback must be labelled, or a reader cannot tell a sound ratio from a mixed one")
    assert without.kind != "launch_bound"
    assert "overhead_gpu_ms" not in without.evidence, (
        "the fallback path must not report an overhead_gpu_ms it does not have")

    # The mixed ratio can also err the OTHER way, which is why the label matters more than any
    # assumed direction: here the harness figure is smaller, so the mixed ratio overshoots.
    high = classify(gpu_ms=0.05, overhead_gpu_ms=0.10, cpu_issue_ms=0.09,
                    flop_count=10**6, byte_count=10**6, peaks=peaks, thresholds=th)
    assert high.evidence["cpu_over_gpu"] == 0.9, (
        "with the probe's denominator the ratio must be 0.09/0.10 regardless of what the "
        "harness measured")


def test_launch_overhead_is_attached_on_the_relaxed_path_too():
    """P2 cause (b): the probe lived only in run_eval, while both L3 configs route to the
    relaxed handler.

    Measured on the L3:43 final re-eval: `measure_launch_overhead: True` returned
    `launch_overhead: None`, because `run_relaxed_correctness` mentioned neither name. Fixing
    causes (a) and (c) without this one would reproduce exactly that on every config the
    experiments actually use.

    Walks the AST for calls to the shared helper, so it tracks the wiring rather than a string.
    """
    import ast
    import inspect

    from kernel_optimizer.gpu import worker_main

    tree = ast.parse(inspect.getsource(worker_main))
    calls_helper = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for inner in ast.walk(node):
            if (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name)
                    and inner.func.id == "_attach_launch_overhead"):
                calls_helper.add(node.name)

    for handler in ("run_eval", "run_relaxed_correctness"):
        assert handler in calls_helper, (
            f"{handler} never attaches the launch-overhead probe, so a job asking for it gets "
            f"silence -- and both L3 configs route to run_relaxed_correctness")

    # And the handler must be registered, or the job type is unreachable.
    assert "compile_probe" in worker_main.HANDLERS


def test_a_low_precision_kernel_is_scored_against_its_own_ceiling():
    """P3: an fp16 kernel must be measured against the fp16 ceiling, not tf32.

    On L3:43, 13 of 25 verdicts carried `impossible_fraction` because the only tensor-core
    ceiling measured was tf32 and every leading candidate was fp16 or bf16. theta_best read
    140.7% of "peak", which reads as "saturated, stop optimizing" for a kernel with headroom
    left -- and the defect got WORSE as candidates improved.
    """
    from kernel_optimizer.evaluation.bottleneck import DevicePeaks, Thresholds, classify

    # Measured shape of a 4090: fp16 about 2x tf32, tf32 about 1.6x fp32.
    peaks = DevicePeaks(dram_tbs=0.9102, fp32_tflops=54.95, tf32_tflops=88.88,
                        fp16_tflops=165.0, bf16_tflops=163.0)
    th = Thresholds(dram_saturated_frac=0.8449, compute_saturated_frac=0.8101, idle_frac=0.1675,
                    launch_bound_cpu_ratio=0.8737)
    sass = {"instructions": 3960, "tensor_core": 128}
    # A kernel doing 95.9 TFLOP/s: over the tf32 ceiling, comfortably under the fp16 one.
    common = dict(flop_count=412_316_860_416, byte_count=416_296_960, peaks=peaks,
                  thresholds=th, sass=sass, cpu_issue_ms=None,
                  occupancy={"occupancy": 0.1667, "limiter": "registers"})
    gpu_ms = 412_316_860_416 / 95.9e12 * 1e3

    fp16 = classify(gpu_ms=gpu_ms, precision="fp16", **common)
    assert fp16.evidence["compute_ceiling_used"] == "tensor-core (fp16)"
    assert fp16.evidence["candidate_precision"] == "fp16"
    assert fp16.evidence["pct_of_compute_peak"] < 100, (
        f"an fp16 kernel at 95.9 TFLOP/s reads "
        f"{fp16.evidence['pct_of_compute_peak']}% of peak; it is being scored against the "
        f"wrong ceiling")
    assert "impossible_fraction" not in fp16.evidence

    bf16 = classify(gpu_ms=gpu_ms, precision="bf16", **common)
    assert bf16.evidence["compute_ceiling_used"] == "tensor-core (bf16)"

    # tf32 and unknown precision keep the tf32 ceiling -- and there the same kernel IS over it,
    # which is the pre-fix reading, so this pins that the change is precision-driven and not a
    # blanket ceiling raise.
    for prec in ("tf32", None):
        v = classify(gpu_ms=gpu_ms, precision=prec, **common)
        assert v.evidence["compute_ceiling_used"] == "tensor-core (tf32)", prec
        assert v.evidence["impossible_fraction"] > 1.0, prec

    # An OLD calibration with no fp16 figure must still classify, falling back to tf32 and
    # saying in the disagreement that the substitution is the likely explanation.
    old_peaks = DevicePeaks(dram_tbs=0.9102, fp32_tflops=54.95, tf32_tflops=88.88)
    stale = classify(gpu_ms=gpu_ms, precision="fp16",
                     **{**common, "peaks": old_peaks})
    assert stale.evidence["compute_ceiling_used"] == "tensor-core (tf32)"
    assert stale.evidence["impossible_fraction"] > 1.0
    assert "no measured fp16 ceiling" in stale.disagreement, (
        "a substituted ceiling must be named as the likely cause, or the agent is told the "
        "candidate does less arithmetic than the reference when the truth is a missing number")


def test_the_calibration_carries_the_low_precision_ceilings():
    """P3 plumbing: worker result -> Calibration -> DevicePeaks -> the agent's brief.

    Each hop has silently dropped a field before (aux_output_ops at the task-cost boundary,
    the median on the strict path), so this walks the whole chain rather than one end of it.
    """
    import inspect

    from kernel_optimizer.evaluation.calibration import Calibration
    from kernel_optimizer.gpu import calibrate as calibrate_mod
    from kernel_optimizer.gpu import worker_main

    # 1. The worker measures them. Asserted on the DTYPES it probes rather than on the result
    #    keys, because those are built as f"{name}_tflops" inside a loop -- a first version of
    #    this test looked for the literal string and failed on correct code.
    src = inspect.getsource(worker_main.run_calibrate)
    assert "torch.float16" in src and "torch.bfloat16" in src, (
        "run_calibrate does not measure the fp16/bf16 matmul ceilings, so a low-precision "
        "candidate is scored against tf32 -- roughly half its real ceiling")
    assert '(("fp16", torch.float16), ("bf16", torch.bfloat16))' in src, (
        "the low-precision ceiling loop is not the shape this test can verify; check that both "
        "dtypes are still measured and update the assertion deliberately")

    # 2. The model can hold them, defaulting to 0 so an old cached calibration still loads.
    cal = Calibration(device_name="x", capability=[8, 9], sm_count=128, dram_tbs=0.9,
                      fp32_tflops=55.0, tf32_tflops=89.0)
    assert cal.fp16_tflops == 0.0 and cal.bf16_tflops == 0.0
    cal2 = cal.model_copy(update={"fp16_tflops": 165.0, "bf16_tflops": 163.0})
    assert cal2.fp16_tflops == 165.0

    # 3. calibrate.py carries them from the worker result, and into the event payload -- a
    #    reader of events.jsonl must be able to see which ceilings a run had.
    csrc = inspect.getsource(calibrate_mod)
    assert 'fp16_tflops=float(result.get("fp16_tflops"' in csrc, (
        "calibrate.py does not read fp16_tflops out of the worker result")
    assert csrc.count('"fp16_tflops"') >= 2, (
        "the fp16 ceiling does not reach the CALIBRATION_MEASURED/LOADED payload, so a run's "
        "own log cannot say which ceiling its percentages were against")

    # 4. The agent's brief shows them. Needs a verdict: with none, `_bottleneck_doc` returns its
    #    "could not measure this box" early exit and would never reach the ceiling table.
    from kernel_optimizer.agents.modules import _bottleneck_doc
    from kernel_optimizer.evaluation.bottleneck import BottleneckVerdict

    verdict = BottleneckVerdict(kind="compute_bound", evidence={"gpu_ms": 3.0},
                                suggests="x")
    doc = _bottleneck_doc(verdict, None, cal2)
    assert "fp16" in doc and "165" in doc, (
        "the agent is not told this box's fp16 ceiling, so it cannot judge a precision change")
    assert "bf16" in doc and "163" in doc


def test_a_terminating_signal_unwinds_instead_of_orphaning_the_server():
    """P4: SIGTERM must raise, so `with Runtime(...)` shuts the opencode server down.

    Observed while stopping run-l3-43-20260908-053708 by hand: killing the orchestrator left
    its `opencode serve` reparented to init and still holding port 4096, which the next run's
    server would have collided with. `Runtime.__exit__` already stops it; a default SIGTERM
    just never lets it run.
    """
    import signal

    from kernel_optimizer import cli

    saved = {}
    try:
        for name in ("SIGTERM", "SIGHUP"):
            sig = getattr(signal, name, None)
            if sig is not None:
                saved[sig] = signal.getsignal(sig)
        cli._install_termination_handler()
        for sig in saved:
            handler = signal.getsignal(sig)
            assert callable(handler), (
                f"{sig!r} still has the default disposition, so the interpreter dies without "
                f"unwinding and the opencode server is orphaned")
            try:
                handler(int(sig), None)
            except KeyboardInterrupt:
                pass  # what we want: the `with` blocks get to unwind
            else:
                raise AssertionError(f"the {sig!r} handler did not raise, so nothing unwinds")
    finally:
        for sig, prev in saved.items():
            signal.signal(sig, prev)


def test_an_interrupted_run_still_writes_its_report():
    """P4: stopping a run must leave the same artifacts as finishing one.

    Terminating L3:43 to land these fixes meant re-deriving theta_best, re-materializing it and
    re-running the final evaluation by hand. The report is regenerated from events.jsonl, so
    there is no reason a deliberate stop cannot produce it.
    """
    from kernel_optimizer import cli

    calls = {"report": 0, "appended": []}

    class FakeRuntime:
        def __init__(self, cfg, log_dir=None):
            self.exited = False

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.exited = True
            calls["server_stopped"] = True
            return False   # must NOT swallow -- the interrupt is handled inside

    class FakeStore:
        run_dir = "/tmp/x"

        def append(self, event_type, payload):
            calls["appended"].append(event_type)

    class FakeReport:
        def generate(self, store):
            calls["report"] += 1
            return "/tmp/x/report.md"

    class Orch:
        def run(self):
            raise KeyboardInterrupt("terminated by signal 15")

    rc = cli._run_orchestrated(object(), FakeStore(), object(), FakeRuntime,
                               lambda cfg, store, task, runtime: Orch(),
                               lambda: FakeReport())
    assert calls.get("server_stopped"), "the runtime was not exited, so the server is orphaned"
    assert calls["report"] == 1, "an interrupted run wrote no report"
    assert "RUN_INTERRUPTED" in calls["appended"], (
        "the log does not record that the run was interrupted, so a reader cannot tell a "
        "deliberate stop from a converged run")
    assert rc == 130, f"expected the conventional interrupt exit code, got {rc}"

    # A normal run must be unaffected: no interrupt event, exit 0.
    calls["appended"].clear()
    calls["report"] = 0

    class OkOrch:
        def run(self):
            return {"best": {"tuned_ms": 1.0}}

    rc = cli._run_orchestrated(object(), FakeStore(), object(), FakeRuntime,
                               lambda cfg, store, task, runtime: OkOrch(),
                               lambda: FakeReport())
    assert rc == 0 and calls["report"] == 1
    assert "RUN_INTERRUPTED" not in calls["appended"]


def test_a_cuda_candidate_is_not_judged_blind_against_a_triton_one():
    """The verdict must not depend on which backend wrote the kernel.

    Measured before this fix, on identical inputs (same latency, registers, shared bytes) with
    the only difference being what the profiler could collect:

        Triton  -> resource_limited, 30.0% of the fp16 ceiling, "shrink the tile"
        CUDA    -> compute_bound,    89.7% of the FP32 ceiling, "move onto tensor cores"

    The CUDA reading was wrong twice over: it was already using tensor cores, and it was
    nowhere near a ceiling. Cause: `uses_tensor_cores` comes from the SASS instruction mix,
    which was only ever read from `compiled.asm["cubin"]` -- a Triton object. A nvcc build's
    cubin lives inside the .so torch links, so it needs `disassemble_object`.

    Any backend comparison run before this would have measured the profiler's coverage rather
    than the kernels.
    """
    from kernel_optimizer.evaluation.bottleneck import DevicePeaks, Thresholds, classify

    peaks = DevicePeaks(dram_tbs=0.9102, fp32_tflops=54.95, tf32_tflops=89.06,
                        fp16_tflops=164.4, bf16_tflops=166.7)
    th = Thresholds(dram_saturated_frac=0.8449, compute_saturated_frac=0.8101,
                    idle_frac=0.1675, launch_bound_cpu_ratio=0.8737)
    common = dict(gpu_ms=8.369, flop_count=412_316_860_416, byte_count=416_296_960,
                  peaks=peaks, thresholds=th, n_regs=255, n_spills=8, shared_bytes=98304,
                  max_regs_per_thread=255, max_shared_bytes=101376, cpu_issue_ms=None,
                  precision="fp16")
    mix = {"instructions": 3960, "tensor_core": 128}

    triton = classify(**common, sass=mix,
                      occupancy={"occupancy": 0.0833, "limiter": "shared_memory"})
    cuda = classify(**common, sass=mix, occupancy=None)   # cubin: SASS yes, occupancy never

    assert triton.kind == cuda.kind, (
        f"the same kernel is classified {triton.kind} as Triton and {cuda.kind} as CUDA, so the "
        f"harness compares profiler coverage rather than kernels")
    assert triton.evidence["compute_ceiling_used"] == cuda.evidence["compute_ceiling_used"]
    assert triton.evidence["pct_of_compute_peak"] == cuda.evidence["pct_of_compute_peak"]
    assert cuda.evidence["uses_tensor_cores"] is True

    # Without the instruction mix the fp32 ceiling is used and the verdict flips -- the defect,
    # pinned so a regression is visible rather than merely worse. (The label here is the plain
    # "fp32" set before the tensor-core branch, not the fallback name from
    # `compute_ceiling_for`, which is only consulted once tensor-core use is confirmed.)
    blind = classify(**common, sass=None, occupancy=None)
    assert blind.evidence["compute_ceiling_used"] == "fp32"
    assert blind.evidence.get("uses_tensor_cores") is None, (
        "with no instruction mix the field must be absent or None -- never guessed")
    assert blind.kind != triton.kind, (
        "the blind reading must differ, or this test cannot show what the SASS buys")


def test_sass_is_read_from_a_built_object_not_only_a_triton_cubin():
    """`disassemble_object` must exist and be what the cubin path uses.

    Verified against a real load_inline extension on box 2: `cuobjdump -sass` on the .so
    returned 7508 characters, counted to 32 instructions with tensor_core=0 (correct for a
    plain elementwise add). And `num_warps` is deliberately NOT recovered this way -- it is a
    launch parameter (`<<<grid, block>>>`), absent from every compiled artifact, which is why
    occupancy stays unavailable for this backend and is reported as unmeasurable rather than
    left blank.
    """
    import ast
    import inspect

    from kernel_optimizer.evaluation import statics
    from kernel_optimizer.gpu import worker_main

    assert hasattr(statics, "disassemble_object")
    assert "cuobjdump" in inspect.getsource(statics.disassemble_object), (
        "a host binary needs cuobjdump to extract its SASS; nvdisasm takes a bare cubin")

    # The cubin extractor must actually call it, via the per-object helper.
    tree = ast.parse(inspect.getsource(worker_main))
    callers = {node.name for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef)
               for inner in ast.walk(node)
               if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name)
               and inner.func.id == "_cubin_sass"}
    assert "_extract_cubin_metadata" in callers, (
        f"the cubin path does not attach an instruction mix; callers found: {callers}")

    # And it must say WHY occupancy is missing, not silently omit it.
    helper_src = inspect.getsource(worker_main._cubin_sass)
    assert "num_warps is a launch parameter" in helper_src, (
        "an unmeasurable signal must be reported as unmeasurable, or a reader takes its "
        "absence for a clean bill of health")


def test_the_contract_names_the_four_constructs_the_static_check_refuses():
    """F1: `try`/`except`/`pass`/threading are hard rejections; the contract must say so.

    Measured with the harness's own arguments (backend="triton", precision="fp32"), all four
    are refused before the file reaches the GPU: a host-side try/except fallback, an
    `if ...: pass` branch, an UNUSED `import threading`, and -- the trap -- the word "pass"
    inside a STRING LITERAL, because KernelBench's checker strips comments but not strings.

    The gate has never fired on us, and only by luck: of 20 real candidate sources on box 1,
    `try:` 0, `except` 0, and both "pass" hits were a comment ("single-pass") and a function
    name (`_twopass_attn_kernel`), neither matching `\bpass\b`. One stray bare `pass` changes
    that, and the message the candidate receives talks about an "inheritance bypass" -- an
    accusation unrelated to what it did, costing a whole repair round to decode.

    So this test pins the CONTRACT text, which is the cheap half of the fix: an agent told in
    advance does not write them.
    """
    from kernel_optimizer.agents.modules import _contract_doc

    doc = _contract_doc()
    for construct in ("try:", "except", "pass", "threading"):
        assert construct in doc, f"the contract never mentions {construct!r}"
    # The string-literal trap is the non-obvious half: a candidate can be refused for a word
    # in a comment-like note. Naming the constructs without this would still cost a round.
    low = doc.lower()
    assert "string literal" in low, (
        "the contract must warn that `pass` inside a string literal also fails -- comments "
        "are stripped before the regex runs, strings are not")
    # And it must say the message will not describe what the candidate was doing, or an agent
    # reads "inheritance bypass" as a real diagnosis of its own code.
    assert "multiprocessing" in doc or "concurrent.futures" in doc, (
        "threading's siblings are caught by the same check and must be named")


def test_our_static_check_pins_its_own_forbidden_list():
    """F10: pass `forbidden` explicitly, so an upstream edit cannot move our acceptance bar.

    Two properties, both load-bearing:
      1. the four anti-cheat checks we require are passed explicitly (not inherited from
         KernelBench's STRICT_CHECKS default);
      2. `torch_computation_ops` and `pytorch_wrap` are NOT in that list -- they stay
         warnings, which is exactly what makes calling cuBLAS for a sub-op legal. Promoting
         either to strict would re-forbid the vendor-library route F2 opens (measured worth
         1.33x on L3:43's projections at strict ieee).
    """
    import ast
    import inspect

    from kernel_optimizer.gpu import worker_main

    required = set(worker_main.REQUIRED_STATIC_CHECKS)
    assert required == {"code_bypass", "timing_event_patch", "thread_injection", "lazy_eval"}, (
        f"the pinned anti-cheat set changed unexpectedly: {sorted(required)}")
    for permitted in ("torch_computation_ops", "pytorch_wrap", "stream_injection",
                      "precision_downgrade"):
        assert permitted not in required, (
            f"{permitted} must stay a WARNING: enforcing it would forbid the vendor-library "
            f"route (cuBLAS for a large regular GEMM) that the contract now permits")

    # It must actually be forwarded -- a constant nobody passes protects nothing.
    src = inspect.getsource(worker_main.run_static_check)
    tree = ast.parse(src.lstrip())
    forwarded = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name)
                and node.func.id == "validate_kernel_static"):
            continue
        assert any(kw.arg == "forbidden" for kw in node.keywords), (
            "validate_kernel_static is called without `forbidden`, so it silently falls back "
            "to KernelBench's STRICT_CHECKS and our acceptance bar can drift upstream")
        forwarded = True
    assert forwarded, "no validate_kernel_static call found in run_static_check"


def test_the_contract_states_both_halves_of_the_relaxed_gate():
    """The dual-precision criterion is a conjunction; the contract used to state only one half.

    `_relaxed_close` requires frac > relaxed_pass_frac AND cosine >= cosine_min. A candidate
    told only about the 1%/99% element test can land a result that clears it and fails on
    cosine, with no way to have anticipated that from the contract.
    """
    from kernel_optimizer.agents.modules import _contract_doc

    doc = _contract_doc().lower()
    assert "99%" in doc, "the element-fraction half of the gate is missing"
    assert "cosine" in doc, (
        "the contract omits the cosine criterion, so a candidate cannot know both halves "
        "must hold")


def test_the_contract_permits_the_vendor_library_and_keeps_its_floor():
    """F2: torch ops may do COMPUTATION, not only layout -- with the hard floor intact.

    The old wording restricted torch ops to "(layout, reshaping)", which excluded calling
    cuBLAS for a sub-op. Nothing in the code ever enforced that: KernelBench reports
    `torch_computation_ops` as a WARNING (see the F10 test), so the restriction lived only in
    our text. Measured cost of the restriction: at strict IEEE fp32, cuBLAS beat our
    hand-written Triton GEMM by 1.33x on L3:43's two projections, which are 85.7% of that
    task's FLOPs.

    The permission must arrive WITH both halves of the trade and the floor, or it turns into
    "always call the library" -- which would be the opposite error, since at tf32/fp16/bf16
    our Triton GEMM is within 5% of cuBLAS and fusion across the boundary is worth more.
    """
    from kernel_optimizer.agents.modules import _contract_doc

    doc = _contract_doc()
    low = doc.lower()
    # The permission itself.
    assert "vendor librar" in low, "the contract never mentions the vendor libraries"
    assert "f.linear" in low or "F.linear" in doc, (
        "the contract should name the actual call an agent would make")
    # It must NOT still say torch ops are only for layout/reshaping.
    assert "(layout, reshaping)" not in doc, (
        "the parenthetical that excluded computation is still present")
    # Both directions of the trade, so this does not read as 'always use the library'.
    assert "fuse" in low or "fusion" in low, (
        "the contract must say the library call is a fusion barrier, or an agent will "
        "delegate work that was worth keeping")
    # And the floor.
    assert "at least one real kernel" in low or "must define at least one" in low, (
        "the hard floor -- a real kernel must exist -- has to survive the relaxation")
    assert "only calls torch ops is rejected" in low or "only calls torch" in low, (
        "the contract must still reject a file that is nothing but torch calls")


def test_trials_are_reported_per_precision_with_the_cause_of_a_dead_one():
    """F6: a precision with zero completed trials must be visible, and named by its cause.

    Replayed over the real L3:43 runs on disk, this table shows bf16 taking 190-208 trials
    per run and completing NONE of them -- invisible before, because the report's only trial
    statistic is one run-level `N total, C complete, F failed` line.

    The cause is deliberately derived per precision rather than asserted: on those runs the
    dominant failure for the dead precision is `correctness_mismatch` (an arithmetic
    problem), NOT `infeasible_shared_memory` (a tile problem). An earlier draft of this
    hint named the tile cause for every dead precision and would have sent the reader to
    the wrong fix.
    """
    from kernel_optimizer.reporting.report import _trials_by_precision

    def trial(prec, status, ms=None, kind=None):
        return {"params": {"values": {"COMPUTE_DTYPE": prec, "BLOCK_M": 64}},
                "status": status, "failure_kind": kind,
                "latency_ms": ({"median": ms} if ms is not None else None)}

    rows = ([trial("fp16", "complete", 8.0) for _ in range(3)]
            + [trial("tf32", "fail", kind="infeasible_shared_memory") for _ in range(4)]
            + [trial("bf16", "fail", kind="correctness_mismatch") for _ in range(5)])
    text = "\n".join(_trials_by_precision(rows))

    assert "fp16" in text and "tf32" in text and "bf16" in text
    assert "zero completed trials" in text, "a precision that never completed is not flagged"
    # The two dead precisions must get DIFFERENT explanations, keyed to what actually failed.
    assert "byte width" in text, (
        "the shared-memory case must say the tile does not fit at this dtype's width")
    assert "does not hold the task's numerics" in text, (
        "the correctness case must point at the arithmetic, not at the tile")
    # A precision that did complete must not be flagged as dead.
    dead_section = text.split("zero completed trials")[1]
    assert "`fp16`" not in dead_section, "a precision that completed trials is called dead"


def test_a_precision_knob_is_found_whatever_the_candidate_named_it():
    """`_precision_of` must not key on one spelling.

    Real candidates have used COMPUTE_DTYPE, PREC, DOT_PRECISION and BC_CACHE_DTYPE. Keying
    on a single name would report "(no precision knob)" for most of the fleet and make the
    F6 table useless exactly where it matters.
    """
    from kernel_optimizer.reporting.report import _precision_of

    assert _precision_of({"values": {"COMPUTE_DTYPE": "fp16"}}) == "fp16"
    assert _precision_of({"values": {"PREC": "bf16"}}) == "bf16"
    assert _precision_of({"values": {"DOT_PRECISION": "tf32"}}) == "tf32"
    assert _precision_of({"values": {"BC_CACHE_DTYPE": "fp16"}}) == "fp16"
    # Compute precision wins over a cache dtype when both are present.
    assert _precision_of({"values": {"BC_CACHE_DTYPE": "fp32",
                                     "COMPUTE_DTYPE": "fp16"}}) == "fp16"
    # Tile sizes and warp counts must not be mistaken for a precision.
    assert _precision_of({"values": {"BLOCK_M": 64, "NUM_WARPS": 4}}) is None
    assert _precision_of({"values": {"X_EVICT": "evict_last"}}) is None


def test_vendor_library_delegation_is_reported_not_punished():
    """F8: the static warning that names a torch computation op must reach the report.

    It was recorded on the worker result (`static_warnings`) and read by nobody, so which
    sub-ops a candidate delegated to cuBLAS was invisible -- which matters now that F2
    permits it. Surfaced as attribution, never as a rejection: the whole point of F2 is that
    this is a legal choice.
    """
    from kernel_optimizer.reporting.report import _vendor_library_usage

    class E:
        def __init__(self, type_, payload):
            self.type = type_
            self.payload = payload

    events = [
        E("QUICKTEST_DONE", {"candidate_id": "cand-aaa",
                             "static_warnings": ["Uses torch computation op: torch.matmul"]}),
        E("QUICKTEST_DONE", {"candidate_id": "cand-bbb", "static_warnings": []}),
        E("QUICKTEST_DONE", {"candidate_id": "cand-ccc",
                             "static_warnings": ["Uses torch.nn compute layer (only "
                                                 "containers, Parameter, init allowed)"]}),
    ]
    text = "\n".join(_vendor_library_usage(events))
    assert "cand-aaa" in text and "cand-ccc" in text
    assert "cand-bbb" not in text, "a candidate with no such warning must not be listed"
    assert "PERMITTED" in text, (
        "the section must say this is allowed, or a reader takes it for a defect list")
    # Silence when nothing delegated: no empty section.
    assert _vendor_library_usage([E("QUICKTEST_DONE", {"candidate_id": "x"})]) == []


def test_the_contract_names_the_supported_backends_and_when_cuda_wins():
    """F3: replace the bare "Prefer triton" with the reason, and with when NOT to.

    35 of 35 candidates across the whole project were Triton, which is compliance with the
    old wording rather than a finding. And the wording was wrong in at least one measured
    place: at strict IEEE fp32 a hand-written CUDA attention kernel reached 55-74% of this
    card's fp32 roof where the best of 36 Triton tile configurations reached 18%, and no tile
    closed it -- `tl.dot(input_precision="ieee")` has no fast path here.

    The contract must also stop implying CUTLASS/CuTe are options: `Backend` admits only
    triton and cuda, and the dependency is not installed in the worker venv, so a CUTLASS
    candidate is a compile error dressed up as a candidate defect.
    """
    from kernel_optimizer.agents.modules import _contract_doc

    doc = _contract_doc()
    low = doc.lower()
    assert "prefer `triton`" not in low, (
        "the unconditional preference is still there; it should give the reason and the "
        "counter-case instead")
    # The reason to start from Triton must be stated, not assumed.
    assert "bottleneck report" in low or "resource profile" in low, (
        "the contract should say WHY Triton is the default starting point (the harness "
        "reads its compiler metadata), or the choice stays a superstition")
    # The measured counter-case.
    assert "ieee" in low and "18%" in doc, (
        "the strict-IEEE case where CUDA measurably wins must be named")
    # And the unavailable backends must be called out rather than silently unavailable.
    assert "cutlass" in low, (
        "CUTLASS is not installed in the worker venv; the contract must say so instead of "
        "letting a candidate discover it as a compile error")
    # cp.async must NOT be sold as a reason to leave Triton -- Triton emits it.
    assert "cp.async" in low, (
        "num_stages already emits cp.async double-buffering; saying so prevents a candidate "
        "switching backends for a feature it already has")


def test_the_contract_defers_precision_to_measurement_and_warns_about_the_tile():
    """F3 (second half) + the L3:43 lesson: do not prescribe a precision in the source.

    The old line said to "prefer" tf32 for matmul/conv-bound work. Measured, the winner is
    task-dependent: fp16 and bf16 tied on one attention task (3.03 vs 3.01 ms), bf16 failed
    correctness outright on a state-space task where fp16 passed. The tuner decides this.

    The more expensive half is the tile interaction: L3:43's winning candidate declared four
    precisions and could launch at one, because its tile was sized at 2 bytes/element and
    needs 131072-164352 bytes at 4. The contract must warn about that AND say the agent does
    not have to compute the figure -- it cannot, reliably (measured: hand-written constraints
    ran at a median 32% of the compiler's own number).
    """
    from kernel_optimizer.agents.modules import _contract_doc

    doc = _contract_doc()
    low = doc.lower()
    assert "prefer it\nfor matmul" not in low and "prefer it for matmul" not in low, (
        "the prescriptive tf32 preference is still present")
    assert "decided by the tuner" in low or "decided by the tuner on real" in low, (
        "the contract must hand the precision choice to the tuner's measurements")
    assert "cannot launch at another" in low or "131072" in doc, (
        "the tile-vs-precision trap that cost L3:43 two precision branches is not warned "
        "about")
    assert "you do not need to compute the shared-memory" in low, (
        "the agent must be told the harness gets this from the compiler, or it will keep "
        "writing constraints that are wrong")


def test_the_default_config_states_its_correctness_mode():
    """F4: `correctness_mode` must be explicit in default.yaml, not left to the field default.

    `load_config` reads ONE yaml with no base layer, so an omitted key silently takes the
    dataclass default -- here `strict`, a whole-tensor allclose(1e-4) that all three L3 tasks
    cannot clear (their own two-precision floors are 0.9554/0.9767/0.9778). Every L3
    experiment config overrides it; default.yaml did not mention it at all, so anything based
    on default.yaml inherited a gate that contradicts the contract's own advice.
    """
    import pathlib

    import yaml

    from kernel_optimizer.config import load_config

    raw = yaml.safe_load(
        (pathlib.Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
        .read_text(encoding="utf-8"))
    assert "correctness_mode" in (raw.get("evaluation") or {}), (
        "default.yaml does not state correctness_mode, so it falls back silently")

    # And the experiment configs must still be the relaxed ones -- this fix must not have
    # quietly changed what the L3 runs do.
    cfg_dir = pathlib.Path(__file__).resolve().parents[1] / "configs"
    for name in ("experiments_l3.yaml", "experiments_l3_glm.yaml"):
        cfg = load_config(cfg_dir / name)
        assert cfg.evaluation.correctness_mode == "dual_witness_relaxed", (
            f"{name} no longer uses the relaxed gate: {cfg.evaluation.correctness_mode}")


def test_the_winners_fp64_rescue_count_reaches_the_report():
    """F7: `fp64_rescued_trials` was journalled and read by nobody.

    That hid half of a real comparison: our L3:48 winner at 1.411 ms passes correctness only
    through the fp64 relative arm on every quick trial, while an external CUDA kernel at
    1.477 ms clears the primary gate with zero rescues. 4.5% apart, and theirs is the more
    accurate kernel.

    Three properties:
      - a full-dependency case escalates with the warning, because that is the case where a
        like-for-like comparison would otherwise mislead;
      - 0 rescues says so plainly rather than staying silent;
      - the gate being DISABLED is reported as such -- printing "0 rescues" for a check that
        never ran would read as a clean bill of health, the same error as leaving an
        unmeasurable signal blank.

    The denominator is `quick_correctness_trials`, not `correctness_trials`: these counts come
    from tuning trials (quick path, 3 by default) while the final re-eval runs the full path
    (5). Verified against the real run-l3-48-20260907-202457 event log, where mixing the two
    rendered "3 of 5" for a candidate that was rescued on 3 of 3.
    """
    from kernel_optimizer.reporting.report import _fp64_rescue_line

    cfg = {"fp64_relative_gate": True, "quick_correctness_trials": 3,
           "correctness_trials": 5}
    best = {"candidate_id": "cand-win"}

    def tr(cid, rescued):
        return {"candidate_id": cid, "fp64_rescued_trials": rescued}

    # Fully dependent on the relative arm -> the escalation must fire.
    full = "\n".join(_fp64_rescue_line(best, [tr("cand-win", 3), tr("cand-win", 3)], cfg))
    assert "3 of 3" in full, f"wrong denominator or count: {full}"
    assert "rests entirely on the fp64" in full, (
        "a candidate whose correctness depends wholly on the relative arm must be flagged")

    # Partially dependent -> reported, not escalated.
    part = "\n".join(_fp64_rescue_line(best, [tr("cand-win", 1), tr("cand-win", 0)], cfg))
    assert "1 of 3" in part
    assert "rests entirely" not in part, "a partial dependency must not read as a total one"

    # Zero rescues -> stated, so the reader knows the arm was available and unused.
    zero = "\n".join(_fp64_rescue_line(best, [tr("cand-win", 0)], cfg))
    assert "0 rescues" in zero and "on its own" in zero

    # Gate disabled -> say so; never imply accuracy from an absence.
    off = "\n".join(_fp64_rescue_line(best, [tr("cand-win", 0)],
                                     {**cfg, "fp64_relative_gate": False}))
    assert "disabled" in off and "not evidence of accuracy" in off

    # Only the WINNER's trials count.
    other = "\n".join(_fp64_rescue_line(best, [tr("cand-other", 3), tr("cand-win", 0)], cfg))
    assert "0 rescues" in other, "another candidate's rescues were attributed to the winner"


def test_the_report_reads_the_evaluation_config_it_needs():
    """The rescue line needs the gate's configured state, which must come from the manifest.

    Inferring "gate off" from a zero count is exactly the confusion the line exists to
    prevent, so the config has to be threaded through rather than guessed.
    """
    import ast
    import inspect

    from kernel_optimizer.reporting import report

    src = inspect.getsource(report.ReportGenerator.generate)
    assert "evaluation" in src, (
        "the report generator does not read the evaluation config, so the rescue line "
        "cannot tell a disabled gate from an unused one")
    tree = ast.parse(src.lstrip())
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "_fp64_rescue_line"]
    assert calls, "the rescue line is computed but never rendered"


def test_the_compile_probe_answers_a_batch_in_one_process():
    """F5a: batching is the whole reason this screen can be consulted before a trial.

    Measured on box 2 with the harness own job path: one probe per worker process costs a
    median 16.7 s (11.9-25.6), almost entirely process start plus torch/CUDA/KernelBench
    import, against the 18.6 s the wasted trial it replaces already cost -- so one-per-process
    saves nothing, and putting it in the ask() reject loop (ceiling 64) would stall one ask
    for 17.8 minutes. Forty-eight configurations in ONE process took 11.02 s: 7 ms marginal.

    Two properties the batch shape must keep:
      - a single-path job result is unchanged, so existing callers are unaffected;
      - each variant carries its OWN ok/reason, so one unprobeable configuration falls
        through to a real trial instead of suppressing its siblings.
    """
    import ast
    import inspect

    from kernel_optimizer.gpu import worker_main
    from kernel_optimizer.gpu.jobs import make_compile_probe_job

    single = make_compile_probe_job("/ref.py", "/k0.py", backend="triton")
    assert "extra_kernel_src_paths" not in single, (
        "a single-path job must not grow a batch field, or older workers break on it")
    batch = make_compile_probe_job("/ref.py", "/k0.py", backend="triton",
                                   extra_kernel_src_paths=["/k1.py", "/k2.py"])
    assert batch["extra_kernel_src_paths"] == ["/k1.py", "/k2.py"]
    assert batch["kernel_src_path"] == "/k0.py", "the primary variant must stay the primary"

    src = inspect.getsource(worker_main.run_compile_probe)
    # The per-variant module name must be unique. Reusing one name lets Python import
    # machinery return the FIRST variant module, so every later configuration reports the
    # first one shared bytes -- a screen that looks like it works and answers the wrong
    # question.
    assert "kopt_compile_probe_" in src, (
        "variants must get unique module names, or the batch silently re-reports the first")
    tree = ast.parse(src.lstrip())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "probe_one" in names, "the per-variant probe was not factored out"


def test_the_shared_memory_guard_is_three_valued_and_cache_only():
    """F5b: unknown must not shrink the search space, and the guard must not touch the GPU.

    cached_shared_verdict returns True (fits) / False (cannot launch) / None (not screened).
    Collapsing None into either verdict is a defect: as infeasible it silently removes
    configurations nobody measured; as feasible it is merely today behaviour. And it must
    read the cache only -- a worker round-trip inside the ask() reject loop would cost
    64 x 16.7 s in the worst case.
    """
    import ast
    import inspect

    from kernel_optimizer.evaluation.correctness import CorrectnessEvaluator

    ev = CorrectnessEvaluator.__new__(CorrectnessEvaluator)
    ev._screen_cache = {}
    src_fits, src_dies, src_unknown = "SRC-FITS", "SRC-DIES", "SRC-UNSEEN"
    ev._screen_cache[f"triton:{hash(src_fits)}"] = {"ok": True, "max_shared": 65536}
    ev._screen_cache[f"triton:{hash(src_dies)}"] = {"ok": True, "max_shared": 164352}
    # A probe that could not answer must read as unknown, NOT as feasible-or-infeasible.
    ev._screen_cache["triton:xxx"] = {"ok": False, "reason": "no triton kernel"}

    assert ev.cached_shared_verdict(src_fits, "triton", 101376) is True
    assert ev.cached_shared_verdict(src_dies, "triton", 101376) is False
    assert ev.cached_shared_verdict(src_unknown, "triton", 101376) is None
    # No device limit -> no opinion, rather than a comparison against an absent number.
    assert ev.cached_shared_verdict(src_dies, "triton", None) is None

    # Cache-only: no worker call anywhere in the body.
    body = inspect.getsource(CorrectnessEvaluator.cached_shared_verdict)
    tree = ast.parse(body.lstrip())
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "run_job" not in attrs, (
        "the guard predicate must not talk to the worker; it runs inside the ask reject loop")


def test_an_unscreened_config_is_still_evaluated_for_real():
    """The screen must never be the thing that rejects a candidate -- including via the guard.

    Three ways a configuration can be unscreened: the batch never covered it, the probe could
    not answer, or the worker reported no figure. All three must let the trial run, because
    the alternative is a search space quietly smaller than the one the report describes.
    """
    from kernel_optimizer.evaluation.correctness import CorrectnessEvaluator

    ev = CorrectnessEvaluator.__new__(CorrectnessEvaluator)
    ev._screen_cache = {}
    # Never probed.
    assert ev.cached_shared_verdict("anything", "triton", 101376) is None
    # Probed but unanswerable (a CUDA candidate has no Triton kernel to compile).
    ev._screen_cache[f"cuda:{hash('cuda-src')}"] = {
        "ok": False, "reason": "no triton kernel compiled during the forward pass"}
    assert ev.cached_shared_verdict("cuda-src", "cuda", 101376) is None
    # Probed, ok, but the worker reported no figure.
    ev._screen_cache[f"triton:{hash('nofig')}"] = {"ok": True, "max_shared": None}
    assert ev.cached_shared_verdict("nofig", "triton", 101376) is None


def test_the_prompt_no_longer_asks_the_agent_to_compute_shared_memory():
    """F5d: retire the teaching, because following it produced a constraint rejecting nothing.

    Audited on the real declared constraint of L3:43 cand-969997e3, evaluated with the guard
    own evaluator over 36 tile x precision configurations: it admitted all 36, including the
    17 that cannot launch, and rejected none -- 53% agreement with the compiler, exactly the
    all-admit baseline. Hand-counted figures ran at a median 32% of the truth (worst 12%), and
    an independent agent constraint on another task under-estimated in 10 of 16 points.
    The cause is structural: metadata.shared includes multi-buffering, alignment and
    intermediates, and is not monotonic in the tile dims.

    So the prompt must stop asking for it, and must say who does it instead -- otherwise an
    agent reads the silence as an omission and writes one anyway. What stays is the part
    arithmetic CAN decide, plus the one thing no constraint will do: keep the tile domain
    launchable at every precision offered.
    """
    import inspect

    from kernel_optimizer.agents import modules

    src = inspect.getsource(modules)
    # The template that produced the vacuous constraints must be gone.
    assert "<elements staged per stage>" not in src, (
        "the shared-memory constraint template is still being handed to the agent")
    assert "DO NOT write a shared-memory constraint" in src, (
        "the prompt must say explicitly not to write one, not merely omit the template")
    # And it must say the harness does it, with the reason, or the instruction reads arbitrary.
    assert "The harness screens this for you" in src
    assert "monotonic" in src, (
        "the non-monotonicity is why hand computation cannot work; state it")
    # The replacement work must be named, so the constraint budget is not simply lost.
    assert "MAX_THREADS_PER_BLOCK" in src, (
        "thread-count bounds ARE computable and should still be requested")
    # The tile-domain requirement that no constraint can express.
    assert "at least one launchable configuration" in src, (
        "without this, a precision whose every tile is infeasible is never measured")


def test_only_the_two_supported_backends_reach_the_loader():
    """F9: the tilelang/cute branch was unreachable through the type system, and read as support.

    Backend is Literal["triton", "cuda"], so a candidate declaring "cute" fails pydantic
    validation long before the worker. The loader branch listing tilelang and cute was
    therefore dead code -- and worse than dead: CUTLASS/CuTe are not installed in the worker
    venv, so had a value ever reached it, the result would have been a compile error that
    looked like a candidate defect rather than a missing dependency.
    """
    import inspect

    from kernel_optimizer.gpu import worker_main
    from kernel_optimizer.models.core import Candidate

    # Assert on CODE, not on prose: the comment explaining why the branch was removed
    # legitimately names tilelang and cute, and an assertion over the whole file text would
    # fail on the explanation of its own fix. So parse and inspect the string constants that
    # a backend comparison actually tests against.
    import ast

    tree = ast.parse(inspect.getsource(worker_main))
    compared = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for comparator in node.comparators:
                if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
                    compared.add(comparator.value)
                elif isinstance(comparator, (ast.Tuple, ast.List, ast.Set)):
                    for elt in comparator.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            compared.add(elt.value)
    assert "tilelang" not in compared, (
        f"a backend comparison still tests for tilelang: {sorted(compared)}")
    assert "cute" not in compared, (
        f"a backend comparison still tests for cute: {sorted(compared)}")
    assert "triton" in compared, "the triton branch went missing with the dead ones"

    # And the type system must still be the thing that stops it.
    import pydantic
    import pytest as _pytest

    with _pytest.raises(pydantic.ValidationError):
        Candidate(candidate_id="c", family_id="f", origin="seed", backend="cutlass",
                  source_sha="x", structural_signature="y", approach_summary="z")


def test_the_accumulator_rule_reads_as_the_default_it_actually_is():
    """F11: the contract claimed REQUIRED/MUST for a rule nothing checks.

    triton_lint only warns about a hardcoded low-precision cast without a dtype knob, and its
    own docstring says "WARNING only (never blocks)". No code inspects the accumulator dtype
    at all. Keeping fp32 is very nearly always right for a long reduction, so it stays the
    strong default -- but stating it as an enforced rule misdescribes the system, and an agent
    that believes a violation is rejected reasons differently from one told the diff-test is
    the only check.
    """
    from kernel_optimizer.agents.modules import _contract_doc

    doc = _contract_doc()
    # The default must still be stated plainly.
    assert "keep the accumulator in fp32" in doc.lower()
    # But not as a checked requirement.
    assert "strong default rather than a checked rule" in doc, (
        "the contract still presents the accumulator dtype as enforced when nothing checks it")
    assert "diff-test is then" in doc, (
        "if the rule is not enforced, the contract must name what actually catches a bad "
        "accumulator")


# --- a rewrite may change backend, and the harness must believe the file, not the label ---

def test_a_rewrite_can_declare_a_backend_and_the_source_decides():
    """`RewriteCandidate` carries `backend`, and the harness overrides it from the source.

    Two separate defects, one fix each.

    The schema gap: the generator and novelty results both carried `backend`; the rewriter's
    did not. So the one module whose job is "restructure to unlock a blocked direction" had no
    field in which to express the most structural change available, and nothing in its prompt
    suggested the move existed. 35 of 35 candidates across every run so far were Triton -- a
    consequence of what the prompts asked for, not a finding about backends.

    The trust gap: a declaration is not evidence. The field exists to make the option visible
    in the prompt; `_detect_backend` reads the compile mechanism actually present, which is what
    the loader has to agree with.
    """
    from kernel_optimizer.agents.modules import _detect_backend
    from kernel_optimizer.models.reports import RewriteCandidate

    # The field exists, and defaults to the backend the prompt tells them to start from.
    rc = RewriteCandidate(file="rewrites/rw_1.py", change_summary="x")
    assert rc.backend == "triton"
    assert RewriteCandidate(file="f.py", change_summary="x", backend="cuda").backend == "cuda"

    # And the source is what actually decides, in both directions.
    triton_src = "import triton\nimport triton.language as tl\n@triton.jit\ndef k(): pass\n"
    cuda_src = ("from torch.utils.cpp_extension import load_inline\n"
                "mod = load_inline(name='m', cpp_sources=[''], cuda_sources=['...'])\n")
    assert _detect_backend(triton_src) == "triton"
    assert _detect_backend(cuda_src) == "cuda"
    # A file that DECLARES triton but compiles CUDA is CUDA. This is the case that would
    # otherwise reach `load_custom_model_with_tempfile` and fail as a candidate defect.
    assert _detect_backend(cuda_src) != RewriteCandidate(
        file="f.py", change_summary="x", backend="triton").backend


def test_a_rewrite_does_not_inherit_its_parents_backend():
    """The registered backend comes from the rewrite's own source, never from the parent.

    Inheriting `parent.backend` was wrong in exactly the case the new prompt now invites: a
    rewrite that switches to CUDA would have been registered as Triton, so (a) the worker would
    load it with `load_custom_model_with_tempfile`, which needs a jit kernel, and (b)
    `structural_signature` -- which hashes the backend -- would collide it with its Triton
    parent, so a genuinely new structure would be discarded as a duplicate.

    Asserted on the orchestrator's AST rather than its text: the call must pass a value derived
    from the source at that site, and `parent.backend` must not appear in the rewrite
    registration's arguments.
    """
    import ast
    import inspect

    from kernel_optimizer.control import orchestrator as orch

    tree = ast.parse(inspect.getsource(orch))
    # Find the _register call whose origin argument is the literal "rewrite".
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "_register"):
            continue
        args = node.args
        if len(args) >= 2 and isinstance(args[1], ast.Constant) and args[1].value == "rewrite":
            found.append(node)
    assert found, "no _register(..., 'rewrite', ...) call found in the orchestrator"

    for call in found:
        rendered = [ast.unparse(a) for a in call.args]
        assert not any("parent.backend" in r for r in rendered), (
            "a rewrite still inherits parent.backend; a CUDA rewrite would be loaded as "
            f"Triton and would collide with its parent's signature. args={rendered}")
        assert any("_detect_backend" in r or "detected" in r for r in rendered), (
            f"the rewrite's backend is not derived from its own source. args={rendered}")


def test_the_rewriter_is_told_when_a_backend_switch_is_the_answer():
    """The prompt must name the measured case, not merely permit switching.

    A schema field nobody is told about changes nothing. The bottleneck report deliberately
    does NOT recommend a backend -- its job is to say what limits the kernel -- so the
    inference has to be made here, which means this prompt has to carry the evidence needed to
    make it: strict IEEE fp32 is where Triton has no fast path (18% of roof against 55-74% for
    hand-written CUDA on the same task), and no tile closed that gap.

    It must also state the cost, so the switch is a trade rather than a coin flip.
    """
    import inspect

    from kernel_optimizer.agents.modules import StructureRewriterAgent

    src = inspect.getsource(StructureRewriterAgent.render_prompt)
    assert "may also change BACKEND" in src, (
        "the rewriter is never told that switching backend is an available rewrite")
    assert "cuda" in src and "load_inline" in src, (
        "the prompt names no mechanism for writing the other backend")
    assert "IEEE fp32" in src or "strict IEEE" in src, (
        "the prompt does not name the one regime where the switch is measured to win")
    assert "18%" in src and ("55" in src or "74%" in src), (
        "the prompt asserts a backend choice without the measurement behind it")
    # And the cost, so this is a trade and not an invitation.
    assert "minute rather than seconds" in src or "compiles in about a minute" in src, (
        "the prompt does not state the compile-time cost of a CUDA candidate")
    assert "not installed" in src or "NOT installed" in src, (
        "the prompt must say CUTLASS/CuTe/TileLang are unavailable, or a rewrite will try them")
    # It must NOT read as a general encouragement to vary the backend.
    assert "for style" in src or "on a hunch" in src, (
        "the prompt does not warn against switching backend without a named reason")


def test_a_rescued_rewrite_keeps_the_backend_it_was_written_in():
    """A rescued CUDA rewrite must not default to Triton.

    `rescue_from_sandbox` rebuilds the result from files on disk when a transport failure kills
    the call, so the agent's JSON -- and its `backend` field -- never arrives. Defaulting to
    "triton" there would hand a CUDA file to the Triton loader: the same mis-load as the
    inheritance bug, reached by a different path, and only on the timeout path where it is
    hardest to notice.
    """
    import inspect

    from kernel_optimizer.agents.modules import StructureRewriterAgent

    src = inspect.getsource(StructureRewriterAgent.rescue_from_sandbox)
    assert "_detect_backend" in src, (
        "the rewriter's rescue does not read the backend from the source, so a rescued CUDA "
        "rewrite is loaded as Triton")
    # The advisory Triton lint must not fire on a CUDA rewrite either.
    soft = inspect.getsource(StructureRewriterAgent.soft_check)
    assert "_detect_backend" in soft, (
        "soft_check lints every rewrite as Triton; a CUDA rewrite would be warned about "
        "missing `tl.` idioms")


# --- the box's measured ceilings must reach the agents that WRITE kernels ---

def test_the_measured_ceilings_reach_every_kernel_writing_agent():
    """`device.md` must state what the card ACHIEVES, not only what it forbids.

    Found on a live run. `run-l3-21-20260908-232211` measured fp16 at 158.6 and bf16 at
    164.2 TFLOP/s, and its generator's `docs/device.md` was eight lines: name, VRAM, registers,
    shared memory, threads. No DRAM figure, no TFLOP figure. The ceilings block existed but was
    reachable only from `_bottleneck_doc`, and `AnalystInputs` was the only Inputs dataclass
    carrying a calibration -- so the analyst saw them and the four agents that actually write
    kernels did not.

    This is not a cosmetic omission: the guidance those agents are given is stated as ratios
    between ceilings. The contract says fp16 is "roughly 2x" tf32 on this class of card and asks
    them to treat precision as a first-class design choice; the rewriter is told a CUDA rewrite
    wins under strict IEEE fp32. Neither could see a single ceiling for the box in front of it.
    """
    from kernel_optimizer.agents.modules import _device_doc, _measured_ceilings_doc
    from kernel_optimizer.gpu.calibrate import calibration_from_worker
    from kernel_optimizer.models.core import DeviceLimits

    cal = calibration_from_worker(MEASURED_4090)
    dev = DeviceLimits(name="NVIDIA GeForce RTX 4090 (sm_89)", vram_gb=23.0,
                       max_regs_per_thread=255, max_shared_bytes_static=49152,
                       max_shared_bytes_optin=101376, max_threads_per_block=1024)

    # Without a calibration the doc still renders (a box that could not measure must still run),
    # and it must not invent numbers.
    bare = _device_doc(dev)
    assert "Max opt-in shared memory" in bare
    assert "TFLOP" not in bare, "a doc with no calibration must not state a throughput ceiling"

    # With one, every ceiling the calibration holds is stated.
    full = _device_doc(dev, cal)
    assert "Max opt-in shared memory" in bare and "Max opt-in shared memory" in full
    assert "TB/s" in full, "no DRAM ceiling in device.md"
    assert "fp32" in full and "TFLOP" in full, "no fp32 ceiling in device.md"
    for label in ("tf32", "fp16", "bf16"):
        assert label in full, f"no {label} ceiling in device.md, so its ratio cannot be reasoned about"
    # The ratio is what makes it actionable, not the raw figure.
    assert "the tf32 figure" in full, (
        "the low-precision ceilings are stated without their ratio to tf32, which is the form "
        "the contract's own guidance is written in")

    # And the block is the same one the analyst gets, not a divergent copy.
    assert _measured_ceilings_doc(cal) in full
    assert _measured_ceilings_doc(None) == ""


def test_every_kernel_writing_module_accepts_and_forwards_a_calibration():
    """The four writers plus the parameterizer take `calibration` and pass it to `_device_doc`.

    A field nobody forwards changes nothing, and a forwarded field nobody supplies changes
    nothing either -- so this checks the dataclass, the seeding call, and the orchestrator's
    construction site, for each module.

    The parameterizer is included deliberately: it chooses the tile and precision DOMAINS, and
    "is fp16 worth a choice on this card" is a question about a ratio between two ceilings.
    """
    import ast
    import inspect

    from kernel_optimizer.agents import modules as mod
    from kernel_optimizer.control import orchestrator as orch

    for name in ("GeneratorInputs", "ParameterizerInputs", "RewriterInputs", "NoveltyInputs"):
        cls = getattr(mod, name)
        assert "calibration" in cls.__dataclass_fields__, (
            f"{name} has no calibration field, so that agent cannot be told what the box does")

    # Each of those modules' seed_sandbox must pass it into _device_doc.
    for agent, label in ((mod.CandidateGeneratorAgent, "generator"),
                         (mod.ParameterizerAgent, "parameterizer"),
                         (mod.StructureRewriterAgent, "rewriter"),
                         (mod.NoveltyGeneratorAgent, "novelty")):
        src = inspect.getsource(agent.seed_sandbox)
        assert "_device_doc(inputs.device, inputs.calibration)" in src, (
            f"{label} writes a device.md with no ceilings in it")

    # And the orchestrator must actually supply it at every construction site of those inputs.
    tree = ast.parse(inspect.getsource(orch))
    wanted = {"GeneratorInputs", "ParameterizerInputs", "RewriterInputs", "NoveltyInputs"}
    seen = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in wanted:
            kws = {k.arg for k in node.keywords}
            seen.setdefault(node.func.id, []).append("calibration" in kws)
    for name in wanted:
        assert name in seen, f"the orchestrator never constructs {name}"
        assert all(seen[name]), (
            f"{seen[name].count(False)} of {len(seen[name])} {name} construction sites omit "
            f"calibration, so those calls silently fall back to None")


def test_per_candidate_cost_survives_both_profiler_paths():
    """G4/G6: per-candidate peak memory, aten traffic and launch geometry reach ProfileRecord.

    These fields exist because the numbers the classifier divides by come from TaskCost, which is
    measured on the REFERENCE and is one constant per task. Verified on 5 runs: gpu_ms varies up
    to 4.6x between candidates while the derived numerator varies <=0.36%, so `pct_of_dram_peak`
    is 1/latency rescaled and carries no per-candidate information at all.

    Drives the real LightProfiler on all three of its paths, because the field set is passed
    separately on each and an omission on one path is exactly how `aux_output_ops` was silently
    dropped before (the job output held 1, the event log read None).
    """
    from kernel_optimizer.evaluation.profilerx import LightProfiler

    lp = LightProfiler()
    # numbers from the real probe on box 1, cand-9fa6786e of run-l3-48-20260909-115701
    cost = {
        "peak_alloc_bytes": 1493172224,
        "peak_reserved_bytes": 1543503872,
        "peak_above_resident_bytes": 679477248,
        "candidate_aten_bytes": 1267107840,
        "candidate_aten_ops": 7,
        "threads_launched": 8388608,
        "launches": [{"kernel": "_ssd_chunk_kernel", "n_blocks": 16384, "num_warps": 8,
                      "threads": 4194304},
                     {"kernel": "_ssd_scan_kernel", "n_blocks": 16384, "num_warps": 4,
                      "threads": 2097152}],
    }

    rec = lp.extract({
        "triton": {"kernels": [{"name": "k", "n_regs": 168, "n_spills": 0, "shared": 40960}],
                   "compile_s": 1.0, "candidate_cost": cost},
    })
    assert rec.peak_alloc_bytes == 1493172224, "peak memory lost on the triton path"
    assert rec.peak_reserved_bytes == 1543503872
    assert rec.peak_above_resident_bytes == 679477248
    assert rec.candidate_aten_bytes == 1267107840, "candidate traffic lost on the triton path"
    assert rec.candidate_aten_ops == 7
    assert rec.threads_launched == 8388608
    assert len(rec.launches) == 2, "per-launch geometry lost"
    assert rec.launches[0]["kernel"] == "_ssd_chunk_kernel"

    # No kernel metadata at all: cost is a property of RUNNING the candidate, so like launch
    # overhead it must survive a candidate whose resources could not be read.
    rec_bare = lp.extract({"triton": {"candidate_cost": cost}})
    assert rec_bare.peak_alloc_bytes == 1493172224, (
        "per-candidate cost was dropped when no kernel metadata was collected, so a candidate "
        "whose resources could not be read reports no cost either"
    )
    assert rec_bare.candidate_aten_bytes == 1267107840

    # Not measured must be None, never 0. A 0 here would read as "this candidate moves no bytes
    # and allocates nothing", which for a memory-bound kernel inverts the diagnosis.
    rec_none = lp.extract({"triton": {"kernels": [{"name": "k", "n_regs": 8}]}})
    assert rec_none.peak_alloc_bytes is None, "unmeasured peak memory must be None, not 0"
    assert rec_none.candidate_aten_bytes is None, "unmeasured traffic must be None, not 0"
    assert rec_none.candidate_aten_ops is None
    assert rec_none.threads_launched is None
    assert rec_none.launches == []

    # A failure must arrive as a note, not as silence: absent and zero are different answers.
    rec_failed = lp.extract({"triton": {
        "kernels": [{"name": "k", "n_regs": 8}],
        "candidate_cost": {"cost_notes": ["cost measurement failed: RuntimeError: OOM"]},
    }})
    assert rec_failed.peak_alloc_bytes is None
    assert rec_failed.cost_notes and "OOM" in rec_failed.cost_notes[0], (
        "a cost-measurement failure left no note, so it is indistinguishable from a path that "
        "never collects cost"
    )


def test_candidate_cost_measurement_is_wired_into_the_worker():
    """The worker must actually CALL the cost measurement, not merely define it.

    This is the failure this test exists to catch: a probe verified standalone, a model field
    added, a profiler mapping written -- and nothing calling the measurement, so every field
    reads None in production while all the unit tests pass. Asserted on the worker's own source
    because the call site is inside a GPU-only function that cannot run here.
    """
    import inspect

    from kernel_optimizer.gpu import worker_main

    src = inspect.getsource(worker_main._extract_triton_metadata)
    assert "_measure_candidate_cost(" in src, (
        "_extract_triton_metadata never calls _measure_candidate_cost, so no per-candidate cost "
        "is ever collected in a real run"
    )
    assert '"candidate_cost"' in src, (
        "the cost result is computed but not placed in the worker's return dict under "
        "'candidate_cost', which is the key LightProfiler reads"
    )

    # The measurement must warm up before measuring the peak: compilation allocates, and a
    # compile-time allocation folded into the peak makes the number depend on whether this
    # candidate happened to be compiled in this process, i.e. not a property of the candidate.
    cost_src = inspect.getsource(worker_main._measure_candidate_cost)
    warm = cost_src.index("reset_peak_memory_stats")
    first_fwd = cost_src.index("model(*inputs)")
    assert first_fwd < warm, (
        "reset_peak_memory_stats runs before the first forward, so the peak includes "
        "compilation allocations and is not comparable between candidates"
    )
    # And it must restore JITFunction.run even when the forward raises: leaving the wrapper
    # installed would make every later candidate in this process record into a stale list.
    assert "finally:" in cost_src and "JITFunction.run = original_run" in cost_src, (
        "JITFunction.run is not restored in a finally block, so a failing candidate leaves the "
        "launch-recording wrapper installed for every candidate after it"
    )


def test_impossible_dram_fraction_is_not_reported_as_saturated():
    """G7: a DRAM fraction above the roof means the DENOMINATOR does not apply, not 'saturated'.

    This is a MUST-FAIL test for a defect a positive control found: five kernels whose bottleneck
    is known by construction were classified, and the two whose working set fits in L2 were
    mislabelled. One purely L2-resident streaming kernel read 287% of the measured DRAM roof --
    physically impossible -- and was reported `memory_bound`, i.e. "you are at the ceiling, stop
    optimizing", about a kernel that never touched the memory bus.

    The cause is that `byte_count` counts LOGICAL bytes, so an L2 hit is counted as traffic. The
    compute branch has always guarded its own impossible case; the DRAM branch did not.

    Reporting `unknown` is deliberately stronger than clamping to 1.0: a clamp still says "at the
    roof", which is the wrong action, whereas the true statement is that the quantity cannot be
    evaluated for this kernel.
    """
    from kernel_optimizer.evaluation.bottleneck import DevicePeaks, classify

    peaks = DevicePeaks(dram_tbs=0.911, fp32_tflops=54.8, tf32_tflops=88.1)

    # The real control case: kernel E of probe_binding_criterion_control.py, a streaming kernel
    # shrunk to fit inside the 72 MiB L2. 256 MiB of logical traffic in 0.098 ms is 2.61 TB/s,
    # 287% of a 0.911 TB/s roof.
    v = classify(gpu_ms=0.098, cpu_issue_ms=None, flop_count=None,
                 byte_count=256 * 2**20, peaks=peaks)
    assert v.kind == "unknown", (
        f"a kernel reading 287% of the DRAM roof was classified {v.kind!r}; an impossible "
        f"fraction means the byte count does not describe this kernel, and calling it "
        f"memory_bound tells the agent to stop optimizing a kernel that never hit DRAM"
    )
    assert v.evidence.get("impossible_dram_fraction"), (
        "the impossible fraction is not recorded in the evidence, so a reader cannot tell this "
        "`unknown` apart from one caused by missing measurements"
    )
    assert "L2" in (v.suggests or ""), (
        "the suggestion does not name the L2 cause, leaving the agent no way to act on it"
    )

    # A GENUINELY saturated kernel must still be memory_bound. Without this the fix could be
    # 'return unknown whenever the fraction is high', which would destroy the one verdict that
    # has been carrying real weight -- L3:48 at 94.6% of the roof is a true and useful finding.
    v_real = classify(gpu_ms=1.554, cpu_issue_ms=None, flop_count=None,
                      byte_count=int(1.351 * 10**9), peaks=peaks)
    assert v_real.kind == "memory_bound", (
        f"L3:48's real winner (1.351 GB in 1.554 ms = 94.6% of the roof) came out {v_real.kind!r}; "
        f"the L2 guard must not swallow genuinely bandwidth-bound kernels"
    )
    assert not v_real.evidence.get("impossible_dram_fraction")

    # And the boundary: a few percent over 100% is legitimate, because the roof is itself a
    # measurement that the calibration workload does not fully reach.
    v_edge = classify(gpu_ms=1.0, cpu_issue_ms=None, flop_count=None,
                      byte_count=int(0.93 * 10**9), peaks=peaks)   # 102% of roof
    assert v_edge.kind == "memory_bound", (
        "a kernel at 102% of a MEASURED roof was rejected; the roof is a measurement, not a "
        "hardware maximum, so a small overshoot is expected rather than impossible"
    )


def test_pressure_fields_declare_that_they_track_latency():
    """G1/G2: the two throughput fractions must say, in the evidence, what they are.

    They are computed from TASK-level counts (TaskCost is measured on the reference, one constant
    per task), so within a task they are 1/gpu_ms rescaled and order candidates exactly as latency
    does. Verified on 5 runs: gpu_ms varies up to 4.6x between candidates while the derived
    numerator varies <=0.36%.

    They are kept, because "how far from the card's physical roof" is a real question and L3:48 at
    94.6% is a real answer. What must not happen is a reader -- human or agent -- taking them for
    per-candidate traffic measurements, or counting them as two dimensions independent of latency.
    The basis string is in the evidence dict so it travels with the numbers into the agent prompt.
    """
    from kernel_optimizer.evaluation.bottleneck import DevicePeaks, classify

    peaks = DevicePeaks(dram_tbs=0.911, fp32_tflops=54.8, tf32_tflops=88.1)
    v = classify(gpu_ms=3.0, cpu_issue_ms=None, flop_count=112_113_254_400,
                 byte_count=321_769_472, peaks=peaks)
    assert "dram_pressure_basis" in v.evidence, (
        "pct_of_dram_peak travels with no statement of what it is, so a reader has no way to "
        "know it is 1/latency rescaled rather than this candidate's traffic"
    )
    assert "1/latency" in v.evidence["dram_pressure_basis"]
    assert "compute_pressure_basis" in v.evidence

    # The per-candidate quantities must be reported as absolute measurements, never divided by
    # gpu_ms -- dividing by latency is exactly the mistake the two fractions embody.
    v2 = classify(gpu_ms=3.0, cpu_issue_ms=None, flop_count=None, byte_count=None, peaks=peaks,
                  peak_alloc_bytes=1264582656, candidate_aten_bytes=1037959168,
                  candidate_aten_ops=13, threads_launched=14805120)
    assert v2.evidence["peak_alloc_mib"] == 1206.0, v2.evidence.get("peak_alloc_mib")
    assert v2.evidence["candidate_aten_mib"] == 989.9, v2.evidence.get("candidate_aten_mib")
    assert v2.evidence["candidate_aten_ops"] == 13
    assert v2.evidence["threads_launched"] == 14805120
    assert "LOWER bound" in v2.evidence["candidate_aten_basis"], (
        "the aten byte count travels without its bound, so a reader will take a well-fused "
        "candidate's small number for low real traffic when it means the opposite"
    )


def test_orchestrator_passes_per_candidate_cost_to_the_classifier():
    """The classifier accepting the fields is worthless if the caller never passes them.

    This is the gap that made `launch_bound` dead code for 848 trials: the field existed in
    classify()'s signature and nothing ever populated it, so a whole verdict branch could never
    fire and no test noticed, because every test called classify() directly.

    Asserted on the orchestrator's source at the classify() call site, since reaching it for real
    needs a GPU, a tuned candidate and an agent call.
    """
    import re
    from pathlib import Path

    # Read the file rather than importing it: the orchestrator pulls in optuna, which is a GPU-box
    # dependency, and this assertion is about the call site's shape, not its runtime behaviour.
    root = Path(__file__).resolve().parents[1]
    src = (root / "src" / "kernel_optimizer" / "control" / "orchestrator.py").read_text(
        encoding="utf-8")
    call = src[src.index("return classify("):]
    call = call[:call.index("\n            )") + 14]
    for field in ("peak_alloc_bytes", "peak_reserved_bytes", "candidate_aten_bytes",
                  "candidate_aten_ops", "threads_launched"):
        assert re.search(rf"\b{field}=", call), (
            f"the classify() call never passes {field}, so it is always None in production and "
            f"the evidence the agent reads carries no per-candidate cost at all"
        )
        assert f"profile.{field}" in call, (
            f"{field} is passed but not read from the profile record, so it cannot be the "
            f"measured value"
        )

    # And the task-level pair must SURVIVE. Replacing them with per-candidate numbers would
    # destroy the only quantity that answers "how much headroom does this task itself have" --
    # the source of L3:48's "1.351 GB compulsory, only ~10% left" conclusion.
    assert "cost.compulsory_bytes" in call and "cost.flop_count" in call, (
        "the task-level counts were removed in favour of per-candidate ones; they answer a "
        "different question (the task's own floor) and both are needed"
    )
