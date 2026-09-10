"""The witness fallback must step AWAY from the failure, not merely to a different config.

The fallback exists for one measured case: a task whose outputs exceed a dtype's range, where the
minimal witness (every knob's `choices[0]`, and the contract asks for the cheapest precision
first) cannot pass for any candidate. On level3/48 that rejected 9 of 9 candidates declaring a
precision knob while publishing 7 of 7 without one.

The defect these tests guard is subtler than "the fallback is missing". It was present, its
out-of-range predicate fired correctly, and it still failed -- because `itertools.product` varies
its LAST factor fastest, so both retries held `COMPUTE_DTYPE` at `fp16`, the dtype that had just
overflowed, and differed only in `NUM_STAGES`. Measured on the real level3/48 space: 2 of 2
retries reused the poisoned dtype. That is a fallback that consumes its budget confirming the
failure it was called to escape, and the candidate is then rejected with `witness_minimal_failed`
after repair had already fixed its algorithm.

These tests drive the real `SpaceValidator._next_witness` with a fake evaluator, because the
recorded failure mode here is a source-text assertion passing on broken code: "the code sorts by
distance" is satisfied by a sort that computes the wrong distance.
"""

from __future__ import annotations

import itertools

import pytest

from kernel_optimizer.config import EvalConfig
from kernel_optimizer.models.core import (
    Candidate,
    DeviceLimits,
    ParamDomain,
    ParameterSpace,
    ParamSet,
    TaskSpec,
)
from kernel_optimizer.paramspace.validation import SpaceValidator, _looks_out_of_range

# level3/48's real space for cand-d2cf7928, in its DECLARED order -- the order is the whole point,
# so a reordered copy would not be the same test.
_DOMAINS = [
    ("COMPUTE_DTYPE", "str", ["fp16", "bf16", "tf32", "ieee"]),
    ("DOT_MODE", "str", ["plain", "split3"]),
    ("BL", "int", [64, 128]),
    ("BN", "int", [128, 256]),
    ("BP", "int", [64, 128]),
    ("NUM_WARPS", "int", [2, 4, 8]),
    ("NUM_STAGES", "int", [1, 2, 3]),
]
_DEFAULT = {"COMPUTE_DTYPE": "ieee", "DOT_MODE": "plain", "BL": 64, "BN": 128, "BP": 64,
            "NUM_WARPS": 4, "NUM_STAGES": 1}
_MINIMAL = {"COMPUTE_DTYPE": "fp16", "DOT_MODE": "plain", "BL": 64, "BN": 128, "BP": 64,
            "NUM_WARPS": 2, "NUM_STAGES": 1}

# The VERBATIM tail from box 3's rejection on 2026-09-11, kept so the predicate cannot drift away
# from what the worker actually emits.
BOX3_REAL_TAIL = (
    "relaxed mismatch on trial 2; gate needs frac_within_tol>0.99 AND cosine>=0.99985\n"
    "  vs ieee ref: {'NON_FINITE_OUTPUT': '19403796 of 134217728 candidate values are not "
    "finite (16070970 NaN, 3332826 +/-Inf) -- THIS is the failure; the statistics below are "
    "computed over the finite values only', 'frac_within_tol': 0.836301, 'cosine': 0.99999995, "
    "'ref_absmax': '1.099e+11'}\n"
    "  reference's OWN ieee-vs-tf32 spread (task noise floor, NOT a bug): "
    "{'frac_within_tol': 0.979825, 'ref_absmax': '1.038e+22'}"
)


def _space() -> ParameterSpace:
    return ParameterSpace(
        space_id="sp-test", candidate_id="cand-test", version=1, source_sha="0" * 64,
        domains=[ParamDomain(name=n, kind=k, choices=c) for n, k, c in _DOMAINS],
        constraints=[],
    )


class _Recorder:
    """Records every config the fallback tests, and can be told which ones pass.

    `accept` decides pass/fail from the params, so a test can say "fp16 always overflows" without
    naming a config -- which is what makes the assertions about the SEARCH rather than about one
    hard-coded pair.
    """

    def __init__(self, accept=lambda vals: True):
        self.seen: list[dict] = []
        self._accept = accept

    def quick_test(self, task, kernel_src_path, tag, backend="triton"):
        import ast
        import re
        src = kernel_src_path.read_text(encoding="utf-8")
        # The real materializer pretty-prints PARAMS across multiple lines, so a single-line
        # regex silently matches nothing and every recorded config comes back {} -- which fails
        # with a KeyError instead of reporting what the fallback chose. Parse the real shape.
        m = re.search(r"^PARAMS = (\{.*?^\})", src, re.S | re.M)
        assert m, "the materialized witness has no parsable PARAMS block:\n%s" % src[:300]
        vals = ast.literal_eval(m.group(1))
        self.seen.append(vals)
        if self._accept(vals):
            # The real `latency_from_result` requires every key; a partial dict raises a KeyError
            # from inside the code under test, which reads as a fallback bug rather than a fake
            # worker's omission.
            return {"ok": True, "compiled": True, "correct": True,
                    "latency_ms": {"mean": 1.0, "std": 0.01, "min": 0.99, "max": 1.01,
                                   "n": 20, "median": 1.0}}
        return {"ok": False, "compiled": True, "correct": False,
                "failure_kind": "correctness_mismatch", "log_tail": BOX3_REAL_TAIL}


def _validator(rec: _Recorder, retries: int = 2) -> SpaceValidator:
    return SpaceValidator(
        correctness=rec,
        device=DeviceLimits(name="test", vram_gb=24,
                            max_regs_per_thread=255, max_shared_bytes_static=49152,
                            max_shared_bytes_optin=101376),
        eval_cfg=EvalConfig(), max_witness_retries=retries,
    )


def _source(vals: dict) -> str:
    return "PARAMS = %r\n\n\nclass ModelNew:\n    pass\n" % (vals,)


def _call(rec: _Recorder, retries: int = 2):
    v = _validator(rec, retries)
    space = _space()
    return v._next_witness(
        space,
        exclude=(ParamSet(values=_DEFAULT), ParamSet(values=_MINIMAL)),
        source=_source(_DEFAULT),
        work_dir=__import__("pathlib").Path(__import__("tempfile").mkdtemp()),
        task=TaskSpec(name="48", level=3, problem_id=48, ref_path="/nonexistent/r.py",
                      ref_src_sha="0" * 64),
        candidate=Candidate(candidate_id="cand-test", family_id="fam", origin="seed",
                            backend="triton", source_sha="0" * 64,
                            structural_signature="0" * 64, approach_summary="x"),
        witness_sources_default=_source(_DEFAULT),
    )


# --- the defect itself ------------------------------------------------------------------------


def test_the_retries_do_not_reuse_the_failed_precision():
    """THE regression. Both failed configs are `COMPUTE_DTYPE` fp16 (minimal) and ieee (default);
    a retry that comes back fp16 is re-testing the overflow.

    Asserted as "none of the retries reuses a failed value of the FIRST knob" rather than
    "the retry is bf16", so it does not encode which dtype happens to be second in the list.
    """
    rec = _Recorder(accept=lambda v: False)  # nothing passes: we want to see the whole walk
    _call(rec)
    assert rec.seen, "the fallback tested nothing at all"
    failed_precisions = {_MINIMAL["COMPUTE_DTYPE"], _DEFAULT["COMPUTE_DTYPE"]}
    reused = [v["COMPUTE_DTYPE"] for v in rec.seen if v["COMPUTE_DTYPE"] in failed_precisions]
    assert not reused, (
        "the fallback spent %d of %d retries on a precision that had already failed (%s) -- it "
        "exists to escape an out-of-range dtype and instead re-confirmed it"
        % (len(reused), len(rec.seen), reused))


def test_each_retry_differs_from_the_failures_in_more_than_one_knob():
    """The mechanism, stated without naming the precision.

    A config differing from a dead witness in exactly one knob is a neighbour of the failure, and
    on the measured case the single differing knob was `NUM_STAGES` -- irrelevant to an overflow.
    The fallback has two attempts; both must be genuine departures.
    """
    rec = _Recorder(accept=lambda v: False)
    _call(rec)
    for vals in rec.seen:
        d_min = sum(1 for k in _MINIMAL if vals.get(k) != _MINIMAL[k])
        d_def = sum(1 for k in _DEFAULT if vals.get(k) != _DEFAULT[k])
        assert min(d_min, d_def) > 1, (
            "retry %r differs from an already-failed witness in only %d knob(s); a neighbour of "
            "the failure is not a step away from it" % (vals, min(d_min, d_def)))


def test_it_finds_a_witness_when_only_the_failed_precision_is_bad():
    """End to end on the real shape: if every fp16 config overflows and everything else passes,
    the fallback must return a witness. Before the fix it returned None and the candidate was
    rejected -- with its algorithm already repaired."""
    rec = _Recorder(accept=lambda v: v["COMPUTE_DTYPE"] != "fp16")
    got = _call(rec)
    assert got is not None, (
        "no witness found although 3 of the 4 precisions pass; this is the rejection that cost "
        "9 of 9 candidates on one task")
    assert got.params.values["COMPUTE_DTYPE"] != "fp16"


# --- the guarantees the fix must not break ----------------------------------------------------


def test_the_retry_budget_is_still_respected():
    """Each retry is a real GPU quick test (30-60s on L3). A reordering that also removed the
    bound would trade a rejected candidate for an unbounded grid walk."""
    rec = _Recorder(accept=lambda v: False)
    _call(rec, retries=2)
    assert len(rec.seen) == 2, "expected exactly 2 attempts, saw %d" % len(rec.seen)
    rec1 = _Recorder(accept=lambda v: False)
    _call(rec1, retries=1)
    assert len(rec1.seen) == 1


def test_an_already_failed_config_is_never_retested():
    """The default and minimal configs are known dead. Re-running either wastes a GPU test and
    could return a stale pass."""
    rec = _Recorder(accept=lambda v: False)
    _call(rec)
    for vals in rec.seen:
        assert vals != _MINIMAL and vals != _DEFAULT, "retested an already-failed config %r" % vals


def test_the_walk_is_deterministic():
    """A run must be replayable from events.jsonl, so the same space must yield the same retries.
    A sort keyed only on distance would leave ties to the input order; this asserts stability."""
    a = _Recorder(accept=lambda v: False); _call(a)
    b = _Recorder(accept=lambda v: False); _call(b)
    assert a.seen == b.seen, "two identical calls chose different retries: %r vs %r" % (a.seen, b.seen)


def test_the_out_of_range_predicate_still_gates_the_fallback():
    """The fallback must stay narrowed to the overflow signature. An ordinary mismatch at the
    cheap corner is plausibly a real defect, and stepping past it would hide the evidence --
    trading a diagnosis problem for a lost-samples problem.
    """
    assert _looks_out_of_range({"log_tail": BOX3_REAL_TAIL}) is True
    assert _looks_out_of_range(
        {"log_tail": "relaxed mismatch; frac_within_tol 0.94; 'ref_absmax': '5.7e+00'"}) is False
    assert _looks_out_of_range({}) is False


def test_a_single_axis_space_degrades_rather_than_misbehaves():
    """When every knob is a tile size and there is no precision to escape, the distance ordering
    has nothing special to find. It must still return a usable witness rather than nothing."""
    rec = _Recorder(accept=lambda v: v["BM"] == 128)
    v = _validator(rec)
    space = ParameterSpace(
        space_id="sp2", candidate_id="c", version=1, source_sha="0" * 64,
        domains=[ParamDomain(name="BM", kind="int", choices=[16, 64, 128]),
                 ParamDomain(name="BN", kind="int", choices=[16, 64])],
        constraints=[])
    dead_a = {"BM": 16, "BN": 16}
    dead_b = {"BM": 64, "BN": 16}
    got = v._next_witness(
        space, exclude=(ParamSet(values=dead_a), ParamSet(values=dead_b)),
        source=_source(dead_a),
        work_dir=__import__("pathlib").Path(__import__("tempfile").mkdtemp()),
        task=TaskSpec(name="t", level=3, problem_id=1, ref_path="/nonexistent/r.py",
                      ref_src_sha="0" * 64),
        candidate=Candidate(candidate_id="c", family_id="f", origin="seed", backend="triton",
                            source_sha="0" * 64, structural_signature="0" * 64,
                            approach_summary="x"),
        witness_sources_default=_source(dead_a))
    assert got is not None and got.params.values["BM"] == 128


def test_no_task_or_dtype_is_named_in_the_ordering_code():
    """The project's standard: a fix naming one task, dtype, or knob would be the case-specific
    change this codebase forbids. The ordering must work from the failed configs alone.
    """
    import inspect
    src = inspect.getsource(SpaceValidator._next_witness)
    for banned in ("fp16", "bf16", "tf32", "COMPUTE_DTYPE", "level3", "l3:48", "48"):
        assert banned not in src.split('"""')[2] if '"""' in src else True, (
            "the ordering logic names %r, making it guidance about one case" % banned)
    # The docstring may (and should) cite the measurement; the CODE may not.
    body = src.split('"""')[2] if src.count('"""') >= 2 else src
    for banned in ("fp16", "bf16", "COMPUTE_DTYPE", "NUM_STAGES"):
        assert banned not in body, "the executable body names %r" % banned
