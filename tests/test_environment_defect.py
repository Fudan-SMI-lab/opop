"""An environment defect must not be reported as a bad kernel.

A missing dependency and a wrong kernel arrive through the same channel -- `failure_kind:
runtime_error` plus a traceback -- and they read identically to an operator and to a repair agent.
The measured cost of that confusion (G29): `kernelbench`'s package __init__ imports a chain reaching
dotenv -> openai -> litellm, and on a venv lacking them EVERY candidate came back `runtime_error`,
so a 6-call re-run produced 12 usable candidates and 0 correct ones. "0 correct" reads exactly like
"the model wrote bad kernels".

It then RECURRED, which is why this exists as code rather than as a package list. Box 1's config
pointed `wsl.venv` at an interpreter holding torch, triton and optuna but not kernelbench's import
chain; the run died at its eager baseline with the same ModuleNotFoundError on the same line.
Installing 25 packages on one box does not generalise to the next venv -- recognising the signature
does.

THE NARROWNESS IS THE DESIGN. `environment_defect` must fire on an import failure inside the
evaluation library's own chain and on nothing else. A candidate that imports something absent (an
agent reaching for CUTLASS or TileLang, neither installed) is a REAL case that must keep arriving as
the candidate's problem and keep getting its repair attempt. Widening the match to any
ModuleNotFoundError would silently reclassify those as box defects and stop the repair loop from
seeing them -- trading a diagnosis problem for a lost-samples problem, which is the worse trade.
"""

from __future__ import annotations

import pytest

from kernel_optimizer.evaluation.benchmark import environment_defect

# The VERBATIM tail from box 1's failure on 2026-09-11, kept as the positive control. A synthetic
# string would let the matcher drift away from what the worker actually produces.
BOX1_REAL_TAIL = """Traceback (most recent call last):
  File "/root/autodl-tmp/work/opop/src/kernel_optimizer/gpu/worker_main.py", line 2637, in main
    result = handler(job)
             ^^^^^^^^^^^^
  File "/root/autodl-tmp/work/opop/src/kernel_optimizer/gpu/worker_main.py", line 1771, in run_baseline
    from kernelbench.timing import measure_ref_program_time
  File "/root/autodl-tmp/opop-workspace/KernelBench/src/kernelbench/__init__.py", line 1, in <module>
    from . import utils  # triggers monkey-patch on torch.randn
    ^^^^^^^^^^^^^^^^^^^
  File "/root/autodl-tmp/opop-workspace/KernelBench/src/kernelbench/utils.py", line 6, in <module>
    from dotenv import load_dotenv
ModuleNotFoundError: No module named 'dotenv'
"""

# G29's original, from `run_static_check` rather than the baseline -- a different entry point into
# the same chain, so the matcher must not be anchored on the baseline frame.
G29_STATIC_CHECK_TAIL = """Traceback (most recent call last):
  File ".../gpu/worker_main.py", line 1722, in run_static_check
    from kernelbench.kernel_static_checker import validate_kernel_static
  File ".../KernelBench/src/kernelbench/__init__.py", line 1
    from . import utils   # triggers monkey-patch on torch.randn
  File ".../kernelbench/utils.py", line 22
    from openai import OpenAI
ModuleNotFoundError: No module named 'openai'
"""


def test_the_real_box1_failure_is_recognised():
    """The positive control, verbatim from the run that died."""
    out = environment_defect(BOX1_REAL_TAIL)
    assert out, "the failure that actually happened is not recognised"
    assert "ENVIRONMENT DEFECT" in out
    assert "dotenv" in out, "the message does not name the missing module, so it is not actionable"


def test_the_g29_static_check_failure_is_recognised_too():
    """Same chain, different entry point (static check, not baseline) and a different missing module
    further down it (openai, not dotenv). Anchoring on either would make this a one-case patch."""
    out = environment_defect(G29_STATIC_CHECK_TAIL)
    assert out, "G29's original signature is not recognised"
    assert "openai" in out


def test_the_message_names_wsl_venv_and_not_the_launching_interpreter():
    """The actionable part. The fix goes into the venv named by `wsl.venv`, which on some boxes is
    deliberately NOT the interpreter that launched the CLI -- box 2 runs the driver from orch-venv
    and the worker from kernel-opt-venv. A message saying only "install the package" sends an
    operator to the wrong interpreter, where the install changes nothing.
    """
    out = environment_defect(BOX1_REAL_TAIL)
    assert "wsl.venv" in out
    assert "NOT the interpreter that launched the CLI" in out
    assert "doctor" in out, "the message does not point at the check that covers this"


# --- the narrowness, which is the part that can cause harm if it drifts ----------------------


def test_a_candidate_importing_an_absent_module_is_NOT_an_environment_defect():
    """The counter-direction, and the one with a real cost if it breaks.

    An agent reaching for CUTLASS or TileLang (neither installed in the worker environment) is a
    MEASURED case -- backend choice has no evidence precisely because those were never available.
    Those failures must stay attributed to the candidate so the repair loop sees them. Classifying
    them as box defects would suppress real samples, which is worse than the diagnosis problem this
    function exists to fix.
    """
    for mod in ("cutlass", "tilelang", "cuda.bindings", "flash_attn"):
        tail = (
            "Traceback (most recent call last):\n"
            '  File "/root/runs/r1/sandboxes/c1/candidates/cand_1.py", line 3, in <module>\n'
            "    import %s\n"
            "ModuleNotFoundError: No module named '%s'\n" % (mod, mod)
        )
        assert environment_defect(tail) is None, (
            "a candidate's own missing import (%s) was reclassified as a box defect, which would "
            "stop the repair loop from seeing it" % mod)


def test_a_harness_module_import_failure_inside_the_worker_is_not_matched_either():
    """G33's case: the worker imports two harness modules for its measurements, and when
    `kernel_optimizer` is not importable those degrade to 0.0 rather than failing a kernel. That is
    a different defect with a different fix (extra_pythonpath), and it is NOT in kernelbench's
    chain, so this must not claim it -- a message pointing at the wrong fix is worse than none.
    """
    tail = ("Traceback (most recent call last):\n"
            '  File ".../gpu/worker_main.py", line 900, in _triton_ceilings\n'
            "    from kernel_optimizer.gpu.tritonmm import measure\n"
            "ModuleNotFoundError: No module named 'kernel_optimizer'\n")
    assert environment_defect(tail) is None


@pytest.mark.parametrize("tail", [
    pytest.param("", id="empty"),
    pytest.param("max abs diff 0.031 exceeds tolerance 0.01", id="absdiff"),
    pytest.param("correctness_mismatch: frac_within_tol 0.94 < 0.99", id="frac"),
    pytest.param("torch.OutOfMemoryError: CUDA out of memory", id="oom"),
    pytest.param("triton.runtime.errors.OutOfResources: out of resource: shared memory",
                 id="shared-oor"),
    pytest.param("RuntimeError: CUDA error: an illegal memory access was encountered", id="illegal"),
    # WITH a kernelbench frame, which is the realistic shape and the one the bare strings above
    # miss: kernelbench IS the evaluator, so a correctness or OOM failure genuinely passes through
    # its frames. A matcher keyed on the frame alone would fire on every one of these.
    pytest.param('Traceback (most recent call last):\n'
                 '  File ".../KernelBench/src/kernelbench/eval.py", line 402, in eval_kernel_against_ref\n'
                 "    raise RuntimeError('correctness mismatch')\n"
                 "RuntimeError: correctness mismatch\n",
                 id="correctness-through-kernelbench-frame"),
    pytest.param('Traceback (most recent call last):\n'
                 '  File ".../kernelbench/utils.py", line 88, in set_seed\n'
                 "    torch.cuda.manual_seed_all(seed)\n"
                 "torch.OutOfMemoryError: CUDA out of memory\n",
                 id="oom-through-kernelbench-utils-frame"),
])
def test_ordinary_failures_are_not_environment_defects(tail):
    """Every one of these is the candidate's or the task's business. A function that fires on them
    would attach a "fix your box" message to correctness and OOM failures, teaching an operator to
    distrust it -- and a warning nobody trusts is not a warning.

    The last two carry kernelbench frames on purpose: the earlier bare strings do not, so on their
    own they would pass against a matcher keyed on the frame alone. That gap was found by a
    revert-check variant, not by inspection. Explicit `id=`s because these tails contain newlines,
    and a node id built from the raw string is unusable in a revert-check's name list.
    """
    assert environment_defect(tail) is None


def test_an_import_error_outside_the_kernelbench_chain_is_not_matched():
    """A ModuleNotFoundError is necessary but NOT sufficient: the failing import has to be inside
    kernelbench's own __init__/utils chain. This is the assertion that keeps the match narrow, and
    the reason the function reads two independent signals rather than one."""
    tail = ("Traceback (most recent call last):\n"
            '  File "/root/x/some_other_lib/thing.py", line 5, in <module>\n'
            "    from dotenv import load_dotenv\n"
            "ModuleNotFoundError: No module named 'dotenv'\n")
    assert environment_defect(tail) is None, (
        "matched on the module name alone; the same module can legitimately be missing outside the "
        "evaluation library, where this message's fix does not apply")


# --- it has to reach both places a failure surfaces -----------------------------------------


def test_the_baseline_raise_carries_the_diagnosis():
    """Drives the real fatal path with a real worker result, rather than asserting on source text.

    The baseline failure is FATAL by design (no baseline means nothing to compare against), so the
    message it raises with is the only thing the operator gets -- and before this change it was a
    bare traceback that cost a whole run to interpret. A source-text assertion would pass on code
    that computed the diagnosis and dropped it, which is a defect this project has hit.
    """
    from kernel_optimizer.config import EvalConfig
    from kernel_optimizer.evaluation.benchmark import Benchmarker
    from kernel_optimizer.models.core import TaskSpec

    class _Worker:
        """Returns the box-1 failure for the eager/ieee job, which is the fatal one."""

        def run_job(self, job, timeout, tag, lock_mode=None):
            return {"ok": False, "compiled": False, "failure_kind": "runtime_error",
                    "log_tail": BOX1_REAL_TAIL}

    task = TaskSpec(name="43_MinGPTCausalAttention", level=3, problem_id=43,
                    ref_path="/nonexistent/ref.py", ref_src_sha="0" * 64)
    b = Benchmarker(_Worker(), evaluator=None, cfg=EvalConfig())

    with pytest.raises(RuntimeError) as exc:
        b.measure_baseline(task)

    msg = str(exc.value)
    assert "ENVIRONMENT DEFECT" in msg, (
        "the fatal baseline failure raises without the diagnosis, so the operator gets the bare "
        "traceback that cost a run to interpret")
    assert "dotenv" in msg and "wsl.venv" in msg


def test_the_baseline_raise_stays_bare_for_an_ordinary_failure():
    """The other direction on the same real path: a genuine kernel/task failure must NOT acquire a
    'fix your box' message. Without this, the test above is satisfied by unconditionally appending
    the text."""
    from kernel_optimizer.config import EvalConfig
    from kernel_optimizer.evaluation.benchmark import Benchmarker
    from kernel_optimizer.models.core import TaskSpec

    class _Worker:
        def run_job(self, job, timeout, tag, lock_mode=None):
            return {"ok": False, "failure_kind": "oom",
                    "log_tail": "torch.OutOfMemoryError: CUDA out of memory"}

    task = TaskSpec(name="t", level=3, problem_id=43, ref_path="/nonexistent/ref.py",
                    ref_src_sha="0" * 64)
    b = Benchmarker(_Worker(), evaluator=None, cfg=EvalConfig())
    with pytest.raises(RuntimeError) as exc:
        b.measure_baseline(task)
    assert "ENVIRONMENT DEFECT" not in str(exc.value)


def test_the_candidate_trial_path_carries_the_diagnosis():
    """The path that matters MORE, because it is where G29 cost 12 candidates and where a repair
    agent gets told 'your kernel crashed'. The baseline kills the run loudly; a per-trial failure is
    silent and gets attributed to the model.

    BEHAVIOURAL, driving the real `Orchestrator._run_trial`. It was a source-text assertion
    (`"environment_defect" in inspect.getsource(...)`) and the revert-check on the A800 proved that
    version was NOT EVIDENCE: with the whole diagnosis block replaced by `pass`, the test still
    passed, because `failure_detail=detail` remained on the TrialRecord below the deleted lines and
    the import remained at module top. Exactly the recorded failure mode -- a source-text assertion
    passing on broken code. Windows could not catch it: the test skipped there for lack of optuna,
    so the variant was reported UNVERIFIED rather than FAIL.
    """
    pytest.importorskip("optuna", reason="orchestrator import needs optuna")

    record = _run_one_trial(BOX1_REAL_TAIL)
    assert record.status == "fail"
    assert "ENVIRONMENT DEFECT" in record.failure_detail, (
        "a per-candidate failure caused by the BOX still reaches the repair agent as a kernel "
        "bug; this is where G29's 12 candidates were lost")
    assert "dotenv" in record.failure_detail, "the message does not name the missing module"
    assert "wsl.venv" in record.failure_detail, "the message does not name the venv to fix"
    # The original excerpt must survive alongside the diagnosis: replacing it would hide the
    # traceback the repair agent needs when the diagnosis turns out not to apply.
    assert "worker_main.py" in record.failure_detail, (
        "the diagnosis REPLACED the traceback excerpt instead of being appended to it")


def test_the_candidate_trial_path_stays_bare_for_an_ordinary_failure():
    """The counter-direction on the same real path. Without it, the test above is satisfied by
    appending the text unconditionally -- and then every correctness miss and OOM tells the
    operator to fix their box, which is how a warning stops being trusted."""
    pytest.importorskip("optuna", reason="orchestrator import needs optuna")

    tail = ("Traceback (most recent call last):\n"
            '  File "/root/runs/r1/sandboxes/c1/candidates/cand_1.py", line 3, in <module>\n'
            "    import cutlass\n"
            "ModuleNotFoundError: No module named 'cutlass'\n")
    record = _run_one_trial(tail)
    assert record.status == "fail"
    assert "ENVIRONMENT DEFECT" not in record.failure_detail, (
        "a candidate's own missing import was labelled a box defect, which would stop the repair "
        "loop from seeing a real failure")
    assert "cutlass" in record.failure_detail, "the actual error was dropped from the detail"


def _run_one_trial(log_tail: str):
    """Drive the real `Orchestrator._run_trial` with a worker that fails the given way.

    Builds the minimum real object graph: `_run_trial` materializes the source, runs the config
    screen, calls the evaluator, and builds the TrialRecord. Only the evaluator and the screen are
    faked -- the code under test is untouched.
    """
    import tempfile
    from pathlib import Path
    from unittest.mock import patch

    from kernel_optimizer.control.orchestrator import Orchestrator
    from kernel_optimizer.models.core import (
        Candidate,
        ParamDomain,
        ParameterSpace,
        ParamSet,
        TaskSpec,
    )

    class _Evaluator:
        def quick_test(self, task, path, tag, backend="triton"):
            return {"ok": False, "compiled": False, "failure_kind": "runtime_error",
                    "log_tail": log_tail}

    source = 'PARAMS = {\n    "BLOCK_M": 64,\n}\n\n\nclass ModelNew:\n    pass\n'
    cand = Candidate(candidate_id="cand-t", family_id="fam", origin="seed", backend="triton",
                     source_sha="0" * 64, structural_signature="0" * 64, approach_summary="x")
    space = ParameterSpace(
        space_id="sp", candidate_id="cand-t", version=1, source_sha="0" * 64,
        domains=[ParamDomain(name="BLOCK_M", kind="int", choices=[32, 64])], constraints=[])

    class _CandRun:
        candidate = cand
        source = globals().get("_src_placeholder") or ""

    crun = _CandRun()
    crun.source = source

    orch = Orchestrator.__new__(Orchestrator)          # no __init__: it wants a whole app
    orch.task = TaskSpec(name="43", level=3, problem_id=43, ref_path="/nonexistent/r.py",
                         ref_src_sha="0" * 64)

    class _Deps:
        evaluator = _Evaluator()

    orch.deps = _Deps()

    with tempfile.TemporaryDirectory() as td:
        # The config screen needs a compiler; it is not what this test is about, and returning
        # None is its documented "no opinion" answer, which lets the real trial run.
        with patch.object(Orchestrator, "_screen_config", return_value=None):
            # The real signature is (crun, space, trial_id, params, trials_dir) -- trial_id BEFORE
            # params. Passing them the other way round raises deep inside the materializer
            # ("'str' object has no attribute 'values'"), and this test SKIPS on the Windows host
            # for want of optuna, so the swap was invisible there and only failed on the A800.
            return Orchestrator._run_trial(
                orch, crun, space, "tr-1", ParamSet(values={"BLOCK_M": 64}), Path(td))


def test_the_two_signals_are_independent_not_one_regex():
    """Structural: the function must require BOTH an import failure AND the kernelbench chain.
    Collapsing them into a single pattern is how this becomes either too broad (reclassifying a
    candidate's import) or too narrow (anchored on one module name or one frame).
    """
    # An import failure with no kernelbench frame: not matched (asserted above).
    # A kernelbench frame with no import failure: also not matched.
    tail = ('  File ".../kernelbench/utils.py", line 6, in <module>\n'
            "RuntimeError: something else entirely\n")
    assert environment_defect(tail) is None
