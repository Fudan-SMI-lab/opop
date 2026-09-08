"""GPU worker — runs INSIDE WSL. stdlib + torch + triton + kernelbench only.

Usage: python worker_main.py --job job.json --out result.json
Always writes a result JSON, even on crash.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
import traceback
from pathlib import Path


def _log_tail(exc: BaseException) -> str:
    return "".join(traceback.format_exception(exc))[-4000:]


def _ensure_optional_deps() -> None:
    """kernelbench.utils imports litellm at module scope for LLM helper calls
    this worker never makes; stub it if absent so eval/timing stay importable."""
    import importlib.util
    import types

    if importlib.util.find_spec("litellm") is None:
        stub = types.ModuleType("litellm")
        stub.completion = None  # type: ignore[attr-defined]
        sys.modules["litellm"] = stub


def _classify_exception(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "out of memory" in text or "cuda oom" in text:
        return "oom"
    return "runtime_error"


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if not n:
        return -1.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def capture_timing_samples() -> bool:
    """Make KernelBench's own timing paths retain their raw samples. Idempotent.

    WHY THIS IS NEEDED AT ALL. The tuning objective must be a median: on level2:37 a 20-sample
    MEAN has a 24-37% coefficient of variation against the median's 3-8%, and on a pair whose
    true costs differ by 7.6% the mean picks the faster configuration 64.8% of the time while
    the median gets 93.2% (scripts/probe_robust_objective.py). `robust_ms` reads the median and
    falls back to the mean, so a missing median is not an error -- it is a SILENT downgrade to
    the estimator that is barely better than a coin flip.

    And it was missing on the main path. `_stats_to_dict` can only compute a median when it is
    handed the samples, and only ONE of the four timing paths had them: the relaxed-correctness
    handler, which times the model itself. The strict `eval_perf` path, the reference-baseline
    path, and `measure_ref_program_time` all read KernelBench's returned summary dict --
    and KernelBench computes `elapsed_times`, passes it to `get_timing_stats`, then drops it
    (eval.py:625-632, 671-678; timing.py:95-103). So every strict-mode run has been tuning on
    the 20-sample mean. Verified on run-l1-42-20260908-015408: all five eval_perf outputs carry
    keys ['max','mean','min','n','std'] and `median: None`.

    WHY INTERCEPT `get_timing_stats` rather than patch the callers. It is the single funnel --
    all four sites that hold an `elapsed_times` list pass it to exactly this function, and both
    modules resolve it as an attribute at call time (`timing.get_timing_stats` in eval.py, a
    module global in timing.py), so one wrapper reaches all of them. Patching call sites would
    mean editing vendored KernelBench, which is pinned at 423217d precisely so the evaluation
    口径 stays comparable across every run.

    The wrapper adds a key and changes nothing else: KernelBench's own mean/std/min/max are
    still ITS numbers, computed by its own code, so the pinned evaluation semantics are intact.
    The extra key rides in the dict KernelBench returns and is read by `_stats_to_dict`.
    Returns True if the patch is in place.
    """
    try:
        from kernelbench import timing as kb_timing
    except Exception:
        return False
    if getattr(kb_timing.get_timing_stats, "_kopt_captures_samples", False):
        return True
    original = kb_timing.get_timing_stats

    def get_timing_stats(elapsed_times, device=None):
        stats = original(elapsed_times, device=device)
        # Never let a diagnostic addition break timing: on any surprise, return KernelBench's
        # dict untouched and let the median stay absent rather than failing the measurement.
        try:
            stats["_samples"] = [float(v) for v in elapsed_times]
        except Exception:
            pass
        return stats

    get_timing_stats._kopt_captures_samples = True
    kb_timing.get_timing_stats = get_timing_stats
    return True


def _stats_to_dict(stats: dict, elapsed: list | None = None) -> dict:
    """Summarize a timing run, keeping the raw samples so robust statistics stay available.

    `median` and `samples` are additions, and both exist because a 20-sample MEAN is not a
    usable tuning objective. Measured on level2:37 at pre-scaling sizes
    (docs/finding-tuning-objective-is-a-20-sample-mean.md): a few 300-700 us scheduling
    stalls drag a 20-sample mean 35-136% above the kernel's real cost, giving the objective
    a 24-53% standard error against a `min_improvement_pct` of 2.0. The same stalls barely
    move the 100-sample baselines (mean/min 1.04-1.15x), so candidates were being penalized
    against baselines by construction.

    Quantified with scripts/probe_robust_objective.py (2000-sample ground truth, 400 windows
    of n=20, this machine): the mean's coefficient of variation at n=20 is 24-37%, the
    median's is 3-8%. On a pair whose true costs differ by 7.6%, a 20-sample mean picks the
    faster config 64.8% of the time -- near a coin flip -- while the median gets 93.2%.

    `min` is deliberately NOT offered as an objective despite looking robust: at n=20 it is
    biased +9.8% to +156% (20 samples rarely contain the true minimum) and it ranked three of
    six config pairs BACKWARDS, below 50% agreement, because it reports the luckiest draw
    rather than the configuration's cost.

    `samples` is retained because without it no estimator choice can be re-examined after the
    fact -- verifying the above required re-running the GPU, since the log held only
    mean/std/min/max. It also honours the intent already stated for the reference side
    ("keep the full distribution for the record"), which was never actually implemented.
    """
    out = {
        "mean": float(stats.get("mean", -1.0)),
        "std": float(stats.get("std", 0.0)),
        "min": float(stats.get("min", -1.0)),
        "max": float(stats.get("max", -1.0)),
        "n": int(stats.get("num_trials", 0)),
    }
    # `elapsed` is the explicit path, used where this worker did its own timing. The
    # `_median`/`_samples` keys are what `capture_timing_samples` smuggles through
    # KernelBench's summary dict on paths where the samples exist inside KernelBench and are
    # discarded before it returns -- see that function for why the interception is necessary.
    if elapsed is None and stats.get("_samples"):
        elapsed = stats["_samples"]
    if elapsed:
        try:
            vals = [float(v) for v in elapsed]
            out["median"] = _median(vals)
            out["samples"] = [round(v, 5) for v in vals]
        except (TypeError, ValueError):
            pass
    return out


def _classify_eval_failure(metadata: dict, compiled: bool, correct: bool) -> tuple[str, str]:
    if not compiled:
        detail = str(metadata.get("compilation_error", "")) or str(
            metadata.get("compilation_error_name", "compile failed")
        )
        return "compile_error", detail
    text = " ".join(str(v) for v in metadata.values()).lower()
    if "out of memory" in text:
        return "oom", text[-2000:]
    if not correct:
        if "runtime_error" in metadata or "runtime_error_name" in metadata:
            detail = str(metadata.get("runtime_error", metadata.get("runtime_error_name", "")))
            if "out of memory" in detail.lower():
                return "oom", detail
            return "runtime_error", detail
        detail = str(metadata.get("correctness_issue", "output mismatch"))
        return "correctness_mismatch", detail
    return "runtime_error", text[-2000:]


# --- triton metadata (duck-typed; pattern validated on triton 3.5 / sm_120) ---


def _import_statics():
    """Import evaluation.statics from inside the worker, whatever PYTHONPATH it was given.

    The worker is launched with PYTHONPATH set to KernelBench only (worker_client._build_command),
    because it is deliberately a stdlib+torch+triton process that does not depend on the harness
    package. Tier 1 broke that assumption: the counting and occupancy arithmetic live in
    `kernel_optimizer.evaluation.statics` so they can be unit-tested without a GPU, and a plain
    import therefore fails in a real run -- silently, since Tier 1 is best-effort. It only worked
    in manual testing because I had added the source root to PYTHONPATH by hand.

    Rather than adding a config knob someone has to remember (or duplicating the logic here, which
    would let the tested copy and the running copy drift), the worker locates its own package:
    worker_main.py lives at <root>/src/kernel_optimizer/gpu/, so the source root is three parents
    up. That works for every launcher -- WSL, the Linux port, a bare `python worker_main.py` -- and
    keeps ONE implementation under test.
    """
    try:
        from kernel_optimizer.evaluation import statics  # noqa: PLC0415

        return statics
    except ImportError:
        pass
    src_root = Path(__file__).resolve().parents[2]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from kernel_optimizer.evaluation import statics  # noqa: PLC0415

    return statics


def _tier1_statics(compiled, row: dict, props) -> dict:
    """SASS instruction mix + analytic occupancy for one compiled Triton kernel (step 5).

    Best-effort by construction: every failure path returns {} or a note, because this is a
    diagnostic and must never turn a working evaluation into a failed one. The counting and
    occupancy arithmetic live in evaluation/statics.py so they are unit-testable without a GPU;
    this function only supplies the cubin and the device limits.
    """
    out: dict = {}
    notes: list[str] = []
    try:
        statics = _import_statics()
        compute_occupancy = statics.compute_occupancy
        count_sass = statics.count_sass
        disassemble_cubin = statics.disassemble_cubin
    except Exception as exc:  # noqa: BLE001 — the worker may run without the package importable
        return {"statics_note": f"tier1 unavailable: {type(exc).__name__}: {exc}"[:200]}

    asm = getattr(compiled, "asm", None) or {}
    cubin = asm.get("cubin") if isinstance(asm, dict) else None
    if cubin:
        sass = disassemble_cubin(cubin)
        if sass:
            out["sass"] = count_sass(sass).model_dump()
        else:
            notes.append("no disassembler available (nvdisasm/cuobjdump not found) or "
                         "disassembly failed; instruction mix unknown")
    else:
        notes.append("no cubin in the compiled kernel's asm dict; instruction mix unknown")

    n_regs, num_warps = row.get("n_regs"), row.get("num_warps")
    if n_regs and num_warps:
        occ = compute_occupancy(
            n_regs, row.get("shared") or 0, num_warps,
            max_threads_per_sm=int(getattr(props, "max_threads_per_multi_processor", 0) or 0),
            regs_per_sm=int(getattr(props, "regs_per_multiprocessor", 65536) or 65536),
            shared_per_sm=int(getattr(props, "shared_memory_per_multiprocessor", 102400)
                              or 102400),
            max_blocks_per_sm=int(getattr(props, "max_blocks_per_multi_processor", 16) or 16),
        )
        if occ is not None:
            out["occupancy"] = occ.model_dump()
    else:
        notes.append("n_regs or num_warps missing; occupancy not computable")

    if notes:
        out["statics_notes"] = notes
    return out


def _extract_triton_metadata(kernel_src: str, ref_src: str, device_index: int) -> dict | None:
    """Load the kernel module fresh, launch forward once, then walk JIT caches."""
    import importlib.util
    import os
    import tempfile

    import torch

    t0 = time.monotonic()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(kernel_src)
        mod_path = f.name
    try:
        spec = importlib.util.spec_from_file_location("kopt_meta_probe", mod_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Reference provides input factories; candidate must not redefine them.
        ref_ctx: dict = {}
        exec(compile(ref_src, "<ref>", "exec"), ref_ctx)
        get_inputs = ref_ctx["get_inputs"]
        get_init_inputs = ref_ctx.get("get_init_inputs", lambda: [])

        device = torch.device(f"cuda:{device_index}")
        torch.cuda.set_device(device)
        with torch.no_grad():
            init_inputs = [
                x.to(device) if isinstance(x, torch.Tensor) else x for x in get_init_inputs()
            ]
            model = module.ModelNew(*init_inputs).to(device)
            inputs = [
                x.to(device) if isinstance(x, torch.Tensor) else x for x in get_inputs()
            ]
            model(*inputs)
            torch.cuda.synchronize(device)
        compile_s = time.monotonic() - t0
        props = torch.cuda.get_device_properties(device_index)

        kernels = []
        for attr_name in dir(module):
            obj = getattr(module, attr_name, None)
            caches = getattr(obj, "device_caches", None)
            if caches is None:
                continue
            try:
                entry = caches[device_index]
            except Exception:
                continue
            if not entry:
                continue
            for compiled in entry[0].values():
                meta = getattr(compiled, "metadata", None)
                row = {
                    "name": getattr(compiled, "name", None)
                    or (getattr(meta, "name", None) if meta else None),
                    "n_regs": _opt_int(getattr(compiled, "n_regs", None)),
                    "n_spills": _opt_int(getattr(compiled, "n_spills", None)),
                    "shared": _opt_int(getattr(meta, "shared", None) if meta else None),
                    "num_warps": _opt_int(
                        getattr(meta, "num_warps", None) if meta else None
                    ),
                    "num_stages": _opt_int(
                        getattr(meta, "num_stages", None) if meta else None
                    ),
                }
                # Tier 1 (step 5): instruction mix from the cubin, and analytic occupancy.
                # Both need no privileges, unlike ncu -- see evaluation/statics.py.
                row.update(_tier1_statics(compiled, row, props=props))
                kernels.append(row)
        return {"kernels": kernels, "compile_s": compile_s}
    finally:
        try:
            os.unlink(mod_path)
        except OSError:
            pass


def _opt_int(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


def _wants_kernel_metadata(job: dict) -> bool:
    """Whether this job should collect per-kernel resource metadata, for ANY backend.

    Reads a backend-neutral key and falls back to the historical Triton-specific one. The old
    name is the defect: callers set `collect_triton_metadata=(backend == "triton")`, so for a
    cuda/cutlass/cute candidate the flag was False and the outer `if` never opened -- which
    made the cubin branch below it, written specifically to serve those backends,
    unreachable. Measured on box 2: a CORRECT `load_inline` candidate returned
    `cubin: None, cubin_error: None`; flipping this one flag by hand on the same job produced
    a full record. So a non-Triton candidate reached the bottleneck classifier with no
    registers, spills, shared bytes or occupancy, and was judged worse than a Triton
    candidate for reasons unrelated to its kernel.

    The old key is still honoured so a replayed or in-flight job keeps working.
    """
    if "collect_kernel_metadata" in job:
        return bool(job["collect_kernel_metadata"])
    return bool(job.get("collect_triton_metadata"))


# --- cubin metadata: CUDA C++, and CUTLASS / CuTe for free ---------------------
#
# WHY THIS IS NOT A CUDA-SPECIFIC PATH. CUDA C++, CUTLASS and CuTe all compile through nvcc
# to a cubin, so ONE reader covers all three: `cuobjdump -res-usage` prints per-kernel
# registers / stack / shared / local out of the compiled object regardless of which of them
# wrote the source. Writing a "CUDA extractor" would mean rewriting it for CUTLASS later.
#
# WHY IT MATTERS THAT THIS EXISTS AT ALL. Before it, `ProfileRecord` was populated only from
# Triton's compiled-kernel object, so a `cuda` candidate got NOTHING: no registers, no spills,
# no shared. That is not a property of the backend, it is a shortcut in our profiler -- and it
# degraded the paper's own feedback loop (tuning evidence -> bottleneck report -> rewrite) on
# the backend with the HIGHER expressiveness ceiling, which is exactly backwards.


def _parse_res_usage(text: str) -> list[dict]:
    """Parse `cuobjdump -res-usage` output into per-kernel resource dicts.

    The format is a `Function <mangled>:` line followed by a line of KEY:VALUE pairs:

        Function _Z6kernelPfS_i:
          REG:42 STACK:0 SHARED:16384 LOCAL:0 CONSTANT[0]:380 TEXTURE:0 ...

    Parsed with a regex over the whole block rather than by column position, because the field
    set varies by architecture and by nvcc version (CONSTANT[n] banks come and go) -- a
    positional parse would silently mis-assign on the next toolkit.
    """
    import re

    out: list[dict] = []
    # Split on the Function header, keeping the name; DOTALL so the body may span lines.
    for match in re.finditer(r"Function\s+([^\s:]+):(.*?)(?=Function\s+[^\s:]+:|\Z)",
                             text, re.DOTALL):
        name, body = match.group(1), match.group(2)
        fields = {k.upper(): int(v) for k, v in re.findall(r"([A-Z]+)(?:\[\d+\])?:(\d+)", body)}
        if "REG" not in fields:
            continue
        out.append({
            "name": name,
            "n_regs": fields.get("REG"),
            # STACK is the per-thread stack frame; a non-zero value means the compiler could
            # not keep everything in registers, which is the same signal as a Triton spill.
            # LOCAL is spilled local memory. Report their sum so the field means the same
            # thing on both backends: "the compiler ran out of registers".
            "n_spills": (fields.get("STACK", 0) or 0) + (fields.get("LOCAL", 0) or 0),
            "shared": fields.get("SHARED"),
            # nvcc does not record a launch geometry in the cubin -- warps/stages are
            # properties of the LAUNCH, not the compiled code. Left None rather than guessed;
            # the runtime path below fills them when a launch is observed.
            "num_warps": None,
            "num_stages": None,
        })
    return out


def _launched_kernel_names(kernel_src: str, ref_src: str, device_index: int) -> list[str]:
    """Which kernels a candidate ACTUALLY launches, observed with torch.profiler.

    Needed because a cubin records no launch: it holds whatever nvcc emitted, and for CUTLASS
    that is dozens of template instantiations of which one runs. Aggregating resources over the
    whole cubin would report a variant that never executed -- the same class of error as timing
    a fallback path and calling it the kernel.

    Uses torch.profiler (CUPTI), NOT `ncu`. Hardware performance counters are unavailable on a
    rented container (ERR_NVGPUCTRPERM needs a host-side kernel-module parameter), but per-kernel
    timing and launch counts do not require them -- so this works on the machines the experiments
    actually run on. Returns [] on any failure: an empty list means "not observed", and the
    caller must not treat it as "nothing launched".
    """
    import importlib.util
    import os
    import tempfile

    import torch
    from torch.profiler import ProfilerActivity, profile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(kernel_src)
        mod_path = f.name
    try:
        spec = importlib.util.spec_from_file_location("kopt_launch_probe", mod_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        ref_ctx: dict = {}
        exec(compile(ref_src, "<ref>", "exec"), ref_ctx)
        get_inputs = ref_ctx["get_inputs"]
        get_init_inputs = ref_ctx.get("get_init_inputs", lambda: [])

        device = torch.device(f"cuda:{device_index}")
        torch.cuda.set_device(device)
        with torch.no_grad():
            init_inputs = [x.to(device) if isinstance(x, torch.Tensor) else x
                           for x in get_init_inputs()]
            model = module.ModelNew(*init_inputs).to(device)
            inputs = [x.to(device) if isinstance(x, torch.Tensor) else x
                      for x in get_inputs()]
            model(*inputs)                      # warm up / compile before profiling
            torch.cuda.synchronize(device)
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                model(*inputs)
                torch.cuda.synchronize(device)
        names = []
        for evt in prof.key_averages():
            if getattr(evt, "self_device_time_total", 0) and evt.key:
                names.append(evt.key)
        return names
    except Exception:  # noqa: BLE001 — observation is best-effort
        return []
    finally:
        try:
            os.unlink(mod_path)
        except OSError:
            pass


def _attach_launch_overhead(job: dict, result: dict, kernel_src: str, ref_src: str) -> None:
    """Attach the launch-overhead probe to a result, if the job asked for it.

    A shared helper because the block used to live only inside `run_eval`, while BOTH L3
    configs set `correctness_mode: dual_witness_relaxed` and therefore route to
    `run_relaxed_correctness` -- a handler that mentioned neither the flag nor the field. So
    passing `measure_launch_overhead: True` on the configs the experiments actually use
    returned nothing, which is what the L3:43 final re-eval measured: flag True, value None.

    That was one of THREE independent reasons `launch_bound` had never fired in any run. The
    other two: only `full_eval` set the flag (and its sole caller is the final re-eval, while
    every verdict comes from a tuning trial), and `classify()` divided by the wrong `gpu_ms`.
    Fixing the other two without this one would have reproduced the same None.
    """
    if not job.get("measure_launch_overhead"):
        return
    try:
        result["launch_overhead"] = _measure_launch_overhead(kernel_src, ref_src, 0)
    except Exception as exc:  # noqa: BLE001 — a diagnostic must never fail the eval
        result["launch_overhead"] = None
        result["launch_overhead_error"] = str(exc)[-1000:]


def run_compile_probe(job: dict) -> dict:
    """Compile a candidate's Triton kernels WITHOUT launching them, and report shared bytes.

    WHY THIS EXISTS. On run-l3-43-20260908-053708, 180 of 1004 tuning trials (18% of the
    budget, 0.93 h of 11.73 h wall clock) failed with Triton's
    `out of resource: shared memory`. Every one was predictable before touching the GPU: the
    compiler fills in `metadata.shared`, and the launch merely compares it against the device
    limit. Probed on box 2 across 6 configurations of a real pipelined `tl.dot` matmul, the
    compile-time figure equals the runtime `Required:` value byte-for-byte.

    Worse than the wasted time, those trials were reported to Optuna as FAIL, which EXCLUDES
    them from the TPE model rather than treating them as bad objectives -- so 18% of the
    samples produced neither a measurement nor an avoidance signal, which is why the
    per-candidate failure rate (21-33%) never decayed over a run.

    WHY NOT A GUARD CONSTRAINT. Shared usage is not a closed-form function of the knobs.
    Audited across all 16 L3:43 candidates, `BLOCK_M * BLOCK_N * stages` has failing-min BELOW
    passing-max in 15 of 15 -- the feasible and infeasible sets overlap in any such product, so
    no arithmetic constraint can separate them and asking the parameterizer agent to write one
    would be asking it to guess. The number has to come from the compiler.

    HOW IT AVOIDS THE LAUNCH. `JITFunction.run` takes a `warmup` flag and returns the compiled
    kernel before its launch block (`if not warmup:`), so forcing it True compiles every kernel
    the forward pass reaches and launches none. The forward then produces garbage or raises --
    both fine and both ignored, since the only output wanted is metadata.

    Returns `{"ok": True, "kernels": [...], "max_shared": N}`. `ok: False` means the probe
    could not answer, and the caller MUST fall through to a real trial: this screen may never
    be the thing that rejects a candidate.
    """
    import importlib.util
    import os
    import tempfile

    import torch

    kernel_src = open(job["kernel_src_path"], encoding="utf-8").read()
    ref_src = open(job["ref_src_path"], encoding="utf-8").read()

    try:
        from triton.runtime.jit import JITFunction
    except Exception as exc:  # noqa: BLE001 — a non-Triton candidate has nothing to probe
        return {"ok": False, "reason": f"triton unavailable: {type(exc).__name__}: {exc}"[:200]}

    seen: list[dict] = []
    original_run = JITFunction.run

    def run_compile_only(self, *args, grid=None, warmup=False, **kwargs):
        kernel = original_run(self, *args, grid=grid, warmup=True, **kwargs)
        try:
            meta = kernel.metadata
            seen.append({
                "name": getattr(self, "__name__", "?"),
                "shared": int(getattr(meta, "shared", 0) or 0),
                "n_regs": _opt_int(getattr(meta, "num_regs", None)),
                "n_spills": _opt_int(getattr(meta, "num_spills", None)),
                "num_warps": _opt_int(getattr(meta, "num_warps", None)),
                "num_stages": _opt_int(getattr(meta, "num_stages", None)),
            })
        except Exception:  # noqa: BLE001 — one unreadable kernel must not lose the others
            seen.append({"name": getattr(self, "__name__", "?"), "shared": None})
        return kernel

    mod_path = None
    try:
        JITFunction.run = run_compile_only
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False,
                                         encoding="utf-8") as f:
            f.write(kernel_src)
            mod_path = f.name
        spec = importlib.util.spec_from_file_location("kopt_compile_probe", mod_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        ref_ctx: dict = {}
        exec(compile(ref_src, "<ref>", "exec"), ref_ctx)  # noqa: S102 — same trust as eval
        get_inputs = ref_ctx["get_inputs"]
        get_init_inputs = ref_ctx.get("get_init_inputs", lambda: [])

        device = torch.device(f"cuda:{job.get('device_index', 0)}")
        torch.cuda.set_device(device)
        with torch.no_grad():
            init_inputs = [x.to(device) if isinstance(x, torch.Tensor) else x
                           for x in get_init_inputs()]
            model = module.ModelNew(*init_inputs).to(device)
            inputs = [x.to(device) if isinstance(x, torch.Tensor) else x
                      for x in get_inputs()]
            try:
                model(*inputs)
            except Exception:  # noqa: BLE001, S110
                # EXPECTED and deliberately ignored. Nothing launched, so downstream torch ops
                # see uninitialized buffers and may raise anything. A raise here says nothing
                # about whether the config is feasible -- only `seen` does.
                pass
    except Exception as exc:  # noqa: BLE001 — probe failure is "cannot answer", never a verdict
        return {"ok": False, "reason": f"{type(exc).__name__}: {exc}"[:300],
                "kernels": seen}
    finally:
        JITFunction.run = original_run
        if mod_path:
            try:
                os.unlink(mod_path)
            except OSError:
                pass

    if not seen:
        # No Triton kernel was reached: a CUDA/CUTLASS candidate, or a forward that failed
        # before its first kernel. Either way the screen has no opinion.
        return {"ok": False, "reason": "no triton kernel compiled during the forward pass"}
    shared_vals = [k["shared"] for k in seen if k.get("shared") is not None]
    return {"ok": True, "kernels": seen,
            "max_shared": max(shared_vals) if shared_vals else None}


def _measure_launch_overhead(kernel_src: str, ref_src: str, device_index: int,
                             iters: int = 50) -> dict | None:
    """CPU-side issue cost and true wall time per call — the harness's timing blind spot.

    WHY THIS IS NOT ALREADY KNOWN. KernelBench times a call as
    `synchronize -> clear_l2_cache() -> start_event.record() -> kernel_fn() -> end_event.record()`
    (timing.py:251-263). The L2 flush is enqueued and never waited on, so it sits in the queue
    while the CPU issues the kernel: the GPU is busy flushing exactly while the launch is being
    submitted, and the CPU cost is hidden behind it. That is a reasonable way to measure a
    kernel's GPU execution, but it means an op whose real cost is dominated by launching is
    reported as fast. Measured on level2:37: 66.6 us of CPU issue time against a reported
    37.9 us total, and an external candidate's claimed 8.85x became 1.53x once the baseline was
    timed under the same convention.

    So this returns three numbers per call, all for the SAME model, and their disagreement is
    the signal:

      cpu_issue_ms  -- CPU time to submit the work, measured with NO synchronization inside the
                       loop. This is what a caller pays even if the GPU were infinitely fast.
      gpu_ms        -- device time, from CUDA events, synchronized (comparable to what the
                       harness reports).
      wall_ms       -- end-to-end wall clock per call with a single sync at the end: what the
                       calling program actually experiences.

    `cpu_issue_ms / gpu_ms >= 1` means the CPU cannot keep the GPU fed and no amount of kernel
    optimization will change what a caller observes -- the `launch_bound` verdict.

    Deliberately NO L2 flush: the flush is what hides the cost, so measuring with it would
    reproduce the blind spot instead of measuring it. That makes `gpu_ms` here warm-cache and
    thus NOT a substitute for the harness's number; it is only the denominator of the ratio.
    """
    import importlib.util
    import os
    import tempfile

    import torch

    mod_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False,
                                         encoding="utf-8") as f:
            f.write(kernel_src)
            mod_path = f.name
        spec = importlib.util.spec_from_file_location("kopt_overhead_probe", mod_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ref_ctx: dict = {}
        exec(compile(ref_src, "<ref>", "exec"), ref_ctx)  # noqa: S102 — same trust as eval
        get_inputs = ref_ctx["get_inputs"]
        get_init_inputs = ref_ctx.get("get_init_inputs", lambda: [])

        device = torch.device(f"cuda:{device_index}")
        with torch.no_grad():
            init_inputs = [x.to(device) if isinstance(x, torch.Tensor) else x
                           for x in get_init_inputs()]
            model = module.ModelNew(*init_inputs).to(device)
            inputs = [x.to(device) if isinstance(x, torch.Tensor) else x
                      for x in get_inputs()]
            for _ in range(10):            # warm up: compile, autotune, allocate
                model(*inputs)
            torch.cuda.synchronize(device)

            # 1) CPU issue cost: no sync inside the loop, so this measures submission only.
            #    If the queue saturates, later iterations block on it and this becomes an upper
            #    bound rather than pure issue cost -- which is still the right thing to compare,
            #    since a caller pays that too.
            t0 = time.perf_counter()
            for _ in range(iters):
                model(*inputs)
            cpu_issue_ms = (time.perf_counter() - t0) / iters * 1e3
            torch.cuda.synchronize(device)

            # 2) Wall clock per call, one sync at the end: what the caller experiences.
            t0 = time.perf_counter()
            for _ in range(iters):
                model(*inputs)
            torch.cuda.synchronize(device)
            wall_ms = (time.perf_counter() - t0) / iters * 1e3

            # 3) Device time from CUDA events, for the ratio's denominator.
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(device)
            start.record()
            for _ in range(iters):
                model(*inputs)
            end.record()
            torch.cuda.synchronize(device)
            gpu_ms = start.elapsed_time(end) / iters

        return {
            "cpu_issue_ms": round(cpu_issue_ms, 6),
            "gpu_ms": round(gpu_ms, 6),
            "wall_ms": round(wall_ms, 6),
            "iters": iters,
            # Stated so a reader does not compare this gpu_ms with the harness's headline
            # figure and conclude one of them is wrong.
            "note": ("no L2 flush between calls (the flush is what hides launch cost), so "
                     "gpu_ms here is warm-cache and is not comparable to the harness's "
                     "reported latency; use it only as the denominator of cpu_issue_ms/gpu_ms"),
        }
    except Exception:  # noqa: BLE001 — best-effort, never fail an eval over a diagnostic
        return None
    finally:
        if mod_path:
            try:
                os.unlink(mod_path)
            except OSError:
                pass


def _kernel_identifiers(name: str) -> set[str]:
    """The bare function identifiers in a kernel name, whether mangled or demangled.

    Exists because the two sides of the launched-kernel filter speak different dialects and
    could therefore never match. `torch.profiler` reports DEMANGLED signatures --
    `addk(float const*, float const*, float*, int)` -- while `cuobjdump -res-usage` reports
    MANGLED symbols -- `_Z4addkPKfS0_Pfi`. A substring test between those always fails, so
    `launched_filter` came back `no_name_match` and the resource record covered every kernel
    in the cubin including 17 that never ran (measured on box 2).

    Both dialects are reduced to identifiers here, and callers compare identifier SETS rather
    than substrings -- a substring test would let a short name like `mm` match an unrelated
    mangled symbol that merely contains those letters.

    Demangled: drop the argument list, drop template arguments, keep the last `::` segment.
    Mangled: Itanium mangling writes each component as <length><chars>, so every
    length-prefixed run is a component; collecting them all covers both plain and nested
    (`N...E`) names without implementing the full grammar.
    """
    out: set[str] = set()
    base = name.split("(", 1)[0].strip()
    if base.startswith("_Z"):
        i, n = 2, len(base)
        if base[i:i + 1] == "N":       # nested name: N <components> E
            i += 1
        while i < n and base[i].isdigit():
            j = i
            while j < n and base[j].isdigit():
                j += 1
            length = int(base[i:j])
            if length <= 0 or j + length > n:
                break
            out.add(base[j:j + length])
            i = j + length
        return out
    # Demangled: strip template arguments, then take the trailing identifier.
    depth, cleaned = 0, []
    for ch in base:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        elif depth == 0:
            cleaned.append(ch)
    ident = "".join(cleaned).strip().split("::")[-1].strip()
    if ident:
        out.add(ident)
    return out


def _cubin_sass(obj_path: str, _cache: dict = {}) -> dict:  # noqa: B006 — intentional memo
    """Instruction mix for a nvcc-built object, so a CUDA candidate is not judged blind.

    Without it `uses_tensor_cores` is None for every non-Triton candidate, and `classify`
    then falls back to the fp32 compute ceiling. Measured consequence on an identical kernel
    (same latency, registers and shared bytes): Triton got `resource_limited` with
    "shrink the tile", CUDA got `compute_bound` at 89.7% of fp32 with "move onto tensor cores"
    -- advice for something the kernel was already doing. The verdict differed by BACKEND, not
    by kernel, which makes any backend comparison meaningless.

    Cached per object because several kernels share one .so and disassembly is not cheap. The
    mix is therefore per-OBJECT, not per-kernel: for a single-kernel extension that is exact,
    and for a multi-kernel one it answers "does this binary use tensor cores at all", which is
    what selects the ceiling. `num_warps` is deliberately NOT recoverable this way -- it is a
    launch parameter chosen at the call site (`<<<grid, block>>>`), so no compiled artifact
    records it, which is why occupancy stays unavailable for this backend.
    """
    if obj_path in _cache:
        return _cache[obj_path]
    out: dict = {}
    try:
        statics = _import_statics()
        text = statics.disassemble_object(obj_path)
        if text:
            out["sass"] = statics.count_sass(text).model_dump()
        else:
            out["statics_notes"] = ["no disassembler available or disassembly of the built "
                                    "object failed; instruction mix unknown"]
    except Exception as exc:  # noqa: BLE001 — a diagnostic must never fail an evaluation
        out["statics_notes"] = [f"tier1 unavailable: {type(exc).__name__}: {exc}"[:150]]
    # A cubin records no launch geometry, so occupancy cannot be computed for this backend.
    # Said explicitly rather than left blank, so a reader does not mistake an unmeasurable
    # signal for a measured-and-fine one.
    out.setdefault("statics_notes", []).append(
        "occupancy not computable for a cubin backend: num_warps is a launch parameter and is "
        "not recorded in any compiled artifact")
    _cache[obj_path] = out
    return out


def _extract_cubin_metadata(launched_names: list[str] | None = None,
                            build_dir: str | None = None) -> dict | None:
    """Resources for the kernels a load_inline / nvcc build produced.

    `launched_names` MUST be passed when known: CUTLASS instantiates many template variants
    and a cubin can hold dozens of kernels of which one actually runs. Aggregating over the
    whole cubin would report a variant that never executed -- the same class of error as
    timing a fallback path and calling it the kernel. When it is given, kernels are kept when
    they share a function identifier with a launched name (see `_kernel_identifiers`: the two
    sides are mangled and demangled respectively, so they are compared as identifier sets).

    Returns None (not an empty dict) when there is nothing to read, so the caller can tell
    "no CUDA backend here" from "a CUDA backend with no resources", which would be a bug.
    """
    import glob
    import os
    import shutil
    import subprocess

    cuobjdump = shutil.which("cuobjdump")
    if not cuobjdump:
        try:
            from torch.utils.cpp_extension import CUDA_HOME

            cand = os.path.join(CUDA_HOME or "", "bin", "cuobjdump")
            cuobjdump = cand if os.path.exists(cand) else None
        except Exception:  # noqa: BLE001
            cuobjdump = None
    if not cuobjdump:
        return None

    # torch's load_inline builds into TORCH_EXTENSIONS_DIR (default ~/.cache/torch_extensions),
    # one directory per extension name, containing the .so. The cubin is embedded in the .so,
    # and cuobjdump reads it directly out of the host binary.
    roots = [build_dir] if build_dir else []
    roots.append(os.environ.get("TORCH_EXTENSIONS_DIR")
                 or os.path.expanduser("~/.cache/torch_extensions"))
    objects: list[str] = []
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        objects.extend(sorted(glob.glob(os.path.join(root, "**", "*.so"), recursive=True),
                              key=os.path.getmtime, reverse=True))
    if not objects:
        return None

    kernels: list[dict] = []
    seen: set[str] = set()
    for obj in objects[:4]:      # newest few: an older extension in the cache is not ours
        try:
            proc = subprocess.run([cuobjdump, "-res-usage", obj],
                                  capture_output=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if proc.returncode != 0:
            continue
        for k in _parse_res_usage(proc.stdout.decode("utf-8", "replace")):
            if k["name"] in seen:
                continue
            seen.add(k["name"])
            k.update(_cubin_sass(obj))
            kernels.append(k)
    if not kernels:
        return None

    if launched_names:
        # Compare identifier sets, not substrings: the profiler's names are demangled and the
        # cubin's are mangled, so a substring test never matched (see _kernel_identifiers).
        want: set[str] = set()
        for ln in launched_names:
            want |= _kernel_identifiers(ln)
        keep = [k for k in kernels if _kernel_identifiers(k["name"]) & want] if want else []
        # Only narrow when the intersection is non-empty. Reporting nothing would be worse
        # than reporting the union, and a genuine mismatch is still possible (a kernel
        # launched from a library, a name the profiler renders unexpectedly). Say which
        # happened so the report cannot present a guess as a measurement.
        if keep:
            return {"kernels": keep, "launched_filter": "applied"}
        return {"kernels": kernels, "launched_filter": "no_name_match"}
    return {"kernels": kernels, "launched_filter": "not_available"}


# --- job handlers -------------------------------------------------------------


def run_env_probe(job: dict) -> dict:
    import torch

    result = {
        "ok": True,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        result["device_name"] = torch.cuda.get_device_name(0)
        result["capability"] = list(torch.cuda.get_device_capability(0))
        free, total = torch.cuda.mem_get_info(0)
        result["vram_free_bytes"] = free
        result["vram_total_bytes"] = total
    try:
        import triton

        result["triton"] = triton.__version__
    except Exception as exc:  # noqa: BLE001
        result["triton"] = None
        result["triton_error"] = str(exc)
    try:
        # Import the SYMBOLS THE HARNESS ACTUALLY CALLS, not just the package. `import
        # kernelbench` only executes the package __init__, which on the pinned 423217d does
        # not reach `kernelbench.utils` -- and that module imports litellm at module scope.
        # So a box missing litellm (or dotenv, tqdm, openai...) passed this probe, doctor
        # reported "kernelbench importable" green, and the run then died at its FIRST
        # baseline: `load_original_model_and_inputs` swallows the ImportError and returns
        # None, so the caller fails several frames away with `TypeError: cannot unpack
        # non-iterable NoneType object` and the traceback never names the missing module.
        # Measured on box 2 (2026-09-07): doctor was green while `from kernelbench.eval
        # import eval_kernel_against_ref` raised ModuleNotFoundError: litellm.
        #
        # Importing the real entry points makes the probe fail for the same reason a run
        # would, and the error text names the module. Generic by construction: it checks
        # whatever those modules transitively need, so a future dependency is covered too.
        from kernelbench.eval import (  # noqa: F401
            eval_kernel_against_ref,
            load_original_model_and_inputs,
        )
        from kernelbench.timing import time_execution_with_cuda_event  # noqa: F401

        result["kernelbench_importable"] = True
    except Exception as exc:  # noqa: BLE001
        result["kernelbench_importable"] = False
        result["kernelbench_error"] = f"{type(exc).__name__}: {exc}"
    # The `cuda` backend compiles through torch.utils.cpp_extension.load_inline, which needs
    # BOTH nvcc and ninja on PATH. Neither is required by the triton backend, so a box can be
    # fully green and still be unable to build a single CUDA candidate -- and the failure
    # arrives as a per-candidate compile_error at the witness gate, which reads like the
    # agent wrote bad code. Measured on box 2 (2026-09-07): ninja was pip-installed into the
    # venv but not on PATH, so `load_inline` raised "Ninja is required to load C++ extensions"
    # for every cuda candidate while doctor reported all green.
    #
    # Reported, not fatal: a Triton-only run is perfectly valid, and every candidate produced
    # across 20 runs so far has been Triton. The point is that the operator learns which
    # backends this box can actually build BEFORE an agent spends a call writing one.
    try:
        from torch.utils.cpp_extension import CUDA_HOME  # noqa: PLC0415

        result["cuda_home"] = CUDA_HOME
        result["nvcc"] = shutil.which("nvcc") or (
            f"{CUDA_HOME}/bin/nvcc" if CUDA_HOME
            and Path(f"{CUDA_HOME}/bin/nvcc").exists() else None
        )
        result["ninja"] = shutil.which("ninja")
        result["cuda_backend_buildable"] = bool(result["nvcc"] and result["ninja"])
    except Exception as exc:  # noqa: BLE001
        result["cuda_backend_buildable"] = False
        result["cuda_backend_error"] = f"{type(exc).__name__}: {exc}"
    return result


def run_calibrate(job: dict) -> dict:
    """Measure this box's ceilings and the four yardstick workloads.

    Every number the bottleneck classifier compares against comes from here. Nothing is read
    from a datasheet, because a datasheet describes a card and this describes a container: the
    same 4090 gives materially different achievable bandwidth depending on clocks and who else
    is on the PCIe root.

    The yardstick workloads exist so the classification THRESHOLDS can also be measured instead
    of guessed. Each has an analytically known bottleneck, so the separation between them is the
    evidence for where each line belongs; `evaluation/calibration.derive_thresholds` turns that
    separation into dimensionless fractions. Two guessed constants were disproved this way on
    the 4090 (a 0.50 compute line against a measured 0.975; a 1.0 launch ratio against a
    measured 0.973 for an indisputably launch-bound workload).

    Sizes are chosen relative to the CARD, not fixed: the streaming buffer is sized off free
    VRAM and the matmul off the L2, so this is not a measurement that only works on 24 GB.
    """
    import statistics as _stats

    import torch

    if not torch.cuda.is_available():
        return {"ok": False, "failure_kind": "worker_crash",
                "log_tail": "no CUDA device; cannot calibrate"}

    device = _pick_device()               # already a torch.device, not an index
    torch.cuda.set_device(device)
    device_index = device.index or 0
    props = torch.cuda.get_device_properties(device_index)

    def timed(fn, n: int, warmup: int) -> float:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize(device)
        samples = []
        for _ in range(n):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize(device)
            samples.append(start.elapsed_time(end))
        return _stats.median(samples)

    def cpu_issue(fn, n: int) -> float:
        torch.cuda.synchronize(device)
        samples = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - t0) * 1e3)
        torch.cuda.synchronize(device)
        return _stats.median(samples)

    result: dict = {
        "ok": True,
        "device_name": props.name,
        "capability": [props.major, props.minor],
        "sm_count": props.multi_processor_count,
        "torch_version": torch.__version__,
        "driver_version": getattr(torch, "version", None)
        and getattr(torch.version, "cuda", "") or "",
        # `L2_cache_size` -- capital L. The lowercase spelling silently returns the default and
        # a calibration would then report an L2 of zero without erroring.
        "l2_bytes": int(getattr(props, "L2_cache_size", 0) or 0),
    }

    # Derived spec bandwidth: 2 (DDR) x memory clock x bus width / 8. Used ONLY to detect a
    # throttled or contended calibration, never as the denominator -- what a kernel can actually
    # get is the honest ceiling.
    mem_clock_khz = int(getattr(props, "memory_clock_rate", 0) or 0)
    bus_width = int(getattr(props, "memory_bus_width", 0) or 0)
    result["spec_dram_tbs"] = (
        (2 * mem_clock_khz * 1e3 * bus_width / 8 / 1e12) if (mem_clock_khz and bus_width) else 0.0
    )

    free_bytes, _total = torch.cuda.mem_get_info(device_index)

    with torch.no_grad():
        # --- DRAM ceiling: a streaming add sized off FREE VRAM, so this works on a 16 GB card
        # as well as a 24 GB one, and never OOMs the box it is measuring.
        stream_elems = int(min(128 * 1024 * 1024, max(4 * 1024 * 1024,
                                                      free_bytes * 0.15 / 4 / 2)))
        big = torch.randn(stream_elems, device=device)
        out = torch.empty_like(big)
        ms = timed(lambda: torch.add(big, 1.0, out=out), n=30, warmup=10)
        result["dram_tbs"] = (2 * stream_elems * 4) / (ms * 1e-3) / 1e12
        result["dram_probe_bytes"] = 2 * stream_elems * 4
        del big, out
        torch.cuda.empty_cache()

        # --- fp32 and tf32 ceilings from the same matmul. Both are needed: comparing a
        # tensor-core kernel against the fp32 ceiling reports >100% of peak, and comparing a
        # scalar kernel against the tf32 ceiling reports it as hopeless. The classifier must
        # know which ceiling applies to the kernel in front of it.
        mm_n = 8192
        while mm_n > 1024 and (3 * mm_n * mm_n * 4) > free_bytes * 0.35:
            mm_n //= 2
        a = torch.randn(mm_n, mm_n, device=device)
        b = torch.randn(mm_n, mm_n, device=device)
        prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            ms = timed(lambda: a @ b, n=20, warmup=8)
            result["fp32_tflops"] = (2 * mm_n ** 3) / (ms * 1e-3) / 1e12
            torch.backends.cuda.matmul.allow_tf32 = True
            ms = timed(lambda: a @ b, n=20, warmup=8)
            result["tf32_tflops"] = (2 * mm_n ** 3) / (ms * 1e-3) / 1e12
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32
        result["matmul_probe_n"] = mm_n
        del a, b
        torch.cuda.empty_cache()

        # --- fp16 and bf16 ceilings (P3). Without these, a low-precision candidate was scored
        # against the tf32 figure, and on a 4090 fp16 dense throughput is roughly 2x tf32. The
        # consequence INVERTS the advice: L3:43's best candidate read 140.7% of its ceiling --
        # "saturated, stop" -- when it was plausibly around 70%. 13 of 25 verdicts in that run
        # carried `impossible_fraction` for this reason, and the defect got worse as candidates
        # got faster, since every candidate good enough to lead was fp16 or bf16.
        #
        # Measured the same way as fp32/tf32 (same shape, same `timed`, same median), so the
        # four numbers are comparable and a ratio between them means something.
        for name, dtype in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
            try:
                ah = torch.randn(mm_n, mm_n, device=device, dtype=dtype)
                bh = torch.randn(mm_n, mm_n, device=device, dtype=dtype)
                ms = timed(lambda: ah @ bh, n=20, warmup=8)  # noqa: B023
                result[f"{name}_tflops"] = (2 * mm_n ** 3) / (ms * 1e-3) / 1e12
                del ah, bh
                torch.cuda.empty_cache()
            except Exception as exc:  # noqa: BLE001 — a missing dtype must not lose the rest
                result[f"{name}_tflops"] = 0.0
                result[f"{name}_error"] = f"{type(exc).__name__}: {exc}"[:200]

        # --- Empty-launch floor: what a launch costs when the body does nothing. This is the
        # input `overhead_floor` has been missing. Without it, a kernel already at the floor
        # shows near-zero throughput fractions and lands in `latency_bound`, so the agent is
        # told to add parallelism to a kernel whose body is no longer what costs.
        #
        # Measured on a real launch (a 1-element op), not on an empty CUDA graph: what matters
        # is the floor a torch-level candidate can reach, which includes torch's own dispatch.
        tiny = torch.zeros(1, device=device)
        tiny_out = torch.empty_like(tiny)
        floor_ms = timed(lambda: torch.add(tiny, 1.0, out=tiny_out), n=200, warmup=50)
        result["empty_launch_floor_ms"] = floor_ms
        result["empty_launch_floor_note"] = (
            "median GPU-event time for a 1-element elementwise op: the smallest latency any "
            "single torch-level launch can have on this box. A kernel at or near this is at the "
            "floor and no tiling or precision change can help it.")
        del tiny, tiny_out

        # --- The four yardsticks. `truth` is decided analytically BEFORE measuring; it is an
        # input to placing the thresholds, not a prediction to be checked against them.
        yardsticks = []

        a = torch.randn(4096, 4096, device=device)
        b = torch.randn(4096, 4096, device=device)
        prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            fn = lambda: a @ b  # noqa: E731
            yardsticks.append({
                "name": "COMPUTE 4096^3 fp32 matmul", "truth": "compute",
                "gpu_ms": timed(fn, n=40, warmup=10), "cpu_issue_ms": cpu_issue(fn, n=40),
                "flop_count": 2 * 4096 ** 3, "byte_count": 3 * 4096 * 4096 * 4,
            })
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32
        del a, b
        torch.cuda.empty_cache()

        copy_elems = int(min(64 * 1024 * 1024, max(2 * 1024 * 1024,
                                                   free_bytes * 0.08 / 4 / 2)))
        big = torch.randn(copy_elems, device=device)
        out = torch.empty_like(big)
        fn = lambda: torch.add(big, 1.0, out=out)  # noqa: E731
        yardsticks.append({
            "name": f"MEMORY {copy_elems*4//(1024*1024)}MB elementwise", "truth": "memory",
            "gpu_ms": timed(fn, n=40, warmup=10), "cpu_issue_ms": cpu_issue(fn, n=40),
            "flop_count": copy_elems, "byte_count": 2 * copy_elems * 4,
        })
        del big, out
        torch.cuda.empty_cache()

        small = torch.randn(256, 256, device=device)

        def many():
            y = small
            for _ in range(40):
                y = torch.relu(y) + 1.0
            return y

        yardsticks.append({
            "name": "LAUNCH 40 tiny ops", "truth": "launch",
            "gpu_ms": timed(many, n=40, warmup=10), "cpu_issue_ms": cpu_issue(many, n=40),
            "flop_count": 40 * 2 * 256 * 256, "byte_count": 40 * 2 * 256 * 256 * 4,
        })

        # A realistic fused-op shape. Its analytic truth is `unsaturated`: neither ceiling is
        # anywhere near reached and the host side is non-trivial. Two jobs here -- it keeps the
        # launch line from swallowing ordinary kernels (its cpu/gpu is high but not launch-bound),
        # and it keeps the idle line honest. NOT labelled "mixed": the classifier has a `mixed`
        # class meaning "partially saturated", and this workload is measurably below the idle
        # line on both axes (3.7% of bandwidth, 2.1% of compute on the 4090). That is also the
        # evidence that the residual classes are the NORM for real fused tasks, not edge cases.
        m_, k_, n_, g_ = 128, 512, 1024, 32
        x = torch.randn(m_, k_, device=device)
        w = torch.randn(n_, k_, device=device)
        wb = torch.randn(n_, device=device)
        bias = torch.randn(n_, device=device)
        gnw = torch.randn(n_, device=device)
        gnb = torch.randn(n_, device=device)

        def mixed():
            z = torch.nn.functional.linear(x, w, wb)
            h = torch.sigmoid(z) * z
            return torch.nn.functional.group_norm(h + bias, g_, gnw, gnb, 1e-5)

        yardsticks.append({
            "name": "UNSATURATED linear+silu+groupnorm", "truth": "unsaturated",
            "gpu_ms": timed(mixed, n=40, warmup=10), "cpu_issue_ms": cpu_issue(mixed, n=40),
            "flop_count": 2 * m_ * k_ * n_,
            "byte_count": (m_ * k_ + n_ * k_ + 3 * m_ * n_) * 4,
        })

    result["yardsticks"] = yardsticks

    # Which profiling tier this box actually supports (step 4). Recorded WITH the calibration
    # rather than only printed by doctor, because a report read weeks later must be able to say
    # why a run's verdicts carry no instruction mix -- "the box had no disassembler" and "the
    # kernels genuinely used no tensor cores" are opposite conclusions that otherwise look
    # identical in the log.
    try:
        find_cuda_tool = _import_statics().find_cuda_tool

        nvdisasm = find_cuda_tool("nvdisasm")
        cuobjdump = find_cuda_tool("cuobjdump")
        result["tiers"] = {
            "tier0_events_and_counts": True,
            "tier1_sass_and_occupancy": bool(nvdisasm or cuobjdump),
            "nvdisasm": nvdisasm,
            "cuobjdump": cuobjdump,
            # Tier 3 is counters. `ncu` being present says nothing about whether counters work:
            # it is installed on both experiment boxes and returns ERR_NVGPUCTRPERM on each.
            "ncu_present": bool(find_cuda_tool("ncu")),
            "tier3_counters": False,
            "tier3_note": ("hardware counters need NVreg_RestrictProfilingToAdminUsers on the "
                           "HOST, which cannot be set from inside a container; ncu being on the "
                           "box does not mean counters are usable"),
        }
    except Exception as exc:  # noqa: BLE001
        result["tiers"] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
    return result


def _measure_task_cost(ref_src: str, device) -> dict:
    """Count the arithmetic and traffic this TASK requires, by running the reference once.

    Two counts from the same dispatch-level view of the program, so they are consistent with each
    other:

      FLOP     via torch.utils.flop_counter.FlopCounterMode -- torch's own accounting, not a
               hand-derived formula. Verified against analytic values for matmul, conv, attention
               and depthwise (ratio 1.0000). It returns 0 for ops with no multiply-accumulate,
               which is the correct roofline answer: a maxpool's ceiling is bandwidth.

      bytes    via a __torch_dispatch__ mode that records every aten op's input and output
               tensors. Module hooks would miss a reference written with functional calls, which
               most KernelBench references are.

    `compulsory_bytes` counts each distinct storage ONCE (by data_ptr), so a tensor read by five
    ops is one unavoidable read -- that is what makes it a floor no implementation can go below.
    `reference_bytes` sums per-op traffic, so the same tensor read five times counts five times;
    the ratio between them is the fusion headroom.
    """
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    notes: list[str] = []

    ref_ctx: dict = {}
    exec(compile(ref_src, "<ref>", "exec"), ref_ctx)  # noqa: S102 — same trust as every eval
    get_inputs = ref_ctx["get_inputs"]
    get_init_inputs = ref_ctx.get("get_init_inputs", lambda: [])
    Model = ref_ctx["Model"]

    with torch.no_grad():
        init_inputs = [x.to(device) if isinstance(x, torch.Tensor) else x
                       for x in get_init_inputs()]
        model = Model(*init_inputs).to(device)
        inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in get_inputs()]

        # Traffic the task cannot avoid: the inputs it is given, the parameters it must read, and
        # the output it must produce. Deduplicated by storage so a shared tensor is one cost.
        seen: set[int] = set()
        compulsory = 0

        def account_once(t) -> int:
            """Bytes for a tensor, counted only the first time its storage is seen.

            Deduplication is by STORAGE, not by tensor object: a view, a transpose and a slice of
            the same buffer are one unavoidable read, and counting them separately would inflate
            the floor above what any implementation could achieve. Every dtype counts -- an int
            index tensor is traffic exactly like a float one.
            """
            if not isinstance(t, torch.Tensor) or t.numel() == 0:
                return 0
            try:
                ptr = t.untyped_storage().data_ptr()
            except Exception:  # noqa: BLE001 — exotic tensors (meta, sparse) have no storage
                return 0
            if ptr in seen:
                return 0
            seen.add(ptr)
            return t.numel() * t.element_size()

        for x in inputs:
            compulsory += account_once(x)
        for p in model.parameters():
            compulsory += account_once(p)
        for b in model.buffers():
            compulsory += account_once(b)

        class ByteCounter(TorchDispatchMode):
            """Sum every dispatched op's tensor traffic. Sees functional calls and module calls
            alike, because it hooks the dispatcher rather than nn.Module boundaries."""

            def __init__(self):
                super().__init__()
                self.total = 0
                self.ops = 0
                # How many dispatched ops returned auxiliary tensors alongside their result.
                # Reported so the exclusion above is visible rather than silent.
                self.aux_outputs = 0

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                out = func(*args, **(kwargs or {}))
                try:
                    self.ops += 1
                    flat = list(args) + list((kwargs or {}).values())
                    for a in flat:
                        if isinstance(a, torch.Tensor):
                            self.total += a.numel() * a.element_size()
                        elif isinstance(a, (list, tuple)):
                            for e in a:
                                if isinstance(e, torch.Tensor):
                                    self.total += e.numel() * e.element_size()
                    # Only the op's PRIMARY output counts as traffic the task requires. Several
                    # aten ops return auxiliary tensors alongside their result -- most visibly
                    # `max_pool2d_with_indices`, which returns (values, indices) -- and those are
                    # implementation bookkeeping, not part of what the task computes. Counting
                    # them made a ONE-op reference report 2.0x its own compulsory traffic
                    # (level1:42 in run-l1-42-20260908-015408: 8564.79 MB against 4286.59 MB),
                    # which `_bottleneck_doc` would have turned into "fusion is your largest
                    # lever" advice on a kernel with nothing whatsoever to fuse.
                    #
                    # Taking the first tensor rather than filtering by name or dtype: the primary
                    # result is first by aten convention, and an int64 index tensor is
                    # indistinguishable from a legitimate integer output on dtype alone.
                    outs = out if isinstance(out, (list, tuple)) else [out]
                    for o in outs[:1]:
                        if isinstance(o, torch.Tensor):
                            self.total += o.numel() * o.element_size()
                    if len(outs) > 1:
                        self.aux_outputs += 1
                except Exception:  # noqa: BLE001 — accounting must never break the forward
                    pass
                return out

        # Warm up outside both modes: the first call may allocate, autotune or compile, and that
        # traffic is not part of the task's cost.
        model(*inputs)
        torch.cuda.synchronize(device)

        counter = ByteCounter()
        try:
            with counter:
                out = model(*inputs)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"byte counting failed: {type(exc).__name__}: {exc}")
            out = model(*inputs)

        # Same rule for the model's own output: the task's result, not an op's bookkeeping.
        final_outs = out if isinstance(out, (list, tuple)) else [out]
        for o in final_outs[:1]:
            compulsory += account_once(o)

        flop_count = 0
        try:
            from torch.utils.flop_counter import FlopCounterMode

            fc = FlopCounterMode(display=False)
            with fc:
                model(*inputs)
            flop_count = int(fc.get_total_flops())
            if flop_count == 0:
                notes.append(
                    "FlopCounterMode reports 0 FLOP: this task has no multiply-accumulate ops "
                    "(e.g. pooling, elementwise, normalization-only). That is the correct "
                    "roofline answer -- its ceiling is bandwidth, not FLOP/s -- NOT a "
                    "measurement failure.")
        except Exception as exc:  # noqa: BLE001
            notes.append(f"FLOP counting unavailable: {type(exc).__name__}: {exc}")

    result_cost = {
        "flop_count": flop_count,
        "compulsory_bytes": compulsory,
        "reference_bytes": counter.total,
        "op_count": counter.ops,
        "notes": notes,
    }
    if counter.aux_outputs:
        # Visible, not silent: an op returning (values, indices) had its auxiliary tensor
        # excluded from the traffic count, and a reader comparing this against a hand
        # calculation needs to know that happened.
        result_cost["aux_output_ops"] = counter.aux_outputs
        notes.append(
            f"{counter.aux_outputs} dispatched op(s) returned auxiliary tensors alongside their "
            f"result (e.g. max_pool2d_with_indices -> values, indices); only the primary output "
            f"is counted as task traffic, since the rest is implementation bookkeeping.")
    return result_cost


def run_task_cost(job: dict) -> dict:
    """Measure the task's required arithmetic and traffic from its REFERENCE.

    On the reference deliberately: these are properties of the TASK, identical for every candidate
    optimizing it, which is what makes them a usable shared denominator. Counting a candidate
    instead would measure the very thing being optimized.
    """
    import torch

    if not torch.cuda.is_available():
        return {"ok": False, "failure_kind": "worker_crash", "log_tail": "no CUDA device"}
    device = _pick_device()
    torch.cuda.set_device(device)
    ref_src = open(job["ref_src_path"], encoding="utf-8").read()
    try:
        cost = _measure_task_cost(ref_src, device)
    except Exception as exc:  # noqa: BLE001 — a diagnostic must not fail the run
        return {"ok": False, "failure_kind": _classify_exception(exc),
                "log_tail": _log_tail(exc)}
    return {"ok": True, "task_cost": cost}


def run_probe_semantics(job: dict) -> dict:
    """Improvement J: report the reference model's runtime eval semantics so the
    agent can reproduce them. Reads the LIVE model object's .training flag (the
    exact state its reference forward runs in), not the source text — so it is
    correct regardless of how the reference is written. Norm layers are found by
    CAPABILITY (running_mean/var buffers) rather than a hardcoded type list, so
    custom BN-like layers are still detected."""
    import torch
    from kernelbench.eval import (
        load_original_model_and_inputs,
        set_seed,
    )

    ref_src = open(job["ref_src_path"], encoding="utf-8").read()
    context: dict = {}
    Model, get_init_inputs, _get_inputs = load_original_model_and_inputs(ref_src, context)
    set_seed(job.get("seed", 0))
    init_inputs = get_init_inputs()
    with torch.no_grad():
        set_seed(job.get("seed", 0))
        ref_model = Model(*init_inputs)

    norm_layers = []
    for m in ref_model.modules():
        has_running = hasattr(m, "running_mean") or hasattr(m, "running_var")
        has_track = hasattr(m, "track_running_stats")
        if not (has_running or has_track):
            continue
        norm_layers.append({
            "type": type(m).__name__,
            "training": bool(getattr(m, "training", False)),
            "has_running_stats": bool(has_running),
            "track_running_stats": (
                bool(m.track_running_stats)
                if hasattr(m, "track_running_stats") else None
            ),
            "momentum": (
                float(m.momentum)
                if getattr(m, "momentum", None) is not None else None
            ),
        })
    return {
        "ok": True,
        "training": bool(ref_model.training),
        "norm_layers": norm_layers,
    }


# The anti-cheat checks we require, stated explicitly rather than inherited from
# KernelBench's STRICT_CHECKS default. Passing this list pins our acceptance criteria: an
# upstream edit to STRICT_CHECKS would otherwise change what we accept with no signal here,
# and the four below are choices we have actually examined:
#
#   code_bypass        bans any `try:` / `except` / bare `pass`. Blocks two real cheats -- a
#                      kernel that falls back to PyTorch inside an exception handler, and a
#                      ModelNew that inherits the reference and `pass`es. Kept STRICT despite
#                      being a blunt regex (it also matches a `try` used for a legitimate
#                      host-side fallback, and the word "pass" inside a string literal, since
#                      KernelBench strips comments but not strings). The contract now warns
#                      about both, which is the cheaper half of the fix.
#   timing_event_patch monkey-patching the timing functions. No legitimate use.
#   thread_injection   threading/multiprocessing. Blocks timing manipulation; also catches an
#                      unused `import threading`, which the contract now mentions.
#   lazy_eval          returning work not yet done, so the timer stops before the compute.
#
# The complement is as deliberate: `torch_computation_ops` and `pytorch_wrap` stay WARNINGS
# (KernelBench's own default), which is what makes calling cuBLAS for a sub-op legal. Those
# warnings are surfaced in the report rather than enforced -- a large regular GEMM is often
# already at the hardware roof, and forbidding the vendor library would cost real time on
# fp32-class tasks (measured on L3:43: cuBLAS beats our hand-written Triton GEMM by 1.33x at
# strict ieee). `stream_injection` and `precision_downgrade` likewise stay warnings: the first
# has legitimate async uses, and the second flags exactly the tensor-core path the contract
# recommends.
REQUIRED_STATIC_CHECKS = ("code_bypass", "timing_event_patch", "thread_injection", "lazy_eval")


def run_static_check(job: dict) -> dict:
    from kernelbench.kernel_static_checker import validate_kernel_static

    code = open(job["kernel_src_path"], encoding="utf-8").read()
    # `forbidden` does NOT need the per-backend implementation check appended --
    # validate_kernel_static adds BACKEND_IMPL_CHECK[backend] itself, so listing it here
    # would only duplicate it.
    valid, errors, warnings = validate_kernel_static(
        code, backend=job["backend"], precision=job["precision"],
        forbidden=list(REQUIRED_STATIC_CHECKS),
    )
    return {
        "ok": valid,
        "compiled": True,
        "correct": valid,
        "errors": errors,
        "warnings": warnings,
        "failure_kind": None if valid else "static_check_failed",
        "log_tail": "; ".join(errors)[-4000:],
    }


def run_baseline(job: dict) -> dict:
    from kernelbench.timing import measure_ref_program_time

    matmul = job.get("matmul_precision")
    if matmul:
        _set_matmul_precision(matmul)
    ref_src = open(job["ref_src_path"], encoding="utf-8").read()
    stats = measure_ref_program_time(
        ref_arch_name="ref",
        ref_arch_src=ref_src,
        num_trials=job["num_trials"],
        timing_method=job["timing_method"],
        use_torch_compile=job["use_torch_compile"],
        precision=job["precision"],
        device=_pick_device(),
    )
    return {"ok": True, "latency_ms": _stats_to_dict(stats)}


def _pick_device():
    import torch

    return torch.device("cuda:0")


def run_eval(job: dict, measure_performance: bool) -> dict:
    import torch
    from kernelbench.eval import eval_kernel_against_ref

    ref_src = open(job["ref_src_path"], encoding="utf-8").read()
    kernel_src = open(job["kernel_src_path"], encoding="utf-8").read()

    exec_result = eval_kernel_against_ref(
        original_model_src=ref_src,
        custom_model_src=kernel_src,
        seed_num=job["seed"],
        num_correct_trials=job["num_correct_trials"],
        num_perf_trials=job["num_perf_trials"],
        measure_performance=measure_performance,
        timing_method=job["timing_method"],
        build_dir=job.get("build_dir"),
        device=torch.device("cuda:0"),
        backend=job["backend"],
        precision=_dtype(job["precision"]),
        check_for_excessive_speedup=True,
        excessive_speedup_threshold=job.get("excessive_speedup_threshold", 10.0),
    )
    if exec_result is None:
        return {
            "ok": False,
            "compiled": False,
            "correct": False,
            "failure_kind": "compile_error",
            "log_tail": "eval returned None (lock-file/concurrent-compile error); retryable",
            "retryable": True,
        }

    metadata = {k: str(v) for k, v in (exec_result.metadata or {}).items()}
    compiled = bool(exec_result.compiled)
    correct = bool(exec_result.correctness)
    result: dict = {
        "ok": compiled and correct,
        "compiled": compiled,
        "correct": correct,
        "metadata": metadata,
        "failure_kind": None,
        "log_tail": "",
    }
    if not result["ok"]:
        kind, detail = _classify_eval_failure(exec_result.metadata or {}, compiled, correct)
        result["failure_kind"] = kind
        result["log_tail"] = detail[-4000:]
        return result

    if measure_performance:
        stats = exec_result.runtime_stats or {}
        result["latency_ms"] = _stats_to_dict(stats)
        result["excessive_speedup"] = bool((exec_result.metadata or {}).get("excessive_speedup"))

    # Launch-overhead measurement, gated by the caller. Requested only on `full_eval` and
    # baselines -- never on the 20-sample tuning trials, where it would add ~150 extra model
    # calls to every one of hundreds of trials. On the paths that do ask for it the run is
    # already doing 100 timed samples, so the marginal cost is close to nothing, and it is the
    # only place the number is needed: the analyst compares a candidate against the baseline,
    # both of which come from these paths.
    _attach_launch_overhead(job, result, kernel_src, ref_src)

    if _wants_kernel_metadata(job):
        if job["backend"] == "triton":
            try:
                result["triton"] = _extract_triton_metadata(kernel_src, ref_src, 0)
            except Exception as exc:  # noqa: BLE001 — best-effort, never fail the eval
                result["triton"] = None
                result["triton_error"] = str(exc)[-1000:]
        else:
            try:
                launched = _launched_kernel_names(kernel_src, ref_src, 0)
                result["cubin"] = _extract_cubin_metadata(launched_names=launched or None)
                result["cubin_launched_observed"] = launched
            except Exception as exc:  # noqa: BLE001 — best-effort, never fail the eval
                result["cubin"] = None
                result["cubin_error"] = str(exc)[-1000:]
    return result


def _dtype(precision: str):
    import torch

    return {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[precision]


def _set_matmul_precision(mode: str) -> None:
    """Switch the fp32 matmul path: 'tf32' (TensorFloat-32) or 'ieee' (full fp32).

    Prefers the torch 2.9+ fp32_precision API and falls back to the older
    allow_tf32 flags on older torch (both silenced under the try)."""
    import torch

    want_tf32 = (mode == "tf32")
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32" if want_tf32 else "ieee"
        torch.backends.cudnn.conv.fp32_precision = "tf32" if want_tf32 else "ieee"
    except (AttributeError, RuntimeError):
        torch.backends.cuda.matmul.allow_tf32 = want_tf32
        torch.backends.cudnn.allow_tf32 = want_tf32
    torch.set_float32_matmul_precision("high" if want_tf32 else "highest")


def _cosine_similarity(ref32, got32) -> float:
    """Cosine similarity that does not overflow on large-magnitude outputs.

    Computing dot()/norm() in fp32 overflows to inf whenever the output magnitude
    exceeds ~1.8e19 (fp32 max is 3.4e38, and the products are squares): level3/48's
    outputs reach 1e22, so dot and both norms became inf, cos became inf/inf = nan, and
    `nan >= cosine_min` is False. That silently rejected candidates whose accuracy was
    excellent -- measured frac_within_1%=0.999983 with median relative error 4e-7,
    against a 0.99 gate. Accumulate in float64 and scale by the larger norm first, so
    the gate judges accuracy instead of dynamic range.
    """
    import torch

    a = ref32.flatten().double()
    b = got32.flatten().double()
    scale = max(a.abs().max().item(), b.abs().max().item())
    if scale == 0.0:
        return 1.0  # both identically zero
    a = a / scale
    b = b / scale
    denom = (a.norm() * b.norm()).item()
    if denom == 0.0 or not math.isfinite(denom):
        return float("nan")
    cos = torch.dot(a, b).item() / denom
    return max(-1.0, min(1.0, cos))


def _relaxed_close(ref, got, elem_tol: float, pass_frac: float, cosine_min: float) -> bool:
    """Improvement A slack gate (mirrors kernelfoundry all_close_with_slack + cosine):
    accept when >pass_frac of elements are within elem_tol relative error AND the
    flattened cosine similarity clears cosine_min. Shape mismatch never passes."""
    import torch

    if ref.shape != got.shape:
        return False
    ref32 = ref.float()
    got32 = got.float()
    rel = (ref32 - got32).abs() / (ref32.abs() + 1e-7)
    frac_ok = (rel < elem_tol).float().mean().item()
    if frac_ok <= pass_frac:
        return False
    cos = _cosine_similarity(ref32, got32)
    if math.isnan(cos):
        # Degenerate cosine (all-zero or non-finite output). Fall back to the
        # elementwise verdict rather than rejecting an otherwise-passing candidate.
        return bool(torch.isfinite(got32).all().item())
    return cos >= cosine_min


_LOW_PRECISION_VALUES = ("fp16", "bf16", "float16", "bfloat16", "half")
_LOW_PRECISION_TOKENS = ("tl.float16", "tl.bfloat16", ".half(", "torch.float16",
                         "torch.bfloat16")


def _computes_low_precision(kernel_src: str) -> bool:
    """Does this materialized candidate actually COMPUTE in fp16/bf16?

    Only used to choose which slack multiplier the fp64 relative gate applies (2.0 vs
    3.0), mirroring the driver's `_detect_candidate_precision` on the signals available
    inside the worker.

    Reads the PARAMS literal FIRST and treats it as authoritative, because a plain token
    scan over the whole file is wrong on the common shape. Candidates keep every dtype
    branch in the kernel body:

        if COMPUTE_DTYPE == "fp16":    ... tl.float16 ...
        elif COMPUTE_DTYPE == "bf16": ... tl.bfloat16 ...
        elif COMPUTE_DTYPE == "tf32": ...

    `COMPUTE_DTYPE` is a `tl.constexpr`, so exactly one branch survives compilation --
    but the file still contains all of them. Scanning the text granted the wider
    low-precision multiplier to a tf32 candidate (verified live: multiplier 3.0 where
    2.0 was intended), and would do the same for an ieee one. The materialized PARAMS
    holds the tuner's actual choice, so it decides.

    Falls back to the token scan only when there is no dtype-valued knob at all -- a
    candidate that hardcodes its cast has no knob to read, and there the tokens are the
    live code.
    """
    text = kernel_src.lower()
    params_block = ""
    if "params" in text:
        head = text.split("params", 1)[1]
        brace = head.find("{")
        close = head.find("}", brace) if brace >= 0 else -1
        if brace >= 0 and close > brace:
            params_block = head[brace:close + 1]

    if params_block:
        # A dtype-valued knob is present iff one of the recognized precision names
        # appears as a value in the literal. If so, it alone decides.
        declares_dtype = any(f'"{v}"' in params_block or f"'{v}'" in params_block
                             for v in _LOW_PRECISION_VALUES + ("tf32", "ieee",
                                                               "float32", "fp32"))
        if declares_dtype:
            return any(f'"{v}"' in params_block or f"'{v}'" in params_block
                       for v in _LOW_PRECISION_VALUES)

    return any(tok in text for tok in _LOW_PRECISION_TOKENS)


def _rmse(ref, got) -> float:
    """Root-mean-square error in fp64. Matches torch._dynamo.utils.rmse."""
    import torch

    return torch.sqrt(torch.mean(torch.square(ref.double() - got.double()))).item()


def _fp64_relative_ok(golden, ref, got, multiplier: float, tol: float) -> tuple[bool, dict]:
    """Accept when the candidate is no worse than `multiplier` x the REFERENCE's own error.

    This is torch._dynamo.utils.same()'s fp64 path, which is what PyTorch's own accuracy
    checker falls back to when plain allclose fails:

        ref_error = rmse(fp64_ref, ref)
        res_error = rmse(fp64_ref, res)
        passes    = res_error <= multiplier * ref_error + tol / 10

    Why this shape rather than our absolute gate:

    - The floor is measured against TRUTH (an fp64 evaluation of the reference), not
      between two imprecise results. On level3/21 the reference's own ieee-vs-tf32 spread
      puts 4.5% of elements outside a 1% tolerance, so an absolute `frac > 0.99` gate is
      unreachable for any low-precision candidate however correct it is. Comparing to
      fp64 separates "this kernel is wrong" from "this task cannot be computed that
      precisely".
    - RMSE is a single aggregate dominated by the large deviations, so it needs one
      threshold rather than one per metric. A candidate that is fine on most elements but
      badly wrong on a few has a large RMSE and still fails.
    - `multiplier > 1` is deliberate and comes from torch, whose comment explains it:
      AMP-vs-fp32 end-to-end accuracy differs by <0.1% while failing a strict check, so
      "it's possible that the correctness check failures for these models are false
      alarms. We use multiplier of 3 instead of 2 to avoid these false alarms."

    Returns (verdict, metrics) so the numbers reach the failure message either way.
    """
    ref_error = _rmse(golden, ref)
    res_error = _rmse(golden, got)
    threshold = multiplier * ref_error + tol / 10.0
    metrics = {
        "reference_rmse_vs_fp64": f"{ref_error:.4e}",
        "candidate_rmse_vs_fp64": f"{res_error:.4e}",
        "multiplier": multiplier,
        "threshold": f"{threshold:.4e}",
        "ratio_to_reference": (f"{res_error / ref_error:.3f}" if ref_error > 0 else "inf"),
    }
    if math.isnan(ref_error) or math.isnan(res_error):
        # A nan on either side makes the comparison meaningless; defer to the
        # absolute gate rather than passing on an unreadable number.
        metrics["skipped"] = "nan in rmse"
        return False, metrics
    return res_error <= threshold, metrics


def _relaxed_metrics(ref, got) -> dict:
    """The numbers the relaxed gate actually decides on, for the failure message.

    Reporting only max-abs-diff is diagnostically useless on a task whose outputs span
    many orders of magnitude: on level3/48 the reference's OWN fp32-vs-fp64 max-abs-diff
    is 1.5e16, so a candidate rejected at "max abs diff 7.3e15" may be well inside the
    reference's own noise while the message reads as catastrophic. On L3:48 that misled
    the repair agent into inventing sign-convention bugs and flip-flopping between
    exp(A) and exp(-exp(A)). Always report frac-within-tol and cosine (the actual gate
    criteria) plus where the error sits relative to the output's own magnitude.
    """
    import torch

    if ref.shape != got.shape:
        return {"shape_ref": tuple(ref.shape), "shape_got": tuple(got.shape)}
    r = ref.float()
    g = got.float()
    # A NaN/Inf ANYWHERE in the candidate's output poisons every derived statistic
    # (median, max, cosine all come back nan), so the message would read "nan" five
    # times without saying why -- indistinguishable from a metric that overflowed, and
    # useless to a repair agent. Count and locate the non-finite values explicitly, and
    # compute the remaining statistics over the finite subset so they stay informative.
    bad = ~torch.isfinite(g)
    n_bad = int(bad.sum().item())
    out: dict = {}
    if n_bad:
        n_nan = int(torch.isnan(g).sum().item())
        out["NON_FINITE_OUTPUT"] = (
            f"{n_bad} of {g.numel()} candidate values are not finite "
            f"({n_nan} NaN, {n_bad - n_nan} +/-Inf) -- THIS is the failure; the "
            f"statistics below are computed over the finite values only"
        )
        finite = ~bad
        r = r[finite]
        g = g[finite]
        if g.numel() == 0:
            out["frac_within_tol"] = 0.0
            return out
    rel = (r - g).abs() / (r.abs() + 1e-7)
    cos = _cosine_similarity(r, g)
    # torch.quantile refuses inputs above ~16M elements, and these tensors are much
    # larger (level3/48's output is 2048*128*8*64 = 134M), so it raised inside the
    # failure-reporting path and turned a correctness mismatch into a runtime_error --
    # a diagnostic that destroyed the diagnosis. Sort a bounded random sample instead:
    # a p99 of the error distribution needs a representative sample, not every element.
    flat = rel.flatten()
    try:
        if flat.numel() > 1_000_000:
            idx = torch.randint(0, flat.numel(), (1_000_000,), device=flat.device)
            sample = flat[idx]
        else:
            sample = flat
        p99 = f"{sample.sort().values[int(sample.numel() * 0.99)].item():.3e}"
    except Exception:  # noqa: BLE001 - never let reporting break the report
        p99 = "n/a"
    # frac_within_tol must be reported against the FULL output, not the finite subset:
    # a kernel that is perfect on 91% of elements and NaN on the rest has not passed
    # 91% of the gate, it has failed. Non-finite values count as outside tolerance.
    frac = (rel < 0.01).float().sum().item() / (rel.numel() + n_bad)
    out.update({
        "frac_within_tol": round(frac, 6),
        "cosine": ("nan" if math.isnan(cos) else round(cos, 8)),
        "median_rel_err": f"{rel.median().item():.3e}",
        "p99_rel_err": p99,
        "max_abs_diff": f"{(r - g).abs().max().item():.3e}",
        "ref_absmax": f"{r.abs().max().item():.3e}",
        "ref_absmedian": f"{r.abs().median().item():.3e}",
    })
    return out


def run_relaxed_correctness(job: dict) -> dict:
    """Dual-precision witness correctness (improvement A). Reuses KernelBench's
    model/input loaders for byte-identical input generation, but computes the
    reference at BOTH tf32 and ieee fp32 and accepts the kernel if it matches
    EITHER under the relaxed slack gate. Does NOT time — correctness only."""
    import torch
    from kernelbench.eval import (
        _process_input_tensor,
        graceful_eval_cleanup,
        load_custom_model,
        load_custom_model_with_tempfile,
        load_original_model_and_inputs,
        set_seed,
    )

    ref_src = open(job["ref_src_path"], encoding="utf-8").read()
    kernel_src = open(job["kernel_src_path"], encoding="utf-8").read()
    backend = job["backend"]
    precision = _dtype(job["precision"])
    device = torch.device("cuda:0")
    seed = job["seed"]
    num_trials = job["num_correct_trials"]
    elem_tol = job.get("relaxed_elem_tol", 0.01)
    pass_frac = job.get("relaxed_pass_frac", 0.99)
    cosine_min = job.get("cosine_min", 0.99985)
    fp64_gate = bool(job.get("fp64_relative_gate", False))
    fp64_mult_lowp = float(job.get("fp64_rel_multiplier_lowp", 3.0) or 3.0)
    fp64_mult = float(job.get("fp64_rel_multiplier", 2.0) or 2.0)

    context: dict = {}
    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(ref_src, context)
    set_seed(seed)
    init_inputs = get_init_inputs()
    init_inputs = [_process_input_tensor(x, device, backend, precision) for x in init_inputs]
    with torch.no_grad():
        set_seed(seed)
        ref_model = Model(*init_inputs)

    tempfile = None
    try:
        if backend.lower() in ("triton", "tilelang", "cute"):
            ModelNew, tempfile = load_custom_model_with_tempfile(kernel_src, "ModelNew")
        else:
            ModelNew = load_custom_model(kernel_src, context, job.get("build_dir"))
        torch.cuda.synchronize(device=device)
    except Exception as exc:  # noqa: BLE001 — compile failure is a first-class outcome
        graceful_eval_cleanup(context, device, tempfile)
        detail = str(exc)
        if "lock" in detail or "No such file or directory" in detail:
            return {"ok": False, "compiled": False, "correct": False,
                    "failure_kind": "compile_error", "retryable": True,
                    "log_tail": detail[-4000:]}
        return {"ok": False, "compiled": False, "correct": False,
                "failure_kind": "compile_error", "log_tail": detail[-4000:]}

    # ModelNew is the class; instantiate it once (seeded, same as KernelBench) — a
    # failure here is a runtime error, not a compile error.
    try:
        with torch.no_grad():
            set_seed(seed)
            custom_model = ModelNew(*init_inputs)
    except Exception as exc:  # noqa: BLE001
        graceful_eval_cleanup(context, device, tempfile)
        return {"ok": False, "compiled": True, "correct": False,
                "failure_kind": _classify_exception(exc), "log_tail": _log_tail(exc)}

    # Same deterministic per-trial seed sequence as KernelBench run_and_check_correctness.
    torch.manual_seed(seed)
    trial_seeds = [torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(num_trials)]

    # fp64 golden reference model, built once per job. Cheap relative to the trials:
    # one extra module instantiation, and one fp64 forward per trial only when the
    # relaxed gate has already failed. If fp64 is unsupported for an op or OOMs, the
    # relative arm is simply unavailable and the absolute gate decides alone -- which is
    # also what torch does (it falls back to cosine, which we already require).
    golden_model = None
    fp64_unavailable = ""
    # Which multiplier applies is decided by the precision the candidate COMPUTES in,
    # which is NOT its output dtype. torch's same() keys off res.dtype because torchbench
    # casts the whole model, so a low-precision run returns low-precision tensors. Here
    # only the dot is cast -- `tl.dot(a.to(bf16), ...)` with an fp32 accumulator returns
    # float32 -- so keying off the output dtype left the low-precision multiplier
    # permanently dead (verified live: a bf16 candidate was scored with 2.0, not 3.0).
    # The MATERIALIZED source carries the answer, since the tuner's chosen value is
    # substituted into the PARAMS literal before the file reaches this worker.
    fp64_mult_effective = fp64_mult_lowp if _computes_low_precision(kernel_src) else fp64_mult
    if fp64_gate:
        try:
            with torch.no_grad():
                set_seed(seed)
                golden_model = Model(*[
                    x.double() if isinstance(x, torch.Tensor) and x.is_floating_point()
                    else x for x in init_inputs
                ]).to(device=device, dtype=torch.float64)
        except Exception as exc:  # noqa: BLE001 -- absence of fp64 is not a failure
            golden_model = None
            fp64_unavailable = f"{type(exc).__name__}: {exc}"

    pass_count = 0
    last_detail = ""
    fp64_metrics_last: dict = {}
    # How many trials were accepted ONLY because of the fp64 relative arm. This is the
    # experiment's dependent variable: without it, the fp64 numbers reach the log only
    # when the arm ALSO failed, so the cases it was added to admit would be invisible.
    fp64_rescued = 0
    fp64_rescue_metrics: dict = {}
    try:
        with torch.no_grad():
            for trial in range(num_trials):
                ts = trial_seeds[trial]
                set_seed(ts)
                inputs = get_inputs()
                inputs = [_process_input_tensor(x, device, backend, precision) for x in inputs]

                set_seed(ts); model = ref_model.to(device=device, dtype=precision)
                set_seed(ts); model_new = custom_model.to(device=device, dtype=precision)

                _set_matmul_precision("tf32")
                out_ref_tf32 = model(*inputs); torch.cuda.synchronize(device=device)
                _set_matmul_precision("ieee")
                out_ref_ieee = model(*inputs); torch.cuda.synchronize(device=device)

                out_kernel = model_new(*inputs); torch.cuda.synchronize(device=device)

                ok = (_relaxed_close(out_ref_tf32, out_kernel, elem_tol, pass_frac, cosine_min)
                      or _relaxed_close(out_ref_ieee, out_kernel, elem_tol, pass_frac, cosine_min))

                # Second acceptance path, and ONLY reachable when the absolute gate has
                # already failed -- so the fp64 forward stays off the happy path and no
                # previously-accepted candidate can become a rejection.
                #
                # This is torch._dynamo.utils.same()'s fp64 arm: judge the candidate
                # against the REFERENCE's own distance from truth rather than against a
                # fixed tolerance that the task itself may not meet. All three L3 tasks
                # measure ieee-vs-tf32 floors below the 0.99 the absolute gate demands
                # (0.9554 / 0.9767 / 0.9778), so a correct low-precision kernel could not
                # pass on those tasks at all.
                if (not ok and golden_model is not None
                        and out_ref_ieee.shape == out_kernel.shape):
                    try:
                        golden = golden_model(*[
                            x.double() if isinstance(x, torch.Tensor)
                            and x.is_floating_point() else x for x in inputs
                        ])
                        torch.cuda.synchronize(device=device)
                        # tf32 is the reference the harness compares against and the
                        # noisier of the two, so it sets the floor.
                        #
                        # Multiplier decided above from the materialized source; the
                        # output dtype is still honoured when it is itself low precision.
                        mult = fp64_mult_effective
                        if out_kernel.dtype in (torch.float16, torch.bfloat16):
                            mult = fp64_mult_lowp
                        ok, fp64_metrics_last = _fp64_relative_ok(
                            golden, out_ref_tf32, out_kernel, mult, elem_tol)
                        del golden
                        if ok:
                            # Accepted by the relative arm alone: the absolute gate had
                            # already failed on this trial.
                            fp64_rescued += 1
                            fp64_rescue_metrics = dict(fp64_metrics_last)
                    except Exception as exc:  # noqa: BLE001 -- absence of fp64 is not a failure
                        fp64_unavailable = f"{type(exc).__name__}: {exc}"
                        golden_model = None

                if ok:
                    pass_count += 1
                else:
                    if out_ref_ieee.shape != out_kernel.shape:
                        last_detail = (f"shape mismatch: ref {tuple(out_ref_ieee.shape)} "
                                       f"vs kernel {tuple(out_kernel.shape)}")
                    else:
                        # Report against BOTH witnesses (the gate accepts either) with the
                        # criteria it actually uses, and include the reference's own
                        # fp32-vs-tf32 spread as the task's noise floor: a candidate whose
                        # error is at or below that floor is not "wrong by 1e16", it is
                        # inside the reference's own reordering noise.
                        #
                        # Wrapped because a diagnostic must never destroy the diagnosis:
                        # torch.quantile's ~16M-element limit raised in here and turned a
                        # correctness_mismatch into an opaque runtime_error, losing the
                        # mismatch entirely. Any failure to compute the rich detail falls
                        # back to the bare numbers rather than propagating.
                        try:
                            m_ieee = _relaxed_metrics(out_ref_ieee, out_kernel)
                            m_tf32 = _relaxed_metrics(out_ref_tf32, out_kernel)
                            floor = _relaxed_metrics(out_ref_ieee, out_ref_tf32)
                            last_detail = (
                                f"relaxed mismatch on trial {trial}; gate needs "
                                f"frac_within_tol>{pass_frac} AND cosine>={cosine_min}\n"
                                f"  vs ieee ref: {m_ieee}\n"
                                f"  vs tf32 ref: {m_tf32}\n"
                                f"  reference's OWN ieee-vs-tf32 spread (task noise "
                                f"floor, NOT a bug): {floor}"
                            )
                            if fp64_metrics_last:
                                last_detail += (
                                    "\n  fp64-relative arm ALSO failed (the candidate "
                                    "must be within multiplier x the reference's own "
                                    "rmse against an fp64 golden reference): "
                                    f"{fp64_metrics_last}"
                                )
                            elif fp64_gate and fp64_unavailable:
                                last_detail += (
                                    "\n  fp64-relative arm unavailable: "
                                    f"{fp64_unavailable}"
                                )
                        except Exception as diag_exc:  # noqa: BLE001
                            md = (out_ref_ieee.float() - out_kernel.float()
                                  ).abs().max().item()
                            last_detail = (
                                f"relaxed mismatch on trial {trial} (max abs diff "
                                f"{md:.3e}); gate needs frac_within_tol>{pass_frac} AND "
                                f"cosine>={cosine_min}. Detailed metrics unavailable: "
                                f"{type(diag_exc).__name__}: {diag_exc}"
                            )
    except Exception as exc:  # noqa: BLE001
        kind = _classify_exception(exc)
        graceful_eval_cleanup(context, device, tempfile)
        return {"ok": False, "compiled": True, "correct": False,
                "failure_kind": kind, "log_tail": _log_tail(exc)}

    correct = pass_count == num_trials

    # Timing (only if correct and requested). Same job as the merged strict eval,
    # but WITHOUT re-running strict correctness — we already judged correctness with
    # the dual-witness gate above; KernelBench's perf path would re-fail a tf32
    # candidate under its strict allclose. Time under ieee precision (honest fp32).
    latency_ms = None
    ref_latency_ms = None
    num_perf = job.get("num_perf_trials", 0)
    if correct and num_perf and num_perf > 0:
        try:
            from kernelbench.timing import get_timing_stats, time_execution_with_cuda_event

            _set_matmul_precision("ieee")
            set_seed(seed)
            perf_inputs = get_inputs()
            perf_inputs = [_process_input_tensor(x, device, backend, precision)
                           for x in perf_inputs]
            model_new = custom_model.to(device=device, dtype=precision)
            with torch.no_grad():
                elapsed = time_execution_with_cuda_event(
                    model_new, perf_inputs, num_warmup=3, num_trials=num_perf,
                    verbose=False, device=device)
            latency_ms = _stats_to_dict(get_timing_stats(elapsed, device=device), elapsed)

            # Anti-reward-hacking: KernelBench's own excessive-speedup check lives in
            # its strict eval path, which this relaxed handler deliberately bypasses
            # (that path re-fails a legitimately tf32 candidate under strict allclose).
            # So the guard has to be reproduced here, or a candidate that skips the
            # real work — caching an output, eliding the compute — is reported as a
            # spectacular win with nothing flagging it.
            #
            # The threshold is 10x, so this screen needs an order-of-magnitude estimate
            # of the reference, not a precise measurement: a few samples suffice and
            # keep the added cost off the hot path (a full re-timing of the reference
            # on every trial would roughly double the timed work per job).
            ref_trials = max(3, min(int(num_perf), 10))
            ref_model = Model(*init_inputs).to(device=device, dtype=precision)
            set_seed(seed)
            with torch.no_grad():
                ref_elapsed = time_execution_with_cuda_event(
                    ref_model, perf_inputs, num_warmup=3, num_trials=ref_trials,
                    verbose=False, device=device)
            ref_latency_ms = _stats_to_dict(
                get_timing_stats(ref_elapsed, device=device), ref_elapsed)
            # The guard compares against a median, not a mean: a single scheduling stall must
            # not decide a verdict. Observed live on L3:48: a 10-sample reference came
            # back mean=609ms with min=29.8ms, max=5760ms, std=1720ms -- one ~5.8s
            # outlier dragged the mean 20x, producing a bogus 115x "speedup".
            # `_stats_to_dict` now computes that median (and keeps the samples, which the
            # bespoke block here promised "for the record" but never actually did), so this
            # side needs no special case -- the candidate above gets the same treatment.
            del ref_model
        except Exception as exc:  # noqa: BLE001 — timing failure is a runtime failure
            kind = _classify_exception(exc)
            graceful_eval_cleanup(context, device, tempfile)
            return {"ok": False, "compiled": True, "correct": True,
                    "failure_kind": kind, "log_tail": _log_tail(exc)}

    graceful_eval_cleanup(context, device, tempfile)
    result: dict = {
        "ok": correct, "compiled": True, "correct": correct,
        "failure_kind": None if correct else "correctness_mismatch",
        "log_tail": "" if correct else last_detail[-4000:],
        "correctness_mode": "dual_witness_relaxed",
        "trials_passed": pass_count, "trials_total": num_trials,
    }
    # Report the relative arm's contribution whenever the gate was enabled, including a
    # plain zero. "Enabled and rescued nothing" and "not enabled" are different findings,
    # and only an explicit field can tell them apart after the fact.
    if fp64_gate:
        result["fp64_gate_enabled"] = True
        result["fp64_rescued_trials"] = fp64_rescued
        if fp64_rescue_metrics:
            result["fp64_rescue_metrics"] = fp64_rescue_metrics
        if fp64_unavailable:
            result["fp64_unavailable"] = fp64_unavailable
    if latency_ms is not None:
        result["latency_ms"] = latency_ms
    if ref_latency_ms is not None:
        result["ref_latency_ms"] = ref_latency_ms
        thr = float(job.get("excessive_speedup_threshold", 10.0) or 10.0)
        cand_mean = latency_ms.get("mean", -1.0) if latency_ms else -1.0
        # Prefer the median reference (robust to a single scheduling stall); fall back to
        # the mean when the median is unavailable.
        ref_mean = ref_latency_ms.get("median") or ref_latency_ms.get("mean", -1.0)
        if cand_mean > 0 and ref_mean > 0:
            speedup = ref_mean / cand_mean
            result["speedup_vs_ref_in_worker"] = speedup
            if speedup >= thr:
                # The guard exists to catch work-SKIPPING (a cached output, an elided
                # compute), which shows up as an implausible speedup. It is NOT a cap on
                # legitimate speed. A candidate that passed every correctness trial has
                # demonstrably produced the reference's values on fresh inputs, so a
                # hard fail here would discard a verified-correct kernel for the offence
                # of being fast -- which is the entire point of the search.
                #
                # On L3:48 that is exactly what happened: four trials of cand-c18203b6
                # were rejected at 11.1x-13.9x with correct=True and trials_passed=3/3,
                # while a neighbouring point at 8.95x was accepted. Same kernel, verdict
                # decided by which side of 10x the noise landed -- and the discarded
                # points were the FASTEST ones, biasing the reported optimum downward.
                #
                # So: correctness decides acceptance, and the speedup only raises a flag.
                # A fast candidate that FAILED correctness is still a hard failure (the
                # timing-cheat fixture caches on tensor identity, so its correctness
                # trials with fresh inputs do not pass).
                result["excessive_speedup"] = True
                result["suspicious_speedup"] = speedup
                if not correct:
                    result["ok"] = False
                    result["failure_kind"] = "excessive_speedup"
                    result["log_tail"] = (
                        f"excessive speedup {speedup:.1f}x vs the reference "
                        f"({ref_mean:.3f} ms -> {cand_mean:.3f} ms) exceeds the "
                        f"{thr:.0f}x threshold AND correctness did not pass "
                        f"({pass_count}/{num_trials} trials); treated as not performing "
                        "the reference computation"
                    )
                else:
                    # Verified correct: keep the measurement, but record the flag so the
                    # report and the final re-eval can scrutinise it.
                    result["excessive_speedup_note"] = (
                        f"{speedup:.1f}x vs reference ({ref_mean:.3f} ms -> "
                        f"{cand_mean:.3f} ms) exceeds the {thr:.0f}x plausibility "
                        f"threshold, but all {pass_count}/{num_trials} correctness "
                        "trials passed on fresh inputs; accepted and flagged for review"
                    )
            else:
                result["excessive_speedup"] = False
    # P2 cause (b): this handler used to mention neither `measure_launch_overhead` nor
    # `launch_overhead`, while BOTH L3 configs route here (correctness_mode
    # dual_witness_relaxed). So the flag was honoured on a path the experiments never take.
    if correct:
        _attach_launch_overhead(job, result, kernel_src, ref_src)
    if correct and _wants_kernel_metadata(job):
        if backend == "triton":
            try:
                result["triton"] = _extract_triton_metadata(kernel_src, ref_src, 0)
            except Exception as exc:  # noqa: BLE001 — metadata is best-effort
                result["triton"] = None
                result["triton_error"] = str(exc)[-1000:]
        else:
            # Every non-triton backend compiles through nvcc to a cubin, so ONE reader serves
            # cuda, cutlass and cute. Before this, `backend == "triton"` gated the whole block
            # and a cuda candidate carried no resources at all.
            try:
                launched = _launched_kernel_names(kernel_src, ref_src, 0)
                result["cubin"] = _extract_cubin_metadata(launched_names=launched or None)
                result["cubin_launched_observed"] = launched
            except Exception as exc:  # noqa: BLE001 — best-effort, never fail the eval
                result["cubin"] = None
                result["cubin_error"] = str(exc)[-1000:]
    return result


HANDLERS = {
    "env_probe": run_env_probe,
    "probe_semantics": run_probe_semantics,
    "static_check": run_static_check,
    "baseline": run_baseline,
    "eval_correctness": lambda job: run_eval(job, measure_performance=False),
    "eval_perf": lambda job: run_eval(job, measure_performance=True),
    "eval_correctness_relaxed": run_relaxed_correctness,
    "calibrate": run_calibrate,
    "task_cost": run_task_cost,
    "compile_probe": run_compile_probe,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    result: dict
    try:
        _ensure_optional_deps()
        # Before any handler runs: make KernelBench retain its raw timing samples, so every
        # timing path can report a median rather than only the paths that time the model
        # themselves. Installed here rather than per-handler so no future handler can miss it.
        # Idempotent and best-effort -- a failure leaves timing exactly as KernelBench does it.
        samples_captured = capture_timing_samples()
        with open(args.job, encoding="utf-8") as f:
            job = json.load(f)
        handler = HANDLERS.get(job.get("job_type"))
        if handler is None:
            result = {
                "ok": False,
                "failure_kind": "worker_crash",
                "log_tail": f"unknown job_type: {job.get('job_type')!r}",
            }
        else:
            result = handler(job)
        # Recorded so a run whose trials lack a median can be told apart from one where the
        # interception failed -- otherwise a silent downgrade to the mean objective looks
        # identical to a task that legitimately has no samples.
        if isinstance(result, dict):
            result.setdefault("timing_samples_captured", samples_captured)
    except BaseException as exc:  # noqa: BLE001 — always write a result
        result = {
            "ok": False,
            "compiled": False,
            "correct": False,
            "failure_kind": _classify_exception(exc),
            "log_tail": _log_tail(exc),
        }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
