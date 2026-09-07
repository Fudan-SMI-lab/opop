"""Build and measure an external .cu candidate against the KernelBench reference, honestly.

Answers one question: does the claimed speedup reproduce, and against WHICH baseline?

The external manifest reports `ref_speed: 0.0903 ms` for eager/TF32 on an Ada card. Our own
harness measures the same reference at 0.0375 ms on a 4090 -- a 2.4x gap that, if real, means
their speedup ratio is inflated by the same factor no matter how good the kernel is. So this
script measures FOUR things on one card in one process:

  1. the reference, timed the way THEIR harness reports it (plain wall-clock loop, no L2 flush)
  2. the reference, timed the way OUR harness does (CUDA events + per-trial L2 flush)
  3. their kernel, timed both ways
  4. correctness against the reference, before any timing is believed

Both timing conventions are run against both kernels, so a difference in the RATIO can be
attributed to the method rather than the hardware. Measuring their kernel with our convention
and ours with theirs is the only way to separate "their kernel is faster" from "their harness
counts differently".

FALLBACK DETECTION: their kernel has an exact-ATen fallback behind a strict gate. A fallback
that silently triggers would be correct AND slow-but-plausible, so the fast path is verified
with a deliberate gate violation: a non-contiguous input must produce the SAME numbers (proving
the fallback works) while a contiguous one must be measurably faster (proving the fast path is
what we timed).

Usage:
    python measure_external_candidate.py --cu <file.cu> [--name cells_27]
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

M, K, N, G = 128, 512, 1024, 32
EPS = 1e-5


def reference(x, w, wb, bias, gnw, gnb):
    z = F.linear(x, w, wb)
    h = torch.sigmoid(z) * z
    t = h + bias
    return F.group_norm(t, G, gnw, gnb, EPS)


def time_wallclock(fn, n=200, warmup=30):
    """Their convention: plain loop, one synchronize at the end of each timed call."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) * 1e3)
    return out


def time_cuda_events_l2flush(fn, n=200, warmup=30):
    """Our convention: CUDA events, with KernelBench's 256 MB L2 flush enqueued per trial."""
    flush = torch.empty(int(256e6 // 4), dtype=torch.float32, device="cuda")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(n):
        flush.zero_()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e))
    return out


def stats(xs):
    xs = sorted(xs)
    return {
        "mean": statistics.mean(xs), "median": statistics.median(xs),
        "min": xs[0], "max": xs[-1],
        "std": statistics.pstdev(xs) if len(xs) > 1 else 0.0, "n": len(xs),
    }


def show(label, s):
    print(f"  {label:34} mean {s['mean']*1e3:8.2f} us   median {s['median']*1e3:8.2f}   "
          f"min {s['min']*1e3:8.2f}   max {s['max']*1e3:8.2f}   std {s['std']*1e3:6.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cu", required=True)
    ap.add_argument("--name", default="ext")
    ap.add_argument("--trials", type=int, default=200)
    args = ap.parse_args()

    p = torch.cuda.get_device_properties(0)
    print(f"device: {p.name} sm_{p.major}{p.minor}   torch {torch.__version__}")
    print(f"shape: M={M} K={K} N={N} G={G}  fp32")
    print()

    torch.manual_seed(42)
    x = torch.randn(M, K, device="cuda")
    w = torch.randn(N, K, device="cuda")
    wb = torch.randn(N, device="cuda")
    bias = torch.randn(N, device="cuda")
    gnw = torch.randn(N, device="cuda")
    gnb = torch.randn(N, device="cuda")

    print("building the external extension (this takes a minute)...")
    mod = load(name=f"ext_{args.name}", sources=[args.cu], verbose=False,
               extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"])
    print("build OK")
    print()

    # ---- correctness FIRST, in both precision modes, before any timing is trusted.
    for tf32 in (False, True):
        torch.backends.cuda.matmul.allow_tf32 = tf32
        ref = reference(x, w, wb, bias, gnw, gnb)
        got = mod.forward(x, w, wb, bias, gnw, gnb, G)
        rel = ((got - ref).abs().max() / ref.abs().max()).item()
        cos = torch.nn.functional.cosine_similarity(
            got.flatten().double(), ref.flatten().double(), dim=0).item()
        tag = "tf32" if tf32 else "ieee"
        print(f"correctness vs reference ({tag}): max rel err {rel:.3e}  cosine {cos:.10f}")

    # An fp64 reference is the only way to say WHICH of the two is more accurate rather than
    # just how far apart they are.
    torch.backends.cuda.matmul.allow_tf32 = False
    ref64 = reference(x.double(), w.double(), wb.double(), bias.double(),
                      gnw.double(), gnb.double())
    ref32 = reference(x, w, wb, bias, gnw, gnb)
    got = mod.forward(x, w, wb, bias, gnw, gnb, G)
    err_ref = ((ref32.double() - ref64).abs().max() / ref64.abs().max()).item()
    err_ext = ((got.double() - ref64).abs().max() / ref64.abs().max()).item()
    print(f"vs fp64 truth: reference itself {err_ref:.3e}   their kernel {err_ext:.3e}   "
          f"ratio {err_ext / err_ref:.2f}x")
    print()

    # ---- FALLBACK DETECTION. A silently-taken ATen fallback is correct but is not their
    # kernel, so prove the fast path is what gets timed.
    xnc = x.t().contiguous().t()            # same values, non-contiguous -> gate must reject
    assert not xnc.is_contiguous()
    slow = stats(time_wallclock(lambda: mod.forward(xnc, w, wb, bias, gnw, gnb, G), n=50))
    fast = stats(time_wallclock(lambda: mod.forward(x, w, wb, bias, gnw, gnb, G), n=50))
    same = torch.allclose(mod.forward(xnc, w, wb, bias, gnw, gnb, G),
                          mod.forward(x, w, wb, bias, gnw, gnb, G), rtol=1e-4, atol=1e-5)
    print(f"fallback control: non-contiguous (ATen path) {slow['median']*1e3:.2f} us vs "
          f"contiguous (fast path) {fast['median']*1e3:.2f} us, "
          f"same numbers={same}")
    if fast["median"] >= slow["median"]:
        print("  WARNING: the 'fast path' is not faster than the fallback -- the gate may be")
        print("  rejecting our inputs, in which case every number below is the ATen path.")
    else:
        print(f"  ok: fast path is {slow['median']/fast['median']:.2f}x the fallback, so the")
        print("  gate accepted and the timings below are their kernel.")
    print()

    results = {}
    for tf32 in (False, True):
        torch.backends.cuda.matmul.allow_tf32 = tf32
        tag = "tf32" if tf32 else "ieee"
        print(f"--- matmul precision: {tag}")
        ref_fn = lambda: reference(x, w, wb, bias, gnw, gnb)   # noqa: E731
        ext_fn = lambda: mod.forward(x, w, wb, bias, gnw, gnb, G)  # noqa: E731

        rw = stats(time_wallclock(ref_fn, n=args.trials))
        ew = stats(time_wallclock(ext_fn, n=args.trials))
        rc = stats(time_cuda_events_l2flush(ref_fn, n=args.trials))
        ec = stats(time_cuda_events_l2flush(ext_fn, n=args.trials))
        show(f"reference   [wall-clock]", rw)
        show(f"their kernel[wall-clock]", ew)
        show(f"reference   [events+L2flush]", rc)
        show(f"their kernel[events+L2flush]", ec)
        print(f"  speedup, wall-clock      : {rw['median']/ew['median']:6.2f}x "
              f"(median) / {rw['mean']/ew['mean']:.2f}x (mean)")
        print(f"  speedup, events+L2 flush : {rc['median']/ec['median']:6.2f}x "
              f"(median) / {rc['mean']/ec['mean']:.2f}x (mean)")
        results[tag] = {"ref_wall": rw, "ext_wall": ew, "ref_ev": rc, "ext_ev": ec}
        print()

    print("=== the comparison that matters")
    print(f"their manifest claims ref_speed 0.0903 ms (eager/TF32) and kernel 0.00832 ms")
    rt = results["tf32"]
    print(f"measured here      ref  {rt['ref_wall']['median']:.4f} ms wall-clock / "
          f"{rt['ref_ev']['median']:.4f} ms events+flush")
    print(f"measured here      them {rt['ext_wall']['median']:.4f} ms wall-clock / "
          f"{rt['ext_ev']['median']:.4f} ms events+flush")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
