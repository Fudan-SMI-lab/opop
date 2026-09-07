"""What can this box actually measure about a kernel, without hardware counters?

`ncu` is installed but returns ERR_NVGPUCTRPERM in this container: NVIDIA GPU performance
counters need a host-side kernel-module permission that cannot be set from inside. So any
bottleneck taxonomy the harness relies on MUST be classifiable without them, on every box.

This probe establishes the achievable signal set. Each candidate signal is tested for:
  (a) does it work here at all,
  (b) does it DISCRIMINATE -- measured on four kernels with deliberately different bottlenecks,
      a signal that returns the same value for all four is useless however cheap it is.

The four reference workloads, chosen so the ground truth is known analytically:
  COMPUTE : 4096^3 fp32 matmul        -- 137 GFLOP, 201 MB traffic  -> ~50 FLOP/byte
  MEMORY  : 256 MB elementwise copy   -- 0 FLOP,    512 MB traffic  -> ~0 FLOP/byte
  LAUNCH  : 40 tiny elementwise ops   -- trivial work, 40 launches
  MIXED   : level2:37 reference       -- 6 ops at 128x512x1024

If a signal cannot separate COMPUTE from MEMORY, it cannot support the taxonomy.
"""
from __future__ import annotations

import json
import statistics
import subprocess
import time

import torch
import torch.nn.functional as F

DEV = "cuda"


def workloads():
    torch.manual_seed(0)
    a = torch.randn(4096, 4096, device=DEV)
    b = torch.randn(4096, 4096, device=DEV)
    # 4096^3 matmul: 2*N^3 FLOP, reads 2 matrices + writes 1
    yield ("COMPUTE 4096^3 matmul", lambda: a @ b,
           2 * 4096 ** 3, 3 * 4096 * 4096 * 4)

    big = torch.randn(64 * 1024 * 1024, device=DEV)   # 256 MB
    out = torch.empty_like(big)
    yield ("MEMORY 256MB copy", lambda: torch.add(big, 1.0, out=out),
           big.numel(), 2 * big.numel() * 4)

    small = torch.randn(256, 256, device=DEV)

    def many():
        y = small
        for _ in range(40):
            y = torch.relu(y) + 1.0
        return y
    yield ("LAUNCH 40 tiny ops", many, 40 * 2 * 256 * 256, 40 * 2 * 256 * 256 * 4)

    M, K, N, G = 128, 512, 1024, 32
    x = torch.randn(M, K, device=DEV)
    w = torch.randn(N, K, device=DEV)
    wb = torch.randn(N, device=DEV)
    bias = torch.randn(N, device=DEV)
    gnw = torch.randn(N, device=DEV)
    gnb = torch.randn(N, device=DEV)

    def l237():
        z = F.linear(x, w, wb)
        h = torch.sigmoid(z) * z
        return F.group_norm(h + bias, G, gnw, gnb, 1e-5)
    yield ("MIXED level2:37 ref", l237,
           2 * M * K * N, (M * K + N * K + 3 * M * N) * 4)


def timed(fn, n=60):
    for _ in range(15):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(n):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e))
    return statistics.median(out)


def cpu_issue_time(fn, n=60):
    torch.cuda.synchronize()
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1e3)
    torch.cuda.synchronize()
    return statistics.median(out)


def kernel_breakdown(fn):
    """torch.profiler: per-kernel GPU time and launch count. Needs CUPTI, NOT counters."""
    from torch.profiler import ProfilerActivity, profile
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
    evts = [e for e in prof.key_averages() if e.device_type.name == "CUDA" or e.self_device_time_total]
    tot = sum(e.self_device_time_total for e in evts)
    names = sorted(((e.self_device_time_total, e.key, e.count) for e in evts), reverse=True)
    return tot / 10.0 / 1e3, [(n, c, t / 10.0) for t, n, c in names[:4]]


def main() -> int:
    p = torch.cuda.get_device_properties(0)
    print(f"device: {p.name} sm_{p.major}{p.minor}  {p.multi_processor_count} SMs")

    # Peak rates, derived not guessed. fp32 FMA: 2 FLOP * 64 lanes/SM * clock.
    clock_hz = torch.cuda.clock_rate() * 1e3 if hasattr(torch.cuda, "clock_rate") else None
    print(f"clock: {clock_hz/1e9 if clock_hz else '?'} GHz")
    print()

    # A measured DRAM ceiling beats a spec number: it is what the kernel can actually get.
    big = torch.randn(128 * 1024 * 1024, device=DEV)     # 512 MB
    o = torch.empty_like(big)
    t = timed(lambda: torch.add(big, 1.0, out=o), n=30)
    peak_bw = (2 * big.numel() * 4) / (t * 1e-3) / 1e12
    del big, o
    torch.cuda.empty_cache()
    print(f"measured DRAM ceiling (512MB stream copy): {peak_bw:.3f} TB/s")

    a = torch.randn(8192, 8192, device=DEV)
    b = torch.randn(8192, 8192, device=DEV)
    torch.backends.cuda.matmul.allow_tf32 = False
    t = timed(lambda: a @ b, n=20)
    peak_flops = (2 * 8192 ** 3) / (t * 1e-3) / 1e12
    del a, b
    torch.cuda.empty_cache()
    print(f"measured fp32 ceiling (8192^3 matmul):     {peak_flops:.2f} TFLOP/s")
    ridge = peak_flops * 1e12 / (peak_bw * 1e12)
    print(f"=> roofline ridge point: {ridge:.1f} FLOP/byte "
          f"(below = memory-bound, above = compute-bound)")
    print()

    rows = []
    for name, fn, flop, byts in workloads():
        gpu_ms = timed(fn)
        cpu_ms = cpu_issue_time(fn)
        prof_ms, top = kernel_breakdown(fn)
        ai = flop / byts                                   # arithmetic intensity
        achieved_bw = byts / (gpu_ms * 1e-3) / 1e12
        achieved_fl = flop / (gpu_ms * 1e-3) / 1e12
        n_launch = sum(c for _, c, _ in top)
        rows.append({
            "name": name, "gpu_ms": gpu_ms, "cpu_issue_ms": cpu_ms,
            "prof_gpu_ms": prof_ms, "ai_flop_per_byte": ai,
            "achieved_tbs": achieved_bw, "pct_of_dram_peak": achieved_bw / peak_bw * 100,
            "achieved_tflops": achieved_fl, "pct_of_fp32_peak": achieved_fl / peak_flops * 100,
            "cpu_over_gpu": cpu_ms / gpu_ms, "kernels": top,
        })
        print(f"--- {name}")
        print(f"  GPU (events)          {gpu_ms*1e3:9.1f} us     "
              f"CPU issue {cpu_ms*1e3:9.1f} us   ratio {cpu_ms/gpu_ms:6.2f}x")
        print(f"  arithmetic intensity  {ai:9.2f} FLOP/byte  "
              f"({'COMPUTE side' if ai > ridge else 'MEMORY side'} of the ridge)")
        print(f"  achieved bandwidth    {achieved_bw:9.3f} TB/s  = "
              f"{achieved_bw/peak_bw*100:5.1f}% of measured peak")
        print(f"  achieved fp32         {achieved_fl:9.2f} TFLOP/s = "
              f"{achieved_fl/peak_flops*100:5.1f}% of measured peak")
        print(f"  launches/call {n_launch:3d}   profiler GPU total {prof_ms*1e3:.1f} us")
        for kn, kc, kt in top[:3]:
            print(f"      {kc:4d}x {kt*1e3:8.1f} us  {kn[:58]}")
        print()

    print("=== DISCRIMINATION CHECK: does each signal separate the four workloads?")
    for key, label in (("ai_flop_per_byte", "arithmetic intensity"),
                       ("pct_of_dram_peak", "% of DRAM peak"),
                       ("pct_of_fp32_peak", "% of fp32 peak"),
                       ("cpu_over_gpu", "CPU-issue / GPU ratio")):
        vals = [r[key] for r in rows]
        spread = max(vals) / min(vals) if min(vals) > 0 else float("inf")
        verdict = "DISCRIMINATES" if spread > 3 else "does NOT discriminate"
        print(f"  {label:26} {[f'{v:.2f}' for v in vals]}  spread {spread:8.1f}x  {verdict}")

    with open("bottleneck_signals.json", "w", encoding="utf-8") as f:
        json.dump({"peak_bw_tbs": peak_bw, "peak_fp32_tflops": peak_flops,
                   "ridge_flop_per_byte": ridge, "rows": rows}, f, indent=2)
    print()
    print("wrote bottleneck_signals.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
