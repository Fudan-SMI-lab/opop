"""Positive control for the binding criterion: can it tell DRAM-bound from occupancy-bound?

This decides whether the A-1 finding means what I said it means.

`multibind_rule_check.py` found that 8 of 14 above-noise successful rewrites relieved no
BINDING dimension, and 6 of those pushed a dimension TOWARD its wall. I read that as "the
winning move is filling a slack resource". But Balakrishnan & Cheng (IJPR 38(6), 2000 --
verified full text) show a case where the UNDER-utilised resource carries the higher shadow
price ($1.8 vs $0.6), i.e. utilisation is a poor criterion for which constraint matters. If
that holds here, the dimensions I labelled "slack" were the real constraints all along and my
label was simply wrong -- no new mechanism, just an unreliable criterion.

Those two readings are indistinguishable from historical data, because my "binding" label IS
utilisation. They are distinguishable with kernels whose bottleneck we know by construction.

Four kernels, each built so one dimension is the limit and the others are not:

  A  stream       pure streaming copy, tiny tile, no math, no shared memory
                  -> DRAM-bound by construction: it moves N bytes and does ~N flops
  B  occupancy    same traffic per thread as A, but a huge register-resident accumulator
                  forces n_regs up and occupancy down, WITHOUT changing bytes moved
                  -> occupancy-bound by construction: same traffic, fewer resident warps
  C  compute      tiny working set that fits in L1, long fp32 ieee dependent-multiply chain
                  -> compute-bound by construction: bytes ~ 0, flops huge
  D  shared       tile sized so shared memory is near the 101376 B limit but traffic is low
                  -> shared-capacity-bound by construction

The criterion passes only if, for each kernel, the dimension we built to be the limit is the
one it reports as most-pressured. Anything else is a mislabel, and a mislabel means S2's two
lists ("walls" / "slack") are two columns of the same error.

Every ceiling is measured on this box first, and every kernel is checked against the measured
ceiling before its label is trusted -- because a control that reads 104% of the fp32 ceiling
(which happened once, from tl.dot silently defaulting to tf32) is not a control.

Prints a verdict table. Exit code 0 = criterion distinguishes all four; nonzero = it does not.
"""
import json
import os
import statistics
import sys
import time

import torch
import triton
import triton.language as tl

SHARED_LIMIT = 101376
REG_LIMIT = 255


# --------------------------------------------------------------------------- ceilings
def measure_ceilings(dev):
    """DRAM roof and fp32 (ieee) compute roof, measured here, now."""
    free_bytes, _ = torch.cuda.mem_get_info(dev)
    n = int(min(128 << 20, max(4 << 20, free_bytes * 0.15 / 4 / 2)))
    a = torch.randn(n, device="cuda")
    out = torch.empty_like(a)

    def timed(fn, iters=30, warmup=10):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            fn()
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e) / iters

    ms = timed(lambda: torch.add(a, 1.0, out=out))
    dram_gbs = (2 * n * 4) / (ms * 1e-3) / 1e9
    del a, out
    torch.cuda.empty_cache()

    mm = 4096
    x = torch.randn(mm, mm, device="cuda")
    y = torch.randn(mm, mm, device="cuda")
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    ms = timed(lambda: torch.mm(x, y), iters=20, warmup=10)
    fp32_tflops = (2 * mm ** 3) / (ms * 1e-3) / 1e12
    torch.backends.cuda.matmul.allow_tf32 = prev
    del x, y
    torch.cuda.empty_cache()

    props = torch.cuda.get_device_properties(dev)
    return dict(dram_gbs=dram_gbs, fp32_tflops=fp32_tflops,
                sms=props.multi_processor_count,
                shared_limit=SHARED_LIMIT)


# --------------------------------------------------------------------------- kernels
@triton.jit
def k_stream(X, Y, n, BLOCK: tl.constexpr):
    """A: pure streaming. One load, one store, one add. No shared memory, no math depth."""
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    tl.store(Y + off, tl.load(X + off, mask=m, other=0.0) + 1.0, mask=m)


@triton.jit
def k_occupancy(X, Y, n, BLOCK: tl.constexpr, NACC: tl.constexpr):
    """B: L2-resident data + very high register pressure + minimal arithmetic.

    Two failed constructions before this one, both of which the criterion called correctly
    while I called them wrong:

      1. Eight chains x DEPTH=64 iterations. That raised registers, but at 512 fma per element
         it was genuinely compute-heavy -- and it read 125% of the measured fp32 roof, which
         the sanity check flagged as impossible (my flop count double-counted fma).
      2. Register tile with traffic identical to the streaming kernel A. It came out at
         DRAM 1.00 and 0.584 ms -- the same time as A. Correct: A already saturates bandwidth,
         so B was bandwidth-bound too. 83% occupancy hides streaming latency perfectly well.

    The lesson generalises: OCCUPANCY ONLY BINDS WHEN IT FAILS TO HIDE LATENCY. So a genuine
    occupancy control needs (a) traffic far below the DRAM roof, or the kernel is bandwidth-
    bound instead, (b) few enough resident warps that latency is exposed, and (c) little
    arithmetic, or it is compute-bound instead. Here: the footprint is small enough to sit in
    L2 (so DRAM pressure is low), NACC is large enough to push registers toward the cap (so
    resident warps are few), and the work is one fma per tile element.
    """
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    v = tl.load(X + off, mask=m, other=0.0)
    rows = tl.arange(0, NACC)
    acc = v[None, :] * (1.0 + rows[:, None] * 1e-6) + rows[:, None]
    tl.store(Y + off, tl.sum(acc, axis=0), mask=m)


@triton.jit
def k_compute(X, Y, n, BLOCK: tl.constexpr, ITERS: tl.constexpr):
    """C: tiny traffic, long dependent fp32 chain. bytes ~ 0, flops huge."""
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    v = tl.load(X + off, mask=m, other=1.0)
    for _ in range(ITERS):
        v = v * 1.0000001 + 0.0000001
    tl.store(Y + off, v, mask=m)


@triton.jit
def k_shared(X, Y, n, BLOCK: tl.constexpr, ROWS: tl.constexpr):
    """D: a large shared-resident tile with low traffic and a SMALL accumulator.

    tl.dot forces its operands through shared memory, so a ROWS x BLOCK fp32 tile lands there.
    The shape is deliberately rectangular (few rows, wide K): shared holds ROWS*BLOCK*4 bytes
    while the accumulator is only ROWS x ROWS. A square 128x128 version allocated the same
    shared memory but spilled 2606 registers, which makes it a register control wearing a
    shared-memory label.
    """
    off = tl.arange(0, BLOCK)
    rows = tl.arange(0, ROWS)
    base = tl.program_id(0) * ROWS * BLOCK
    idx = base + rows[:, None] * BLOCK + off[None, :]
    tile = tl.load(X + idx, mask=idx < n, other=0.0)
    acc = tl.dot(tile, tl.trans(tile), input_precision="ieee")
    r = tl.sum(acc, axis=1)
    tl.store(Y + tl.program_id(0) * ROWS + rows,
             r, mask=(tl.program_id(0) * ROWS + rows) < n)


# --------------------------------------------------------------------------- runner
def run_one(name, fn, grid, args, consts, bytes_moved, flops, ceilings):
    """Launch, read resources, time it, and turn that into per-dimension pressures."""
    compiled = fn.warmup(*args, **consts, grid=grid)
    fn[grid](*args, **consts)
    torch.cuda.synchronize()

    def timed(iters=50, warmup=20):
        for _ in range(warmup):
            fn[grid](*args, **consts)
        torch.cuda.synchronize()
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            fn[grid](*args, **consts)
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e) / iters

    ms = statistics.median([timed(iters=20, warmup=10) for _ in range(5)])

    md = compiled.metadata
    shared = getattr(md, "shared", 0) or 0
    n_regs = compiled.n_regs
    n_spills = compiled.n_spills
    num_warps = getattr(md, "num_warps", 0) or 0

    # occupancy from the register limit, the way Triton's own limit works out:
    # 65536 registers per SM / (regs per thread * 32 threads per warp) = resident warps
    warps_by_reg = 65536 // max(1, n_regs * 32) if n_regs else None
    max_warps = 48  # Ada: 1536 threads/SM / 32
    occ = min(1.0, warps_by_reg / max_warps) if warps_by_reg else None

    achieved_gbs = bytes_moved / (ms * 1e-3) / 1e9
    achieved_tflops = flops / (ms * 1e-3) / 1e12

    pres = {
        "dram": achieved_gbs / ceilings["dram_gbs"],
        "compute": achieved_tflops / ceilings["fp32_tflops"],
        "occupancy": (1.0 - occ) if occ is not None else None,
        "shared": shared / ceilings["shared_limit"],
        "registers": n_regs / REG_LIMIT if n_regs else None,
    }
    return dict(name=name, ms=ms, shared=shared, n_regs=n_regs, n_spills=n_spills,
                num_warps=num_warps, occupancy=occ,
                achieved_gbs=achieved_gbs, achieved_tflops=achieved_tflops,
                pressures=pres)


def main():
    dev = torch.cuda.current_device()
    print("device: %s" % torch.cuda.get_device_name(dev))
    print("measuring ceilings on this box first (a control checked against a guessed "
          "ceiling is not a control)...")
    C = measure_ceilings(dev)
    print("  DRAM roof      %.1f GB/s" % C["dram_gbs"])
    print("  fp32 ieee roof %.1f TFLOP/s" % C["fp32_tflops"])
    print("  SMs %d   shared limit %d B\n" % (C["sms"], C["shared_limit"]))

    N = 64 << 20                       # 64 Mi elements = 256 MiB per buffer, past L2
    X = torch.randn(N, device="cuda", dtype=torch.float32)
    Y = torch.empty(N, device="cuda", dtype=torch.float32)

    cases = []

    # A: streaming -- 2 arrays touched once each, ~1 flop per element
    BLOCK = 1024
    cases.append(("A stream (built: DRAM-bound)", "dram", k_stream,
                  (triton.cdiv(N, BLOCK),), (X, Y, N), dict(BLOCK=BLOCK),
                  2 * N * 4, float(N)))

    # B: occupancy -- L2-resident footprint (low DRAM pressure), high register pressure,
    # minimal arithmetic. NB is sized to fit inside the 4090's 72 MiB L2 so the kernel is not
    # bandwidth-bound; NACC is large so registers approach the cap and resident warps are few.
    # See k_occupancy's docstring for the two constructions that failed before this one.
    NACC = 128
    NB = 4 << 20                       # 4 Mi elements = 16 MiB per buffer, inside L2
    gridB = triton.cdiv(NB, BLOCK)
    launchedB = gridB * BLOCK
    cases.append(("B occupancy (built: occupancy-bound)", "occupancy", k_occupancy,
                  (gridB,), (X, Y, NB), dict(BLOCK=BLOCK, NACC=NACC),
                  2 * NB * 4, float(launchedB) * NACC))

    # C: compute -- small n so traffic is negligible, long dependent chain
    NC = 1 << 20
    ITERS = 512
    cases.append(("C compute (built: compute-bound)", "compute", k_compute,
                  (triton.cdiv(NC, BLOCK),), (X, Y, NC), dict(BLOCK=BLOCK, ITERS=ITERS),
                  2 * NC * 4, float(NC) * ITERS * 2))

    # D: shared -- rectangular tile so shared memory is large while the accumulator stays
    # small. ROWS x BW fp32 through tl.dot puts ROWS*BW*4 bytes in shared; the accumulator is
    # ROWS x ROWS. The square 128x128 version allocated the same shared memory but spilled 2606
    # registers, i.e. it was really a register-pressure control.
    ROWS, BW = 32, 512
    NS = (N // (ROWS * BW)) * (ROWS * BW)
    cases.append(("D shared (built: shared-capacity-bound)", "shared", k_shared,
                  (NS // (ROWS * BW),), (X, Y, NS), dict(BLOCK=BW, ROWS=ROWS),
                  NS * 4, float(NS) * ROWS))

    # E: control FOR THE CONTROL. Same streaming kernel as A, but sized to sit inside the
    # 4090's 72 MiB L2. Its traffic is served by cache, so it is NOT DRAM-bound -- yet our
    # pressure metric computes bytes/time against the DRAM roof and cannot tell the difference.
    # If E reads high "dram" pressure, the criterion conflates L2-served with DRAM traffic.
    NE = 4 << 20                       # 16 MiB per buffer, both buffers inside L2
    cases.append(("E stream-in-L2 (built: NOT DRAM-bound)", "none", k_stream,
                  (triton.cdiv(NE, BLOCK),), (X, Y, NE), dict(BLOCK=BLOCK),
                  2 * NE * 4, float(NE)))

    rows = []
    for name, expect, fn, grid, args, consts, byts, flops in cases:
        try:
            r = run_one(name, fn, grid, args, consts, byts, flops, C)
        except Exception as e:
            print("  %-40s FAILED TO RUN: %s: %s" % (name, type(e).__name__, str(e)[:80]))
            rows.append(dict(name=name, expect=expect, err=str(e)[:120]))
            continue
        r["expect"] = expect
        rows.append(r)

    print("MEASURED (pressure: 1.0 = against that wall; occupancy is inverted)")
    print("%-40s %8s %7s %6s %7s | %6s %7s %6s %7s %7s  %s"
          % ("kernel", "ms", "shared", "regs", "occ",
             "dram", "compute", "occ", "shared", "regs", "verdict"))
    ok = True
    for r in rows:
        if "err" in r:
            ok = False
            continue
        p = r["pressures"]
        top = max((k for k in p if p[k] is not None), key=lambda k: p[k])
        if r["expect"] == "none":
            # E is diagnostic, not a labelled case: it asks whether an L2-resident kernel is
            # wrongly reported as bandwidth-pressured.
            hit = p["dram"] is not None and p["dram"] < 0.5
        else:
            hit = (top == r["expect"])
        ok = ok and hit

        def f(k):
            v = p.get(k)
            return "%.2f" % v if v is not None else "  -"
        print("%-40s %8.3f %7d %6s %7s | %6s %7s %6s %7s %7s  %s"
              % (r["name"][:40], r["ms"], r["shared"], r["n_regs"],
                 ("%.2f" % r["occupancy"]) if r["occupancy"] else "-",
                 f("dram"), f("compute"), f("occupancy"), f("shared"), f("registers"),
                 ("OK   top=%s" % top) if hit
                 else ("L2 TRAFFIC READ AS DRAM %.2f" % p["dram"]
                       if r["expect"] == "none"
                       else "MISLABEL top=%s want=%s" % (top, r["expect"]))))

    print()
    print("=" * 108)
    print("SECOND ARM: the SAME five kernels through the HARNESS'S OWN classify() (G26 / J2-8)")
    print("=" * 108)
    print("The pressures above are computed by this probe. That tests the criterion's CONCEPT, not")
    print("the code every verdict actually goes through -- so a fix in bottleneck.py would not show")
    print("up here at all. This arm feeds the same measurements to the real classifier.")
    print()
    try:
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
        from kernel_optimizer.evaluation.bottleneck import DevicePeaks, classify

        l2_bytes = int(getattr(torch.cuda.get_device_properties(dev), "L2_cache_size", 0) or 0)
        print("this card's measured L2: %.1f MiB%s"
              % (l2_bytes / 2**20,
                 "" if l2_bytes else "   <-- UNMEASURED, so the working-set gate cannot fire"))
        peaks = DevicePeaks(dram_tbs=C["dram_gbs"] / 1000.0,
                            fp32_tflops=C["fp32_tflops"], l2_bytes=l2_bytes)
        print()
        print("%-40s %10s %9s %-16s %s"
              % ("kernel", "bytes MiB", "frac_bw", "classify() kind", "dram applicable?"))
        arm2_ok = True
        for r, (name, expect, _fn, _g, _a, _c, byts, flops) in zip(rows, cases):
            if "err" in r:
                continue
            v = classify(gpu_ms=r["ms"], cpu_issue_ms=None, flop_count=int(flops),
                         byte_count=int(byts), peaks=peaks,
                         n_regs=r["n_regs"], n_spills=r["n_spills"],
                         shared_bytes=r["shared"], precision="fp32")
            applicable = v.evidence.get("dram_applicable")
            print("%-40s %10.1f %9.2f %-16s %s"
                  % (name[:40], byts / 2**20, r["pressures"]["dram"], v.kind,
                     "NO (L2-resident)" if applicable is False else "yes"))
            # The gate that matters: E must NOT come out memory_bound.
            if expect == "none" and v.kind == "memory_bound":
                arm2_ok = False
                print("      ^^ FAIL: an L2-resident kernel was classified memory_bound. This is "
                      "J2-8, and S2 cannot")
                print("         start while it holds: per-dimension output would just split this "
                      "mislabelling into two columns.")
        print()
        if arm2_ok:
            print("ARM 2 VERDICT: PASS -- classify() does not report the L2-resident kernel as "
                  "memory_bound.")
            print("  Note this is the arm that tracks the production path; arm 1 above reflects "
                  "this probe's own")
            print("  pressure arithmetic, which is deliberately left unfixed so the two can be "
                  "compared.")
        else:
            print("ARM 2 VERDICT: FAIL -- J2-8 is not satisfied.")
    except Exception as exc:  # noqa: BLE001 -- arm 2 must not break arm 1's result
        print("ARM 2 could not run: %s: %s" % (type(exc).__name__, str(exc)[:200]))
        print("  (that is a probe failure, not a pass -- do not read the absence of a FAIL as one)")

    print()
    print("SANITY (a control that exceeds a measured ceiling is broken, not fast)")
    for r in rows:
        if "err" in r:
            continue
        flags = []
        if r["pressures"]["dram"] and r["pressures"]["dram"] > 1.05:
            flags.append("DRAM %.0f%% of measured roof" % (100 * r["pressures"]["dram"]))
        if r["pressures"]["compute"] and r["pressures"]["compute"] > 1.05:
            flags.append("compute %.0f%% of measured roof"
                         % (100 * r["pressures"]["compute"]))
        if r["n_spills"]:
            flags.append("%d spills" % r["n_spills"])
        print("  %-40s %s" % (r["name"][:40], "; ".join(flags) if flags else "within ceilings"))

    print()
    n_scored = len([r for r in rows if "err" not in r])
    # An empty or short result set must never read as a pass. My edit once deleted the
    # execution loop and this block printed "identified the built-in limit in all 0 kernels ...
    # that supports reading the A-1 result as filling slack" -- a vacuous truth presented as
    # evidence, which is the same failure shape as a probe returning a plausible constant.
    if n_scored < len(cases):
        print("VERDICT: INCONCLUSIVE -- only %d of %d controls ran. A probe that did not run"
              % (n_scored, len(cases)))
        print("  is not a probe that passed. Fix the failures above before reading anything.")
        return 2
    if ok:
        print("VERDICT: the utilisation-based binding criterion identified the built-in limit")
        print("  in all %d kernels. That supports reading the A-1 result as 'filling slack',"
              % n_scored)
        print("  because the labels it produces are trustworthy on cases we know the answer to.")
        return 0
    print("VERDICT: the criterion MISLABELLED at least one kernel whose limit we built in.")
    print("  Then the A-1 reading 'winning rewrites fill slack' is NOT supported: the")
    print("  'slack' dimensions may be the real constraints, as Balakrishnan & Cheng's")
    print("  shadow-price example predicts. S2's two lists would be two columns of one error.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
