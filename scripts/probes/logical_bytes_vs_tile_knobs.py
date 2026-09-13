"""Is `logical_bytes` independent of the tile knobs, or just a restatement of them?

WHY THIS IS THE DECIDING QUESTION, not "can we parse 2-D IR". Parsing is engineering. What decides
whether the dimension is worth adding is whether its number carries information we do not already
have for free. Twice now this project has added a "dimension" that turned out to be an existing
quantity rescaled -- `pct_of_dram_peak` is 1/latency times a task constant (rho +1.000 over 4 runs)
and speed-of-light headroom ranked candidates exactly as 1/latency did -- and both were caught only
by asking for a correlation instead of accepting a plausible story. So the same test is applied
before, not after, building this one.

THE SPECIFIC WORRY. For a tiled matmul the logical traffic of one program instance is

    per_instance = (BM*BK + BK*BN) * width * trips + BM*BN * width,   trips = K/BK

and the grid is (M/BM)*(N/BN). Multiplying out, the TOTAL logical bytes reduce to something close to
M*N*K*width*(1/BN + 1/BM) -- i.e. a function of the tile knobs and the problem shape and nothing
else. If that is what it is, then `logical_bytes` is an algebraic rearrangement of BM and BN, both of
which the sampler already varies directly and both of which `shared_bytes` already tracks. It would
then be a third name for a knob, not a resource dimension.

This probe measures that on a real compiled matmul across a real grid of tile choices, and reports
Spearman against three things we already have: the tile product, shared_bytes from the compiler, and
measured latency. It also reports whether the TOTAL (as opposed to per-instance) varies at all --
for an elementwise kernel it provably does not, which was already measured (BLOCK 256..2048 all gave
exactly 201326592), and a dimension that is constant under every knob cannot be truncated by any
knob and therefore cannot be a wall.
"""
import re
import statistics
import sys

import torch
import triton
import triton.language as tl
from triton.runtime.jit import JITFunction

_W = {"f32": 4, "f16": 2, "bf16": 2, "f64": 8, "i8": 1, "i16": 2, "i32": 4, "i64": 8, "i1": 1}


def bytes_per_instance(ir: str) -> tuple[int, bool]:
    """(bytes one instance touches in ONE pass, whether a loop was present).

    Loop trips are NOT multiplied in here: the trip count is not always a literal in the IR, and
    guessing it would be the same class of error as guessing an events payload path. The loop flag
    is returned instead, which is what tells a reader the figure is a lower bound.
    """
    per, loop = 0, "scf.for" in ir
    for line in ir.splitlines():
        if "tt.load" not in line and "tt.store" not in line:
            continue
        m2 = [(int(a), int(b), c) for a, b, c in
              re.findall(r"tensor<(\d+)x(\d+)x!tt\.ptr<([a-z0-9]+)>>", line) if c in _W]
        if m2:
            n, m, t = max(m2)
            per += n * m * _W[t]
            continue
        m1 = [(int(a), b) for a, b in
              re.findall(r"tensor<(\d+)x!tt\.ptr<([a-z0-9]+)>>", line) if b in _W]
        if m1:
            n, t = max(m1)
            per += n * _W[t]
    return per, loop


def spearman(xs, ys):
    n = len(xs)
    if n < 4 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else None


@triton.jit
def mm(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        offs_k = k + tl.arange(0, BK)
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
        b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
        acc += tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)


M = N = K = 1024
a = torch.randn(M, K, device="cuda", dtype=torch.float16)
b = torch.randn(K, N, device="cuda", dtype=torch.float16)
c = torch.empty(M, N, device="cuda", dtype=torch.float32)

rows = []
original = JITFunction.run
for BM in (32, 64, 128):
    for BN in (32, 64, 128):
        for BK in (32, 64):
            captured = []

            def run_capturing(self, *args, grid=None, warmup=False, **kw):
                k = original(self, *args, grid=grid, warmup=warmup, **kw)
                captured.append(k)
                return k

            try:
                JITFunction.run = run_capturing
                grid = (M // BM, N // BN)
                mm[grid](a, b, c, M, N, K, BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=2)
                torch.cuda.synchronize()
            except Exception as exc:  # noqa: BLE001 — an infeasible tile is a datum, not a crash
                print(f"BM={BM:4d} BN={BN:4d} BK={BK:3d}  REFUSED {type(exc).__name__}: "
                      f"{str(exc)[:60]}")
                continue
            finally:
                JITFunction.run = original
            if not captured:
                continue
            k = captured[0]
            ir = (getattr(k, "asm", {}) or {}).get("ttir", "")
            per, loop = bytes_per_instance(ir)
            trips = K // BK
            meta = getattr(k, "metadata", None)
            shared = int(getattr(meta, "shared", 0) or 0)
            # Total logical bytes: the load ops are inside the K loop, the store is not. Rather than
            # try to attribute per-op position from the IR, bound it both ways and report both --
            # the honest form when the trip count cannot be attached to individual ops.
            total_lo = per * grid[0] * grid[1]
            total_hi = per * trips * grid[0] * grid[1]
            # Timing, so independence from latency can be checked too.
            for _ in range(3):
                mm[grid](a, b, c, M, N, K, BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=2)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(20):
                mm[grid](a, b, c, M, N, K, BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=2)
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end) / 20.0
            rows.append({"BM": BM, "BN": BN, "BK": BK, "per": per, "trips": trips,
                         "instances": grid[0] * grid[1], "total_lo": total_lo,
                         "total_hi": total_hi, "shared": shared, "ms": ms,
                         "tile_product": BM * BN * BK, "loop": loop})
            print(f"BM={BM:4d} BN={BN:4d} BK={BK:3d}  per_inst={per:8d} trips={trips:3d} "
                  f"inst={grid[0] * grid[1]:5d} total_hi={total_hi:12d} shared={shared:6d} "
                  f"ms={ms:7.4f}")

print()
print(f"rows={len(rows)}  compulsory_bytes={(M * K + K * N) * 2 + M * N * 4}")
if len(rows) < 4:
    print("too few rows to correlate")
    raise SystemExit(0)

print()
print("=== does the TOTAL vary at all, or is it fixed like the elementwise case? ===")
tot = [r["total_hi"] for r in rows]
print(f"  total_hi  min={min(tot)}  max={max(tot)}  ratio={max(tot) / min(tot):.2f}x")
print("  => " + ("it VARIES with the tile, so a knob can move it"
                 if max(tot) / min(tot) > 1.05 else
                 "it is EFFECTIVELY CONSTANT -- no knob moves it, so no knob can be walled by it"))

print()
print("=== INDEPENDENCE: is it distinguishable from what we already have? ===")
for name, key in (("tile product BM*BN*BK", "tile_product"),
                  ("shared_bytes (compiler)", "shared"),
                  ("latency ms", "ms")):
    rho = spearman([float(r["total_hi"]) for r in rows], [float(r[key]) for r in rows])
    if rho is None:
        print(f"  vs {name:24s} rho None")
        continue
    verdict = ("**RESTATEMENT** -- adds nothing over a quantity already in hand"
               if abs(rho) > 0.9 else
               "strongly related -- little new information" if abs(rho) > 0.6 else
               "moderately related" if abs(rho) > 0.3 else
               "essentially independent -- carries its own information")
    print(f"  vs {name:24s} rho {rho:+.3f}   {verdict}")

print()
print("=== the same, for the PER-INSTANCE figure (what the IR gives directly) ===")
for name, key in (("tile product", "tile_product"), ("shared_bytes", "shared")):
    rho = spearman([float(r["per"]) for r in rows], [float(r[key]) for r in rows])
    print(f"  per_instance vs {name:14s} rho "
          + ("None" if rho is None else f"{rho:+.3f}"))

print()
print("=== how far above the COMPULSORY bytes does the logical figure sit? ===")
comp = (M * K + K * N) * 2 + M * N * 4
for r in sorted(rows, key=lambda r: r["total_hi"]):
    print(f"  BM={r['BM']:4d} BN={r['BN']:4d} BK={r['BK']:3d}  "
          f"total_hi/compulsory = {r['total_hi'] / comp:7.1f}x   "
          f"total_lo/compulsory = {r['total_lo'] / comp:5.2f}x")
print("  => the gap IS the L2 blindness: those re-reads mostly hit cache, and a logical count "
      "cannot tell which did.")
