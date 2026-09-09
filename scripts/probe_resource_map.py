"""Is the local resource map extrapolable, or must it be measured every time?

`probe_resource_read_cost.py` established that a resource point costs 614.6 ms (cold compile +
one untimed launch) = 1/30th of a timed trial on this box, and that n_regs/n_spills ARE readable
after one launch. So a MAP of ~30 resource points costs about one timed trial.

That settles affordability but not necessity. If shared memory and registers were smooth and
monotone in the tile knobs, a couple of points plus a rule would do, and measuring a map would
be waste. This probe tests the rule directly on the same 108-config sweep.

Three questions, each with a decision attached:

  Q1 Is shared_bytes a closed-form function of (BM,BN,BK,stages,dtype)?
     Triton computes it by liveness analysis + graph colouring with fixed-point iteration
     (lib/Analysis/Allocation.cpp), total = max(offset+size), so theory says no closed form.
     Tested here as: does the best-fit linear model  a*BM*BK + b*BK*BN + c*stages + d  reproduce
     it exactly? Any nonzero residual kills prediction and confirms "compile and read".

  Q2 Is n_regs monotone in tile size?
     If yes, direction-of-change is predictable and we only need the sign. If it is
     non-monotone, an agent reasoning "bigger tile => more registers" is wrong somewhere in
     the space, and the only correct answer is to read it.

  Q3 How local is the map? For each pair of configs differing in exactly ONE knob by one step,
     record the resource delta. If the same one-step move gives a consistent delta everywhere,
     one measurement generalises. If the delta depends on the other knobs' values, the map is
     not separable and a local map is the only honest artefact.

Q3 is the one that decides the v3 design: a separable map can be measured once per candidate;
a non-separable map must be re-measured around the current point after every structural change
-- which is exactly what the user said happens ("每次算子结构改动后, 算子本身的资源占用率都会发生改变").
"""
import itertools
import json
import os
import statistics
import sys
import time

import torch
import triton
import triton.language as tl


@triton.jit
def gemm(A, B, C, M, N, K,
         sam, sak, sbk, sbn, scm, scn,
         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
         GROUP: tl.constexpr):
    pid = tl.program_id(0)
    n_m = tl.cdiv(M, BM)
    n_n = tl.cdiv(N, BN)
    per_group = GROUP * n_n
    gid = pid // per_group
    first_m = gid * GROUP
    gsize = min(n_m - first_m, GROUP)
    pid_m = first_m + ((pid % per_group) % gsize)
    pid_n = (pid % per_group) // gsize
    om = (pid_m * BM + tl.arange(0, BM)) % M
    on = (pid_n * BN + tl.arange(0, BN)) % N
    ok = tl.arange(0, BK)
    pa = A + (om[:, None] * sam + ok[None, :] * sak)
    pb = B + (ok[:, None] * sbk + on[None, :] * sbn)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(pa, mask=ok[None, :] < K - k * BK, other=0.0)
        b = tl.load(pb, mask=ok[:, None] < K - k * BK, other=0.0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
        pa += BK * sak
        pb += BK * sbk
    cm = pid_m * BM + tl.arange(0, BM)
    cn = pid_n * BN + tl.arange(0, BN)
    pc = C + scm * cm[:, None] + scn * cn[None, :]
    tl.store(pc, acc, mask=(cm[:, None] < M) & (cn[None, :] < N))


M = N = K = 1024
# Every axis needs >= 3 values or the monotonicity question is vacuous on it: any two points
# are trivially ordered. The first version gave BK and num_warps two values each, reported
# "monotone everywhere", and quietly excluded the axis with the LARGEST register spread
# (num_warps, 127 registers). Both are widened here so Q2 covers all five.
AXES = {"BM": (32, 64, 128), "BN": (32, 64, 128), "BK": (16, 32, 64),
        "num_stages": (2, 3, 4), "num_warps": (2, 4, 8)}
ORDER = ["BM", "BN", "BK", "num_stages", "num_warps"]
CONFIGS = [dict(zip(ORDER, v)) for v in itertools.product(*(AXES[k] for k in ORDER))]


def measure(cfg, a, b, c):
    """One resource point: compile, one untimed launch, read every field."""
    k = gemm.warmup(
        torch.empty(1, device="cuda"), torch.empty(1, device="cuda"),
        torch.empty(1, device="cuda"), M, N, K, K, 1, N, 1, N, 1,
        BM=cfg["BM"], BN=cfg["BN"], BK=cfg["BK"], GROUP=8,
        num_stages=cfg["num_stages"], num_warps=cfg["num_warps"], grid=(1,))
    grid = (triton.cdiv(M, cfg["BM"]) * triton.cdiv(N, cfg["BN"]),)
    gemm[grid](a, b, c, M, N, K, K, 1, N, 1, N, 1,
               BM=cfg["BM"], BN=cfg["BN"], BK=cfg["BK"], GROUP=8,
               num_stages=cfg["num_stages"], num_warps=cfg["num_warps"])
    torch.cuda.synchronize()
    md = k.metadata
    return dict(shared=getattr(md, "shared", None), n_regs=k.n_regs,
                n_spills=k.n_spills, n_max_threads=getattr(k, "n_max_threads", None))


def key(cfg):
    return tuple(cfg[k] for k in ORDER)


def main():
    dev = torch.cuda.current_device()
    print("device: %s" % torch.cuda.get_device_name(dev))
    print("map: %d configs over %s\n" % (len(CONFIGS), {k: list(v) for k, v in AXES.items()}))
    a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b = torch.randn(K, N, device="cuda", dtype=torch.float32)
    c = torch.empty(M, N, device="cuda", dtype=torch.float32)

    t0 = time.perf_counter()
    R = {}
    for cfg in CONFIGS:
        try:
            R[key(cfg)] = measure(cfg, a, b, c)
        except Exception as e:
            R[key(cfg)] = dict(err=type(e).__name__ + ": " + str(e)[:70])
    wall = time.perf_counter() - t0
    ok = {k: v for k, v in R.items() if "err" not in v}
    print("measured %d/%d points in %.1f s  (%.0f ms per point, = %.1f timed trials at 18.6 s)"
          % (len(ok), len(R), wall, 1e3 * wall / len(R), wall / 18.6))

    # ---------- Q1: closed form for shared? ----------
    print("\nQ1  IS shared_bytes A CLOSED FORM OF THE TILE KNOBS?")
    print("    least-squares fit of  shared ~ a*BM*BK + b*BK*BN + c*stages + d  (fp32 = 4 B/elt)")
    rows = [(k, v["shared"]) for k, v in ok.items() if v.get("shared") is not None]
    if rows:
        # normal equations, no numpy dependency
        X = [[k[0] * k[2], k[2] * k[1], k[3], 1.0] for k, _ in rows]
        y = [float(s) for _, s in rows]
        n_f = 4
        A_ = [[sum(X[r][i] * X[r][j] for r in range(len(X))) for j in range(n_f)]
              for i in range(n_f)]
        b_ = [sum(X[r][i] * y[r] for r in range(len(X))) for i in range(n_f)]
        # gaussian elimination
        for i in range(n_f):
            p = max(range(i, n_f), key=lambda r: abs(A_[r][i]))
            A_[i], A_[p] = A_[p], A_[i]
            b_[i], b_[p] = b_[p], b_[i]
            if abs(A_[i][i]) < 1e-12:
                continue
            for r in range(i + 1, n_f):
                f = A_[r][i] / A_[i][i]
                for cc in range(i, n_f):
                    A_[r][cc] -= f * A_[i][cc]
                b_[r] -= f * b_[i]
        coef = [0.0] * n_f
        for i in reversed(range(n_f)):
            if abs(A_[i][i]) < 1e-12:
                continue
            coef[i] = (b_[i] - sum(A_[i][j] * coef[j] for j in range(i + 1, n_f))) / A_[i][i]
        pred = [sum(X[r][i] * coef[i] for i in range(n_f)) for r in range(len(X))]
        resid = [abs(pred[r] - y[r]) for r in range(len(y))]
        exact = sum(1 for r in resid if r < 1.0)
        rel = [100 * resid[r] / y[r] for r in range(len(y)) if y[r]]
        print("    coefficients: a=%.3f b=%.3f c=%.1f d=%.1f" % tuple(coef))
        print("    EXACT hits (residual < 1 byte): %d / %d" % (exact, len(y)))
        print("    residual: median %.0f B (%.1f%%), max %.0f B (%.1f%%)"
              % (statistics.median(resid), statistics.median(rel),
                 max(resid), max(rel)))
        print("    => %s" % ("A CLOSED FORM FITS -- prediction would be safe here"
                             if exact == len(y) else
                             "NO CLOSED FORM. Predicting shared memory from the knobs is wrong "
                             "somewhere in this space; compile and read."))

    # ---------- Q2: monotone registers? ----------
    print("\nQ2  IS n_regs MONOTONE IN TILE SIZE?")
    viol = []
    tested_axes, skipped_axes = [], []
    for axis_i, axis in enumerate(ORDER):
        vals = AXES[axis]
        # A 2-value axis cannot be non-monotone by construction: any two points are trivially
        # ordered. Reporting "monotone everywhere" while silently skipping those axes would
        # overstate the result, so they are named as untested.
        if len(vals) < 3:
            skipped_axes.append("%s(%d values)" % (axis, len(vals)))
            continue
        tested_axes.append(axis)
        for other in itertools.product(*(AXES[k] for k in ORDER if k != axis)):
            seq = []
            for v in vals:
                kk = list(other)
                kk.insert(axis_i, v)
                r = ok.get(tuple(kk))
                if r and r.get("n_regs") is not None:
                    seq.append((v, r["n_regs"]))
            if len(seq) < 3:
                continue
            ups = sum(1 for i in range(len(seq) - 1) if seq[i + 1][1] > seq[i][1])
            downs = sum(1 for i in range(len(seq) - 1) if seq[i + 1][1] < seq[i][1])
            if ups and downs:
                viol.append((axis, seq))
    print("    axes tested: %s" % (", ".join(tested_axes) or "none"))
    print("    axes NOT testable (fewer than 3 values, monotone is vacuous): %s"
          % (", ".join(skipped_axes) or "none"))
    print("    non-monotone 1-D slices: %d" % len(viol))
    for axis, seq in viol[:6]:
        print("      along %-11s %s" % (axis, " -> ".join("%s:%d" % s for s in seq)))
    print("    => %s" % ("monotone on the tested axes (%s) -- the SIGN of the change is "
                         "predictable there. Says nothing about %s."
                         % (", ".join(tested_axes), ", ".join(skipped_axes) or "-")
                         if not viol else
                         "NOT MONOTONE. 'bigger tile => more registers' is false somewhere, so "
                         "an agent reasoning about direction alone will be wrong."))

    # ---------- Q3: is the map separable? ----------
    print("\nQ3  IS THE MAP SEPARABLE? (does one 1-step move give a consistent delta everywhere?)")
    print("    %-12s %-8s %6s %9s %9s %9s %9s" %
          ("knob", "step", "n", "d_shared", "spread", "d_regs", "spread"))
    sep = {}
    for axis_i, axis in enumerate(ORDER):
        vals = AXES[axis]
        for lo, hi in zip(vals, vals[1:]):
            ds, dr = [], []
            for other in itertools.product(*(AXES[k] for k in ORDER if k != axis)):
                k1 = list(other); k1.insert(axis_i, lo)
                k2 = list(other); k2.insert(axis_i, hi)
                r1, r2 = ok.get(tuple(k1)), ok.get(tuple(k2))
                if not r1 or not r2:
                    continue
                if r1.get("shared") is not None and r2.get("shared") is not None:
                    ds.append(r2["shared"] - r1["shared"])
                if r1.get("n_regs") is not None and r2.get("n_regs") is not None:
                    dr.append(r2["n_regs"] - r1["n_regs"])
            if not ds and not dr:
                continue
            s_spread = (max(ds) - min(ds)) if ds else None
            r_spread = (max(dr) - min(dr)) if dr else None
            sep["%s:%s->%s" % (axis, lo, hi)] = dict(
                n=len(ds), d_shared_median=statistics.median(ds) if ds else None,
                d_shared_spread=s_spread,
                d_regs_median=statistics.median(dr) if dr else None,
                d_regs_spread=r_spread)
            print("    %-12s %-8s %6d %9s %9s %9s %9s"
                  % (axis, "%s->%s" % (lo, hi), max(len(ds), len(dr)),
                     ("%+d" % statistics.median(ds)) if ds else "-",
                     ("%d" % s_spread) if s_spread is not None else "-",
                     ("%+.1f" % statistics.median(dr)) if dr else "-",
                     ("%d" % r_spread) if r_spread is not None else "-"))

    n_sep_shared = sum(1 for v in sep.values()
                       if v["d_shared_spread"] == 0 and v["n"])
    n_sep_regs = sum(1 for v in sep.values()
                     if v["d_regs_spread"] == 0 and v["n"])
    tot = len([v for v in sep.values() if v["n"]])
    print("\n    one-step moves whose shared delta is IDENTICAL everywhere: %d / %d"
          % (n_sep_shared, tot))
    print("    one-step moves whose n_regs delta is IDENTICAL everywhere: %d / %d"
          % (n_sep_regs, tot))
    print("    => shared is %s; registers are %s"
          % ("separable (measure once, reuse)" if n_sep_shared == tot
             else "NOT separable (the same move costs different amounts depending on the "
                  "other knobs)",
             "separable" if n_sep_regs == tot
             else "NOT separable -- a local map must be re-measured around the current point"))

    out = dict(device=torch.cuda.get_device_name(dev),
               n_points=len(R), n_ok=len(ok), wall_s=wall,
               ms_per_point=1e3 * wall / len(R),
               trials_equivalent=wall / 18.6,
               nonmonotone_slices=len(viol), separability=sep,
               sep_shared=n_sep_shared, sep_regs=n_sep_regs, sep_total=tot,
               map={"|".join(map(str, k)): v for k, v in R.items()})
    p = os.environ.get("OUT", "/root/probe_resource_map.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nwrote %s" % p)

    # positive control: a sweep this wide must move both resources
    sh = {v["shared"] for v in ok.values() if v.get("shared") is not None}
    rg = {v["n_regs"] for v in ok.values() if v.get("n_regs") is not None}
    print("\nPOSITIVE CONTROL  distinct shared=%d  distinct n_regs=%d" % (len(sh), len(rg)))
    if len(sh) < 2 or len(rg) < 2:
        print("FAIL: a resource that does not move across this sweep means the probe is broken.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
