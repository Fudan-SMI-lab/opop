"""What does ONE resource reading cost, when we do not measure latency?

This is the load-bearing number for the whole v3 "many collection algorithms" architecture.
The user's brief says the agent does not know what the other resources are doing, and does not
know what its change will do to them. Both gaps are answerable by *reading more resource
points* -- but only if a resource point is much cheaper than a timed trial.

Known costs already measured on this box:
  timed trial (median of 20 samples + correctness)  18.6 s
  compile alone, standalone process                 1.17 s
  compile alone, batched 48-in-one-process          7 ms marginal

Unknown, and measured here:
  (A) compile + read every pre-launch field, batched                  -> ? ms per config
  (B) compile + ONE launch + read n_regs/n_spills, batched            -> ? ms per config
  (C) same as B but with a correctness check on one input             -> ? ms per config

If (B) is on the order of tens of ms, then ~100 resource points cost less than one timed
trial, and giving the agent a local resource MAP instead of a single point is affordable.
If (B) is on the order of seconds, the map is unaffordable and v3 must work from single points.

The probe deliberately uses a real Triton GEMM whose tile knobs move shared memory and
registers in opposite directions, because that is the case the design has to serve
(L3:48 was simultaneously DRAM-bound at 93.7% and occupancy-bound at 17%).

Positive control, per the standing discipline that a negative result needs a probe that would
fail loudly: the shared-memory reading must CHANGE across tiles (a probe that returns a
constant is broken, which is how the first occupancy probe failed -- it used tl.zeros(), which
lives in registers and allocates no shared memory at all). Asserted at the end; the script
exits nonzero if shared memory is constant across tiles.
"""
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
CONFIGS = []
for bm in (32, 64, 128):
    for bn in (32, 64, 128):
        for bk in (32, 64):
            for ns in (2, 3, 4):
                for nw in (4, 8):
                    CONFIGS.append(dict(BM=bm, BN=bn, BK=bk, num_stages=ns, num_warps=nw))

PRELAUNCH = ("shared", "num_warps", "num_stages", "num_ctas", "cluster_dims")
POSTLAUNCH = ("n_regs", "n_spills", "n_max_threads")


def fields(k):
    if k is None:
        return {f: None for f in PRELAUNCH + POSTLAUNCH}
    out = {}
    md = getattr(k, "metadata", None)
    for f in PRELAUNCH:
        v = getattr(md, f, None) if md is not None else None
        if v is None:
            v = getattr(k, f, None)
        out[f] = v
    for f in POSTLAUNCH:
        v = getattr(k, f, None)
        if v is None and md is not None:
            v = getattr(md, f, None)
        out[f] = v
    return out


def compile_one(cfg):
    """Ahead-of-time compile with no launch. Returns the CompiledKernel."""
    return gemm.warmup(
        torch.empty(1, device="cuda"), torch.empty(1, device="cuda"),
        torch.empty(1, device="cuda"),
        M, N, K, K, 1, N, 1, N, 1,
        BM=cfg["BM"], BN=cfg["BN"], BK=cfg["BK"], GROUP=8,
        num_stages=cfg["num_stages"], num_warps=cfg["num_warps"],
        grid=(1,),
    )


def _kernel_cache():
    entry = gemm.device_caches[torch.cuda.current_device()]
    d = entry[0] if isinstance(entry, (tuple, list)) else entry
    return d if isinstance(d, dict) else {}


def launch_one(cfg, a, b, c):
    """Launch this config. Returns nothing -- the caller re-reads its OWN kernel object.

    Two accessor bugs found here, both of which produced plausible-looking wrong answers:

      1. Reading `device_caches[dev][0]` hands back the cache DICT, not a kernel. Every field
         then reads None, which is indistinguishable from "the hardware does not expose it".

      2. Diffing cache keys before/after the launch finds nothing, because warmup() already
         populated the cache for this very config. The fallback then returned the same first
         entry for all 108 configs -- n_regs came out as the constant 56, which reads as a
         real (and wrong) finding: "registers do not vary with tile size".

    The correct read is on the CompiledKernel that warmup() returned for THIS config: launching
    it runs _init_handles(), which fills n_regs/n_spills in place on that same object.
    """
    grid = (triton.cdiv(M, cfg["BM"]) * triton.cdiv(N, cfg["BN"]),)
    gemm[grid](a, b, c, M, N, K, K, 1, N, 1, N, 1,
               BM=cfg["BM"], BN=cfg["BN"], BK=cfg["BK"], GROUP=8,
               num_stages=cfg["num_stages"], num_warps=cfg["num_warps"])


def main():
    print("configs under test: %d  (M=N=K=%d, fp32 ieee dot)" % (len(CONFIGS), M))
    dev = torch.cuda.current_device()
    print("device: %s\n" % torch.cuda.get_device_name(dev))

    a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b = torch.randn(K, N, device="cuda", dtype=torch.float32)
    c = torch.empty(M, N, device="cuda", dtype=torch.float32)
    ref = None

    rows = []
    # ---- (A) compile only, batched, COLD cache ----
    t_compile = []
    compiled = {}
    handles = {}          # keep the CompiledKernel per config; (B) re-reads these in place
    for cfg in CONFIGS:
        key = tuple(sorted(cfg.items()))
        t0 = time.perf_counter()
        try:
            k = compile_one(cfg)
        except Exception as e:
            rows.append(dict(cfg=cfg, stage="compile", err=type(e).__name__ + ": " + str(e)[:90]))
            continue
        t_compile.append(time.perf_counter() - t0)
        compiled[key] = fields(k)
        handles[key] = k

    # ---- (A2) same compiles again, WARM cache ----
    # This separates "generating machine code" from "reading a field". Only the cold number
    # is the cost of a NEW resource point; the warm number is the cost of re-reading one.
    t_warm = []
    for cfg in CONFIGS:
        if tuple(sorted(cfg.items())) not in compiled:
            continue
        t0 = time.perf_counter()
        try:
            compile_one(cfg)
        except Exception:
            continue
        t_warm.append(time.perf_counter() - t0)

    # ---- (B) one launch, batched, read post-launch fields ----
    t_launch = []
    launched = {}
    for cfg in CONFIGS:
        key = tuple(sorted(cfg.items()))
        if key not in compiled:
            continue
        t0 = time.perf_counter()
        try:
            launch_one(cfg, a, b, c)
            torch.cuda.synchronize()
        except Exception as e:
            rows.append(dict(cfg=cfg, stage="launch", err=type(e).__name__ + ": " + str(e)[:90]))
            continue
        t_launch.append(time.perf_counter() - t0)
        # re-read THIS config's own kernel object: the launch filled its handles in place
        launched[key] = fields(handles[key])

    # ---- (C) launch + correctness on one input ----
    t_check = []
    for cfg in CONFIGS[: min(24, len(CONFIGS))]:
        key = tuple(sorted(cfg.items()))
        if key not in launched:
            continue
        t0 = time.perf_counter()
        try:
            launch_one(cfg, a, b, c)
            torch.cuda.synchronize()
            if ref is None:
                ref = (a @ b)
            ok = torch.allclose(c, ref, rtol=1e-3, atol=1e-3)
        except Exception:
            continue
        t_check.append(time.perf_counter() - t0)
        if not ok:
            rows.append(dict(cfg=cfg, stage="check", err="mismatch"))

    def rep(name, ts):
        if not ts:
            print("%-46s no successful samples" % name)
            return None
        med = statistics.median(ts)
        print("%-46s n=%3d  median %8.1f ms   min %7.1f  max %8.1f  total %6.2f s"
              % (name, len(ts), med * 1e3, min(ts) * 1e3, max(ts) * 1e3, sum(ts)))
        return med

    print("COST PER CONFIG")
    m_c = rep("(A) compile only, COLD cache", t_compile)
    m_w = rep("(A2) same compile, WARM cache", t_warm)
    m_l = rep("(B) + one launch, post-launch fields", t_launch)
    m_k = rep("(C) + correctness on one input", t_check)

    TRIAL_S = 18.6
    print()
    print("COMPARED TO A TIMED TRIAL (%.1f s measured on this box)" % TRIAL_S)
    for name, med in (("compile cold", m_c), ("compile warm", m_w),
                      ("launch (post-compile)", m_l), ("correctness (post-compile)", m_k)):
        if med:
            print("  %-27s %8.1f ms  =>  %8.1f points per timed trial"
                  % (name, med * 1e3, TRIAL_S / med))
    if m_c and m_l:
        print("  %-27s %8.1f ms  =>  %8.1f points per timed trial"
              % ("cold compile + launch", (m_c + m_l) * 1e3, TRIAL_S / (m_c + m_l)))

    # ---- what is knowable at each stage ----
    print()
    print("FIELD AVAILABILITY (n=%d compiled, %d launched)" % (len(compiled), len(launched)))
    for f in PRELAUNCH + POSTLAUNCH:
        pre = sum(1 for v in compiled.values() if v.get(f) is not None)
        post = sum(1 for v in launched.values() if v.get(f) is not None)
        print("  %-16s after compile %3d/%-3d    after launch %3d/%-3d"
              % (f, pre, len(compiled), post, len(launched)))

    # ---- positive control: shared memory must vary across tiles ----
    shared_vals = sorted({v["shared"] for v in compiled.values() if v.get("shared") is not None})
    regs_vals = sorted({v["n_regs"] for v in launched.values() if v.get("n_regs") is not None})
    print()
    print("POSITIVE CONTROL")
    print("  distinct shared_bytes across tiles: %d  %s"
          % (len(shared_vals), shared_vals[:8] + (["..."] if len(shared_vals) > 8 else [])))
    print("  distinct n_regs across tiles:       %d  %s"
          % (len(regs_vals), regs_vals[:8] + (["..."] if len(regs_vals) > 8 else [])))

    out = dict(
        device=torch.cuda.get_device_name(dev),
        n_configs=len(CONFIGS), n_compiled=len(compiled), n_launched=len(launched),
        median_compile_cold_ms=(m_c * 1e3 if m_c else None),
        median_compile_warm_ms=(m_w * 1e3 if m_w else None),
        median_launch_ms=(m_l * 1e3 if m_l else None),
        median_check_ms=(m_k * 1e3 if m_k else None),
        distinct_shared=len(shared_vals), distinct_regs=len(regs_vals),
        errors=rows[:20],
        points_per_trial_cold=(TRIAL_S / (m_c + m_l) if (m_c and m_l) else None),
    )
    p = os.environ.get("OUT", "/root/probe_resource_read_cost.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nwrote %s" % p)

    rc = 0
    if len(shared_vals) < 2:
        print("\nFAIL: shared_bytes is constant across tiles -- the probe is broken, "
              "not the hardware. (This is exactly how the first occupancy probe failed.)")
        rc = 2
    n_regs_read = sum(1 for v in launched.values() if v.get("n_regs") is not None)
    if n_regs_read == 0:
        print("\nFAIL: n_regs read None for all %d launched configs. Either the accessor is "
              "wrong or the field does not exist. DO NOT quote this as a hardware finding "
              "until the accessor is proven on at least one config." % len(launched))
        rc = 3
    elif len(regs_vals) < 2:
        # This is the failure that actually happened: the accessor returned the SAME kernel
        # object for all 108 configs, so n_regs was a plausible constant (56) rather than
        # None. A None is obviously broken; a constant looks like a finding. Requiring
        # variation across a tile sweep that provably changes shared memory 18 ways is the
        # control that separates the two.
        print("\nFAIL: n_regs is the constant %s across %d configs whose shared_bytes takes "
              "%d distinct values. A tile sweep this wide cannot leave register count fixed, "
              "so the accessor is reading one kernel repeatedly. This is the shape of a bug "
              "that reads as a result." % (regs_vals[0], n_regs_read, len(shared_vals)))
        rc = 4
    return rc


if __name__ == "__main__":
    sys.exit(main())
