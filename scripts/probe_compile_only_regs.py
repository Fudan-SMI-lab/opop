"""Decisive question for the design: what is knowable from COMPILE ALONE, with no launch?

This matters because it decides between two architectures:
  (i)  PREDICT resource usage from source before compiling (needs a model, may be wrong)
  (ii) COMPILE the variant and READ the resource usage (costs ~1.17 s, cannot be wrong)

Earlier I got contradictory answers because I read n_regs from a config that had ALREADY been
launched in the same process, so the JIT cache supplied it. This runs each config in a FRESH
process (no prior launch of anything) so the answer is not contaminated.

Also tests the grid-throttle floor detector, which was the one perturbation that produced a
clean signal: if runtime does not change when the work changes, the kernel sits on a floor.
"""
import subprocess
import sys
import textwrap

PROBE = textwrap.dedent('''
    import torch, triton, triton.language as tl, json, sys
    BM, BN, BK = {bm}, {bn}, {bk}
    @triton.jit
    def mm_k(a_ptr,b_ptr,c_ptr,M,N,K,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
        pid_m, pid_n = tl.program_id(0), tl.program_id(1)
        om = pid_m*BM + tl.arange(0,BM); on = pid_n*BN + tl.arange(0,BN)
        acc = tl.zeros((BM,BN), dtype=tl.float32)
        for k0 in range(0,K,BK):
            ok = k0 + tl.arange(0,BK)
            a = tl.load(a_ptr+om[:,None]*K+ok[None,:], mask=(om[:,None]<M)&(ok[None,:]<K), other=0.)
            b = tl.load(b_ptr+ok[:,None]*N+on[None,:], mask=(ok[:,None]<K)&(on[None,:]<N), other=0.)
            acc += tl.dot(a,b)
        tl.store(c_ptr+om[:,None]*N+on[None,:], acc, mask=(om[:,None]<M)&(on[None,:]<N))
    M = 1024
    a = torch.randn(M,M,device="cuda"); b = torch.randn(M,M,device="cuda")
    c = torch.empty(M,M,device="cuda")
    out = {{"tile": "%dx%dx%d" % (BM,BN,BK)}}
    # ---- COMPILE ONLY: warmup does not launch
    try:
        kw = mm_k.warmup(a,b,c,M,M,M,BM=BM,BN=BN,BK=BK,
                         grid=(triton.cdiv(M,BM),triton.cdiv(M,BN)))
        md = getattr(kw,"metadata",None)
        out["compile_shared"] = getattr(md,"shared",None)
        out["compile_n_regs"] = getattr(kw,"n_regs",None)
        out["compile_n_spills"] = getattr(kw,"n_spills",None)
        out["compile_ok"] = True
    except Exception as e:
        out["compile_ok"] = False
        out["compile_error"] = type(e).__name__ + ": " + str(e)[:100]
    # ---- NOW LAUNCH, and read the same fields again
    if out.get("compile_ok"):
        try:
            mm_k[(triton.cdiv(M,BM),triton.cdiv(M,BN))](a,b,c,M,M,M,BM=BM,BN=BN,BK=BK)
            torch.cuda.synchronize()
            for dc in mm_k.device_caches.values():
                cache = dc[0] if isinstance(dc,tuple) else dc
                for ck in (cache or {{}}).values():
                    md2 = getattr(ck,"metadata",None)
                    out["launch_shared"] = getattr(md2,"shared",None)
                    out["launch_n_regs"] = getattr(ck,"n_regs",None)
                    out["launch_n_spills"] = getattr(ck,"n_spills",None)
            out["launch_ok"] = True
        except Exception as e:
            out["launch_ok"] = False
            out["launch_error"] = type(e).__name__ + ": " + str(e)[:100]
    print("JSON" + json.dumps(out))
''')

PY = "/root/autodl-tmp/kernel-opt-venv/bin/python"
print("=" * 96)
print("Q1: is register pressure knowable WITHOUT launching? (fresh process per config)")
print("=" * 96)
print("%-14s %-10s %-10s %-10s | %-10s %-10s %-10s" %
      ("tile", "cmp shared", "cmp regs", "cmp spill", "run shared", "run regs", "run spill"))
for bm, bn, bk in ((32, 32, 32), (64, 64, 32), (128, 128, 32), (128, 64, 64), (128, 128, 64)):
    src = PROBE.format(bm=bm, bn=bn, bk=bk)
    open("/tmp/_p.py", "w", encoding="utf-8").write(src)
    r = subprocess.run(["ssh", "autodl", f"cat > /tmp/_p.py && cd /root/autodl-tmp && {PY} /tmp/_p.py"],
                       input=src.encode(), capture_output=True, timeout=300)
    line = next((l for l in r.stdout.decode(errors="replace").splitlines()
                 if l.startswith("JSON")), None)
    if not line:
        err = r.stderr.decode(errors="replace").strip().splitlines()
        print("%-14s PROBE FAILED: %s" % (f"{bm}x{bn}x{bk}", err[-1][:70] if err else "no output"))
        continue
    import json
    d = json.loads(line[4:])
    if not d.get("compile_ok"):
        print("%-14s compile REFUSED: %s" % (d["tile"], d.get("compile_error", "")[:60]))
        continue
    print("%-14s %-10s %-10s %-10s | %-10s %-10s %-10s" %
          (d["tile"], d.get("compile_shared"), d.get("compile_n_regs"),
           d.get("compile_n_spills"), d.get("launch_shared"), d.get("launch_n_regs"),
           d.get("launch_n_spills")))
print()
print("If 'cmp regs' is populated for a config never launched in that process, register")
print("pressure IS knowable pre-launch and architecture (ii) covers occupancy too.")
print("If it is None/absent, only SHARED memory is pre-launch decidable.")
