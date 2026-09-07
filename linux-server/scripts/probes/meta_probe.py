import torch, triton, triton.language as tl
@triton.jit
def k(x, o, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    tl.store(o + off, tl.load(x + off, mask=m, other=0.) * 2.0, mask=m)
x = torch.randn(4096, device="cuda"); o = torch.empty_like(x)
c = k[(16,)](x, o, 4096, BLOCK=256)
torch.cuda.synchronize()
meta = getattr(c, "metadata", None)
print("has .metadata:", meta is not None)
for a in ("shared", "num_warps", "num_stages"):
    print("  metadata.%s = %s" % (a, getattr(meta, a, "ABSENT")))
print("  compiled.n_regs =", getattr(c, "n_regs", "ABSENT"), " .n_spills =", getattr(c, "n_spills", "ABSENT"))
