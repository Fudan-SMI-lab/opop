import sys, time

sys.path.insert(0, "/mnt/d/Pyhon_projects/opop/KernelBench/src")
t0 = time.time()
import torch
import triton
import triton.language as tl

# worker_main stubs litellm before importing kernelbench; mirror that here so the
# kernelbench import cost is measured the way the real worker pays it.
import importlib.util
import types

if importlib.util.find_spec("litellm") is None:
    stub = types.ModuleType("litellm")
    stub.completion = None
    sys.modules["litellm"] = stub
import kernelbench.eval  # noqa: F401

t1 = time.time()


@triton.jit
def _add(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    tl.store(o_ptr + off, tl.load(x_ptr + off, mask=m) + tl.load(y_ptr + off, mask=m), mask=m)


n = 4096
x = torch.randn(n, device="cuda")
y = torch.randn(n, device="cuda")
o = torch.empty_like(x)
_add[(triton.cdiv(n, 256),)](x, y, o, n, BLOCK=256)
torch.cuda.synchronize()
t2 = time.time()
ok = torch.allclose(o, x + y)
print("imports(incl kernelbench) %5.1fs | triton compile+launch %5.1fs | TOTAL %5.1fs | correct=%s"
      % (t1 - t0, t2 - t1, t2 - t0, ok))
