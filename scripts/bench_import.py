import sys, time
sys.path.insert(0, "/mnt/d/Pyhon_projects/opop/KernelBench/src")
t0 = time.time()
import torch
t1 = time.time()
import triton  # noqa: F401
t2 = time.time()
err = ""
try:
    import kernelbench.eval  # noqa: F401
except Exception as e:
    err = " kb=%s" % type(e).__name__
t3 = time.time()
torch.zeros(1, device="cuda")
t4 = time.time()
print("torch %5.1fs | triton %4.1fs | kernelbench %5.1fs | cuda %4.1fs | TOTAL %5.1fs%s"
      % (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0, err))
