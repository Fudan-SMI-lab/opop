"""Was the L3:21 comparison affected by the cudnn.allow_tf32 default?

A parallel session recorded that torch 2.9 defaults matmul.allow_tf32=False but
cudnn.allow_tf32=TRUE, so a convolutional reference silently runs tf32 convolutions. L3:21 is
convolution-dense, and both L3:21 numbers in my comparison (external 4.833 ms, ours 4.414 ms) were
timed through the harness. If the harness left cudnn tf32 on for one side and off for the other, the
two timings would not be comparable.

Check both things that could bite:
  (1) what the harness's own precision setter actually does to BOTH flags;
  (2) whether the reference timing changes materially with cudnn tf32 on vs off -- if it does, the
      baseline number depends on the flag and must be quoted with it.
"""
import sys
import torch
sys.path.insert(0, "/root/autodl-tmp/opop-workspace/opop/src")

print(f"defaults on this box: matmul={torch.backends.cuda.matmul.allow_tf32} "
      f"cudnn={torch.backends.cudnn.allow_tf32}")

from kernel_optimizer.gpu import worker_main as wm
fn = getattr(wm, "_apply_precision", None) or getattr(wm, "_set_precision", None)
print("harness precision setter:", fn.__name__ if fn else "NOT FOUND")
for want in ("fp32", "tf32"):
    if fn:
        try:
            fn(want)
            print(f"  after {want:5s}: matmul={torch.backends.cuda.matmul.allow_tf32} "
                  f"cudnn={torch.backends.cudnn.allow_tf32}")
        except Exception as e:
            print(f"  {want}: {type(e).__name__}: {e}")

# (2) does the L3:21 reference care?
ref_ctx = {}
exec(compile(open("/root/autodl-tmp/opop-workspace/KernelBench/KernelBench/level3/"
                  "21_EfficientNetMBConv.py").read(), "ref", "exec"), ref_ctx)
dev = torch.device("cuda")
torch.manual_seed(0)
m = ref_ctx["Model"](112, 192, 5, 2, 6).to(dev)
x = torch.rand(10, 112, 224, 224, device=dev)

def med(f, n=50, warmup=15):
    with torch.no_grad():
        for _ in range(warmup): f()
        torch.cuda.synchronize()
        s = []
        for _ in range(n):
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record(); f(); b.record(); torch.cuda.synchronize(); s.append(a.elapsed_time(b))
    s.sort(); return s[len(s)//2]

print()
for label, mm, cud in (("both OFF (true ieee)", False, False),
                       ("cudnn ON only (torch default)", False, True),
                       ("both ON", True, True)):
    torch.backends.cuda.matmul.allow_tf32 = mm
    torch.backends.cudnn.allow_tf32 = cud
    print(f"  L3:21 reference, {label:30s} {med(lambda: m(x)):7.3f} ms")
