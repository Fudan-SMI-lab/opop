"""Measure the real level2:37 baseline on this machine, and state the physical floor.

The reported numbers (90.30 us baseline on a 4090, 8.37 us at 11.80x) need checking against
the task's own tensor sizes before any run is planned around them.
"""
import sys, time, importlib.util, torch

sys.path.insert(0, "/mnt/d/Pyhon_projects/opop/KernelBench/src")
REF = "/mnt/d/Pyhon_projects/opop/KernelBench/KernelBench/level2/37_Matmul_Swish_Sum_GroupNorm.py"

spec = importlib.util.spec_from_file_location("ref", REF)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

dev = torch.device("cuda")
print(f"device: {torch.cuda.get_device_name(0)}")
props = torch.cuda.get_device_properties(0)
print(f"VRAM: {props.total_memory/2**30:.1f} GiB")

m = mod.Model(*mod.get_init_inputs()).to(dev)
inputs = [t.to(dev) for t in mod.get_inputs()]
x = inputs[0]
b, i = x.shape
o = m.matmul.out_features
print(f"\nshapes: input {tuple(x.shape)}  output ({b}, {o})")
print(f"input  {x.numel()*4/2**30:.3f} GiB")
print(f"output {b*o*4/2**30:.3f} GiB")

flops = 2 * b * i * o
traffic = (b * i + b * o) * 4
print(f"\nmatmul FLOPs      {flops/1e12:.3f} TFLOP")
print(f"min DRAM traffic  {traffic/2**30:.3f} GiB (read input + write output, nothing else)")

def bench(fn, n=50, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    ts = []
    for _ in range(n):
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts)//2], ts[0]

with torch.no_grad():
    med, mn = bench(lambda: m(x))
print(f"\neager  median {med*1000:.1f} us   min {mn*1000:.1f} us")
print(f"       -> {flops/(med*1e-3)/1e12:.1f} TFLOP/s, {traffic/(med*1e-3)/1e12:.3f} TB/s")

# Pure-copy control: the floor this hardware can do on that much traffic.
out = torch.empty(b, o, device=dev)
src = torch.empty(b, o, device=dev)
cmed, _ = bench(lambda: out.copy_(src))
print(f"\ncontrol: copy {b*o*4/2**30:.3f} GiB device-to-device median {cmed*1000:.1f} us")
print(f"       -> effective {2*b*o*4/(cmed*1e-3)/1e12:.3f} TB/s (read+write)")
