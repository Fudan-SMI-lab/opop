"""Is the old-size measurement usable, or is it all launch overhead and noise?

The first pass gave median 46.88 us with std 54.18 us -- std > median means the distribution
is dominated by outliers, so a "speedup" at this size may be measuring scheduler jitter. Two
checks: (1) a much longer sample with outlier trimming, (2) how much of the time is kernel
work at all, measured by timing an empty-ish graph replay for comparison.
"""
import sys, torch, torch.nn as nn
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

class Model(nn.Module):
    def __init__(s, i, o, g, bs):
        super().__init__()
        s.matmul = nn.Linear(i, o); s.bias = nn.Parameter(torch.randn(bs))
        s.group_norm = nn.GroupNorm(g, o)
    def forward(s, x):
        x = s.matmul(x); x = torch.sigmoid(x) * x; x = x + s.bias
        return s.group_norm(x)

torch.manual_seed(0)
dev = torch.device("cuda")
m = Model(512, 1024, 32, (1024,)).to(dev)
x = torch.randn(128, 512, device=dev)
print(f"device: {torch.cuda.get_device_name(0)}")

def samples(fn, n, warm=200):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    out = []
    for _ in range(n):
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        out.append(s.elapsed_time(e) * 1000.0)
    return sorted(out)

with torch.no_grad():
    v = samples(lambda: m(x), 1000)
n = len(v)
p = lambda q: v[int(q * (n - 1))]
print(f"\nOLD sizes, n=1000 (us):")
print(f"   min {v[0]:8.2f}   p10 {p(.10):8.2f}   median {p(.50):8.2f}   "
      f"p90 {p(.90):8.2f}   max {v[-1]:8.2f}")
trimmed = v[int(.05*n):int(.95*n)]
mean = sum(trimmed)/len(trimmed)
sd = (sum((t-mean)**2 for t in trimmed)/len(trimmed))**.5
print(f"   5-95% trimmed mean {mean:.2f} us, std {sd:.2f} us ({sd/mean*100:.1f}%)")

# How many kernels, and what is the pure launch cost of that many?
from torch.profiler import profile, ProfilerActivity
with torch.no_grad(), profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(20): m(x)
    torch.cuda.synchronize()
ks = [e for e in prof.key_averages() if e.device_time_total > 0]
print(f"\n   distinct CUDA kernels per forward: ~{sum(e.count for e in ks)//20}")
tot = sum(e.device_time_total for e in ks) / 20
print(f"   summed kernel device time: {tot:.2f} us  vs wall median {p(.50):.2f} us")
print(f"   -> {(p(.50)-tot)/p(.50)*100:.0f}% of wall time is NOT kernel execution "
      f"(launch/sync overhead)")
