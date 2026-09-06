"""Baseline latency for level2:37 at the OLD (pre-b08d959) sizes vs the CURRENT ones.

Another team reports 90.30 us baseline and 8.37 us at 11.80x on a 4090 for this task. At the
sizes KernelBench ships today (batch 32768, in 1024, out 4096) those numbers are below the
memory-bandwidth floor, so they must come from the pre-scaling version. KernelBench commit
b08d959 ("scaled up the tensor sizes for level 1, 2 and 3, so that all problems run above
1ms") changed 128/512/1024/32 -> 32768/1024/4096/64 and also swapped `torch.randn` for
`torch.rand` in get_inputs.

This measures both configurations on this machine so the comparison has a local number, and
reports the bandwidth floor alongside each so an impossible claim is visible as impossible.

Run inside the WSL venv:
  wsl.exe -d Ubuntu -- bash -lc "source ~/kernel-opt-venv/bin/activate && \
      python /mnt/d/Pyhon_projects/opop/v2/scripts/probe_l2_37_old_vs_new.py"
"""
from __future__ import annotations

import sys

import torch
import torch.nn as nn

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)


class Model(nn.Module):
    """Verbatim from KernelBench level2/37 -- identical in both revisions."""

    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)

    def forward(self, x):
        x = self.matmul(x)
        x = torch.sigmoid(x) * x           # Swish
        x = x + self.bias
        x = self.group_norm(x)
        return x


# (label, batch, in, out, groups, input generator) -- the generator differs between revisions
CONFIGS = [
    ("OLD  (bb27f27, pre-scaling)", 128, 512, 1024, 32, torch.randn),
    ("CURRENT (b08d959..423217d)", 32768, 1024, 4096, 64, torch.rand),
]

# Their reported figures, for the OLD sizes on a 4090.
THEIRS = {"baseline": 90.30, "8.85x": 90.30 / 8.85, "11.80x": 8.37}
BW_4090 = 1.008e12       # bytes/s, RTX 4090 spec
DEV_NAME = None


def bench(fn, n: int = 100, warm: int = 20) -> tuple[float, float, float]:
    """Median / min / std in milliseconds, CUDA-event timed, matching the harness."""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    samples = []
    for _ in range(n):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    mean = sum(samples) / len(samples)
    var = sum((s - mean) ** 2 for s in samples) / len(samples)
    return samples[len(samples) // 2], samples[0], var ** 0.5


def main() -> int:
    global DEV_NAME
    if not torch.cuda.is_available():
        print("no CUDA device")
        return 1
    DEV_NAME = torch.cuda.get_device_name(0)
    print(f"device: {DEV_NAME}")
    print(f"torch:  {torch.__version__}")
    print()

    results = {}
    for label, b, i, o, g, gen in CONFIGS:
        torch.manual_seed(0)
        dev = torch.device("cuda")
        m = Model(i, o, g, (o,)).to(dev)
        x = gen(b, i, device=dev)

        flops = 2 * b * i * o
        traffic = (b * i + b * o) * 4     # read input + write output, the hard floor

        print(f"=== {label}")
        print(f"    batch={b} in={i} out={o} groups={g}  input gen={gen.__name__}")
        print(f"    input {b * i * 4 / 2**20:9.2f} MiB   output {b * o * 4 / 2**20:9.2f} MiB")
        print(f"    matmul {flops / 1e9:8.3f} GFLOP   min DRAM traffic "
              f"{traffic / 2**20:8.2f} MiB")

        with torch.no_grad():
            med, mn, sd = bench(lambda: m(x))
        us = med * 1000.0
        results[label] = us
        print(f"    eager   median {us:10.2f} us   min {mn * 1000:10.2f} us   "
              f"std {sd * 1000:8.2f} us")
        print(f"            -> {flops / (med * 1e-3) / 1e12:6.1f} TFLOP/s, "
              f"{traffic / (med * 1e-3) / 1e12:6.3f} TB/s")

        # torch.compile, the harness's second baseline.
        try:
            cm = torch.compile(m)
            with torch.no_grad():
                cmed, cmn, _ = bench(lambda: cm(x), n=100, warm=30)
            print(f"    compile median {cmed * 1000:10.2f} us   min {cmn * 1000:10.2f} us")
            results[label + " [compile]"] = cmed * 1000.0
        except Exception as exc:                      # noqa: BLE001
            print(f"    compile FAILED: {type(exc).__name__}: {str(exc)[:90]}")

        floor_us = traffic / BW_4090 * 1e6
        print(f"    4090 bandwidth floor for this traffic: {floor_us:.2f} us")
        print()

        del m, x
        torch.cuda.empty_cache()

    old_label = CONFIGS[0][0]
    print("=" * 76)
    print("COMPARISON WITH THE REPORTED FIGURES (theirs: 4090, OLD sizes)")
    print("=" * 76)
    ours = results[old_label]
    print(f"  their baseline        {THEIRS['baseline']:9.2f} us  (4090)")
    print(f"  ours, same sizes      {ours:9.2f} us  ({DEV_NAME})")
    print(f"  ratio                 {ours / THEIRS['baseline']:9.2f}x  "
          f"(>1 means we are slower)")
    print()
    for k in ("8.85x", "11.80x"):
        tgt = THEIRS[k]
        print(f"  their {k:7s} result  {tgt:9.2f} us   -> to match it from OUR baseline "
              f"we would need {ours / tgt:6.2f}x")
    print()
    cur = results[CONFIGS[1][0]]
    print(f"  CURRENT sizes on this machine: {cur:.2f} us "
          f"({cur / ours:.0f}x the old-size time)")
    print(f"  -> a speedup measured at one size does not transfer to the other; the "
          f"bottleneck moves.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
