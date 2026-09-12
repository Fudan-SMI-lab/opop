"""Does a co-resident card's load show up in THIS card's measured latency?

The question this answers: box 4 has two 4090s, so two arms could run at once, one pinned per card.
The project's standing rule is "two runs on ONE box may never overlap", and its stated reason is
"each would time the other's kernels" -- which is a statement about one CARD. Two physical cards do
not share SMs, VRAM or DRAM bandwidth, so that reason does not obviously carry over. What they DO
share is a chassis: one power budget and one thermal envelope. If card 0 down-clocks because card 1
is drawing 450 W, then an arm's latency depends on what the other arm happens to be doing, and the
arms are no longer comparable.

That is measurable, so it should be measured rather than argued. Three phases:

    solo0   card 0 times its kernel, card 1 idle
    solo1   card 1 times its kernel, card 0 idle
    dual    both cards time their kernels simultaneously

If dual medians match solo medians within the noise this project has already quantified (per-trial
std was 16% of the mean on L3:48, near-ties span 9%, and the independent re-eval gap is +-2-4% with
an unstable sign), co-residency does not perturb timing and two arms may share the box on separate
cards. If they diverge beyond that, they may not.

WHY A COMPUTE-BOUND MATMUL. Power and heat are the only coupling, so the stress must be the kind
that draws power. Two cards do not contend for DRAM bandwidth -- each has its own -- so a
bandwidth-bound kernel would under-state the risk. fp32 without tf32 mirrors the project's own
strict-IEEE path and is the heaviest thing these cards actually do in a run.

WHY A SUSTAINED LOOP AND NOT 100 SAMPLES. A cold card does not throttle; a card 60 s into a run
does. Samples from the first half are discarded and the median is taken over the steady-state tail,
which is the regime a 12 h run spends its time in.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time


def smi(fields: str) -> list[str]:
    """nvidia-smi is queried per-process and reports ALL cards, not just the visible one.

    So the row is selected by the PHYSICAL index passed in, not by the CUDA ordinal -- under
    CUDA_VISIBLE_DEVICES=1 torch sees one card at ordinal 0 while nvidia-smi still calls it 1.
    """
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    ).stdout.decode("utf-8", "replace")
    return [ln.strip() for ln in out.strip().splitlines() if ln.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phys", type=int, required=True, help="physical card index, for nvidia-smi")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch

    # The project's strict path: no tf32 anywhere, including cudnn -- leaving cudnn's tf32 on is a
    # recorded way to make a measurement mean something other than what it says.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    dev = torch.device("cuda:0")  # ordinal 0 of whatever CUDA_VISIBLE_DEVICES exposed
    name = torch.cuda.get_device_name(0)
    n = args.n
    a = torch.randn(n, n, device=dev, dtype=torch.float32)
    b = torch.randn(n, n, device=dev, dtype=torch.float32)

    def once() -> None:
        torch.mm(a, b)

    for _ in range(20):
        once()
    torch.cuda.synchronize()

    samples: list[tuple[float, float]] = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < args.seconds:
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        once()
        e.record()
        torch.cuda.synchronize()
        samples.append((time.monotonic() - t0, s.elapsed_time(e)))

    half = args.seconds / 2.0
    tail = [ms for t, ms in samples if t >= half]
    if not tail:                      # a very slow card could produce nothing in the tail
        tail = [ms for _, ms in samples]

    row = smi("clocks.sm,power.draw,power.limit,temperature.gpu,utilization.gpu")
    mine = row[args.phys].split(",") if args.phys < len(row) else ["?"] * 5

    result = {
        "phys": args.phys,
        "device_name": name,
        "n_samples": len(samples),
        "n_tail": len(tail),
        "median_ms": statistics.median(tail),
        "mean_ms": statistics.fmean(tail),
        "min_ms": min(tail),
        "max_ms": max(tail),
        "p90_ms": sorted(tail)[int(0.9 * (len(tail) - 1))],
        "stdev_pct": (statistics.pstdev(tail) / statistics.fmean(tail) * 100.0
                      if len(tail) > 1 else 0.0),
        "clocks_sm_mhz": mine[0].strip(),
        "power_draw_w": mine[1].strip(),
        "power_limit_w": mine[2].strip(),
        "temp_c": mine[3].strip(),
        "util_pct": mine[4].strip(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)"),
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
