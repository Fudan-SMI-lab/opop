"""Does more warmup remove the first-sample artifact, and how much is enough?

Measured fact this probe exists to act on: in run-l2-37-20260907-020707 every one of 7 trials
had sample #1 at 285-448 us while samples 2-20 sat inside a 1-3 us band (e.g. 14.0-16.1 us).
7/7, with zero other outliers. So the task's timing noise is almost entirely ONE
deterministic artifact at the first timed launch, and `num_warmup=3`
(gpu/worker_main.py, time_execution_with_cuda_event calls) does not absorb it.

That matters beyond noise: candidates are summarized by a median (robust to the artifact)
while baselines come from KernelBench's summary-only path and have only a mean, so the
headline comparison is across two different statistics. If more warmup removes the artifact,
mean and median converge and the comparison becomes same-statistic again -- a root-cause fix
rather than a defence.

The question is empirical and has three parts:
  Q1 Does raising num_warmup actually remove it, or does the first TIMED launch always pay
     the cost regardless of how many untimed ones preceded it?
  Q2 If it works, how many warmups are needed? (Cost is paid on every trial's hot path.)
  Q3 Is the artifact per-process, per-kernel-compile, or per-timing-loop? A fresh worker
     process per job (which this harness uses) means anything cached at process level is
     paid once per job no matter what.

Run inside the WSL venv:
  wsl.exe -d Ubuntu -- bash -lc "source ~/kernel-opt-venv/bin/activate && \
      python /mnt/d/Pyhon_projects/opop/v2/scripts/probe_warmup_artifact.py"
"""
from __future__ import annotations

import statistics as st
import sys

import torch
import torch.nn as nn

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

BATCH, IN, OUT, GROUPS = 128, 512, 1024, 32     # level2:37, pre-scaling sizes
N_TIMED = 20                                     # what quick_perf_trials uses
WARMUPS = [0, 1, 3, 5, 10, 20, 50]               # 3 is today's value


class Model(nn.Module):
    def __init__(self, in_f, out_f, groups):
        super().__init__()
        self.matmul = nn.Linear(in_f, out_f)
        self.bias = nn.Parameter(torch.randn(out_f))
        self.gn = nn.GroupNorm(groups, out_f)

    def forward(self, x):
        x = self.matmul(x)
        x = torch.sigmoid(x) * x
        x = x + self.bias
        return self.gn(x)


def timed(fn, n: int, warm: int) -> list[float]:
    """Exactly the harness's shape: `warm` untimed calls, then n CUDA-event timings (ms)."""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    out = []
    for _ in range(n):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        out.append(start.elapsed_time(end))
    return out


def describe(s: list[float]) -> str:
    us = [v * 1000 for v in s]
    rest = us[1:]
    med = st.median(us)
    return (f"first {us[0]:7.1f} | rest {min(rest):5.1f}-{max(rest):5.1f} | "
            f"median {med:5.1f} | mean {sum(us)/len(us):6.1f} | "
            f"mean/median {(sum(us)/len(us))/med:4.2f}x | "
            f"first/median {us[0]/med:5.1f}x")


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA device")
        return 1
    print(f"device: {torch.cuda.get_device_name(0)}   torch {torch.__version__}")
    print(f"timed samples per measurement: {N_TIMED}")
    print()

    dev = torch.device("cuda")
    torch.manual_seed(0)
    model = Model(IN, OUT, GROUPS).to(dev)
    x = torch.randn(BATCH, IN, device=dev)
    torch.backends.cuda.matmul.allow_tf32 = True

    # ---- Q1/Q2: does more warmup remove it, and how much is enough? -------------------
    # Each row repeats the FULL sequence (warmup then timing) so the artifact, if it is a
    # property of the first timed launch, has a fresh chance to appear every time.
    print("=" * 104)
    print("Q1/Q2  warmup sweep -- same process, sequence repeated per row")
    print("=" * 104)
    with torch.no_grad():
        for w in WARMUPS:
            s = timed(lambda: model(x), N_TIMED, w)
            print(f"  warmup={w:<3} {describe(s)}")
    print()
    print("  Read: if `first/median` collapses toward 1.0 as warmup grows, warmup is the")
    print("  cure. If it stays high at every warmup count, the cost belongs to the first")
    print("  TIMED launch (not to a cold kernel) and warmup cannot fix it -- the median")
    print("  stays the right answer and the mean cannot be repaired by warming up.")
    print()

    # ---- Q3: is it per-process? -------------------------------------------------------
    # If the artifact is a one-off process-level cost (allocator arena, CUDA graph capture,
    # cuBLAS workspace, autotune cache), then a SECOND measurement in the same process
    # should be clean regardless of warmup. The harness runs one fresh process per job, so a
    # per-process artifact is paid once per job and warmup inside that job would fix it.
    print("=" * 104)
    print("Q3  repeated measurements in ONE process, warmup=3 (today's value) each time")
    print("=" * 104)
    with torch.no_grad():
        for i in range(6):
            s = timed(lambda: model(x), N_TIMED, 3)
            print(f"  measurement {i + 1}: {describe(s)}")
    print()
    print("  Read: if only measurement 1 shows the artifact, it is per-PROCESS -- and since")
    print("  the harness spawns a fresh worker per job, every job pays it exactly once, on")
    print("  its first timed sample. That is consistent with 7/7 trials showing it.")
    print()

    # ---- Does it follow a change of shape/config, i.e. is it per-compile? --------------
    print("=" * 104)
    print("Q3b  does a NEW configuration re-trigger it inside the same process?")
    print("=" * 104)
    with torch.no_grad():
        for i, (b, tf32) in enumerate([(BATCH, True), (BATCH, False),
                                       (BATCH * 2, True), (BATCH, True)]):
            torch.backends.cuda.matmul.allow_tf32 = tf32
            xx = torch.randn(b, IN, device=dev)
            model(xx)                     # one untimed call so any compile happens here
            torch.cuda.synchronize()
            s = timed(lambda: model(xx), N_TIMED, 3)
            print(f"  cfg {i + 1} (batch={b}, tf32={tf32}): {describe(s)}")
    print()
    print("  Read: an artifact that reappears for each new shape/precision is a per-compile")
    print("  or per-workspace cost; one that appears only on cfg 1 is per-process.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
