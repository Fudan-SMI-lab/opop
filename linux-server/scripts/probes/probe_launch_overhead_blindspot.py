"""Measure how much of a fused kernel's real advantage the harness's timing hides.

THE BLIND SPOT, in the code (KernelBench timing.py:251-263):

    torch.cuda.synchronize()          # queue is EMPTY here
    clear_l2_cache(device)            # enqueues a 256 MB fill on the SAME stream
    start_event.record()              # <- timing starts
    kernel_fn(*args)                  # CPU issues the launches...
    end_event.record()                # <- timing ends
    torch.cuda.synchronize()

`clear_l2_cache` enqueues ~256 MB of GPU work before the timed region opens. The GPU is busy
with it while the CPU races ahead issuing `kernel_fn`'s launches, so by the time the GPU reaches
`start_event` every launch is already queued. The events therefore measure GPU execution with
the CPU launch cost hidden behind the flush.

That is fine for comparing two candidates. It is NOT fine for crediting fusion: fusion's whole
mechanism on a small task is REMOVING LAUNCHES, and this measurement gives the multi-launch
baseline a free ride on exactly that cost.

THE 2x2 that separates the two explanations. If cold L2 were the cause, arm C (cold cache,
drained queue) would look like arm A. If queue depth is the cause, C looks like B.

    A  cold L2 + deep queue   <- what the harness measures
    B  warm L2 + shallow queue
    C  cold L2 + DRAINED queue (synchronize between flush and timing)
    D  warm L2 + deep queue

Plus F: the pure CPU time to ISSUE the op sequence, with no GPU wait at all. If F is large
relative to A, the launch cost is real and the harness is hiding it.

The internal control that needs no external reference: run the same 2x2 on a MILLISECOND task.
There the CPU can never outrun the GPU, so queue depth must be worth ~nothing -- if the pattern
appears there too, the explanation is wrong.

Usage:
    python probe_launch_overhead_blindspot.py
"""
from __future__ import annotations

import statistics
import time

import torch
import torch.nn.functional as F

EPS = 1e-5


def make_l2_37():
    """The overhead-bound case: 6 PyTorch ops, ~29 us of GPU work."""
    M, K, N, G = 128, 512, 1024, 32
    torch.manual_seed(0)
    x = torch.randn(M, K, device="cuda")
    w = torch.randn(N, K, device="cuda")
    wb = torch.randn(N, device="cuda")
    bias = torch.randn(N, device="cuda")
    gnw = torch.randn(N, device="cuda")
    gnb = torch.randn(N, device="cuda")

    def fn():
        z = F.linear(x, w, wb)
        h = torch.sigmoid(z) * z
        t = h + bias
        return F.group_norm(t, G, gnw, gnb, EPS)
    return "level2:37 (128x512x1024, 6 ops)", fn


def make_big_matmul():
    """The CONTROL: one op, milliseconds of GPU work. Queue depth cannot matter here."""
    torch.manual_seed(0)
    a = torch.randn(4096, 4096, device="cuda")
    b = torch.randn(4096, 4096, device="cuda")

    def fn():
        return a @ b
    return "4096^3 matmul (1 op, ms-scale)", fn


def make_many_small():
    """The other extreme: 40 tiny ops. Launch-dominated by construction."""
    torch.manual_seed(0)
    x = torch.randn(256, 256, device="cuda")

    def fn():
        y = x
        for _ in range(40):
            y = torch.relu(y) + 1.0
        return y
    return "40 tiny elementwise ops", fn


def flush_l2():
    dummy = torch.empty((32, 1024, 1024), dtype=torch.int64, device="cuda")
    dummy.fill_(42)
    del dummy


def med(xs):
    return statistics.median(xs)


def arm_A(fn, n):
    """cold L2 + deep queue -- EXACTLY what KernelBench/our harness does."""
    out = []
    for _ in range(n):
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        flush_l2()                    # enqueued, NOT waited on
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e))
    return out


def arm_B(fn, n):
    """warm L2 + shallow queue: no flush, and the queue is empty when timing opens."""
    out = []
    for _ in range(n):
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e))
    return out


def arm_C(fn, n):
    """cold L2 + DRAINED queue: flush, then WAIT for it, then time."""
    out = []
    for _ in range(n):
        torch.cuda.synchronize()
        flush_l2()
        torch.cuda.synchronize()      # <- the only difference from arm A
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e))
    return out


def arm_D(fn, n):
    """warm L2 + deep queue: fill the queue with cheap work instead of a 256 MB flush."""
    filler = torch.randn(2048, 2048, device="cuda")
    out = []
    for _ in range(n):
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        for _ in range(3):
            filler.mul_(1.0)          # enqueued, not waited on
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e))
    return out


def arm_F(fn, n):
    """CPU time to ISSUE the sequence: no GPU wait at all."""
    torch.cuda.synchronize()
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1e3)
    torch.cuda.synchronize()
    return out


def arm_wall(fn, n):
    """Wall clock per call, GPU included: what a user actually experiences."""
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) * 1e3)
    return out


def main() -> int:
    p = torch.cuda.get_device_properties(0)
    print(f"device: {p.name} sm_{p.major}{p.minor}  torch {torch.__version__}")
    print()
    for label, fn in (make_l2_37(), make_many_small(), make_big_matmul()):
        for _ in range(30):
            fn()
        torch.cuda.synchronize()
        n = 120
        a, b, c, d = (med(arm_A(fn, n)), med(arm_B(fn, n)),
                      med(arm_C(fn, n)), med(arm_D(fn, n)))
        f, wall = med(arm_F(fn, n)), med(arm_wall(fn, n))
        print(f"--- {label}")
        print(f"  A cold L2 + deep queue   (THE HARNESS) : {a*1e3:9.1f} us")
        print(f"  B warm L2 + shallow queue              : {b*1e3:9.1f} us")
        print(f"  C cold L2 + DRAINED queue              : {c*1e3:9.1f} us")
        print(f"  D warm L2 + deep queue                 : {d*1e3:9.1f} us")
        print(f"  F CPU time to ISSUE (no GPU wait)      : {f*1e3:9.1f} us")
        print(f"    wall clock per call                  : {wall*1e3:9.1f} us")
        print(f"  queue depth is worth : {(c-a)*1e3:+9.1f} us   (C - A)")
        print(f"  cold L2 is worth     : {(a-d)*1e3:+9.1f} us   (A - D)")
        print(f"  cold/warm ratio      : {a/d if d else 0:9.2f}x")
        print(f"  HIDDEN by the harness: {(wall-a)*1e3:+9.1f} us   "
              f"({wall/a if a else 0:.2f}x the reported number)")
        print()
    print("READ THIS AS: on the ms-scale control, queue depth must be ~0 and the cold/warm")
    print("ratio ~1.00x. If it is, the launch-overhead explanation holds for the small task.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
