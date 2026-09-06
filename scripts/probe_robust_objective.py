"""Does a robust statistic actually beat the mean as a tuning objective, or just look better?

The tuning objective is `record.latency_ms.mean` over `quick_perf_trials: 20` samples
(tuning/tpe.py:85, control/orchestrator.py:988). On level2:37 at pre-scaling sizes that mean
carries a 24-53% standard error against a `min_improvement_pct` of 2.0, because a few
300-700 us scheduling stalls drag a 20-sample mean by 35-136%
(docs/finding-tuning-objective-is-a-20-sample-mean.md).

Replacing it with a trimmed mean or median is the obvious fix, but "obvious" is not
evidence. The risks that matter are specific and testable:

  R1 BIAS -- a robust statistic estimates a different quantity than the mean. If real
     throughput cost includes occasional stalls, discarding them reports a latency the user
     will never observe. Measured here as: does the statistic track the TRUE cost (estimated
     from a large sample) better or worse than the mean?
  R2 STABILITY -- the point of the change. Does it actually reduce run-to-run variance of
     the objective at n=20?
  R3 RANKING -- does it preserve the ordering of genuinely different configurations? A
     statistic that is stable but ranks wrongly is worse than a noisy one that ranks right.
  R4 DISCRIMINATION -- can it still separate configurations whose true costs differ by
     around min_improvement_pct (2%)? Over-smoothing could erase real small differences.
  R5 MASKING -- a kernel that is genuinely bimodal (e.g. a slow path taken 30% of the time)
     has a real cost the median would hide. Does trimming hide a defect we want to see?

Method: time several real configurations of the same shape as the harness's own trials, with
a LARGE sample count to establish ground truth, then repeatedly draw n=20 windows and compare
estimators against that truth. No synthetic noise -- the stalls are whatever this machine
actually does.

Run inside the WSL venv:
  wsl.exe -d Ubuntu -- bash -lc "source ~/kernel-opt-venv/bin/activate && \
      python /mnt/d/Pyhon_projects/opop/v2/scripts/probe_robust_objective.py"
"""
from __future__ import annotations

import statistics as st
import sys

import torch
import torch.nn as nn

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

BATCH, IN, OUT, GROUPS = 128, 512, 1024, 32     # level2:37, pre-scaling sizes
N_TRUTH = 2000                                   # ground-truth sample count
N_WINDOW = 20                                    # what the harness actually uses
N_DRAWS = 400                                    # how many n=20 windows to simulate


class Model(nn.Module):
    """level2/37: Matmul + Swish + bias + GroupNorm."""

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


def raw_samples(fn, n: int, warm: int = 30) -> list[float]:
    """n CUDA-event timings in ms, exactly the harness's timing method."""
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


def trimmed_mean(xs: list[float], frac: float = 0.1) -> float:
    s = sorted(xs)
    k = int(len(s) * frac)
    core = s[k:len(s) - k] if len(s) - 2 * k > 0 else s
    return sum(core) / len(core)


ESTIMATORS = {
    "mean":      lambda xs: sum(xs) / len(xs),
    "median":    st.median,
    "trim10%":   lambda xs: trimmed_mean(xs, 0.10),
    "trim20%":   lambda xs: trimmed_mean(xs, 0.20),
    "min":       min,
}


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA device")
        return 1
    print(f"device: {torch.cuda.get_device_name(0)}   torch {torch.__version__}")
    print(f"ground truth n={N_TRUTH}, window n={N_WINDOW}, draws={N_DRAWS}")
    print()

    dev = torch.device("cuda")
    torch.manual_seed(0)
    model = Model(IN, OUT, GROUPS).to(dev)
    x = torch.randn(BATCH, IN, device=dev)

    # Several genuinely different configurations, so ranking can be tested. eager and
    # compiled variants differ by a real amount; the tf32/ieee split differs by more.
    configs: dict[str, object] = {}
    with torch.no_grad():
        torch.backends.cuda.matmul.allow_tf32 = False
        configs["eager_ieee"] = ("eager", False)
        torch.backends.cuda.matmul.allow_tf32 = True
        configs["eager_tf32"] = ("eager", True)
        try:
            compiled = torch.compile(model)
            compiled(x)                       # trigger compilation outside timing
            torch.cuda.synchronize()
            configs["compile_tf32"] = ("compile", True)
            configs["compile_ieee"] = ("compile", False)
        except Exception as exc:              # noqa: BLE001
            print(f"torch.compile unavailable: {type(exc).__name__}: {str(exc)[:80]}")
            compiled = None

    truth: dict[str, list[float]] = {}
    for name, (kind, tf32) in configs.items():
        torch.backends.cuda.matmul.allow_tf32 = tf32
        m = compiled if kind == "compile" else model
        with torch.no_grad():
            truth[name] = raw_samples(lambda: m(x), N_TRUTH)
        s = sorted(truth[name])
        print(f"=== {name}")
        print(f"    n={len(s)}  min {s[0]*1000:7.2f}  p10 {s[len(s)//10]*1000:7.2f}  "
              f"median {st.median(s)*1000:7.2f}  p90 {s[9*len(s)//10]*1000:7.2f}  "
              f"max {s[-1]*1000:8.2f} us")
        print(f"    mean {sum(s)/len(s)*1000:7.2f} us   mean/min "
              f"{(sum(s)/len(s))/s[0]:5.2f}x   "
              f"samples >2x median: {sum(1 for v in s if v > 2*st.median(s))}"
              f" ({sum(1 for v in s if v > 2*st.median(s))/len(s)*100:.1f}%)")
    print()

    # ---- R1 BIAS + R2 STABILITY -------------------------------------------------------
    # Ground truth per config = the estimator applied to ALL N_TRUTH samples. Note each
    # estimator has its OWN truth: a median's target is the true median, not the true mean.
    # Comparing a median estimate to a mean truth would manufacture a bias that is really
    # just a definition difference, which is the trap this section avoids.
    print("=" * 96)
    print("R1 BIAS / R2 STABILITY -- each estimator vs its own large-sample value, "
          f"over {N_DRAWS} windows of n={N_WINDOW}")
    print("=" * 96)
    print(f"{'config':<14} {'estimator':<10} {'truth(us)':>10} {'window mean':>12} "
          f"{'bias%':>7} {'window std':>11} {'CV%':>7}")
    import random
    rng = random.Random(0)
    stability: dict[str, dict[str, float]] = {}
    for name, samples in truth.items():
        for est_name, est in ESTIMATORS.items():
            gt = est(samples)
            draws = [est(rng.sample(samples, N_WINDOW)) for _ in range(N_DRAWS)]
            mu = sum(draws) / len(draws)
            sd = st.pstdev(draws)
            stability.setdefault(name, {})[est_name] = sd / mu * 100
            print(f"{name:<14} {est_name:<10} {gt*1000:10.2f} {mu*1000:12.2f} "
                  f"{(mu-gt)/gt*100:+7.2f} {sd*1000:11.2f} {sd/mu*100:7.2f}")
        print()

    print("CV% is the number that matters for tuning: it is the run-to-run spread of the")
    print("objective at n=20. min_improvement_pct is 2.0, so any estimator with CV% above")
    print("~2 cannot resolve a 2% improvement from a single 20-sample measurement.")
    print()

    # ---- R3 RANKING + R4 DISCRIMINATION ----------------------------------------------
    print("=" * 96)
    print("R3 RANKING / R4 DISCRIMINATION -- how often a window ranks two configs "
          "the same way the ground truth does")
    print("=" * 96)
    names = list(truth)
    print(f"{'pair':<30} {'true gap%':>10}   " +
          "  ".join(f"{e:>9}" for e in ESTIMATORS))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            row = []
            for est_name, est in ESTIMATORS.items():
                ta, tb = est(truth[a]), est(truth[b])
                want = ta < tb
                agree = 0
                for _ in range(N_DRAWS):
                    ea = est(rng.sample(truth[a], N_WINDOW))
                    eb = est(rng.sample(truth[b], N_WINDOW))
                    if (ea < eb) == want:
                        agree += 1
                row.append(agree / N_DRAWS * 100)
            gap = abs(st.median(truth[a]) - st.median(truth[b])) / \
                min(st.median(truth[a]), st.median(truth[b])) * 100
            print(f"{a + ' vs ' + b:<30} {gap:9.1f}%   " +
                  "  ".join(f"{v:8.1f}%" for v in row))
    print()
    print("100% = the estimator always agrees with the truth about which is faster.")
    print("50%  = a coin flip, i.e. the objective carries no information about this pair.")
    print()

    # ---- R5 MASKING -------------------------------------------------------------------
    print("=" * 96)
    print("R5 MASKING -- what a robust statistic would HIDE")
    print("=" * 96)
    for name, samples in truth.items():
        s = sorted(samples)
        med = st.median(s)
        slow_frac = sum(1 for v in s if v > 1.5 * med) / len(s)
        tail_share = sum(v for v in s if v > 1.5 * med) / sum(s)
        print(f"  {name:<14} samples >1.5x median: {slow_frac*100:5.1f}%   "
              f"share of total time: {tail_share*100:5.1f}%")
    print()
    print("A 10% trim discards the top 10% of samples. If a config's slow path is taken")
    print("MORE than 10% of the time, trimming hides a real cost -- read the column above")
    print("before trusting a trimmed number. If it is well under 10%, the tail is stalls,")
    print("not the kernel.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
