"""Did co-residency cost the two box-4 arms throughput? Compare job_wall_s, not trial counts.

WHY NOT TRIAL COUNTS. arm1 got 840 trials in 12.25h (68.6/h) against arm2's 748 and arm3's 736 in
12.44h (60.1 and 59.1/h) -- co-resident arms ~13% slower. But that comparison is confounded: trial
cost is measured to scale with the kernel's own latency, and the three arms tuned DIFFERENT candidates
with different latencies, so a slower trial rate can simply mean slower kernels.

`job_wall_s` (added in step 1b) is the fair unit: the host-stamped wall time of one GPU job. Comparing
its distribution answers "did the same operation take longer on the shared box", which is what
co-residency could plausibly have cost and what the dual-card probe did NOT measure -- that probe
timed the GPU kernel, not the whole compile-and-evaluate pipeline, and compilation is CPU-bound and
happens simultaneously in both arms.

Still not a controlled experiment (different kernels compile differently), so the reading is a flag,
not a verdict. Reported as percentiles because the mean over this distribution names its tail: the
project has measured 13.5 min mean against 1.1 min median on job durations.
"""
import json
import os
import statistics
import sys

def jobs(rd):
    out = []
    for ln in open(os.path.join(rd, "events.jsonl"), encoding="utf-8"):
        ln = ln.strip()
        if not ln or "TRIAL_DONE" not in ln:
            continue
        try:
            e = json.loads(ln)
        except Exception:
            continue
        t = (e.get("payload") or {}).get("trial") or {}
        v = t.get("job_wall_s")
        if isinstance(v, (int, float)) and v > 0:
            out.append((float(v), t.get("status"), bool(t.get("job_timed_out"))))
    return out


def pct(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * (len(s) - 1)))]


print(f"{'arm':22s} {'n':>6s} {'p50':>8s} {'p90':>8s} {'p99':>9s} {'mean':>8s} "
      f"{'max':>9s} {'timeouts':>9s}")
rows = []
for spec in sys.argv[1:]:
    label, rd = spec.split("=", 1)
    js = jobs(rd)
    if not js:
        print(f"{label:22s} {'0':>6s}   (no job_wall_s -- this run predates step 1b)")
        continue
    xs = [v for v, _, _ in js]
    to = sum(1 for _, _, t in js if t)
    rows.append((label, xs))
    print(f"{label:22s} {len(xs):6d} {pct(xs,0.5):8.2f} {pct(xs,0.9):8.2f} {pct(xs,0.99):9.2f} "
          f"{statistics.fmean(xs):8.2f} {max(xs):9.2f} {to:9d}")

if len(rows) >= 2:
    base_label, base = rows[0]
    print()
    print(f"relative to {base_label} (p50 / p90):")
    for label, xs in rows[1:]:
        d50 = (pct(xs, 0.5) - pct(base, 0.5)) / pct(base, 0.5) * 100.0
        d90 = (pct(xs, 0.9) - pct(base, 0.9)) / pct(base, 0.9) * 100.0
        print(f"  {label:20s} p50 {d50:+7.1f}%   p90 {d90:+7.1f}%")
    print()
    print("A large positive shift on the SHARED box would be a co-residency cost the dual-card")
    print("probe missed. A shift near zero means the trial-rate gap is the kernels, not the sharing.")
