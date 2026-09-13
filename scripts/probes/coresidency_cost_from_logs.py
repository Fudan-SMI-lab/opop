"""Did co-residency slow the box-4 arms? Compare each arm's jobs when the OTHER arm was idle.

THE NATURAL EXPERIMENT. arm2 and arm3 share one box, and each spends long stretches inside agent
calls -- minutes with no GPU job at all. During those stretches the other arm has the box's CPU and
both cards to itself. So each arm's own trials split into two populations, SOLO and CONTENDED, on the
same box, same toolchain, same kernels, same clock. That is the comparison the cross-box numbers
cannot give: arm1's p50 job_wall_s is 16.81 s against 21.1/21.3 s on box 4, but box 1 is a different
physical machine, so that +26% is not attributable to sharing.

WHY THIS AND NOT THE DUAL-CARD PROBE. That probe timed the GPU kernel and measured a worst median
shift of +1.20%. `job_wall_s` is the whole job: process start, imports, Triton compile, correctness,
timing. Compilation is CPU-bound and both arms do it at once, so the pipeline can lose throughput the
kernel timing never sees. If it does, arm-to-arm search volume differs for a reason that has nothing
to do with the treatment, and per-arm cost figures are not comparable.

BOTH ARMS ARE ON ONE CLOCK. Same host, so `ts` is directly comparable between the two logs -- this
would be invalid across boxes.

POSITIVE CONTROL. The split is only informative if both populations exist: if 0% or 100% of jobs are
contended, the comparison is vacuous and the script says so rather than printing a difference of
zero. It also reports the same split computed against the other arm's intervals shifted by a large
offset -- under a shift the contention labels should become uninformative, so a "difference" that
survives the shift is an artifact of the two arms' differing kernels, not of contention.
"""
import json
import os
import statistics
import sys


def load(rd):
    evs = []
    with open(os.path.join(rd, "events.jsonl"), encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                try:
                    evs.append(json.loads(ln))
                except Exception:
                    pass
    return evs


def job_spans(evs):
    """(start, end, wall) per GPU job. TRIAL_DONE stamps completion, so start = ts - job_wall_s."""
    out = []
    for e in evs:
        if e.get("type") != "TRIAL_DONE":
            continue
        t = (e.get("payload") or {}).get("trial") or {}
        w = t.get("job_wall_s")
        if isinstance(w, (int, float)) and w > 0:
            out.append((e["ts"] - float(w), e["ts"], float(w)))
    return out


def merge(spans):
    """Union of intervals, so an overlap test is a scan rather than a quadratic sweep."""
    if not spans:
        return []
    s = sorted((a, b) for a, b, *_ in spans)
    out = [list(s[0])]
    for a, b in s[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [tuple(x) for x in out]


def overlaps(a, b, iv):
    for x, y in iv:
        if y <= a:
            continue
        if x >= b:
            return False
        return True
    return False


def pct(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * (len(s) - 1)))]


def describe(name, xs):
    if not xs:
        return f"    {name:12s} n=0"
    return (f"    {name:12s} n={len(xs):4d}  p50 {pct(xs, 0.5):7.2f}  p90 {pct(xs, 0.9):7.2f}  "
            f"mean {statistics.fmean(xs):7.2f}")


def split(mine, other_busy, shift=0.0):
    iv = [(a + shift, b + shift) for a, b in other_busy]
    solo, cont = [], []
    for a, b, w in mine:
        (cont if overlaps(a, b, iv) else solo).append(w)
    return solo, cont


def report(label, mine, other_busy, shift_s):
    solo, cont = split(mine, other_busy)
    frac = len(cont) / max(1, len(cont) + len(solo))
    print(f"  {label}: {frac * 100:.0f}% of its jobs overlapped the other arm's GPU work")
    print(describe("SOLO", solo))
    print(describe("CONTENDED", cont))
    if not solo or not cont:
        print("    *** one population is empty -- no comparison is possible from this run, and a")
        print("        printed 0% difference here would be an artifact of the split, not a result")
        return
    d50 = (pct(cont, 0.5) - pct(solo, 0.5)) / pct(solo, 0.5) * 100.0
    d90 = (pct(cont, 0.9) - pct(solo, 0.9)) / pct(solo, 0.9) * 100.0
    print(f"    contended vs solo: p50 {d50:+.1f}%   p90 {d90:+.1f}%")
    ssolo, scont = split(mine, other_busy, shift=shift_s)
    if ssolo and scont:
        s50 = (pct(scont, 0.5) - pct(ssolo, 0.5)) / pct(ssolo, 0.5) * 100.0
        print(f"    positive control (other arm's intervals shifted {shift_s / 3600:.1f}h): "
              f"p50 {s50:+.1f}%  <- should be near zero; if it tracks the real number, the split "
              f"is not measuring contention")


a_evs, b_evs = load(sys.argv[1]), load(sys.argv[2])
a_name = sys.argv[3] if len(sys.argv) > 3 else "armA"
b_name = sys.argv[4] if len(sys.argv) > 4 else "armB"
a_jobs, b_jobs = job_spans(a_evs), job_spans(b_evs)
a_busy, b_busy = merge(a_jobs), merge(b_jobs)
lo = max(min(s for s, _, _ in a_jobs), min(s for s, _, _ in b_jobs))
hi = min(max(e for _, e, _ in a_jobs), max(e for _, e, _ in b_jobs))
print(f"overlapping window: {(hi - lo) / 3600:.2f} h   "
      f"{a_name} {len(a_jobs)} jobs, {b_name} {len(b_jobs)} jobs")
# Restrict to the window both arms were actually running in, or the arm that started earlier gets
# spurious SOLO jobs from a period when the other arm simply did not exist yet.
a_jobs = [j for j in a_jobs if j[0] >= lo and j[1] <= hi]
b_jobs = [j for j in b_jobs if j[0] >= lo and j[1] <= hi]
print(f"in-window: {a_name} {len(a_jobs)}, {b_name} {len(b_jobs)}")
print()
report(a_name, a_jobs, b_busy, shift_s=3.0 * 3600)
print()
report(b_name, b_jobs, a_busy, shift_s=3.0 * 3600)
