"""Is the cross-box job_wall_s gap the CPU, or the kernels? Split the job into its two halves.

THE FINDING TO EXPLAIN. arm 1's median GPU job takes 16.81 s on box 1; the two box-4 arms take 21.1
and 21.3 s. That +26% is not co-residency -- measured from the two logs' own overlap, contention costs
less than the +-8% resolution of the shifted-interval control -- and it matters anyway, because arm 1
is the CONTROL arm. If box 1 simply runs jobs faster, the control bought more search volume than the
treatment arms, and "equal config" has already been measured not to mean "equal search".

THE CANDIDATE CAUSE. The boxes have different CPUs: box 1 is a Xeon 8358P at 2.60 GHz, box 4 an 8352V
at 2.10 GHz -- 23.8% slower, close enough to +26% to be worth testing rather than assuming. Both
boxes hold the same RTX 4090 at the same 3105 MHz max clock.

HOW TO TELL. A job splits cleanly into a CPU half and a GPU half:
  * `profile.compile_s` is Triton compilation -- CPU-bound
  * `latency_ms.robust_ms` is the timed kernel -- GPU-bound, same card on both boxes
If the CPU is the cause, the gap sits in compile_s and the wall, and NOT in the kernel latency. If it
appears in the latency too, the arms are simply tuning kernels of different speeds and this comparison
says nothing about the boxes. The latency is per-kernel and the arms tune DIFFERENT kernels, so it is
here as the discriminator, not as a claim about which arm is better.

FIELD NAMES ARE READ FROM THE EMITTER, NOT GUESSED. A first version of this script filtered on
`status == "ok"` and looked for a top-level `compile_s`; the real values are `"complete"` and
`profile.compile_s`, and it returned n=0 on all three arms. That was the good failure -- loud and
immediate -- but the same guess could have returned plausible-looking numbers, so the payload is now
dumped and read before being filtered (`dump_trial_payload.py`).

WHY `job_wall_s` IS SOMETIMES NULL. It is stamped by the HOST around `run_job` (`worker_client.py:67`),
so a trial whose result never came from a job has none. 160 of arm 3's 760 trials are in that state.
The script counts them rather than dropping them silently, because "no job ran" is a real and
different thing from "the job was not measured", and the first is a saving while the second is a gap.
"""
import json
import os
import statistics
import sys
from collections import Counter


def rows(rd):
    out = []
    kinds = Counter()
    with open(os.path.join(rd, "events.jsonl"), encoding="utf-8") as fh:
        for ln in fh:
            if "TRIAL_DONE" not in ln:
                continue
            try:
                e = json.loads(ln)
            except Exception:
                continue
            t = (e.get("payload") or {}).get("trial") or {}
            kinds[t.get("status")] += 1
            prof = t.get("profile") or {}
            lat = t.get("latency_ms") or {}
            # `robust_ms` is a @PROPERTY on LatencyStats (`models/core.py:133`), so it is computed,
            # never serialized -- reading it from JSON yields None on every record. Reproduce the
            # property's own rule instead: median when present and positive, else mean. Getting this
            # wrong printed "kernel latency n=0" on all three arms, which is the loud failure; the
            # quiet version would have been reading `mean` and silently comparing the wrong estimator
            # (mean's rank-correctness is 64.8% against median's 93.2%).
            med = lat.get("median") if isinstance(lat, dict) else None
            mean = lat.get("mean") if isinstance(lat, dict) else None
            robust = med if isinstance(med, (int, float)) and med > 0 else mean
            out.append({
                "status": t.get("status"),
                "wall": t.get("job_wall_s"),
                "compile_s": prof.get("compile_s"),
                "lat_ms": robust,
                "lat_is_median": isinstance(med, (int, float)) and med > 0,
            })
    return out, kinds


def pct(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * (len(s) - 1)))]


def num(rs, key, status=None):
    return [float(r[key]) for r in rs
            if isinstance(r.get(key), (int, float))
            and (status is None or r["status"] == status)]


def line(name, xs, unit=""):
    if not xs:
        return f"    {name:16s} n=0"
    return (f"    {name:16s} n={len(xs):4d}  p50 {pct(xs, 0.5):8.3f}{unit}  "
            f"p90 {pct(xs, 0.9):8.3f}{unit}  mean {statistics.fmean(xs):8.3f}{unit}")


results = {}
for spec in sys.argv[1:]:
    label, rd = spec.split("=", 1)
    rs, kinds = rows(rd)
    walls = num(rs, "wall")
    comps = num(rs, "compile_s")
    lats = num(rs, "lat_ms")
    no_wall = sum(1 for r in rs if not isinstance(r.get("wall"), (int, float)))
    results[label] = {"wall": walls, "compile_s": comps, "lat_ms": lats}
    print(f"=== {label}   trials {len(rs)}  status {dict(kinds)}")
    print(f"    job_wall_s absent on {no_wall} of {len(rs)} "
          f"({100.0 * no_wall / max(1, len(rs)):.0f}%) -- no host job ran for those")
    print(line("job_wall_s", walls, "s"))
    print(line("compile_s", comps, "s"))
    print(line("kernel latency", lats, "ms"))
    n_med = sum(1 for r in rs if r.get("lat_is_median"))
    if lats:
        print(f"                     ({n_med} of {len(lats)} are a true median, "
              f"the rest fall back to the mean)")
    # The CPU half, isolated: compile plus everything that is not the timed kernel. The kernel is
    # milliseconds against a job of seconds, so this is nearly the whole wall -- said out loud so the
    # near-equality with `job_wall_s` is not read as a bug.
    cpu = [r["wall"] - (r["lat_ms"] or 0.0) / 1000.0 for r in rs
           if isinstance(r.get("wall"), (int, float))]
    print(line("wall - latency", cpu, "s"))

if len(results) >= 2:
    base = list(results)[0]
    print()
    print(f"relative to {base} (p50):")
    for label in list(results)[1:]:
        for field, unit in (("wall", "s"), ("compile_s", "s"), ("lat_ms", "ms")):
            b, x = results[base][field], results[label][field]
            if not b or not x:
                print(f"  {label:6s} {field:10s} (missing on one side)")
                continue
            d = (pct(x, 0.5) - pct(b, 0.5)) / pct(b, 0.5) * 100.0
            print(f"  {label:6s} {field:10s} {d:+7.1f}%")
    print()
    print("READING: a gap in compile_s and wall, with kernel latency NOT tracking it, says the")
    print("slower box is slower at the CPU half. A gap that appears in latency too says the arms")
    print("are tuning different kernels and this says nothing about the box.")
