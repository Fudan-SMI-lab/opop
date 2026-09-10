"""Go / no-go for a control pair that spans two boxes: do their measured ceilings agree?

A same-box pair gets this for free. A cross-box pair does not, and this is the gap the
resource-map digest does NOT close: `sweep_resource_map.py` compares what the COMPILER decides
(registers, shared, spills, which tiles launch), which was byte-identical on the two 4090s. Ceilings
are different in kind -- they are TIMED measurements, so the two boxes cannot produce identical
numbers even when nothing is wrong, and S2/S3 divide by them to produce every per-dimension verdict.

So the question is not "are they equal" but "are they closer together than the difference the run is
trying to detect". The run's yardstick is the task's ieee-vs-tf32 noise floor, and J2-5 asks whether
the final result got worse by more than that. If a ceiling differs between the arms by more than the
noise floor, a difference in the final number has two candidate explanations -- the switch, or the
denominator -- and no way to separate them. That has to be settled BEFORE launch: afterwards it is
an unattributable result, which is the outcome this whole pairing exists to avoid.

Run `kernel-opt --config <arm config> calibrate` on each box first, then pass the two
calibration.json files here.

    python scripts/compare_calibrations.py box1.json box2.json --tol 0.023

`--tol` is the fraction the ceilings may differ by. Default 0.0235 = the L3:43 noise floor (1 -
0.9765) measured on this hardware, because that is the difference J2-5 must be able to see. Pass the
floor of whichever task the pair will run.

Exit 0 = the pair is usable. Exit 1 = it is NOT; fall back to one box, 24 h serial.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The ceilings a verdict actually divides by. `dram_tbs` and the four compute roofs are the
# denominators in pct_of_dram_peak / pct_of_compute_peak; the four *_triton_* are what G10 made the
# ceiling (the max over MEASURED backends, since cuBLAS overstates the fp32 roof by 19% and
# understates fp16 -- Triton reached 109.6% of it). `empty_launch_floor_ms` gates the overhead class.
_CEILINGS = (
    "dram_tbs",
    "fp32_tflops", "tf32_tflops", "fp16_tflops", "bf16_tflops",
    "fp32_triton_tflops", "tf32_triton_tflops", "fp16_triton_tflops", "bf16_triton_tflops",
    "empty_launch_floor_ms",
)

# Must match EXACTLY, not within a tolerance: these are identity, not measurement. Two boxes that
# disagree here are not the same hardware and no tolerance makes the pair valid.
_IDENTITY = ("device_name", "capability", "sm_count", "l2_bytes")

# Derived thresholds. Dimensionless by design ("so they travel to the next card"), but they are
# derived FROM the yardstick timings, so they inherit their noise -- and every band boundary in
# every per-dimension verdict is one of these. Compared at the same tolerance as the ceilings.
_THRESHOLDS = ("dram_saturated_frac", "compute_saturated_frac", "idle_frac",
               "launch_bound_cpu_ratio")


def _rel(a: float, b: float) -> float:
    """Relative difference against the LARGER magnitude, so the answer does not depend on argument
    order -- a go/no-go gate that changed verdict when the files were swapped would be worthless."""
    m = max(abs(a), abs(b))
    return 0.0 if m == 0 else abs(a - b) / m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("left", type=Path)
    ap.add_argument("right", type=Path)
    ap.add_argument("--tol", type=float, default=0.0235,
                    help="fraction the ceilings may differ by; default 0.0235 = the L3:43 "
                         "ieee-vs-tf32 noise floor (1 - 0.9765) measured on this hardware")
    args = ap.parse_args()

    a = json.loads(args.left.read_text(encoding="utf-8"))
    b = json.loads(args.right.read_text(encoding="utf-8"))

    print("LEFT   %s  schema %s  torch %s triton %s  measured %s"
          % (a.get("device_name"), a.get("schema_version"), a.get("torch_version"),
             a.get("triton_version"), a.get("measured_at")))
    print("RIGHT  %s  schema %s  torch %s triton %s  measured %s"
          % (b.get("device_name"), b.get("schema_version"), b.get("torch_version"),
             b.get("triton_version"), b.get("measured_at")))
    print()

    fatal: list[str] = []

    # A stale or half-written calibration must not read as agreement -- an absent number compared
    # against an absent number is not a match, and 0.0 is the value box 1's schema-3 file carried
    # for all four Triton fractions.
    for tag, d in (("LEFT", a), ("RIGHT", b)):
        if d.get("schema_version") != 5:
            fatal.append("%s is calibration schema %s, not 5: it predates the fp16/bf16 "
                         "measurements and its Triton fractions are 0.0" % (tag, d.get("schema_version")))
        missing = [k for k in _CEILINGS if d.get(k) in (None, 0, 0.0)]
        if missing:
            fatal.append("%s has absent or zero ceilings %s -- an unmeasured roof must not be "
                         "compared as if it were measured" % (tag, missing))

    if a.get("torch_version") != b.get("torch_version") or \
            a.get("triton_version") != b.get("triton_version"):
        fatal.append("TOOLCHAIN MISMATCH: %s/%s against %s/%s. Ceiling agreement cannot rescue "
                     "this -- compare the resource-map digests (scripts/sweep_resource_map.py) "
                     "before pairing these boxes at all"
                     % (a.get("torch_version"), a.get("triton_version"),
                        b.get("torch_version"), b.get("triton_version")))

    print("--- identity (must match exactly; these are not measurements) ---")
    for k in _IDENTITY:
        va, vb = a.get(k), b.get(k)
        ok = va == vb
        print("  %-14s %-34s %-34s %s" % (k, va, vb, "ok" if ok else "**DIFFERS**"))
        if not ok:
            fatal.append("%s differs (%s vs %s): these are not the same hardware" % (k, va, vb))
    print()

    over: list[str] = []
    print("--- measured ceilings (tolerance %.4f = %.2f%%) ---" % (args.tol, args.tol * 100))
    for k in _CEILINGS:
        va, vb = a.get(k), b.get(k)
        if not isinstance(va, (int, float)) or not isinstance(vb, (int, float)):
            continue
        r = _rel(float(va), float(vb))
        flag = "ok" if r <= args.tol else "**OVER**"
        print("  %-22s %14.6f %14.6f  %6.2f%%  %s" % (k, va, vb, r * 100, flag))
        if r > args.tol:
            over.append("%s differs by %.2f%%" % (k, r * 100))
    print()

    print("--- derived thresholds (every band boundary in every verdict) ---")
    ta, tb = a.get("thresholds") or {}, b.get("thresholds") or {}
    for k in _THRESHOLDS:
        va, vb = ta.get(k), tb.get(k)
        if not isinstance(va, (int, float)) or not isinstance(vb, (int, float)):
            print("  %-24s ABSENT on one side -- not comparable" % k)
            fatal.append("threshold %s is absent on one side" % k)
            continue
        r = _rel(float(va), float(vb))
        flag = "ok" if r <= args.tol else "**OVER**"
        print("  %-24s %10.4f %10.4f  %6.2f%%  %s" % (k, va, vb, r * 100, flag))
        if r > args.tol:
            over.append("threshold %s differs by %.2f%%" % (k, r * 100))
    print()

    if fatal:
        print("NO-GO -- structural problems, tolerance is irrelevant:")
        for f in fatal:
            print("  * %s" % f)
        print("\nRun the two arms on ONE box, serially.")
        return 1
    if over:
        print("NO-GO -- %d quantity/quantities differ by more than the noise floor the run must "
              "be able to see:" % len(over))
        for o in over:
            print("  * %s" % o)
        print("\nA difference in the final result would have two candidate explanations (the "
              "switch, or the denominator) and no way to separate them. Run the two arms on ONE "
              "box, serially -- 24 h instead of 12 h, which is the cost of an attributable result.")
        return 1

    print("GO -- every ceiling and threshold agrees within %.2f%%, the toolchains match, and the "
          "hardware identity is identical." % (args.tol * 100))
    print("The two arms may run CONCURRENTLY, one per box. They must never share a box: two runs "
          "timing kernels on one GPU corrupt each other's measurements.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
