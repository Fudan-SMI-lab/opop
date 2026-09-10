"""What is THIS box's per-task ieee-vs-tf32 noise floor? (the box-1 numbers are not transferable)

WHY THIS EXISTS. The relaxed correctness gate demands `frac_within_tol >= relaxed_pass_frac` (0.99
in the shipped config). But the REFERENCE ITSELF does not meet that against itself when run at the
two fp32 matmul precisions the harness compares against: measured on box 1, the three L3 tasks sit
at 0.9554 / 0.9767 / 0.9778, all BELOW the line. That is why the fp64 relative arm exists -- without
it a correct low-precision candidate cannot pass those tasks at all.

Those three numbers are a (CARD, TASK) property, not a task property. The A800 has a different
tensor-core path (tf32/fp32 = 5.9x here versus 1.6x on the 4090), a different L2, and a different
Triton, so its floors have to be measured rather than carried over -- the same rule as every ceiling
in `calibration.py`, applied to the other side of the gate.

WHAT IT MEASURES, and why this cannot be confounded by a candidate. The reference is run TWICE per
trial on the same inputs, once with the fp32 matmul path in tf32 mode and once in ieee mode, and the
two outputs are compared with the same `_relaxed_metrics` the gate itself uses. No candidate is
involved, so a low floor cannot be blamed on a kernel: it is the task's own precision sensitivity on
this card.

TWO THINGS THAT WOULD MAKE THE READING A LIE, both checked rather than assumed:

  RNG IN forward(). If the reference draws random numbers inside its forward pass, the two calls see
  different noise and the "precision floor" is really RNG noise. The loop re-seeds immediately before
  EACH of the two calls, so an RNG-using reference still gets identical draws; and a separate control
  runs the reference twice at the SAME precision, which must come back essentially 1.0. If that
  control is not ~1.0, the number this script prints means nothing and it says so.

  A VACUOUS PASS. A floor of exactly 1.0 on every task would look like good news and is more likely
  to mean the precision switch never took effect (an old torch where the API moved, or a task with no
  matmul/conv at all). So the same-precision control is printed alongside, and a task whose two
  precisions agree to 1.0 while the control also reads 1.0 is reported as INCONCLUSIVE rather than as
  "floor 1.0".

Run it on any new box before trusting a floor number, and record the output next to the box's
calibration.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kernel_optimizer.config import load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tasks", default="level3:21,level3:43,level3:48",
                    help="comma-separated; the three L3 tasks by default")
    ap.add_argument("--trials", type=int, default=5,
                    help="reference-vs-reference comparisons per task. The floor is reported as the "
                         "MINIMUM over trials, because the gate is applied per trial: a floor that "
                         "holds on average still rejects on its worst draw.")
    ap.add_argument("--out", default=None, help="write the readings to this JSON file too")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    from kernel_optimizer.gpu.worker_client import WslGpuWorker
    from kernel_optimizer.tasks.kernelbench import parse_task_arg
    from kernel_optimizer.wiring import load_task

    runs_dir = Path(cfg.run.runs_dir)
    worker = WslGpuWorker(cfg.wsl, cfg.gpu.concurrency,
                          jobs_dir=runs_dir / "_noise-floor-jobs")

    readings: list[dict] = []
    for task_arg in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        level, pid = parse_task_arg(task_arg)
        task = load_task(cfg, level, pid)
        job = {"job_type": "probe_noise_floor",
               "ref_src_path": str(task.ref_path),
               "num_trials": args.trials,
               "elem_tol": cfg.evaluation.relaxed_elem_tol}
        # Exclusive: it runs the reference model, and a neighbour job perturbs nothing about
        # correctness but this is cheap enough that there is no reason to share the card.
        res = worker.run_job(job, timeout_s=900.0, tag="noisefloor", lock_mode="exclusive")
        if not res.get("ok"):
            print("%-14s FAILED: %s" % (task_arg, str(res.get("log_tail", ""))[-300:]))
            readings.append({"task": task_arg, "ok": False,
                             "error": str(res.get("log_tail", ""))[-500:]})
            continue
        readings.append({"task": task_arg, "ok": True, **res})

    print()
    print("%-14s %-10s %-10s %-10s %-10s %s" % (
        "task", "floor", "cosine", "control", "absmax", "verdict"))
    print("-" * 78)
    for r in readings:
        if not r.get("ok"):
            print("%-14s %s" % (r["task"], "FAILED"))
            continue
        floor = r.get("floor_frac_within_tol")
        control = r.get("control_frac_within_tol")
        # The control decides whether the floor means anything at all. Both branches are stated
        # rather than one, because "inconclusive" and "no sensitivity" are opposite conclusions
        # that produce the same number.
        if control is None or control < 0.9999:
            verdict = "INVALID (control %.6f != 1.0; RNG in forward?)" % (control or -1)
        elif floor is not None and floor >= 0.9999:
            verdict = "INCONCLUSIVE (no measurable precision sensitivity)"
        else:
            verdict = "measured"
        print("%-14s %-10s %-10s %-10s %-10s %s" % (
            r["task"],
            ("%.6f" % floor) if floor is not None else "-",
            ("%.6f" % r["floor_cosine"]) if r.get("floor_cosine") is not None else "-",
            ("%.6f" % control) if control is not None else "-",
            r.get("ref_absmax", "-"),
            verdict))

    print()
    print("The floor is the MINIMUM over %d trials, because the gate is applied per trial." %
          args.trials)
    print("A floor BELOW evaluation.relaxed_pass_frac (%.4f in this config) means the absolute "
          "gate is unreachable" % cfg.evaluation.relaxed_pass_frac)
    print("for a low-precision candidate on that task, and acceptance depends on the fp64 relative "
          "arm.")

    if args.out:
        Path(args.out).write_text(json.dumps(readings, indent=2), encoding="utf-8")
        print("\nwritten: %s" % args.out)
    return 0 if all(r.get("ok") for r in readings) else 1


if __name__ == "__main__":
    raise SystemExit(main())
