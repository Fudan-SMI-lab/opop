"""Can this box actually EVALUATE a kernel? (the check that G9's wasted run needed)

WHY THIS EXISTS. The A800 passed 422 unit tests, made a real agent call, validated its config, and
still could not evaluate a single kernel: `kernelbench`'s package __init__ imports a chain that
reached `dotenv`, `openai`, then `litellm`, none of which were installed. Every candidate came back
`runtime_error` from `run_static_check`, so a 6-call G9 re-run produced 12 usable candidates and
0 correct ones -- and "0 correct" reads exactly like "the model wrote bad kernels".

The unit suite could not catch it because it never imports kernelbench; the config validator could
not, because the package is present on disk and only fails on import. So the gate has to be an
ACTUAL EVALUATION of a known-good kernel through the harness's own evaluator.

Run this on any new box before spending agent calls on it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kernel_optimizer.config import load_config  # noqa: E402
from kernel_optimizer.store.read import latency_ms_of  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--task", default="level3:21")
    ap.add_argument("--candidate", default=None,
                    help="a known-CORRECT kernel. Omitted: the task's own reference is wrapped as "
                         "ModelNew, which must pass by construction.")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    from kernel_optimizer.store.run_store import RunStore
    from kernel_optimizer.tasks.kernelbench import parse_task_arg
    from kernel_optimizer.wiring import build_gpu_stack, load_task

    level, pid = parse_task_arg(args.task)
    task = load_task(cfg, level, pid)
    import time
    runs = Path(cfg.run.runs_dir)
    store = RunStore.create(runs, "evalcheck-%s" % time.strftime("%Y%m%d-%H%M%S"),
                            {"purpose": "verify this box can evaluate a kernel at all"})
    worker, evaluator, _bench, _prof = build_gpu_stack(cfg, store)

    work = store.run_dir / "candidates"
    work.mkdir(parents=True, exist_ok=True)

    if args.candidate:
        src = Path(args.candidate).read_text(encoding="utf-8")
        label = args.candidate
    else:
        # The reference itself, renamed. It cannot be wrong, so a failure here is the HARNESS.
        ref = Path(task.ref_path).read_text(encoding="utf-8")
        src = ref.replace("class Model(", "class ModelNew(")
        if "class ModelNew(" not in src:
            print("FAIL: could not derive a ModelNew from %s" % task.ref_path)
            return 1
        src = "PARAMS = {}\n\n" + src
        label = "the task's own reference, wrapped as ModelNew"

    path = work / "evalcheck.py"
    path.write_text(src, encoding="utf-8")
    print("task      : %s" % task.name)
    print("candidate : %s" % label)
    print("evaluating through the harness's own quick_test ...")

    try:
        res = evaluator.quick_test(task, path, "evalcheck")
    except Exception as exc:  # noqa: BLE001
        print()
        print("FAIL: the evaluator raised %s: %s" % (type(exc).__name__, str(exc)[:400]))
        return 1

    ok = bool(res.get("ok"))
    kind = res.get("failure_kind")
    ms = latency_ms_of(res)
    print("ok        : %s" % ok)
    print("failure   : %s" % (kind or "-"))
    print("latency   : %s ms" % (("%.4f" % ms) if ms is not None else "-"))

    if not ok:
        tail = str(res.get("log_tail", ""))[:900]
        print()
        print("log tail:")
        print(tail)
        print()
        if "ModuleNotFoundError" in tail or "ImportError" in tail:
            missing = tail.rsplit("No module named", 1)[-1].strip().strip("'\"") if \
                "No module named" in tail else "?"
            print("VERDICT: NOT READY -- this is an ENVIRONMENT gap, not a kernel problem.")
            print("  Missing module: %s" % missing)
            print("  kernelbench's package __init__ imports a chain (dotenv -> openai -> litellm ...)")
            print("  and any missing link makes EVERY candidate fail as runtime_error, which reads")
            print("  identically to 'the model wrote bad kernels'. Install the package and re-run.")
            return 1
        print("VERDICT: NOT READY -- a kernel that should pass did not. Read the tail above; if the")
        print("  candidate was the task's own reference, the fault is in the harness or the box.")
        return 1

    print()
    print("VERDICT: READY -- this box compiled, checked and timed a kernel end to end through the")
    print("  harness's own evaluator. Agent calls spent here can now produce meaningful verdicts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
