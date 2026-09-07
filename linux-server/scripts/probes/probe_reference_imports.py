"""Check every task's reference model can be imported, before a run spends hours proving it cannot.

The reference model is `exec`'d from source at EVALUATION time, so its third-party imports are
never checked by `doctor` or by a successful config load. `load_original_model_and_inputs`
swallows the ImportError, prints it to stdout and returns None; the caller then dies several
frames away with `TypeError: cannot unpack non-iterable NoneType object`, and the traceback
never names the missing module.

Parses each task file's OWN imports rather than checking a hardcoded list, so a task added later
is covered without editing this script.

Run with the WORKER venv's python (the one evaluation uses). Exit 0 = all resolve.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import sys
from pathlib import Path

# level3 33/35/41 are excluded on purpose elsewhere (their forward() calls randn, so a
# correctness comparison is impossible); this script does not care -- point it at whatever
# you intend to run.
DEFAULT_TASKS = {
    "level2:37": "KernelBench-l2-37-orig/KernelBench/level2/37_Matmul_Swish_Sum_GroupNorm.py",
    "level3:21": "KernelBench/KernelBench/level3/21_EfficientNetMBConv.py",
    "level3:43": "KernelBench/KernelBench/level3/43_MinGPTCausalAttention.py",
    "level3:48": "KernelBench/KernelBench/level3/48_Mamba2ReturnY.py",
}


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.add(node.module.split(".")[0])
    return mods


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/autodl-tmp/opop-workspace",
                    help="workspace root holding the KernelBench trees")
    ap.add_argument("--task", action="append", metavar="NAME=PATH",
                    help="extra task as name=path (path relative to --root); repeatable")
    ap.add_argument("--only", metavar="LEVEL:ID",
                    help="check ONLY this task, resolving its file under --kb-root. Use this "
                         "from a launcher so the task being launched is the task checked.")
    ap.add_argument("--kb-root", metavar="DIR",
                    help="KernelBench tree for --only (the config's kernelbench_root). "
                         "Required with --only: L2:37 lives in a different tree.")
    args = ap.parse_args()

    root = Path(args.root)
    if args.only:
        # Resolve the ACTUAL launched task rather than trusting a hardcoded list. The list
        # below is a convenience for a manual sweep; as a launcher guard it was worse than
        # nothing -- planting `import definitely_not_installed_xyz` into level1:42 left the
        # guard passing, because level1:42 was not in the list. A guard that checks something
        # other than what you are about to run cannot protect the run.
        if not args.kb_root:
            print("--only requires --kb-root (the config's kernelbench_root)")
            return 2
        level, _, pid = args.only.partition(":")
        matches = sorted(Path(args.kb_root).glob(f"KernelBench/{level}/{pid}_*.py"))
        if not matches:
            print(f"{args.only:12s} MISSING FILE under {args.kb_root}")
            return 1
        tasks = {args.only: str(matches[0].relative_to(root))
                 if str(matches[0]).startswith(str(root)) else str(matches[0])}
    else:
        tasks = dict(DEFAULT_TASKS)
        for spec in args.task or []:
            name, _, rel = spec.partition("=")
            tasks[name] = rel

    bad = 0
    for task, rel in tasks.items():
        path = Path(rel) if Path(rel).is_absolute() else root / rel
        if not path.is_file():
            print(f"{task:12s} MISSING FILE: {path}")
            bad += 1
            continue
        mods = imported_modules(path)
        missing = sorted(m for m in mods if importlib.util.find_spec(m) is None)
        print(f"{task:12s} {sorted(mods)}  missing={missing or 'none'}")
        bad += len(missing)

    # NEGATIVE CONTROL: a check that cannot report a miss proves nothing. If find_spec ever
    # starts resolving arbitrary names (a sys.meta_path hook, a namespace-package surprise),
    # every "missing=none" above becomes meaningless and this catches it.
    if importlib.util.find_spec("definitely_not_installed_xyz") is not None:
        print("CONTROL BROKEN: a nonexistent module resolved, so no result above is "
              "trustworthy")
        return 2
    print("control OK: a nonexistent module is correctly reported missing")

    # Second control, on the positive side: a module that certainly IS importable must be
    # found, so the check cannot pass by reporting everything as absent-but-ignored.
    if importlib.util.find_spec("sys") is None:
        print("CONTROL BROKEN: the stdlib is reported missing")
        return 2
    print("control OK: a known-present module is found")

    if bad:
        print(f"\n{bad} unresolved import(s) -- pip install them into the WORKER venv "
              f"before launching, or the run dies at its first baseline.")
        return 1
    print("\nall reference models resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
