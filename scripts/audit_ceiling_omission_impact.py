#!/usr/bin/env python3
"""Did the missing ceilings measurably change what generator/parameterizer produced?

The severity question is not "was information withheld" -- it was -- but "did withholding it
change the artifacts". Answerable from completed runs on disk: every seed and every published
space is recorded, so the reachability of the tensor-core path can be counted rather than
argued about.

Reports across all runs given: how many seeds declared a precision knob, how many published
spaces carried a precision domain, and what the winning trial actually used. If seeds reliably
declared precision knobs WITHOUT the ceilings, the omission cost information but not behaviour,
and the fix is an improvement rather than a repair.
"""
import ast
import json
import pathlib
import sys

PREC = {"fp16", "bf16", "tf32", "ieee", "fp32", "float16", "bfloat16", "float32",
        "tf32x3", "half"}


def params_of(src):
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "PARAMS":
                    try:
                        return ast.literal_eval(node.value)
                    except Exception:  # noqa: BLE001
                        return None
    return None


def prec_knob(P):
    for k, v in (P or {}).items():
        if isinstance(v, str) and v.lower() in PREC:
            return k, v
    return None, None


tot_seeds = tot_with = 0
tot_spaces = tot_spaces_with = 0
rows = []
for arg in sys.argv[1:]:
    run = pathlib.Path(arg)
    if not (run / "events.jsonl").exists():
        continue
    seeds = list(run.glob("sandboxes/generator-*/candidates/*.py")) + \
        list(run.glob("sandboxes/novelty-*/novel/*.py"))
    n_seed = n_knob = 0
    for f in seeds:
        P = params_of(f.read_text(encoding="utf-8", errors="replace"))
        if P is None:
            continue
        n_seed += 1
        k, _ = prec_knob(P)
        if k:
            n_knob += 1

    n_sp = n_sp_prec = 0
    win = None
    for ln in (run / "events.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(ln)
        except Exception:  # noqa: BLE001
            continue
        if e.get("type") == "SPACE_PUBLISHED":
            sp = (e["payload"].get("space") or {})
            doms = sp.get("domains") or []
            n_sp += 1
            if any(any(str(c).lower() in PREC for c in (d.get("choices") or []))
                   for d in doms):
                n_sp_prec += 1
        if e.get("type") == "TRIAL_DONE":
            t = (e["payload"].get("trial") or {})
            lat = (t.get("latency_ms") or {})
            m = lat.get("median") or lat.get("mean")
            if t.get("status") == "complete" and m is not None:
                if win is None or m < win[0]:
                    _, v = prec_knob((t.get("params") or {}).get("values") or {})
                    win = (m, v)

    rows.append((run.name, n_seed, n_knob, n_sp, n_sp_prec,
                 f"{win[0]:.3f}/{win[1]}" if win else "-"))
    tot_seeds += n_seed
    tot_with += n_knob
    tot_spaces += n_sp
    tot_spaces_with += n_sp_prec

print(f"{'run':<34} {'seeds':>6} {'w/knob':>7} {'spaces':>7} {'w/prec':>7}  best/prec")
print("-" * 92)
for r in rows:
    print(f"{r[0]:<34} {r[1]:>6} {r[2]:>7} {r[3]:>7} {r[4]:>7}  {r[5]}")
print("-" * 92)
if tot_seeds:
    print(f"{'TOTAL':<34} {tot_seeds:>6} {tot_with:>7} {tot_spaces:>7} {tot_spaces_with:>7}")
    print(f"\nseeds declaring a precision knob : {tot_with}/{tot_seeds} "
          f"({100*tot_with/tot_seeds:.0f}%)")
if tot_spaces:
    print(f"spaces exposing a precision domain: {tot_spaces_with}/{tot_spaces} "
          f"({100*tot_spaces_with/tot_spaces:.0f}%)")
print("\nAll of the above was produced WITHOUT the measured ceilings in device.md.")
