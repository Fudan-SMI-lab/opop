#!/usr/bin/env python3
"""Did this run's seeds make the tensor-core path reachable? Evidence, not impression.

Motivation: the generator for run-l3-21-20260908-232211 received a device.md with NO measured
ceilings (the fix routing them landed after the run started). The contract still told it to
treat dot precision as a first-class knob, so the question is whether the omission actually
changed the candidates -- which is answerable from the seeds on disk.

Reports, per candidate: whether PARAMS declares a precision knob, what its default is, what
choices the parameterizer gave it, and whether low-precision tokens appear in the body.
"""
import ast
import pathlib
import re
import sys

PREC_VALUES = {"fp16", "bf16", "tf32", "ieee", "fp32", "float16", "bfloat16",
               "float32", "tf32x3", "half"}
LOWP_TOKENS = ("tl.float16", "tl.bfloat16", "torch.float16", "torch.bfloat16", ".half(")


def params_of(src: str) -> dict | None:
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


run = pathlib.Path(sys.argv[1])
files = sorted(run.glob("sandboxes/generator-*/candidates/*.py"))
if not files:
    print("no generator candidates on disk yet")
    sys.exit(0)

print(f"{'file':<26} {'prec knob':<20} {'default':<8} {'lowp in body':<13} {'dot precision'}")
print("-" * 92)
for f in files:
    src = f.read_text(encoding="utf-8", errors="replace")
    P = params_of(src) or {}
    knob, default = "-", "-"
    for k, v in P.items():
        if isinstance(v, str) and v.lower() in PREC_VALUES:
            knob, default = k, v
            break
    lowp = ",".join(t for t in LOWP_TOKENS if t in src) or "-"
    dots = sorted(set(re.findall(r'input_precision\s*=\s*[\'"]?(\w+)', src))) or \
        (["<from knob>"] if knob != "-" else ["-"])
    print(f"{f.name:<26} {knob:<20} {default:<8} {lowp[:12]:<13} {','.join(dots)}")

print()
# What the tuner was actually given, per published space.
import json  # noqa: E402
evs = []
p = run / "events.jsonl"
if p.exists():
    for ln in p.read_text(encoding="utf-8").splitlines():
        try:
            evs.append(json.loads(ln))
        except Exception:  # noqa: BLE001
            continue
pub = [e["payload"] for e in evs if e.get("type") == "SPACE_PUBLISHED"]
if pub:
    print("published spaces -- precision domains the tuner may explore:")
    for sp in pub:
        space = sp.get("space") or {}
        doms = space.get("domains") or []
        cand = sp.get("candidate_id", "?")
        hit = False
        for d in doms:
            ch = [str(c) for c in (d.get("choices") or [])]
            if any(c.lower() in PREC_VALUES for c in ch):
                print(f"  {cand}: {d.get('name')} = {ch}")
                hit = True
        if not hit:
            print(f"  {cand}: NO precision domain -- the tuner cannot compare precisions")
else:
    print("no SPACE_PUBLISHED yet")
