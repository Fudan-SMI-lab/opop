"""Would a `logical_bytes` dimension carry information, or is it a restatement of the tile size?

WHY THIS QUESTION AND NOT "CAN WE PARSE 2-D IR". Parsing is engineering; the thing that decides
whether the dimension is worth having is whether the number it produces is INDEPENDENT of quantities
we already have. This project has been burned twice by adding a "dimension" that turned out to be an
existing one rescaled: `pct_of_dram_peak` is 1/latency times a constant (rho +1.000 across 4 runs),
and speed-of-light headroom ranked candidates identically to 1/latency. A third instance would be
the same mistake, and the shape of the risk here is specific and easy to state:

    logical_bytes = (bytes one instance touches) x (grid) x (loop trips)

For an ELEMENTWISE kernel that is a constant -- the same data however it is tiled, which this
probe already measured (BLOCK 256..2048 all gave exactly 201326592). A constant cannot be a wall:
no knob moves it, so no knob can be truncated by it. For a TILED kernel it does vary with the tile,
but then the question is whether it says anything beyond "the tile got smaller", which
`BLOCK_M * BLOCK_N` already says for free and which shared_bytes already covers.

So the probe asks three things of REAL candidate sources and REAL parameter sets from finished runs,
never of a synthetic kernel:

  1. SPREAD. Across a candidate's own parameter sets, does logical_bytes vary at all? A dimension
     that is constant within a candidate cannot be walled.
  2. INDEPENDENCE. Where it does vary, is it distinguishable from the obvious cheap proxies --
     the product of the tile knobs, and shared_bytes? Reported as Spearman, and a |rho| near 1
     against either means the dimension is a restatement.
  3. COVERAGE. On how many real candidates can the number be computed at all? A dimension that
     only exists for hand-picked kernels is not a dimension.

Reads candidate sources and parameter sets out of events.jsonl, materializes each with the project's
own materializer (never a string substitution), compiles with the compile-probe path, and derives
bytes from the TTIR. Any candidate that will not compile is reported as uncomputable rather than
skipped silently -- "we could not measure it" is a result about coverage.
"""
import json
import os
import re
import statistics
import sys
from collections import defaultdict

if len(sys.argv) < 3:
    print(__doc__)
    print("usage: logical_bytes_is_it_a_real_dimension.py <checkout-src> <run_dir> [<run_dir>...]")
    raise SystemExit(2)

sys.path.insert(0, sys.argv[1])
from kernel_optimizer.models.core import ParamSet  # noqa: E402
from kernel_optimizer.paramspace import materializer  # noqa: E402

_W = {"f32": 4, "f16": 2, "bf16": 2, "f64": 8, "i8": 1, "i16": 2, "i32": 4, "i64": 8, "i1": 1}


def bytes_from_ttir(ir: str) -> tuple[int, int, int]:
    """(bytes per instance, n memory ops counted, n ops skipped because untyped).

    Handles 1-D and 2-D pointer tensors. The element type is nested INSIDE the pointer
    (`tensor<64x64x!tt.ptr<f32>>`), which is the spelling a naive pattern misses -- and missing it
    reads 0 while matching the offsets on the same line charges index arithmetic as traffic.

    LOOP TRIPS ARE NOT APPLIED HERE. A `tt.load` inside `scf.for` executes once per trip, and the
    trip count is not always a literal in the IR. Rather than guess it, this returns the
    single-pass figure and reports separately whether a loop was present -- an unguessed number
    beats a wrong one, and the loop flag is what tells the reader the figure is a lower bound.
    """
    per, counted, skipped = 0, 0, 0
    for line in ir.splitlines():
        if "tt.load" not in line and "tt.store" not in line:
            continue
        m2 = re.findall(r"tensor<(\d+)x(\d+)x!tt\.ptr<([a-z0-9]+)>>", line)
        m1 = re.findall(r"tensor<(\d+)x!tt\.ptr<([a-z0-9]+)>>", line)
        if m2:
            n, mm, t = max(((int(a), int(b), c) for a, b, c in m2 if c in _W), default=(0, 0, ""))
            if t:
                per += n * mm * _W[t]
                counted += 1
                continue
        if m1:
            n, t = max(((int(a), b) for a, b in m1 if b in _W), default=(0, ""))
            if t:
                per += n * _W[t]
                counted += 1
                continue
        skipped += 1
    return per, counted, skipped


def spearman(xs, ys):
    n = len(xs)
    if n < 4 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else None


# ---- gather real candidates and their real parameter sets -------------------------------------
sources: dict[str, str] = {}
trials: dict[str, list] = defaultdict(list)
spaces: dict[str, dict] = {}
for rd in sys.argv[2:]:
    path = os.path.join(rd, "events.jsonl")
    if not os.path.exists(path):
        print(f"MISSING {path}")
        continue
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                e = json.loads(ln)
            except Exception:
                continue
            t, p = e.get("type"), (e.get("payload") or {})
            if t == "CANDIDATE_REGISTERED":
                src = p.get("source") or (p.get("candidate") or {}).get("source")
                cid = p.get("candidate_id") or (p.get("candidate") or {}).get("candidate_id")
                if src and cid:
                    sources[cid] = src
            elif t == "SPACE_PUBLISHED":
                sp = p.get("space") or {}
                if sp.get("candidate_id"):
                    spaces[sp["candidate_id"]] = {
                        d["name"]: (d.get("kind"), list(d.get("choices") or []))
                        for d in (sp.get("domains") or []) if isinstance(d, dict)}
            elif t == "TRIAL_DONE":
                tr = p.get("trial") or {}
                cid = tr.get("candidate_id")
                if cid and tr.get("status") == "complete" and tr.get("params"):
                    trials[cid].append(tr)

print(f"candidates with source: {len(sources)}")
print(f"candidates with completed trials: {len(trials)}")
print(f"candidates with BOTH: {len(set(sources) & set(trials))}")
usable = sorted(set(sources) & set(trials))
if not usable:
    print()
    print("NO CANDIDATE SOURCE IN THESE LOGS -- cannot answer on real operators.")
    print("This is a COVERAGE result, not a pass: the question stays open rather than answered.")
    raise SystemExit(0)

# ---- compile each (candidate, parameter set) and derive bytes ----------------------------------
import torch  # noqa: E402
import triton  # noqa: E402
from triton.runtime.jit import JITFunction  # noqa: E402

sys.path.insert(0, os.environ.get("KB_SRC", "/root/KernelBench/src"))

rows = []
uncomputable = defaultdict(int)
for cid in usable:
    src = sources[cid]
    space = spaces.get(cid) or {}
    # Take up to 12 distinct parameter sets, spread over the trial order rather than the first 12,
    # so the sample is not all from the sampler's random start-up phase.
    seen, picks = set(), []
    tl = trials[cid]
    for tr in tl[:: max(1, len(tl) // 12)]:
        key = json.dumps(tr["params"]["values"], sort_keys=True)
        if key not in seen:
            seen.add(key)
            picks.append(tr)
        if len(picks) >= 12:
            break
    for tr in picks:
        vals = tr["params"]["values"]
        try:
            mat = materializer.materialize(src, ParamSet(values=vals))
        except Exception as exc:
            uncomputable[f"materialize:{type(exc).__name__}"] += 1
            continue
        captured = []
        original = JITFunction.run

        def run_capturing(self, *a, grid=None, warmup=False, **kw):
            k = original(self, *a, grid=grid, warmup=True, **kw)
            captured.append(k)
            return k

        ns: dict = {}
        try:
            JITFunction.run = run_capturing
            exec(compile(mat, "<cand>", "exec"), ns)  # noqa: S102
            get_init = ns.get("get_init_inputs", lambda: [])
            get_in = ns["get_inputs"]
            model = ns["ModelNew"](*get_init()).cuda()
            with torch.no_grad():
                model(*[t.cuda() if hasattr(t, "cuda") else t for t in get_in()])
        except Exception as exc:
            uncomputable[f"run:{type(exc).__name__}"] += 1
            continue
        finally:
            JITFunction.run = original
        if not captured:
            uncomputable["no_triton_kernel"] += 1
            continue
        total_per, total_ops, has_loop = 0, 0, False
        for k in captured:
            ir = (getattr(k, "asm", {}) or {}).get("ttir", "")
            if not ir:
                continue
            per, cnt, _ = bytes_from_ttir(ir)
            total_per += per
            total_ops += cnt
            has_loop = has_loop or ("scf.for" in ir)
        if total_ops == 0:
            uncomputable["no_typed_memory_op"] += 1
            continue
        # The cheap proxies this must be shown to differ from.
        tile_product = 1
        n_tile_knobs = 0
        for name, v in vals.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            if re.search(r"BLOCK|TILE|^B[MNK]$|_B[MNK]$", name, re.I):
                tile_product *= int(v)
                n_tile_knobs += 1
        prof = (tr.get("profile") or {})
        rows.append({
            "cand": cid, "per_instance": total_per, "n_kernels": len(captured),
            "has_loop": has_loop, "tile_product": tile_product if n_tile_knobs else None,
            "shared": prof.get("shared_bytes"),
            "ms": ((tr.get("latency_ms") or {}).get("median")
                   or (tr.get("latency_ms") or {}).get("mean")),
        })

print()
print(f"computed rows: {len(rows)}")
print(f"uncomputable: {dict(uncomputable)}")
if not rows:
    print("=> COVERAGE FAILURE: the number could not be produced for any real candidate.")
    raise SystemExit(0)

print()
print("=== Q1 SPREAD: does logical_bytes vary across a candidate's OWN parameter sets? ===")
print("    (a dimension constant within a candidate cannot be walled by any knob)")
by_cand = defaultdict(list)
for r in rows:
    by_cand[r["cand"]].append(r)
flat = varying = 0
for cid, rs in sorted(by_cand.items()):
    vals = [r["per_instance"] for r in rs]
    if len(vals) < 2:
        continue
    spread = (max(vals) - min(vals)) / max(vals) if max(vals) else 0.0
    tag = "FLAT" if spread < 0.01 else "varies"
    if tag == "FLAT":
        flat += 1
    else:
        varying += 1
    print(f"  {cid} n={len(vals):2d} min={min(vals):9d} max={max(vals):9d} "
          f"spread={spread * 100:5.1f}%  {tag}  loop={rs[0]['has_loop']}")
print(f"  => {varying} vary, {flat} FLAT")

print()
print("=== Q2 INDEPENDENCE: is it distinguishable from the cheap proxies? ===")
for proxy in ("tile_product", "shared", "ms"):
    xs, ys = [], []
    for r in rows:
        if isinstance(r[proxy], (int, float)) and r[proxy]:
            xs.append(float(r["per_instance"]))
            ys.append(float(r[proxy]))
    rho = spearman(xs, ys)
    if rho is None:
        print(f"  vs {proxy:13s} rho None (n={len(xs)}, or a side constant)")
        continue
    verdict = ("**RESTATEMENT** -- adds nothing over a quantity we already have"
               if abs(rho) > 0.9 else
               "strongly related" if abs(rho) > 0.6 else
               "moderately related" if abs(rho) > 0.3 else
               "essentially independent -- carries its own information")
    print(f"  vs {proxy:13s} rho {rho:+.3f}  n={len(xs)}  {verdict}")

print()
print("=== Q3 COVERAGE: on how many real candidates is it computable? ===")
n_ok = len(by_cand)
print(f"  computable on {n_ok} of {len(usable)} candidates that have both source and trials")
print(f"  kernels per candidate: {sorted({r['n_kernels'] for r in rows})}")
print(f"  candidates whose IR contains a LOOP (figure is then a LOWER BOUND): "
      f"{sum(1 for rs in by_cand.values() if rs[0]['has_loop'])} of {n_ok}")
