"""Given that hardware counters are refused, what CAN be measured per candidate? Two routes.

Route A -- Triton Proton. Triton ships its own profiler (`triton.profiler`). The question that
decides whether it helps us: does it report BYTES or only TIME, and does it need the same counter
permission ncu was refused? If Proton reports per-kernel bytes without counters, it is exactly the
missing measurement. If it only reports time, it is latency again and worth nothing here.

Route B -- compile-time logical bytes from the Triton IR. Every `tl.load`/`tl.store` has a shape
and a dtype, so the bytes one kernel instance touches are derivable from the compiled artefact
with no runtime tool at all. This would be a LOGICAL byte count -- it cannot see L2 hits, so it
over-counts DRAM traffic exactly the way `pct_of_dram_peak` already does. But unlike the current
number it would be PER CANDIDATE and PER PARAMETER SET, which is the property that is missing.

Both are measured on a real Triton kernel whose true byte traffic is known by construction, so a
reading can be checked rather than trusted: a 4096x4096 fp32 elementwise add reads two matrices and
writes one, 3 * 4096 * 4096 * 4 = 201326592 bytes.
"""
import json
import os
import subprocess
import sys

KNOWN_BYTES = 3 * 4096 * 4096 * 4

KERNEL = '''
import torch, triton, triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    y = tl.load(y_ptr + offs, mask=m)
    tl.store(o_ptr + offs, x + y, mask=m)

N = 4096 * 4096
x = torch.randn(N, device="cuda")
y = torch.randn(N, device="cuda")
o = torch.empty_like(x)
BLOCK = 1024
grid = (triton.cdiv(N, BLOCK),)
'''


def run(cmd, timeout=300):
    try:
        p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=timeout)
        return p.returncode, p.stdout.decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        return -9, "TIMEOUT"
    except Exception as exc:  # noqa: BLE001
        return -1, f"{type(exc).__name__}: {exc}"


py = sys.executable
os.makedirs("/root/probe-clean", exist_ok=True)
print(f"KNOWN_TRUE_BYTES={KNOWN_BYTES}")
print()

print("=== ROUTE A: Triton Proton -- what does it report, and does it need counters? ===")
pa = "/root/probe-clean/proton_probe.py"
with open(pa, "w", encoding="utf-8") as fh:
    fh.write(KERNEL + '''
import triton.profiler as proton
print("PROTON_ATTRS", [a for a in dir(proton) if not a.startswith("_")][:25])
try:
    session = proton.start("prof_out", context="shadow")
    add_kernel[grid](x, y, o, N, BLOCK=BLOCK)
    torch.cuda.synchronize()
    proton.finalize()
    print("PROTON_RAN=1")
except Exception as exc:
    print("PROTON_RAISED", type(exc).__name__, str(exc)[:300])
''')
rc, out = run(f"cd /root/probe-clean && {py} {pa}")
print(f"PROTON_RC={rc}")
for line in out.splitlines()[:25]:
    print("   ", line)
# Proton writes a hatchet/JSON trace; look at what fields it actually contains.
rc2, out2 = run("ls -la /root/probe-clean/prof_out* 2>/dev/null")
print("PROTON_FILES:", out2.strip() or "none")
rc3, out3 = run("head -c 1500 /root/probe-clean/prof_out.hatchet 2>/dev/null || "
                "head -c 1500 /root/probe-clean/prof_out.json 2>/dev/null")
if out3.strip():
    print("PROTON_TRACE_HEAD:")
    print("   ", out3.strip()[:1200])
    print(f"PROTON_MENTIONS_BYTES={'byte' in out3.lower()}")

print()
print("=== ROUTE B: are per-kernel LOGICAL bytes derivable from the compiled artefact? ===")
pb = "/root/probe-clean/ir_bytes_probe.py"
with open(pb, "w", encoding="utf-8") as fh:
    fh.write(KERNEL + '''
compiled = add_kernel[grid](x, y, o, N, BLOCK=BLOCK)
torch.cuda.synchronize()
k = add_kernel.cache[list(add_kernel.cache.keys())[0]]
k = list(k.values())[0] if isinstance(k, dict) else k
print("CACHED_TYPE", type(k).__name__)
print("ASM_KEYS", sorted((getattr(k, "asm", {}) or {}).keys()))
meta = getattr(k, "metadata", None)
print("META_FIELDS", sorted([f for f in dir(meta) if not f.startswith("_")])[:40] if meta else None)
ttir = (getattr(k, "asm", {}) or {}).get("ttir", "")
ttgir = (getattr(k, "asm", {}) or {}).get("ttgir", "")
for name, ir in (("TTIR", ttir), ("TTGIR", ttgir)):
    if not ir:
        print(f"{name}_ABSENT")
        continue
    loads = ir.count("tt.load")
    stores = ir.count("tt.store")
    print(f"{name}_LOADS={loads} {name}_STORES={stores} {name}_CHARS={len(ir)}")
    # The tensor shape and element type appear in the op's type annotation, e.g.
    # tensor<1024xf32>. If those are present, bytes-per-instance is computable.
    import re
    shapes = re.findall(r"tensor<(\\d+)x([a-z0-9]+)>", ir)
    print(f"{name}_TYPED_TENSORS={shapes[:8]} (n={len(shapes)})")
with open("/root/probe-clean/dump_ttir.txt", "w", encoding="utf-8") as g:
    g.write(ttir or "")
print("PTX_PRESENT", bool((getattr(k, "asm", {}) or {}).get("ptx")))
''')
rc, out = run(f"cd /root/probe-clean && {py} {pb}")
print(f"IR_RC={rc}")
for line in out.splitlines()[:30]:
    print("   ", line)

print()
print("=== ROUTE B2: the arithmetic check -- do IR-derived bytes match the known truth? ===")
pc = "/root/probe-clean/ir_bytes_math.py"
with open(pc, "w", encoding="utf-8") as fh:
    fh.write(KERNEL + f'''
import re
compiled = add_kernel[grid](x, y, o, N, BLOCK=BLOCK)
torch.cuda.synchronize()
entry = add_kernel.cache[list(add_kernel.cache.keys())[0]]
k = list(entry.values())[0] if isinstance(entry, dict) else entry
ir = (getattr(k, "asm", {{}}) or {{}}).get("ttir", "")
_W = {{"f32": 4, "f16": 2, "bf16": 2, "i32": 4, "i64": 8, "i8": 1, "i1": 1, "f64": 8}}
per_instance = 0
detail = []
for line in ir.splitlines():
    if "tt.load" not in line and "tt.store" not in line:
        continue
    # The VALUE type is the last tensor<...> on the line for a load, and the stored operand's
    # type for a store. Take every typed tensor on the line and use the widest element type
    # with the largest count -- the data tensor, not the index tensor (which is i32/i64).
    cands = re.findall(r"tensor<(\\d+)x([a-z0-9]+)>", line)
    data = [(int(n), t) for n, t in cands if t in _W and t not in ("i32", "i64")]
    if not data:
        data = [(int(n), t) for n, t in cands if t in _W]
    if not data:
        continue
    n, t = max(data)
    b = n * _W[t]
    per_instance += b
    detail.append((("load" if "tt.load" in line else "store"), n, t, b))
n_instances = {(4096 * 4096 + 1023) // 1024}
total = per_instance * n_instances
print("PER_INSTANCE_BYTES", per_instance)
print("N_INSTANCES", n_instances)
print("IR_DERIVED_TOTAL", total)
print("KNOWN_TRUE_BYTES", {KNOWN_BYTES})
print("RATIO", round(total / {KNOWN_BYTES}, 4) if total else None)
for d in detail:
    print("   OP", d)
''')
rc, out = run(f"cd /root/probe-clean && {py} {pc}")
print(f"MATH_RC={rc}")
for line in out.splitlines()[:25]:
    print("   ", line)
