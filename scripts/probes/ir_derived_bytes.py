"""Are per-candidate LOGICAL bytes derivable from the compiled Triton artefact, with no counters?

WHY THIS MATTERS. Hardware counters are refused on this box (`RmProfilingAdminOnly: 1`, and the
container drops both `cap_sys_admin` and `cap_perfmon`), so `dram__bytes.sum` is unreachable. Triton
Proton was measured and reports only `time (ns)` -- latency again, which is the thing we already
have. This probe tests the remaining route: every `tt.load`/`tt.store` in the compiled IR carries a
tensor shape and an element type, so the bytes ONE kernel instance touches should be computable at
compile time, and multiplying by the grid gives a per-candidate, per-parameter-set figure.

WHAT SUCH A NUMBER WOULD AND WOULD NOT BE. It is a LOGICAL byte count: it cannot see an L2 hit, so
against DRAM it over-counts exactly the way `pct_of_dram_peak` already does. What makes it worth
measuring anyway is the property the current number lacks -- it varies with the CANDIDATE and with
the PARAMETER SET, instead of being one constant per task.

THE CHECK IS ARITHMETIC, NOT PLAUSIBILITY. The kernel is a 4096*4096 fp32 elementwise add: two
matrices read, one written, 3 * 4096 * 4096 * 4 = 201326592 bytes, known by construction. A route
that cannot reproduce a number this simple cannot be trusted on a real candidate. Reported as a
ratio so it can be read against 1.000.

Reaching the compiled object uses the pattern already proven in `worker_main.py`: wrap
`JITFunction.run`, which returns the compiled kernel. `JITFunction.cache` does not exist in Triton
3.7 -- guessing that attribute is what an earlier version of this probe got wrong.
"""
import re
import sys

TRUE_BYTES = 3 * 4096 * 4096 * 4
_W = {"f32": 4, "f16": 2, "bf16": 2, "f64": 8, "i8": 1, "i16": 2, "i32": 4, "i64": 8, "i1": 1}

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from triton.runtime.jit import JITFunction  # noqa: E402


@triton.jit
def add_kernel(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    y = tl.load(y_ptr + offs, mask=m)
    tl.store(o_ptr + offs, x + y, mask=m)


captured = []
_original = JITFunction.run


def run_capturing(self, *args, grid=None, warmup=False, **kwargs):
    kernel = _original(self, *args, grid=grid, warmup=warmup, **kwargs)
    captured.append((getattr(self, "__name__", "?"), kernel))
    return kernel


def bytes_per_instance(ir: str) -> tuple[int, list]:
    """Bytes one program instance loads and stores, from the typed tensors on each memory op.

    THE TYPE IS INSIDE THE POINTER, which is what an earlier version of this probe got wrong and
    why it read a confident 0. A real `tt.load` line from Triton 3.7 looks like:

        %x_5 = tt.load %x_4, %m_3 : tensor<1024x!tt.ptr<f32>> loc(#loc25)

    so the element type is `f32` nested in `!tt.ptr<...>`, and a pattern expecting
    `tensor<1024xf32>` matches nothing on the very lines that carry the traffic. The plain form does
    appear in the IR -- `tensor<1024xi32>` for the offsets and `tensor<1024xf32>` on the `arith.addf`
    -- so a naive pattern silently charges arithmetic and index math as memory traffic while missing
    every actual load and store. Both spellings are handled here, pointer form first.

    Reading zero was the useful failure: `a-constant-reading-is-a-broken-probe` says the dangerous
    outcome is a plausible constant, and 0 bytes for a kernel that obviously moves bytes is loud.
    """
    total = 0
    detail = []
    for line in ir.splitlines():
        is_load, is_store = "tt.load" in line, "tt.store" in line
        if not (is_load or is_store):
            continue
        # Pointer form: tensor<Nx!tt.ptr<T>> -- the authoritative one for a memory op.
        cands = [(int(n), t) for n, t in
                 re.findall(r"tensor<(\d+)x!tt\.ptr<([a-z0-9]+)>>", line) if t in _W]
        if not cands:
            # Plain form, for builds/ops that annotate the value type directly. Integer and boolean
            # types are dropped so an offset or mask tensor is never charged as traffic.
            cands = [(int(n), t) for n, t in re.findall(r"tensor<(\d+)x([a-z0-9]+)>", line)
                     if t in _W and t not in ("i1", "i8", "i16", "i32", "i64")]
        if not cands:
            continue
        n, t = max(cands)
        nbytes = n * _W[t]
        total += nbytes
        detail.append(("load" if is_load else "store", n, t, nbytes))
    return total, detail


N = 4096 * 4096
BLOCK = 1024
x = torch.randn(N, device="cuda")
y = torch.randn(N, device="cuda")
o = torch.empty_like(x)
grid = (triton.cdiv(N, BLOCK),)

JITFunction.run = run_capturing
try:
    add_kernel[grid](x, y, o, N, BLOCK=BLOCK)
    torch.cuda.synchronize()
finally:
    JITFunction.run = _original

print(f"CAPTURED={len(captured)}")
if not captured:
    print("FAILED: no compiled kernel captured -- the wrap did not fire")
    raise SystemExit(1)

name, kernel = captured[0]
asm = getattr(kernel, "asm", {}) or {}
print(f"KERNEL={name}")
print(f"ASM_KEYS={sorted(asm.keys())}")
meta = getattr(kernel, "metadata", None)
for f in ("shared", "num_warps", "num_stages", "num_regs", "num_spills"):
    print(f"META_{f}={getattr(meta, f, None) if meta else None}")

ttir = asm.get("ttir", "")
if not ttir:
    print("NO_TTIR: cannot derive bytes from the IR on this Triton build")
    raise SystemExit(1)

per_inst, detail = bytes_per_instance(ttir)
n_inst = grid[0]
derived = per_inst * n_inst
print()
print(f"PER_INSTANCE_BYTES={per_inst}")
print(f"N_INSTANCES={n_inst}")
print(f"IR_DERIVED_TOTAL={derived}")
print(f"KNOWN_TRUE_BYTES={TRUE_BYTES}")
print(f"RATIO={derived / TRUE_BYTES:.4f}" if derived else "RATIO=None")
for d in detail:
    print(f"   OP {d}")

print()
print("=== does the derived number MOVE with a parameter, which is the whole point? ===")
# A constant that happens to be right on one configuration is worthless: the reason this route is
# being tested is that the CURRENT number is a task-level constant. So vary BLOCK and check the
# per-instance figure responds, and that the TOTAL stays invariant (the same data is touched however
# it is tiled) -- the invariance is itself the correctness property.
for blk in (256, 512, 2048):
    captured.clear()
    JITFunction.run = run_capturing
    try:
        add_kernel[(triton.cdiv(N, blk),)](x, y, o, N, BLOCK=blk)
        torch.cuda.synchronize()
    finally:
        JITFunction.run = _original
    if not captured:
        print(f"BLOCK={blk}: not captured")
        continue
    ir = (getattr(captured[0][1], "asm", {}) or {}).get("ttir", "")
    pi, _ = bytes_per_instance(ir)
    tot = pi * triton.cdiv(N, blk)
    print(f"BLOCK={blk:5d}  per_instance={pi:7d}  instances={triton.cdiv(N, blk):6d}  "
          f"total={tot:12d}  ratio={tot / TRUE_BYTES:.4f}")

print()
print("=== POSITIVE CONTROL: a kernel with KNOWN DIFFERENT traffic must read differently ===")
# Without this, a ratio of 1.000 above could be a coincidence of the constants involved. This
# kernel reads ONE tensor and writes one, so its true traffic is 2/3 of the add kernel's. A parser
# that is actually reading the IR must report that; one that is echoing an assumption will not.
# `probe-needs-a-positive-control`: a negative or confirming result needs a control that would fail
# loudly.


@triton.jit
def scale_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    tl.store(o_ptr + offs, x * 2.0, mask=m)


CONTROL_TRUE = 2 * 4096 * 4096 * 4
captured.clear()
JITFunction.run = run_capturing
try:
    scale_kernel[(triton.cdiv(N, BLOCK),)](x, o, N, BLOCK=BLOCK)
    torch.cuda.synchronize()
finally:
    JITFunction.run = _original
if captured:
    ir = (getattr(captured[0][1], "asm", {}) or {}).get("ttir", "")
    pi, det = bytes_per_instance(ir)
    tot = pi * grid[0]
    print(f"CONTROL_PER_INSTANCE={pi}  (add kernel was {per_inst})")
    print(f"CONTROL_TOTAL={tot}  CONTROL_TRUE={CONTROL_TRUE}  "
          f"ratio={tot / CONTROL_TRUE:.4f}")
    print(f"CONTROL_DISTINGUISHES={'YES' if pi != per_inst else 'NO -- parser is not reading the IR'}")
    for d in det:
        print(f"   OP {d}")

print()
print("=== A HARDER CASE: does it survive a 2-D tiled matmul, where operands are re-read? ===")
# The elementwise kernels above touch each byte once, so they cannot expose the route's real limit.
# A tiled matmul re-reads A and B tiles across the K loop, so LOGICAL bytes far exceed the
# compulsory bytes -- and that gap is the honest answer about what this route measures. Reported,
# not judged: the number is what it is, and the gap is a property of tiling rather than an error.
M = K = Nn = 1024


@triton.jit
def mm_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        offs_k = k + tl.arange(0, BK)
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
        b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
        acc += tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)


a2 = torch.randn(M, K, device="cuda")
b2 = torch.randn(K, Nn, device="cuda")
c2 = torch.empty(M, Nn, device="cuda")
BM = BN = BK = 64
captured.clear()
JITFunction.run = run_capturing
try:
    mm_kernel[(M // BM, Nn // BN)](a2, b2, c2, M, Nn, K, BM=BM, BN=BN, BK=BK)
    torch.cuda.synchronize()
except Exception as exc:  # noqa: BLE001 — a matmul that will not compile is not this probe's subject
    print(f"MM_COMPILE_FAILED={type(exc).__name__}: {str(exc)[:200]}")
finally:
    JITFunction.run = _original
if captured:
    ir = (getattr(captured[0][1], "asm", {}) or {}).get("ttir", "")
    # 2-D tensors are spelled tensor<64x64x!tt.ptr<f32>>, which the 1-D pattern above cannot read.
    twod = re.findall(r"tensor<(\d+)x(\d+)x!tt\.ptr<([a-z0-9]+)>>", ir)
    print(f"MM_2D_POINTER_TENSORS={twod[:6]} (n={len(twod)})")
    print(f"MM_LOADS={ir.count('tt.load')}  MM_STORES={ir.count('tt.store')}  "
          f"MM_HAS_LOOP={'scf.for' in ir}")
    pi, det = bytes_per_instance(ir)
    print(f"MM_1D_PARSER_READS={pi}  <- 0 or wrong means the parser needs 2-D and loop-trip support")
    compulsory = (M * K + K * Nn + M * Nn) * 4
    print(f"MM_COMPULSORY_BYTES={compulsory}")
    print("NOTE: a per-instance figure must be multiplied by the K-loop trip count "
          f"({K // BK}) and the grid ({(M // BM) * (Nn // BN)}), so logical bytes will EXCEED "
          "compulsory bytes -- that gap is what this route can and cannot tell you.")
