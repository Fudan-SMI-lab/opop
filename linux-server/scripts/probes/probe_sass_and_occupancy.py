"""Step 5 probe: can we recover SASS instruction mix and occupancy without counters?

The claim to test, from the counter-free capability research: `nvdisasm` needs NO permissions
(unlike `ncu`, which returns ERR_NVGPUCTRPERM in these containers and cannot be enabled from
inside one), so disassembling a compiled kernel's cubin should recover:

  HMMA / IMMA / BMMA / OMMA  tensor-core instructions -- is the kernel using them at all?
  STL / LDL                  local-memory traffic = register spills, the highest-hit-rate
                             signal in KernelPro's data (18.2%)
  LDS / STS                  shared-memory traffic
  BAR.SYNC                   barriers
  LDG / STG                  global loads/stores, and their width (.128 / .64 / .32) which is
                             the vectorization signal

And that theoretical occupancy is ANALYTICALLY computable from fields already collected
(n_regs, shared, num_warps) plus device properties -- with a `limiter` saying which resource
binds. KernelPro reports occupancy as their highest-GAIN signal (1.48x).

Both claims are mine and untested on this box, so this probe is the positive control: it runs a
kernel deliberately built to spill and one built to use tensor cores, and FAILS LOUDLY if the
signals do not separate them. A probe that reports "no spills found" on a kernel written to
spill is measuring its own breakage, which is a mistake I have already made once here.
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import triton
import triton.language as tl

DEV = "cuda"


# ---- a kernel that MUST use tensor cores: tl.dot on fp16 -----------------------------------
@triton.jit
def tc_kernel(A, B, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    om = pid_m * BM + tl.arange(0, BM)
    on = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        ok = k0 + tl.arange(0, BK)
        a = tl.load(A + om[:, None] * K + ok[None, :], mask=om[:, None] < M, other=0.0)
        b = tl.load(B + ok[:, None] * N + on[None, :], mask=on[None, :] < N, other=0.0)
        acc += tl.dot(a, b)
    tl.store(C + om[:, None] * N + on[None, :], acc,
             mask=(om[:, None] < M) & (on[None, :] < N))


# ---- a kernel that MUST spill ---------------------------------------------------------------
# Two earlier attempts did NOT spill, and Triton's own n_spills agreed with the SASS at 0 both
# times -- so the detector was right and my synthetic case was wrong. What actually spills is a
# large tl.dot tile: the MMA accumulator has to stay live across the whole K loop, and at
# 128x128 with only 4 warps that is 4096 floats per thread of accumulator alone. This is also the
# realistic shape of the failure, since an over-large tile is exactly what an agent produces when
# it maximizes BLOCK_M/BLOCK_N.
@triton.jit
def spill_kernel(A, B, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    om = pid_m * BM + tl.arange(0, BM)
    on = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        ok = k0 + tl.arange(0, BK)
        a = tl.load(A + om[:, None] * K + ok[None, :], mask=om[:, None] < M, other=0.0)
        b = tl.load(B + ok[:, None] * N + on[None, :], mask=on[None, :] < N, other=0.0)
        acc += tl.dot(a, b)
    tl.store(C + om[:, None] * N + on[None, :], acc,
             mask=(om[:, None] < M) & (on[None, :] < N))


# ---- a kernel that MUST use tensor cores is above; a plain scalar one below -----------------


# ---- a plain scalar kernel: neither tensor cores nor spills --------------------------------
@triton.jit
def plain_kernel(X, Y, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    tl.store(Y + off, tl.load(X + off, mask=m, other=0.0) * 2.0 + 1.0, mask=m)


SASS_PATTERNS = {
    # Tensor cores. All four families, because a bf16/fp16 kernel emits HMMA, an int8 one IMMA,
    # and Hopper's async variants appear as different mnemonics -- matching only "HMMA" would
    # report "no tensor cores" on a kernel that is full of them.
    "tensor_core": re.compile(r"\b(HMMA|IMMA|BMMA|OMMA|QMMA)\b"),
    # Local memory = the spill space. STL/LDL are the only way a spill becomes visible without
    # counters, and n_spills from Triton metadata is bytes, not instruction count, so these are
    # complementary rather than redundant.
    "spill_store": re.compile(r"\bSTL\b"),
    "spill_load": re.compile(r"\bLDL\b"),
    "shared_store": re.compile(r"\bSTS\b"),
    "shared_load": re.compile(r"\bLDS\b"),
    "barrier": re.compile(r"\bBAR\.SYNC\b|\bBARRIER\b"),
    "global_load": re.compile(r"\bLDG\b"),
    "global_store": re.compile(r"\bSTG\b"),
    # Width of global access: .128 is fully vectorized, .32 is scalar. KernelPro rates
    # vectorization as their LOWEST-hit signal (4.0%), so this is collected but not prioritized.
    "vec_128": re.compile(r"\b(LDG|STG)[^\s;]*\.128\b"),
    "vec_64": re.compile(r"\b(LDG|STG)[^\s;]*\.64\b"),
}


def find_tool(name: str) -> str | None:
    """Locate a CUDA binary. PATH first, then the toolkit directories.

    NOT just shutil.which: measured on box 2, nvdisasm and cuobjdump are installed at
    /usr/local/cuda/bin/ but absent from the login shell's PATH, so a PATH-only lookup reports
    "unavailable" for a tool that is present and working. That is the same class of failure as
    the opencode PATH bug -- an available capability read as a missing one.
    """
    found = shutil.which(name)
    if found:
        return found
    roots = [os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"), "/usr/local/cuda"]
    try:
        from torch.utils.cpp_extension import CUDA_HOME as TORCH_CUDA_HOME

        roots.append(TORCH_CUDA_HOME)
    except Exception:  # noqa: BLE001
        pass
    # Any versioned toolkit, newest first, so a box with several picks the most recent.
    roots += sorted(glob.glob("/usr/local/cuda-*"), reverse=True)
    for root in roots:
        if not root:
            continue
        cand = Path(root) / "bin" / name
        if cand.exists():
            return str(cand)
    return None


def disassemble(cubin: bytes) -> str | None:
    """SASS text for a cubin via nvdisasm. Returns None when nvdisasm is unavailable."""
    exe = find_tool("nvdisasm")
    if exe is None:
        return None
    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as f:
        f.write(cubin)
        path = f.name
    try:
        out = subprocess.run([exe, "-c", path], capture_output=True, timeout=120)
        if out.returncode != 0:
            # A cubin holding several architectures is not directly disassemblable; cuobjdump
            # extracts the SASS for whichever it holds.
            dump = find_tool("cuobjdump")
            if dump:
                out2 = subprocess.run([dump, "-sass", path], capture_output=True, timeout=120)
                if out2.returncode == 0:
                    return out2.stdout.decode("utf-8", errors="replace")
            return None
        return out.stdout.decode("utf-8", errors="replace")
    finally:
        Path(path).unlink(missing_ok=True)


def count_sass(sass: str) -> dict:
    counts = {k: 0 for k in SASS_PATTERNS}
    total = 0
    for line in sass.splitlines():
        # Instruction lines carry a predicate/mnemonic and end in ';'. Section headers and
        # branch labels do not, so this avoids counting a symbol name as an instruction.
        if ";" not in line:
            continue
        total += 1
        for key, pat in SASS_PATTERNS.items():
            if pat.search(line):
                counts[key] += 1
    counts["instructions"] = total
    return counts


def occupancy(n_regs: int, shared_bytes: int, num_warps: int, props) -> dict:
    """Theoretical occupancy and WHICH resource limits it. Purely analytic -- no counters.

    The limiter is the point. "Occupancy 25%" tells an agent nothing actionable; "occupancy 25%,
    limited by registers at 168/thread" tells it which knob to turn.
    """
    warp_size = 32
    max_warps_per_sm = props.max_threads_per_multi_processor // warp_size
    regs_per_sm = getattr(props, "regs_per_multiprocessor", 65536)
    shared_per_sm = getattr(props, "shared_memory_per_multiprocessor", 102400)
    max_blocks_per_sm = getattr(props, "max_blocks_per_multi_processor", 16) or 16

    threads_per_block = num_warps * warp_size
    # Registers are allocated per warp in granular chunks; using the per-thread figure directly
    # is the standard approximation and errs slightly optimistic.
    regs_per_block = max(1, n_regs) * threads_per_block
    by_regs = regs_per_sm // regs_per_block if regs_per_block else max_blocks_per_sm
    by_shared = (shared_per_sm // shared_bytes) if shared_bytes > 0 else max_blocks_per_sm
    by_warps = max_warps_per_sm // num_warps if num_warps else max_blocks_per_sm

    blocks = max(0, min(by_regs, by_shared, by_warps, max_blocks_per_sm))
    active_warps = blocks * num_warps
    occ = active_warps / max_warps_per_sm if max_warps_per_sm else 0.0

    binding = min(by_regs, by_shared, by_warps, max_blocks_per_sm)
    if binding == by_regs and by_regs < max_blocks_per_sm:
        limiter = "registers"
    elif binding == by_shared and by_shared < max_blocks_per_sm:
        limiter = "shared_memory"
    elif binding == by_warps and by_warps <= max_blocks_per_sm:
        limiter = "warps_per_block"
    else:
        limiter = "blocks_per_sm"
    return {"occupancy": round(occ, 4), "active_warps": active_warps,
            "max_warps_per_sm": max_warps_per_sm, "blocks_per_sm": blocks,
            "limiter": limiter,
            "by_regs": by_regs, "by_shared": by_shared, "by_warps": by_warps}


def probe_kernel(label, launch, jitted, props) -> dict:
    launch()
    torch.cuda.synchronize()
    got = []
    for entry in jitted.device_caches.values():
        if not entry:
            continue
        for compiled in entry[0].values():
            asm = getattr(compiled, "asm", {}) or {}
            cubin = asm.get("cubin")
            meta = getattr(compiled, "metadata", None)
            row = {
                "label": label,
                "name": getattr(compiled, "name", None),
                "n_regs": getattr(compiled, "n_regs", None),
                "n_spills": getattr(compiled, "n_spills", None),
                "shared": getattr(meta, "shared", None) if meta else None,
                "num_warps": getattr(meta, "num_warps", None) if meta else None,
                "asm_keys": sorted(asm.keys()),
                "cubin_bytes": len(cubin) if cubin else 0,
            }
            if cubin:
                sass = disassemble(cubin)
                row["nvdisasm_ok"] = sass is not None
                if sass:
                    row["sass"] = count_sass(sass)
            if row.get("n_regs") and row.get("num_warps"):
                row["occ"] = occupancy(row["n_regs"], row.get("shared") or 0,
                                       row["num_warps"], props)
            got.append(row)
    return got


def main() -> int:
    props = torch.cuda.get_device_properties(0)
    print(f"device: {props.name} sm_{props.major}{props.minor}")
    print(f"nvdisasm: {find_tool('nvdisasm')}")
    print(f"cuobjdump: {find_tool('cuobjdump')}")
    print(f"ncu: {find_tool('ncu')} (exists but FAILS on counter permissions in a container)")
    print(f"max_threads_per_multi_processor {props.max_threads_per_multi_processor}, "
          f"regs_per_multiprocessor {getattr(props, 'regs_per_multiprocessor', '?')}, "
          f"shared_per_sm {getattr(props, 'shared_memory_per_multiprocessor', '?')}")
    print()

    torch.manual_seed(0)
    rows = []

    # 1) tensor cores, fp16
    M = N = K = 512
    a = torch.randn(M, K, device=DEV, dtype=torch.float16)
    b = torch.randn(K, N, device=DEV, dtype=torch.float16)
    c = torch.empty(M, N, device=DEV, dtype=torch.float32)
    rows += probe_kernel("TENSOR_CORE fp16 tl.dot",
                         lambda: tc_kernel[(M // 64, N // 64)](a, b, c, M, N, K,
                                                               BM=64, BN=64, BK=32),
                         tc_kernel, props)

    # 2) deliberate spill: a 128x128 fp32 accumulator on only 4 warps. The accumulator alone is
    #    4096 floats/thread, so the register file cannot hold it.
    S = 256
    sa = torch.randn(S, S, device=DEV)
    sb = torch.randn(S, S, device=DEV)
    sc = torch.empty(S, S, device=DEV)
    rows += probe_kernel("SPILL 128x128 fp32 acc on 4 warps",
                         lambda: spill_kernel[(S // 128, S // 128)](sa, sb, sc, S, S, S,
                                                                    BM=128, BN=128, BK=16,
                                                                    num_warps=4),
                         spill_kernel, props)

    # 3) plain
    n = 1 << 20
    x = torch.randn(n, device=DEV)
    y = torch.empty_like(x)
    rows += probe_kernel("PLAIN elementwise",
                         lambda: plain_kernel[(n // 1024,)](x, y, n, BLOCK=1024),
                         plain_kernel, props)

    for r in rows:
        print(f"--- {r['label']}  ({r.get('name')})")
        print(f"    n_regs {r.get('n_regs')}  n_spills {r.get('n_spills')}  "
              f"shared {r.get('shared')}  num_warps {r.get('num_warps')}")
        print(f"    asm keys: {r.get('asm_keys')}  cubin {r.get('cubin_bytes')} B  "
              f"nvdisasm_ok {r.get('nvdisasm_ok')}")
        s = r.get("sass")
        if s:
            print(f"    SASS {s['instructions']} instrs: "
                  f"tensor_core={s['tensor_core']} STL={s['spill_store']} LDL={s['spill_load']} "
                  f"STS={s['shared_store']} LDS={s['shared_load']} BAR={s['barrier']} "
                  f"LDG={s['global_load']} STG={s['global_store']} "
                  f"vec128={s['vec_128']} vec64={s['vec_64']}")
        o = r.get("occ")
        if o:
            print(f"    occupancy {o['occupancy']*100:.1f}% "
                  f"({o['active_warps']}/{o['max_warps_per_sm']} warps, "
                  f"{o['blocks_per_sm']} blocks/SM) limited by {o['limiter']} "
                  f"[regs->{o['by_regs']} shared->{o['by_shared']} warps->{o['by_warps']}]")
        print()

    # ---- POSITIVE CONTROLS. Without these, "no spills detected" is indistinguishable from a
    # broken detector -- a mistake already made once in this project.
    print("=== POSITIVE CONTROLS")
    ok = True

    def check(label, passed, detail=""):
        nonlocal ok
        print(f"  [{'OK  ' if passed else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
        ok = ok and passed

    tc = [r for r in rows if r["label"].startswith("TENSOR") and r.get("sass")]
    sp = [r for r in rows if r["label"].startswith("SPILL") and r.get("sass")]
    pl = [r for r in rows if r["label"].startswith("PLAIN") and r.get("sass")]

    check("nvdisasm produced SASS for every kernel",
          len(tc) + len(sp) + len(pl) == len(rows) and all(r.get("nvdisasm_ok") for r in rows),
          f"{len(rows)} kernels, sass for {len(tc)+len(sp)+len(pl)}")
    if tc:
        check("an fp16 tl.dot kernel shows tensor-core instructions",
              tc[0]["sass"]["tensor_core"] > 0,
              f"found {tc[0]['sass']['tensor_core']}")
    if pl:
        check("a plain elementwise kernel shows NO tensor-core instructions",
              pl[0]["sass"]["tensor_core"] == 0,
              f"found {pl[0]['sass']['tensor_core']}")
    # The spill detector's positive control is CROSS-VALIDATION against Triton's own n_spills,
    # not a synthetic kernel. Two attempts at a deliberately-spilling kernel both failed, and
    # Triton's n_spills agreed with the SASS at 0 each time -- so the detector was right and my
    # synthetic case was wrong. The finding behind that: Triton's register allocator prefers to
    # CAP registers and lose occupancy over spilling (the 128x128 tile lands at 218 regs, 0
    # spills, 16.7% occupancy). So in Triton, extreme register pressure shows up as low
    # occupancy, and spills are comparatively rare -- which makes occupancy the higher-value
    # signal here and matches KernelPro rating it their highest-gain one.
    #
    # Two independent sources agreeing on a nonzero count is a stronger control than either
    # alone: STL/LDL comes from disassembled SASS, n_spills from the Triton compiler.
    spillers = [r for r in rows
                if r.get("sass") and (r.get("n_spills") or 0) > 0]
    if spillers:
        r = spillers[0]
        stl = r["sass"]["spill_store"] + r["sass"]["spill_load"]
        check("SASS local-memory traffic agrees with Triton's own n_spills (cross-validated)",
              stl > 0,
              f"{r['label']}: STL={r['sass']['spill_store']} LDL={r['sass']['spill_load']} "
              f"vs triton n_spills={r['n_spills']}")
    else:
        check("at least one kernel spilled, so the STL/LDL detector could be validated", False,
              "no kernel in this set spilled; the detector is UNVALIDATED on this box")
    zero_spill = [r for r in rows if r.get("sass") and (r.get("n_spills") or 0) == 0]
    check("kernels Triton reports as spill-free show no local-memory traffic either",
          all(r["sass"]["spill_store"] + r["sass"]["spill_load"] == 0 for r in zero_spill),
          f"{len(zero_spill)} spill-free kernels checked")
    check("occupancy separates a large-tile kernel from a small one",
          len({r["occ"]["occupancy"] for r in rows if r.get("occ")}) >= 2,
          str(sorted({r["occ"]["occupancy"] for r in rows if r.get("occ")})))
    if tc:
        check("a tl.dot kernel shows shared-memory traffic and barriers",
              tc[0]["sass"]["shared_load"] + tc[0]["sass"]["shared_store"] > 0
              and tc[0]["sass"]["barrier"] > 0)
    occs = [r["occ"]["occupancy"] for r in rows if r.get("occ")]
    check("occupancy is computed and in (0, 1]", bool(occs) and all(0 < o <= 1 for o in occs),
          str(occs))
    limiters = {r["occ"]["limiter"] for r in rows if r.get("occ")}
    check("the occupancy limiter is identified", bool(limiters), str(limiters))

    print()
    print("ALL CONTROLS PASSED" if ok else "SOME CONTROLS FAILED -- do not trust these signals")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
