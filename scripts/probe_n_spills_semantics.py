"""Is Triton's `n_spills` a COUNT of spill instructions, or local-memory bytes / 4?

Why it matters: `evaluation/statics.py` and `bottleneck.py` both claim the disassembly's spill
signal is "cross-validated against Triton's own n_spills (STL=2/LDL=1 against n_spills=2,
agreeing exactly)". If `n_spills` is actually `CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES / 4`, then it
counts 32-bit local-memory SLOTS -- all local memory, not only spills -- and comparing it to a
COUNT of instructions is a unit error. The two would agree only by coincidence, and must diverge
as soon as any spill is wider than 4 bytes: an `STL.128` moves 16 bytes in one instruction.

The claim is load-bearing for the paper, which cites dual-source cross-validation of the
counter-free spill signal, so it gets tested rather than argued.

Method: generate kernels with a rising number of independent live accumulators, so register
pressure climbs until the allocator has to use local memory. For each, print `n_regs`,
`n_spills`, the STL/LDL instruction counts from `nvdisasm`, and the access widths seen. If
`n_spills` were an instruction count it would track STL+LDL; if it is bytes/4 it will exceed
that count whenever the widths are wider than `.32`.
"""
import os
import re
import subprocess
import tempfile

import torch
import triton  # noqa: F401 -- imported for the generated source's namespace

SRC_TEMPLATE = '''
import torch, triton, triton.language as tl

@triton.jit
def k(X, Y, N, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    a = tl.load(X + off, mask=m, other=0.0)
{decls}
    s = {ssum}
    tl.store(Y + off, s, mask=m)
'''


def build(nacc: int) -> str:
    """Source for a kernel holding `nacc` values live across a transcendental each.

    tl.sin/tl.cos are used deliberately: they are long-latency and their operands must stay live
    across the call, which is what pushes the allocator into local memory. Plain arithmetic gets
    rescheduled and the pressure never materialises.
    """
    decls = "\n".join(
        "    v{i} = a * {c}.0 + tl.sin(a * {i}.5)".format(i=i, c=i + 1) for i in range(nacc))
    ssum = " + ".join(
        "v{i} * tl.cos(v{j})".format(i=i, j=(i + 1) % nacc) for i in range(nacc))
    return SRC_TEMPLATE.format(decls=decls, ssum=ssum)


def disasm_counts(cubin: bytes) -> tuple:
    """STL/LDL instruction counts and the set of access widths, from nvdisasm."""
    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as f:
            f.write(cubin)
            path = f.name
        for exe in ("/usr/local/cuda/bin/nvdisasm", "nvdisasm"):
            try:
                r = subprocess.run([exe, "-c", path], capture_output=True, text=True, timeout=120)
            except FileNotFoundError:
                continue
            txt = r.stdout or ""
            if not txt:
                continue
            stl = len(re.findall(r"\bSTL(?:\.\d+)?\s", txt))
            ldl = len(re.findall(r"\bLDL(?:\.\d+)?\s", txt))
            widths = sorted(set(re.findall(r"\b(?:STL|LDL)(\.\d+)?", txt)))
            return stl, ldl, [w or ".32" for w in widths]
        return None, None, ["nvdisasm unavailable"]
    finally:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass


def main() -> int:
    x = torch.randn(8192, device="cuda")
    y = torch.empty_like(x)
    print("%-6s %-7s %-8s %-10s %-6s %-6s %-9s %s"
          % ("NACC", "BLOCK", "n_regs", "n_spills", "STL", "LDL", "STL+LDL", "widths"))
    rows = []
    for nacc, block in ((4, 256), (16, 512), (40, 1024), (80, 1024), (140, 1024)):
        try:
            # Triton refuses a @jit function defined by exec ("@jit functions should be
            # defined in a Python file"): it re-reads the decorated function's source from disk
            # to build its cache key. So the generated kernel has to be a real module file.
            import importlib.util
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                             encoding="utf-8") as f:
                f.write(build(nacc))
                gen_path = f.name
            spec = importlib.util.spec_from_file_location("spill_probe_%d" % nacc, gen_path)
            gen = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(gen)
            c = gen.k.warmup(x, y, 8192, BLOCK=block, grid=(1,))
            c._init_handles()
            stl, ldl, widths = disasm_counts(c.asm.get("cubin") or b"")
            tot = (stl + ldl) if isinstance(stl, int) and isinstance(ldl, int) else None
            rows.append((nacc, c.n_regs, c.n_spills, stl, ldl, tot, widths))
            print("%-6s %-7s %-8s %-10s %-6s %-6s %-9s %s"
                  % (nacc, block, c.n_regs, c.n_spills, stl, ldl, tot, widths))
        except Exception as exc:  # noqa: BLE001 -- one failing size must not lose the sweep
            print("%-6s %-7s FAILED %s: %s" % (nacc, block, type(exc).__name__, str(exc)[:70]))

    print()
    spilling = [r for r in rows if r[2] and r[2] > 0]
    if not spilling:
        print("VERDICT: INCONCLUSIVE -- no configuration spilled, so the two quantities were")
        print("  never both nonzero and nothing was compared. A probe that produced no positive")
        print("  case has not tested the claim.")
        return 2
    print("configurations that actually spilled: %d" % len(spilling))
    mismatch = [r for r in spilling if r[5] is not None and r[2] != r[5]]
    for nacc, nr, nsp, stl, ldl, tot, widths in spilling:
        note = "n_spills == STL+LDL" if tot == nsp else "n_spills != STL+LDL"
        print("  NACC=%-4d regs=%-4s n_spills=%-6s STL+LDL=%-6s %s  widths=%s"
              % (nacc, nr, nsp, tot, note, widths))
    if mismatch:
        print()
        print("VERDICT: n_spills is NOT an instruction count -- %d of %d spilling configurations"
              % (len(mismatch), len(spilling)))
        print("  disagree with STL+LDL. The 'cross-validated, agreeing exactly' claim in")
        print("  statics.py and bottleneck.py describes a coincidence on one kernel, not a")
        print("  validated correspondence, and must be corrected.")
    else:
        print()
        print("VERDICT: n_spills tracked STL+LDL on every spilling configuration here (%d)."
              % len(spilling))
        print("  That is consistent with an instruction count OR with bytes/4 where every access")
        print("  happened to be 4 bytes wide -- check the widths column before concluding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
