"""Tier 1 static analysis: SASS instruction mix and analytic occupancy, without counters.

WHY THIS TIER EXISTS. `ncu` is installed on these boxes and returns ERR_NVGPUCTRPERM: hardware
counters need a host-side kernel-module parameter that cannot be set from inside a container, and
rented containers are where these experiments run. So the signals here come from two sources that
need no privileges at all:

  DISASSEMBLY (nvdisasm / cuobjdump on the compiled cubin) recovers the instruction MIX:
    tensor cores  HMMA/IMMA/BMMA/OMMA/QMMA -- is the kernel using them at all? Measured
                  separation on box 2: an fp16 tl.dot kernel shows 16, a scalar elementwise
                  kernel shows 0.
    spills        STL/LDL, i.e. local-memory traffic. Cross-validated against Triton's own
                  n_spills on box 2: STL=2/LDL=1 against n_spills=2, agreeing exactly.
    shared        LDS/STS and BAR.SYNC -- how much staging and synchronizing the kernel does.
    vectorization LDG/STG width (.128/.64/.32).

  ARITHMETIC on fields already collected (n_regs, shared, num_warps) plus device properties gives
  theoretical occupancy AND a `limiter` naming the binding resource. "Occupancy 17%" tells an
  agent nothing; "occupancy 17%, limited by registers at 218/thread" names the knob.

A MEASURED FINDING that changes the priorities. Triton's register allocator prefers to CAP
registers and lose occupancy over spilling: a 128x128 fp32 accumulator on 4 warps compiles to 218
regs, ZERO spills, and 16.7% occupancy. So in Triton, extreme register pressure surfaces as low
occupancy rather than as spills, which makes occupancy the higher-value signal on this backend --
consistent with KernelPro rating occupancy their highest-GAIN tool (1.48x). Spills stay collected
(they are their highest-HIT tool at 18.2%, and CUDA-backend candidates do spill) but occupancy is
what fires on our Triton candidates.

WHAT THIS STILL CANNOT SEE, stated so the set is not mistaken for complete: bank conflicts, warp
divergence, instruction-cache pressure, L2 hit rate, and stall reasons. Those need counters. The
absence is reported explicitly to the analyst rather than left as a silent gap, because an agent
told "nothing is wrong" reasons differently from one told "this cannot be measured here".
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

# NO pydantic import at module scope, and that is a hard constraint rather than a style choice.
# This module is imported BY THE GPU WORKER, which is documented as stdlib + torch + triton +
# kernelbench only (worker_main.py's own docstring) and whose venv does not have the harness
# installed. pydantic happens to be present in the worker venv on box 2, so a top-level import
# works there -- by luck. On a box provisioned strictly to the stated contract it would raise, and
# since Tier 1 is best-effort the failure would be SILENT: every kernel would carry a
# `statics_note` and no instruction mix, indistinguishable in the log from a box with no
# disassembler.
#
# So the counting and occupancy arithmetic below use plain dataclasses, which are stdlib. The
# pydantic models that the rest of the harness validates against are built from these by
# `SassCounts.model_validate(...)` at the call sites that already depend on pydantic.
from dataclasses import asdict, dataclass, field

# Instruction families, matched on disassembled SASS. Each pattern covers the whole family
# deliberately: matching only "HMMA" would report "no tensor cores" on an int8 kernel full of
# IMMA, and that false negative is worse than no signal, since it would send the agent to add
# tensor cores a kernel already has.
SASS_PATTERNS: dict[str, re.Pattern] = {
    "tensor_core": re.compile(r"\b(HMMA|IMMA|BMMA|OMMA|QMMA)\b"),
    "spill_store": re.compile(r"\bSTL\b"),
    "spill_load": re.compile(r"\bLDL\b"),
    "shared_store": re.compile(r"\bSTS\b"),
    "shared_load": re.compile(r"\bLDS\b"),
    "barrier": re.compile(r"\bBAR\.SYNC\b|\bBARRIER\b"),
    "global_load": re.compile(r"\bLDG\b"),
    "global_store": re.compile(r"\bSTG\b"),
    "vec_128": re.compile(r"\b(?:LDG|STG)[^\s;]*\.128\b"),
    "vec_64": re.compile(r"\b(?:LDG|STG)[^\s;]*\.64\b"),
}


@dataclass(frozen=True)
class SassCounts:
    """Instruction counts for one kernel's SASS. All zero is a legitimate result for a simple
    kernel; `instructions == 0` means disassembly failed and is the only "unknown".

    A stdlib dataclass, not a pydantic model, because the GPU worker imports this module -- see the
    note on the dataclass import above. `model_dump`/`model_validate` are provided so call sites
    read the same as they do for the harness's genuine pydantic models.
    """

    instructions: int = 0
    tensor_core: int = 0
    spill_store: int = 0
    spill_load: int = 0
    shared_store: int = 0
    shared_load: int = 0
    barrier: int = 0
    global_load: int = 0
    global_store: int = 0
    vec_128: int = 0
    vec_64: int = 0

    def model_dump(self) -> dict:
        return asdict(self)

    @classmethod
    def model_validate(cls, data: dict) -> "SassCounts":
        """Build from a dict, ignoring unknown keys.

        Tolerant on purpose: these dicts cross a process boundary (worker -> host) and are replayed
        from event logs written by earlier versions, so a field added later must not make an older
        record unreadable.
        """
        names = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in (data or {}).items() if k in names})

    @property
    def uses_tensor_cores(self) -> bool:
        return self.tensor_core > 0

    @property
    def spill_instructions(self) -> int:
        return self.spill_store + self.spill_load

    @property
    def vectorized_frac(self) -> float:
        """Share of global accesses that are 128-bit. 0.0 with no global accesses at all, which is
        why the caller must check `global_load + global_store` before reading anything into it."""
        total = self.global_load + self.global_store
        if total <= 0:
            return 0.0
        return min(1.0, self.vec_128 / total)


@dataclass(frozen=True)
class Occupancy:
    """Theoretical occupancy and the resource that binds it. Analytic -- no counters."""

    occupancy: float
    active_warps: int
    max_warps_per_sm: int
    blocks_per_sm: int
    # "registers" | "shared_memory" | "warps_per_block" | "blocks_per_sm". The actionable half:
    # a bare percentage names no knob.
    limiter: str
    by_regs: int
    by_shared: int
    by_warps: int

    def model_dump(self) -> dict:
        return asdict(self)

    @classmethod
    def model_validate(cls, data: dict) -> "Occupancy":
        names = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in (data or {}).items() if k in names})


@dataclass(frozen=True)
class KernelStatics:
    """Everything Tier 1 knows about one compiled kernel."""

    name: str = ""
    n_regs: int | None = None
    n_spills: int | None = None
    shared_bytes: int | None = None
    num_warps: int | None = None
    sass: SassCounts | None = None
    occupancy: Occupancy | None = None
    notes: list[str] = field(default_factory=list)


def find_cuda_tool(name: str) -> str | None:
    """Locate a CUDA binary: PATH first, then the toolkit directories.

    NOT `shutil.which` alone. Measured on box 2: nvdisasm and cuobjdump are installed under
    /usr/local/cuda/bin/ and absent from the login shell's PATH, so a PATH-only lookup reports a
    working tool as unavailable -- the same class of bug as the opencode PATH failure, where an
    available capability was read as a missing one and a whole signal silently vanished.
    """
    found = shutil.which(name)
    if found:
        return found
    roots: list[str | None] = [os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"),
                               "/usr/local/cuda"]
    try:
        from torch.utils.cpp_extension import CUDA_HOME  # noqa: PLC0415

        roots.append(CUDA_HOME)
    except Exception:  # noqa: BLE001 — torch may be absent where this is only parsed
        pass
    roots += sorted(glob.glob("/usr/local/cuda-*"), reverse=True)
    for root in roots:
        if not root:
            continue
        cand = Path(root) / "bin" / name
        if cand.exists():
            return str(cand)
    return None


def disassemble_cubin(cubin: bytes, timeout_s: float = 120.0) -> str | None:
    """SASS text for a cubin, or None when no disassembler is available or it fails.

    Tries nvdisasm, then falls back to cuobjdump: a fat cubin holding several architectures is not
    directly disassemblable by nvdisasm, and cuobjdump extracts the SASS it holds.
    """
    exe = find_cuda_tool("nvdisasm")
    dump = find_cuda_tool("cuobjdump")
    if exe is None and dump is None:
        return None
    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as f:
            f.write(cubin)
            path = f.name
        if exe is not None:
            out = subprocess.run([exe, "-c", path], capture_output=True, timeout=timeout_s)
            if out.returncode == 0:
                return out.stdout.decode("utf-8", errors="replace")
        if dump is not None:
            out = subprocess.run([dump, "-sass", path], capture_output=True, timeout=timeout_s)
            if out.returncode == 0:
                return out.stdout.decode("utf-8", errors="replace")
        return None
    except Exception:  # noqa: BLE001 — a diagnostic must never fail an evaluation
        return None
    finally:
        if path:
            try:
                Path(path).unlink()
            except OSError:
                pass


def count_sass(sass: str) -> SassCounts:
    """Count instruction families in disassembled SASS.

    Only lines carrying a ';' are counted as instructions: section headers, symbol names and
    branch labels do not, so a kernel named `..._hmma_...` cannot inflate the tensor-core count.
    """
    counts = dict.fromkeys(SASS_PATTERNS, 0)
    total = 0
    for line in sass.splitlines():
        if ";" not in line:
            continue
        total += 1
        for key, pat in SASS_PATTERNS.items():
            if pat.search(line):
                counts[key] += 1
    return SassCounts(instructions=total, **counts)


def compute_occupancy(
    n_regs: int,
    shared_bytes: int,
    num_warps: int,
    *,
    max_threads_per_sm: int,
    regs_per_sm: int,
    shared_per_sm: int,
    max_blocks_per_sm: int = 16,
) -> Occupancy | None:
    """Theoretical occupancy from resource use, plus which resource binds it.

    Pure arithmetic on values the harness already collects, so it works on every backend and needs
    no privileges. Device limits are PARAMETERS rather than constants read from a global, which is
    what lets this be tested against several cards without a GPU present.

    The register figure is the per-thread count multiplied by block size, which is the standard
    approximation; real allocation is granular per warp, so this errs slightly optimistic. That
    direction is deliberate -- an occupancy estimate that is too pessimistic would have the agent
    chasing a limit that is not there.
    """
    warp_size = 32
    if num_warps <= 0 or max_threads_per_sm <= 0:
        return None
    max_warps_per_sm = max_threads_per_sm // warp_size
    if max_warps_per_sm <= 0:
        return None

    threads_per_block = num_warps * warp_size
    regs_per_block = max(1, n_regs) * threads_per_block
    by_regs = regs_per_sm // regs_per_block if regs_per_block > 0 else max_blocks_per_sm
    by_shared = (shared_per_sm // shared_bytes) if shared_bytes > 0 else max_blocks_per_sm
    by_warps = max_warps_per_sm // num_warps

    blocks = max(0, min(by_regs, by_shared, by_warps, max_blocks_per_sm))
    active_warps = blocks * num_warps
    occ = active_warps / max_warps_per_sm

    # Which constraint actually binds. Ties resolve toward the resource an agent can change:
    # registers and shared memory are tunable through the tile shape, whereas the per-SM block
    # cap is not, so naming the latter when a tunable one is equally tight would be useless.
    binding = min(by_regs, by_shared, by_warps, max_blocks_per_sm)
    if by_regs == binding and by_regs < max_blocks_per_sm:
        limiter = "registers"
    elif by_shared == binding and by_shared < max_blocks_per_sm:
        limiter = "shared_memory"
    elif by_warps == binding:
        limiter = "warps_per_block"
    else:
        limiter = "blocks_per_sm"

    return Occupancy(occupancy=round(occ, 4), active_warps=active_warps,
                     max_warps_per_sm=max_warps_per_sm, blocks_per_sm=blocks,
                     limiter=limiter, by_regs=by_regs, by_shared=by_shared, by_warps=by_warps)


def statics_from_worker(result: dict) -> list[KernelStatics]:
    """Rebuild Tier 1 statics from a raw worker result. Pure, so a recorded result replays."""
    out = []
    for k in (result.get("kernel_statics") or []):
        sass = SassCounts.model_validate(k["sass"]) if k.get("sass") else None
        occ = Occupancy.model_validate(k["occupancy"]) if k.get("occupancy") else None
        out.append(KernelStatics(
            name=k.get("name") or "", n_regs=k.get("n_regs"), n_spills=k.get("n_spills"),
            shared_bytes=k.get("shared_bytes"), num_warps=k.get("num_warps"),
            sass=sass, occupancy=occ, notes=list(k.get("notes") or []),
        ))
    return out


# What Tier 1 cannot see. Reported to the analyst verbatim: an agent told "nothing is wrong"
# reasons differently from one told "this is not measurable on this box", and the second is the
# true statement. KernelPro's data also shows raw counter dumps DEGRADE LLM performance
# (NoFeedback beat raw ncu, p=0.0007), so naming the gap is better than filling it with noise.
UNMEASURABLE_ON_THIS_TIER = [
    "shared-memory bank conflicts",
    "warp divergence / branch efficiency",
    "L2 and L1 hit rates",
    "stall reason breakdown (memory throttle, barrier wait, instruction fetch)",
    "achieved (as opposed to theoretical) occupancy",
]


def unmeasurable_note() -> str:
    return ("Not measurable on this box (hardware counters need a host-side kernel-module "
            "permission that cannot be set from inside a container): "
            + ", ".join(UNMEASURABLE_ON_THIS_TIER)
            + ". Do not infer these are fine -- they are unknown.")
