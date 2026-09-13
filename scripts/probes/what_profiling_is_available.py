"""What profiling routes are actually AVAILABLE on this box? Facts only, no recommendations.

WHY THIS RUNS ON THE BOX INSTEAD OF READING DOCS. The claim to be tested is not "does ncu exist"
but "can THIS container read hardware performance counters". That is decided by a kernel-module
parameter on the HOST plus the container's capabilities, so a document cannot answer it and a
`--version` check cannot either -- ncu is present and still refused. Every line below is a
measurement of this machine.

Prints one `KEY=VALUE` per line so the caller can read it without parsing prose.
"""
import os
import shutil
import subprocess
import sys


def run(cmd, timeout=120):
    """(rc, stdout+stderr). Bytes decoded explicitly: text=True uses the locale codec."""
    try:
        p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=timeout)
        return p.returncode, p.stdout.decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        return -9, "TIMEOUT"
    except Exception as exc:  # noqa: BLE001
        return -1, f"{type(exc).__name__}: {exc}"


print("=== 1. the driver's profiling restriction (the thing that causes ERR_NVGPUCTRPERM) ===")
# This file is the authoritative statement of whether counters are admin-only on this HOST.
try:
    with open("/proc/driver/nvidia/params", encoding="utf-8") as fh:
        for line in fh:
            if "Profil" in line or "profil" in line:
                print("PARAM:", line.strip())
except Exception as exc:  # noqa: BLE001
    print(f"PARAMS_UNREADABLE={type(exc).__name__}: {exc}")

print()
print("=== 2. are we root, and do we have the capability a counter read needs? ===")
print(f"EUID={os.geteuid()}")
rc, out = run("cat /proc/self/status | grep -E '^Cap(Eff|Prm)'")
print(out.strip() or f"rc={rc}")
# CAP_SYS_ADMIN is bit 21. A container usually drops it; without it, even root cannot lift the gate.
rc, out = run("capsh --print 2>/dev/null | grep -i 'sys_admin' || echo 'capsh unavailable or no sys_admin'")
print("SYS_ADMIN:", out.strip())

print()
print("=== 3. which tools are present at all ===")
for tool in ("ncu", "nsys", "nvidia-smi", "dcgmi", "compute-sanitizer", "cuobjdump", "nvdisasm"):
    path = shutil.which(tool)
    if not path:
        # The CUDA install may not be on PATH; look where it normally lands.
        for root in ("/usr/local/cuda/bin", "/usr/local/cuda/nsight-compute", "/opt/nvidia"):
            rc, out = run(f"find {root} -maxdepth 3 -name {tool} -type f 2>/dev/null | head -1")
            if out.strip():
                path = out.strip()
                break
    print(f"TOOL_{tool}={path or 'ABSENT'}")

print()
print("=== 4. THE DECISIVE TEST: does a counter read actually succeed? ===")
# A real kernel, then ncu asking for ONE dram metric. This is what would give per-candidate
# bandwidth. Anything other than a number here means the route is closed on this box.
probe = "/root/probe-clean/ncu_target.py"
os.makedirs("/root/probe-clean", exist_ok=True)
with open(probe, "w", encoding="utf-8") as fh:
    fh.write(
        "import torch\n"
        "a = torch.randn(4096, 4096, device='cuda')\n"
        "b = torch.randn(4096, 4096, device='cuda')\n"
        "for _ in range(3):\n"
        "    c = a @ b\n"
        "torch.cuda.synchronize()\n"
        "print('target ok', float(c.sum()))\n")

py = sys.executable
ncu = None
for cand in (shutil.which("ncu"), "/usr/local/cuda/bin/ncu"):
    if cand and os.path.exists(cand):
        ncu = cand
        break
if ncu is None:
    rc, out = run("find /usr/local /opt -maxdepth 4 -name ncu -type f 2>/dev/null | head -1")
    ncu = out.strip() or None

if ncu:
    # dram__bytes.sum is exactly the quantity we lack: bytes actually moved to/from DRAM.
    cmd = (f"{ncu} --metrics dram__bytes.sum --target-processes all --csv "
           f"--print-summary per-kernel {py} {probe}")
    rc, out = run(cmd, timeout=420)
    print(f"NCU_RC={rc}")
    print("NCU_OUT_HEAD:")
    for line in out.splitlines()[:40]:
        print("   ", line)
    print(f"NCU_HAS_ERR_NVGPUCTRPERM={'ERR_NVGPUCTRPERM' in out}")
    print(f"NCU_HAS_PERMISSION_WORD={'ermission' in out}")
    print(f"NCU_HAS_DRAM_NUMBER={'dram__bytes' in out}")
else:
    print("NCU_RC=absent")

print()
print("=== 5. CUPTI: the library ncu and nsys are built on ===")
# Two different CUPTI capabilities, and only ONE of them needs counter permission:
#   Activity API  -> kernel names/durations/memcpy sizes. No counters.
#   Profiling API -> hardware metrics like dram__bytes. Gated.
rc, out = run("find / -maxdepth 6 -name 'libcupti*' 2>/dev/null | head -5")
print("CUPTI_LIBS:")
print(out.strip() or "   none found")

print()
print("=== 6. does torch's own profiler get kernel-level data without counters? ===")
# If this works it proves the Activity path is open even when the Profiling path is shut, which
# is the distinction that decides what is recoverable.
tprobe = "/root/probe-clean/torch_prof.py"
with open(tprobe, "w", encoding="utf-8") as fh:
    fh.write(
        "import torch\n"
        "from torch.profiler import profile, ProfilerActivity\n"
        "a = torch.randn(2048, 2048, device='cuda')\n"
        "b = torch.randn(2048, 2048, device='cuda')\n"
        "torch.cuda.synchronize()\n"
        "with profile(activities=[ProfilerActivity.CUDA]) as prof:\n"
        "    c = a @ b\n"
        "    torch.cuda.synchronize()\n"
        "evs = [e for e in prof.key_averages() if e.device_time_total]\n"
        "print('CUDA_EVENTS', len(evs))\n"
        "for e in evs[:5]:\n"
        "    print('   ', e.key[:60], e.device_time_total)\n")
rc, out = run(f"{py} {tprobe}", timeout=420)
print(f"TORCH_PROF_RC={rc}")
for line in out.splitlines()[-12:]:
    print("   ", line)

print()
print("=== 7. device-level DRAM utilisation sampling (no counters needed) ===")
# nvidia-smi's utilization.memory is a SAMPLED percentage of time the memory interface was busy,
# not bytes. It needs no counter permission. Whether it is usable per candidate depends on
# whether it moves at all under load, which is what this measures.
rc, out = run("nvidia-smi --query-gpu=utilization.gpu,utilization.memory,clocks.sm,clocks.mem "
              "--format=csv,noheader -l 1 -c 3")
print(f"SMI_RC={rc}")
print(out.strip())

print()
print("=== 8. what the Triton compiler already hands us per kernel ===")
# The point of this check: if the IR carries load/store widths, per-candidate LOGICAL bytes are
# derivable at compile time with no runtime tool at all -- a different route from counters.
try:
    import triton
    print(f"TRITON_VERSION={triton.__version__}")
    try:
        import triton.profiler as proton  # noqa: F401
        print("TRITON_PROTON=present")
    except Exception as exc:  # noqa: BLE001
        print(f"TRITON_PROTON=absent ({type(exc).__name__})")
except Exception as exc:  # noqa: BLE001
    print(f"TRITON=absent ({type(exc).__name__})")
