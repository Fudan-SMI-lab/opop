"""Measure the device block for a config, and say which values are NOT readable.

The `device:` block is written verbatim into every agent sandbox by `_device_doc()` and is
exposed inside agent-authored constraint expressions, so a wrong value makes the constraint
guard admit configs that then fail to compile with OutOfResources -- and tells the model the
wrong architecture. Three of the six values are measurable; three are not, and this script
says which is which rather than printing six numbers that look equally authoritative.

Run with the WORKER venv's python (the one with torch), not the orchestrator's.
"""
import torch

p = torch.cuda.get_device_properties(0)
sm = f"{p.major}{p.minor}"

print("=== measured from torch.cuda.get_device_properties(0)")
print(f"  name                    : {p.name}")
print(f"  compute capability      : sm_{sm}")
print(f"  total_memory            : {p.total_memory} bytes = {p.total_memory / 2**30:.2f} GiB")
print(f"  vram_gb (floor)         : {p.total_memory // 2**30}")
print(f"  shared_memory_per_block : {p.shared_memory_per_block}")
print(f"  ...per_block_optin      : {p.shared_memory_per_block_optin}")
print(f"  multi_processor_count   : {p.multi_processor_count}")
print(f"  regs_per_multiprocessor : {p.regs_per_multiprocessor}   (PER-SM, not per-thread)")
print(f"  max_threads_per_mp      : {p.max_threads_per_multi_processor}  (per-SM, not per-block)")

print()
print("=== readability, checked rather than assumed (it CHANGED between torch versions)")
# The handoff doc states neither of these is a readable device property. On torch 2.13.0
# `max_threads_per_block` IS one (returns 1024); on torch 2.9 it was not. So read it when
# present and fall back to the architectural constant, rather than hardcoding either way.
mtpb = getattr(p, "max_threads_per_block", None)
if mtpb is None:
    print("  max_threads_per_block   : ABSENT -> use 1024 (CUDA limit, every current arch)")
    mtpb = 1024
else:
    print(f"  max_threads_per_block   : {mtpb}  (READABLE on this torch -- measured, not assumed)")

mrpt = getattr(p, "max_regs_per_thread", None)
if mrpt is None:
    print("  max_regs_per_thread     : ABSENT -> use 255 (ISA limit, all sm_7x/8x/9x)")
    mrpt = 255
else:
    print(f"  max_regs_per_thread     : {mrpt}  (READABLE on this torch)")
print("  Never substitute regs_per_multiprocessor for max_regs_per_thread: it is per-SM")
print("  (65536), 256x larger, and would let the guard admit every register-pressure config.")

# CONTROL: a name that cannot exist must come back ABSENT, proving the getattr probes above
# are real lookups and not a shim that returns a value for anything asked of it.
assert getattr(p, "definitely_not_a_property_xyz", None) is None, \
    "CONTROL BROKEN: device properties answer to any attribute name, so no ABSENT result " \
    "above can be trusted"
print("  control OK: a nonexistent attribute is reported absent, so the checks above are real")

print()
print("=== the yaml block to paste")
print("device:")
print(f"  name: {p.name} (sm_{sm})")
print(f"  vram_gb: {p.total_memory // 2**30}")
print(f"  max_regs_per_thread: {mrpt}")
print(f"  max_shared_bytes_static: {p.shared_memory_per_block}")
print(f"  max_shared_bytes_optin: {p.shared_memory_per_block_optin}")
print(f"  max_threads_per_block: {mtpb}")
