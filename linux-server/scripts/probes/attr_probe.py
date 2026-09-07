import torch
p = torch.cuda.get_device_properties(0)
for a in ("max_threads_per_block", "max_regs_per_thread", "regs_per_block"):
    print(f"{a:26} = {getattr(p, a, 'ABSENT')}")
# control: a name that must NOT exist
print(f"{'definitely_not_a_prop':26} = {getattr(p, 'definitely_not_a_prop', 'ABSENT')}")
