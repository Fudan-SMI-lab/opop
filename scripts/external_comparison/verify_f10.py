"""F10 的判据:显式传 forbidden 后,真实候选的判定一个都不该翻转。

传入的清单就是 KernelBench 当前的 STRICT_CHECKS,所以行为应当完全相同 —— 这个测试证明"pin 住"
不等于"改变"。同时验证 BACKEND_IMPL_CHECK 仍被自动追加(不能因为我们传了 forbidden 就丢掉它:
那会让一个没有 @triton.jit 的文件通过 triton 后端的检查)。
"""
import glob, pathlib, sys
sys.path.insert(0, "/root/autodl-tmp/opop-workspace/KernelBench/src")
from kernelbench.kernel_static_checker import validate_kernel_static

REQUIRED = ["code_bypass", "timing_event_patch", "thread_injection", "lazy_eval"]

roots = ["/root/autodl-tmp/opop-workspace/opop-glm/runs-l3/*/candidates/*/source.py",
         "/root/autodl-tmp/opop-workspace/opop-glm/runs-smoke42/*/candidates/*/source.py"]
files = [f for r in roots for f in glob.glob(r)]
print(f"comparing default vs explicit `forbidden` over {len(files)} real candidates")
flips = 0
for f in files:
    src = pathlib.Path(f).read_text(encoding="utf-8", errors="replace")
    a_valid, a_err, a_warn = validate_kernel_static(src, backend="triton", precision="fp32")
    b_valid, b_err, b_warn = validate_kernel_static(src, backend="triton", precision="fp32",
                                                    forbidden=list(REQUIRED))
    if a_valid != b_valid or sorted(a_err) != sorted(b_err):
        flips += 1
        print(f"  FLIP {f.split('/')[-2]}: {a_valid}->{b_valid}  {a_err} -> {b_err}")
print(f"  verdict flips: {flips}/{len(files)}   (0 expected: we pinned the current default)")

print("\nbackend impl check must still be appended automatically:")
NO_JIT = '''
import torch, torch.nn as nn
PARAMS = {"A": 1, "B": 2}
class ModelNew(nn.Module):
    def forward(self, x):
        return x * 2.0
'''
v, e, w = validate_kernel_static(NO_JIT, backend="triton", precision="fp32",
                                 forbidden=list(REQUIRED))
print(f"  a file with no @triton.jit, backend=triton, explicit forbidden: valid={v}")
print(f"    errors={e}")
assert not v, "REGRESSION: passing `forbidden` dropped the backend implementation check"
print("  -> backend impl check survives an explicit forbidden list")

print("\nand the four we require still fire:")
for label, code in (
    ("try/except", NO_JIT.replace("        return x * 2.0",
                                  "        try:\n            return x * 2.0\n"
                                  "        except Exception:\n            return x")),
    ("threading", NO_JIT.replace("import torch,", "import threading\nimport torch,")),
):
    v, e, _ = validate_kernel_static(code, backend="cuda", precision="fp32",
                                     forbidden=list(REQUIRED))
    hit = [x for x in e if "try" in x.lower() or "thread" in x.lower()]
    print(f"  {label:12s} refused={not v}  matching error={hit}")
