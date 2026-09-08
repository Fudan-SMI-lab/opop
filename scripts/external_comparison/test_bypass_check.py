"""审计报告称 KernelBench 的 STRICT `code_bypass` 检查禁掉任何 try/except/pass,而我们的契约从未
告知 agent。如果成立,这比 cuBLAS 那条影响面大得多 —— 但必须实测,不能靠读正则推断。

三个问题:
  1. 一个合法候选(有真 kernel、有 try/except 做 host 侧回退)是否真的被拒?
  2. 我们**已经接受过**的真实候选里有多少含 try/except/pass?如果有,说明这条门实际没在拦我们
     (可能因为传的 backend/precision 不同,或候选恰好都不写 try)。
  3. 字符串字面量里的 "pass" 会不会误报?(_strip_comments 只剥注释不剥字符串)
"""
import sys, pathlib, glob
sys.path.insert(0, "/root/autodl-tmp/opop-workspace/KernelBench/src")
from kernelbench.kernel_static_checker import validate_kernel_static

MINIMAL = '''
import torch
import torch.nn as nn
import triton
import triton.language as tl

PARAMS = {"BLOCK": 256, "NUM_WARPS": 4}

@triton.jit
def _k(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=m) * 2.0, mask=m)

class ModelNew(nn.Module):
    def forward(self, x):
        y = torch.empty_like(x)
        n = x.numel()
        _k[(triton.cdiv(n, PARAMS["BLOCK"]),)](x, y, n, BLOCK=PARAMS["BLOCK"],
                                               num_warps=PARAMS["NUM_WARPS"])
        return y
'''

VARIANTS = {
    "baseline (no try/pass)": MINIMAL,
    "host-side try/except around a cache": MINIMAL.replace(
        "        y = torch.empty_like(x)",
        "        try:\n            y = self._buf\n        except AttributeError:\n"
        "            y = torch.empty_like(x)"),
    "'pass' in an if-branch": MINIMAL.replace(
        "        y = torch.empty_like(x)",
        "        if x.numel() == 0:\n            pass\n        y = torch.empty_like(x)"),
    "the word pass inside a STRING": MINIMAL.replace(
        'PARAMS = {"BLOCK": 256, "NUM_WARPS": 4}',
        'PARAMS = {"BLOCK": 256, "NUM_WARPS": 4}\n_NOTE = "we pass tiles through shared memory"'),
    "'passed' (word-boundary check)": MINIMAL.replace(
        'PARAMS = {"BLOCK": 256, "NUM_WARPS": 4}',
        'PARAMS = {"BLOCK": 256, "NUM_WARPS": 4}\n_NOTE = "tiles are passed in"'),
    "import threading (unused)": MINIMAL.replace(
        "import torch\n", "import torch\nimport threading\n"),
}

print("### 1+3. 合法写法是否被拒(backend=triton, precision=fp32,与 harness 一致)")
for label, code in VARIANTS.items():
    valid, errors, warnings = validate_kernel_static(code, backend="triton", precision="fp32")
    print(f"  {label:38s} valid={str(valid):5s}  errors={errors}")

print("\n### 2. 我们已接受过的真实候选里有多少含 try/except/pass?")
pats = ("try:", "except", "pass")
roots = ["/root/autodl-tmp/opop-workspace/opop-glm/runs-l3/*/candidates/*/source.py",
         "/root/autodl-tmp/opop-workspace/opop-glm/runs-smoke42/*/candidates/*/source.py"]
files = [f for r in roots for f in glob.glob(r)]
print(f"  scanning {len(files)} candidate sources on this box")
hits = {p: 0 for p in pats}
rejected = 0
for f in files:
    src = pathlib.Path(f).read_text(encoding="utf-8", errors="replace")
    for p in pats:
        if p in src:
            hits[p] += 1
    valid, errors, _ = validate_kernel_static(src, backend="triton", precision="fp32")
    if not valid:
        rejected += 1
print(f"  contains 'try:'  : {hits['try:']}")
print(f"  contains 'except': {hits['except']}")
print(f"  contains 'pass'  : {hits['pass']}")
print(f"  would FAIL validate_kernel_static NOW: {rejected}/{len(files)}")
