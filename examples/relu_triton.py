import torch
import torch.nn as nn
import triton
import triton.language as tl

# ---- TUNABLE PARAMETERS (harness-materialized; literals only) ----
PARAMS = {
    "BLOCK": 1024,
    "NUM_WARPS": 4,
}
# ------------------------------------------------------------------


@triton.jit
def _relu_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, tl.maximum(x, 0.0), mask=mask)


class ModelNew(nn.Module):
    """Triton ReLU for KernelBench level1/19_ReLU."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        out = torch.empty_like(x)
        n = x.numel()
        grid = (triton.cdiv(n, PARAMS["BLOCK"]),)
        _relu_kernel[grid](x, out, n, BLOCK=PARAMS["BLOCK"],
                           num_warps=PARAMS["NUM_WARPS"])
        return out
