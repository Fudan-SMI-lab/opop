import torch
import torch.nn as nn
import triton
import triton.language as tl


PARAMS = {
    "ROW_BLOCK_SIZE": 512,
    "NUM_WARPS": 4,
}


@triton.jit
def _relu_row_persistent_kernel(
    x_ptr,
    output_ptr,
    n_cols,
    ROW_BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    lane_offsets = tl.arange(0, ROW_BLOCK_SIZE)
    row_start = row * n_cols

    for col_start in range(0, n_cols, ROW_BLOCK_SIZE):
        cols = col_start + lane_offsets
        mask = cols < n_cols
        values = tl.load(x_ptr + row_start + cols, mask=mask)
        tl.store(
            output_ptr + row_start + cols,
            tl.maximum(values, 0.0),
            mask=mask,
        )


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(x)
        n_rows = x.shape[0]
        n_cols = x.numel() // n_rows
        _relu_row_persistent_kernel[(n_rows,)](
            x,
            output,
            n_cols,
            ROW_BLOCK_SIZE=PARAMS["ROW_BLOCK_SIZE"],
            num_warps=PARAMS["NUM_WARPS"],
        )
        return output
