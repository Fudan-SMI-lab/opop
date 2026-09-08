"""KernelBench wrapper around the external CUDA candidate for L3:48 (Mamba2 SSD, return Y).

Same purpose as the other two wrappers: measure the external kernel through OUR pipeline -- same
reference, same input factory, same 100-sample CUDA-event timing, same correctness gate.

The .cu file is used byte-for-byte (sha256 checked below); only this Python shim is new.

A, B, C ARE PARAMETERS, NOT INPUTS. The reference builds them in `__init__` as `nn.Parameter`s and
`get_inputs()` returns only X, so they must be constructed here in the same order with the same
shapes for the RNG draw to line up. `initial_states` defaults to None, which the reference treats
as zeros and the kernel accepts as a None py::object.

The manifest claims 99.2x vs eager for this candidate. That is the number under scrutiny here:
the task moves a fixed ~1.35 GB, so 99.2x over eager would require an effective bandwidth far above
this card's measured 0.91 TB/s roof unless the eager baseline is itself pathological. The reference
IS naive -- it materializes the (l,s) decay matrix per block and runs 6-index einsums -- so a large
speedup is plausible; measuring it here settles the size.
"""

import hashlib
import os
import pathlib

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

PARAMS = {"UNUSED": 0}  # the external kernel has no PARAMS contract; nothing is tuned here

# NOT `pathlib.Path(__file__)`: the harness loads a candidate with `exec` on the source text.
_SRC = pathlib.Path(os.environ.get("EXT_KERNEL_CU")
                    or "/root/autodl-tmp/ext-eval/L3_48/best_kernel.cu")
_EXPECTED_SHA = "b07b30cb8aa9736b84bcfab1c55377d4833c10c7f810adeb8b7939cf6a6d76fb"
_actual = hashlib.sha256(_SRC.read_bytes()).hexdigest()
assert _actual == _EXPECTED_SHA, (
    f"best_kernel.cu is not the file the manifest describes\n"
    f"  expected {_EXPECTED_SHA}\n  actual   {_actual}")

_ext = load(name="ext_l3_48_cuda", sources=[str(_SRC)],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"], verbose=False)


class ModelNew(nn.Module):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        super().__init__()
        assert seq_length % block_len == 0
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.block_len = block_len
        # Same order and shapes as the reference's __init__ so the RNG draws match.
        self.A = nn.Parameter(torch.randn(batch_size, seq_length, n_heads))
        self.B = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        self.C = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))

    def forward(self, X, initial_states=None):
        return _ext.forward(X, self.A, self.B, self.C, self.block_len, initial_states)
