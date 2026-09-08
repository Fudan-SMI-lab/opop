"""KernelBench wrapper around the external CUDA candidate for L3:43.

Purpose: measure the external kernel with OUR pipeline -- same reference, same input factory,
same 100-sample CUDA-event timing, same exclusive GPU lock -- so the comparison against our
3.0126 ms Triton result is one number measured one way, not two numbers from two harnesses.

The .cu file is used byte-for-byte (sha256 checked below); only this Python shim is new. It
mirrors the reference `Model.__init__` so the same weights are constructed and the same
`get_inputs()` feeds both.
"""

import hashlib
import pathlib

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

PARAMS = {"UNUSED": 0}  # the external kernel has no PARAMS contract; nothing is tuned here

# NOT `pathlib.Path(__file__)`: the harness loads a candidate with `exec` on the source text,
# so `__file__` is undefined and the first attempt died with a compile_error that looked like a
# defect in the external kernel. An env var with a literal fallback keeps it runnable both ways.
import os
_SRC = pathlib.Path(os.environ.get("EXT_KERNEL_CU")
                    or "/root/autodl-tmp/ext-eval/L3_43/best_kernel.cu")
_EXPECTED_SHA = "e9f5821bf3a50a7250dcaef4507a7b72149efc74031b14b54540b0eeff0e2633"
_actual = hashlib.sha256(_SRC.read_bytes()).hexdigest()
assert _actual == _EXPECTED_SHA, (
    f"best_kernel.cu is not the file the manifest describes\n"
    f"  expected {_EXPECTED_SHA}\n  actual   {_actual}")

_ext = load(name="ext_l3_43_cuda", sources=[str(_SRC)],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
            verbose=False)


class ModelNew(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen))
                                          .view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        return _ext.forward(x, self.c_attn.weight, self.c_attn.bias,
                            self.c_proj.weight, self.c_proj.bias,
                            self.bias, self.n_head, self.n_embd, self.training)
