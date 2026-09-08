"""KernelBench wrapper around the external CUDA candidate for L3:21 (EfficientNet MBConv).

Same purpose as the L3:43 wrapper: measure the external kernel through OUR pipeline -- same
reference, same input factory, same 100-sample CUDA-event timing, same correctness gate -- so the
comparison against our Triton result is one number measured one way.

The .cu file is used byte-for-byte (sha256 checked below); only this Python shim is new. It mirrors
the reference `Model.__init__` so the same weights are constructed and the same `get_inputs()`
feeds both.

TRAIN-MODE BATCHNORM. The reference is evaluated in train mode, so all three BatchNorms use BATCH
statistics, not the running buffers -- the failure that repeatedly broke our own L3:21 candidates.
The external kernel takes `training` and its own running-stat pointers and computes batch stats
itself when training is true, so this shim only has to forward `self.training` honestly rather than
hardcode a mode.
"""

import hashlib
import os
import pathlib

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

PARAMS = {"UNUSED": 0}  # the external kernel has no PARAMS contract; nothing is tuned here

# NOT `pathlib.Path(__file__)`: the harness loads a candidate with `exec` on the source text, so
# `__file__` is undefined and the first attempt on L3:43 died with a compile_error that looked like
# a defect in the external kernel. An env var with a literal fallback keeps it runnable both ways.
_SRC = pathlib.Path(os.environ.get("EXT_KERNEL_CU")
                    or "/root/autodl-tmp/ext-eval/L3_21/best_kernel.cu")
_EXPECTED_SHA = "393007a36ebb45515008368c5b6e3377c90cbd0ea834e528684ee439dc62f38f"
_actual = hashlib.sha256(_SRC.read_bytes()).hexdigest()
assert _actual == _EXPECTED_SHA, (
    f"best_kernel.cu is not the file the manifest describes\n"
    f"  expected {_EXPECTED_SHA}\n  actual   {_actual}")

_ext = load(name="ext_l3_21_cuda", sources=[str(_SRC)],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"], verbose=False)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio
        self.stride = stride
        # Built as nn.Sequential with the same submodule order as the reference, so the state dict
        # and the RNG draw order match: the external kernel is fed these exact tensors.
        self.expand_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        )
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride,
                      padding=(kernel_size - 1) // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        )
        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        e, eb = self.expand_conv[0], self.expand_conv[1]
        d, db = self.depthwise_conv[0], self.depthwise_conv[1]
        p, pb = self.project_conv[0], self.project_conv[1]
        return _ext.forward(
            x, e.weight, eb.weight, eb.bias, eb.running_mean, eb.running_var,
            d.weight, db.weight, db.bias, db.running_mean, db.running_var,
            p.weight, pb.weight, pb.bias, pb.running_mean, pb.running_var,
            self.stride, self.training, self.use_residual)
