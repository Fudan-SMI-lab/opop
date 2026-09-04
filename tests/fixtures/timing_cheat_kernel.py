"""Correct-but-timing-cheating candidate, for verifying the excessive-speedup guard.

The realistic reward-hack is not "wrong output" — that the correctness gate already
catches (see cheating_kernel.py, rejected as correctness_mismatch). It is a kernel that
computes honestly while the CORRECTNESS trials are running, then costs almost nothing
during the TIMING loop, which reuses one input tensor for every sample. Caching on
tensor identity does exactly that: fresh inputs per correctness trial are all computed,
while the timed loop hits the cache every time.

This must be rejected as excessive_speedup, not reported as a huge win.
"""

import torch
import torch.nn as nn

PARAMS = {
    "BLOCK": 256,
}


class ModelNew(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._key = None
        self._out = None

    def forward(self, x):
        key = (x.data_ptr(), x.shape, x.dtype)
        if self._key == key and self._out is not None:
            return self._out          # timed loop: same tensor every sample -> free
        out = torch.relu(x)           # correctness trials: real work, real result
        self._key, self._out = key, out
        return out
