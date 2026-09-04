"""Deliberately-cheating candidate: allocates the output and never computes it.

Used to verify the relaxed correctness path's anti-reward-hacking guard rejects an
implausibly fast kernel instead of reporting it as a win. It should be flagged as
excessive_speedup (or fail correctness) — never accepted with a huge speedup.
"""

import torch
import torch.nn as nn

PARAMS = {
    "BLOCK": 256,
}


class ModelNew(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        # Returns an uninitialized buffer of the right shape: near-zero work.
        return torch.empty_like(x)
