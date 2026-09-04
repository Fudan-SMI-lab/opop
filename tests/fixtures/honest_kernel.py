"""Honest but very fast candidate: a real elementwise ReLU, no cheating.

Verifies the anti-reward-hacking guard does NOT fire on a legitimate kernel — the
guard must reject work-skipping, not merely fast code.
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
        return torch.relu(x)
