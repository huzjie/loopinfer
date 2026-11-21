# -*- coding: utf-8 -*-
"""Dense FFN for non-looped layers."""

from __future__ import annotations

from loopinfer.core.state import Random
from loopinfer.model.activations import gelu


class DenseFFN:
    def __init__(self, dim: int, hidden_ffn: int, rng: Random = None):
        self.dim = dim
        self.hidden_ffn = hidden_ffn
        rng = rng or Random(9)
        self.w1 = [[rng.gauss(0.0, 0.02) for _ in range(dim)] for _ in range(hidden_ffn)]
        self.w2 = [[rng.gauss(0.0, 0.02) for _ in range(dim)] for _ in range(hidden_ffn)]

    def forward(self, x):
        h = [gelu(sum(self.w1[i][j] * x[j] for j in range(self.dim))) for i in range(self.hidden_ffn)]
        return [sum(self.w2[i][k] * h[i] for i in range(self.hidden_ffn)) for k in range(self.dim)]
