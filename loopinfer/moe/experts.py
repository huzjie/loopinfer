# -*- coding: utf-8 -*-
"""Expert FFN: w1/w2 share the shape [hidden_ffn][dim]; out[k] = sum_i w2[i][k] * h[i]."""

from __future__ import annotations

from loopinfer.core.state import Random


def _relu(v):
    return v if v > 0.0 else 0.0


class ExpertFFN:
    def __init__(self, dim: int, hidden_ffn: int, rng: Random = None):
        self.dim = dim
        self.hidden_ffn = hidden_ffn
        rng = rng or Random(3)
        self.w1 = [[rng.gauss(0.0, 0.02) for _ in range(dim)] for _ in range(hidden_ffn)]
        self.w2 = [[rng.gauss(0.0, 0.02) for _ in range(dim)] for _ in range(hidden_ffn)]

    def forward(self, x):
        h = [_relu(sum(self.w1[i][j] * x[j] for j in range(self.dim))) for i in range(self.hidden_ffn)]
        out = [sum(self.w2[i][k] * h[i] for i in range(self.hidden_ffn)) for k in range(self.dim)]
        return out
