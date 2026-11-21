# -*- coding: utf-8 -*-
"""Layer normalization (pure python)."""

from __future__ import annotations


class LayerNorm:
    def __init__(self, dim: int, eps: float = 1e-5):
        self.dim = dim
        self.eps = eps
        self.gamma = [1.0] * dim
        self.beta = [0.0] * dim

    def forward(self, x):
        n = len(x)
        mean = sum(x) / n
        var = sum((v - mean) ** 2 for v in x) / n
        inv = 1.0 / ((var + self.eps) ** 0.5)
        return [g * (v - mean) * inv + b for g, b, v in zip(self.gamma, self.beta, x)]
