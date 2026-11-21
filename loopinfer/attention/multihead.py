# -*- coding: utf-8 -*-
"""Multi-head attention with sliding-window KV cache."""

from __future__ import annotations

from loopinfer.core.state import Random
from loopinfer.attention.config import AttentionConfig
from loopinfer.attention.cache import KVCache
from loopinfer.attention.head import head_attention


def _matvec(w, x):
    return [sum(wi * xi for wi, xi in zip(row, x)) for row in w]


class MultiHeadAttention:
    def __init__(self, config: AttentionConfig, rng: Random = None):
        self.config = config
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        rng = rng or Random(5)
        self.wq = [[rng.gauss(0.0, 0.02) for _ in range(self.dim)] for _ in range(self.dim)]
        self.wk = [[rng.gauss(0.0, 0.02) for _ in range(self.dim)] for _ in range(self.dim)]
        self.wv = [[rng.gauss(0.0, 0.02) for _ in range(self.dim)] for _ in range(self.dim)]
        self.wo = [[rng.gauss(0.0, 0.02) for _ in range(self.dim)] for _ in range(self.dim)]
        self.cache = KVCache(self.n_heads, config.window_size)

    def _split(self, vec):
        return [vec[h * self.head_dim:(h + 1) * self.head_dim] for h in range(self.n_heads)]

    def forward(self, x):
        q_full = _matvec(self.wq, x)
        k_full = _matvec(self.wk, x)
        v_full = _matvec(self.wv, x)
        qs = self._split(q_full)
        ks = self._split(k_full)
        vs = self._split(v_full)

        concat = []
        for h in range(self.n_heads):
            self.cache.append(h, ks[h], vs[h])
            out_h = head_attention(qs[h], self.cache.k[h], self.cache.v[h])
            concat.extend(out_h)

        return _matvec(self.wo, concat)

    def reset_cache(self):
        self.cache.clear()
