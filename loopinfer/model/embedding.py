# -*- coding: utf-8 -*-
"""Token embedding + sinusoidal positional encoding."""

from __future__ import annotations

import math

from loopinfer.core.state import Random


class Embedding:
    def __init__(self, vocab_size: int, dim: int, max_seq_len: int, rng: Random = None):
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        rng = rng or Random(1)
        self.table = [[rng.gauss(0.0, 0.02) for _ in range(dim)] for _ in range(vocab_size)]
        self.pos = self._sinusoidal(max_seq_len, dim)

    @staticmethod
    def _sinusoidal(max_len: int, dim: int):
        enc = []
        for pos in range(max_len):
            row = [0.0] * dim
            for i in range(dim):
                if i % 2 == 0:
                    row[i] = math.sin(pos / (10000 ** (i / dim)))
                else:
                    row[i] = math.cos(pos / (10000 ** ((i - 1) / dim)))
            enc.append(row)
        return enc

    def forward(self, token_id: int, pos: int):
        tok = self.table[token_id]
        p = self.pos[min(pos, self.max_seq_len - 1)]
        return [t + pp for t, pp in zip(tok, p)]

    def vector(self, token_id: int):
        return list(self.table[token_id])
