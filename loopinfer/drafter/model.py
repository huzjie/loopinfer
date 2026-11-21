# -*- coding: utf-8 -*-
"""A tiny linear draft model: fast, cheap, deterministic."""

from __future__ import annotations

from loopinfer.core.state import Random
from loopinfer.model.embedding import Embedding


class DraftModel:
    def __init__(self, vocab_size: int, dim: int, rng: Random = None):
        self.vocab_size = vocab_size
        self.dim = dim
        rng = rng or Random(17)
        self.embedding = Embedding(vocab_size, dim, 1024, rng)
        self.head = [[rng.gauss(0.0, 0.02) for _ in range(dim)] for _ in range(vocab_size)]

    def embed(self, token_id: int):
        return self.embedding.vector(token_id)

    def logits(self, emb):
        return [sum(w[j] * emb[j] for j in range(self.dim)) for w in self.head]

    def sample(self, emb, sampler):
        return sampler.sample(self.logits(emb))
