# -*- coding: utf-8 -*-
"""Deterministic mock trainer: skill monotonically approaches 1.

The "deterministic trainable mock" lets you verify the train -> improve loop without
a real model or data. Each train_step memorizes an (x, y) pair AND nudges the lm_head,
so the top-1 accuracy on the (subset) test set climbs monotonically to 1.0.
"""

from __future__ import annotations

from loopinfer.core.sampler import argmax


class MockTrainer:
    def __init__(self, model, lr: float = 0.5):
        self.model = model
        self.lr = lr
        self.steps = 0
        self.mem = {}

    def train_step(self, x: int, y: int):
        # soft-update the lm_head toward x's embedding (keeps the model "in the loop")
        ex = self.model.embedding.vector(x)
        head = self.model.lm_head
        for j in range(len(ex)):
            head[y][j] += self.lr * ex[j]
        self.mem[x] = y
        self.steps += 1

    def skill(self, pairs):
        if not pairs:
            return 0.0
        return sum(1 for x, y in pairs if self.mem.get(x) == y) / len(pairs)

    def train(self, pairs, epochs: int = 1):
        for _ in range(epochs):
            for x, y in pairs:
                self.train_step(x, y)
        return self
