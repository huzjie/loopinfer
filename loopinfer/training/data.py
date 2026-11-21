# -*- coding: utf-8 -*-
"""Deterministic synthetic task: predict next token = (x + 1) % vocab.

The test set is a subset of the train set, so a memory-style mock can reach
skill = 1.0 monotonically — proving the train -> improve loop works, end to end.
"""

from __future__ import annotations


class SyntheticData:
    def __init__(self, vocab_size: int = 64, n_train: int = 48, n_test: int = 32):
        self.vocab_size = vocab_size
        all_pairs = [(i, (i + 1) % vocab_size) for i in range(vocab_size)]
        self.train = all_pairs[:n_train]
        self.test = all_pairs[:n_test]  # subset of train
