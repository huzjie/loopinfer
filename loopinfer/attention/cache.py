# -*- coding: utf-8 -*-
"""Sliding-window KV cache: keeps only the last `window_size` tokens per head."""

from __future__ import annotations


class KVCache:
    def __init__(self, n_heads: int, window_size: int = 512):
        self.n_heads = n_heads
        self.window_size = window_size
        self.k = [[] for _ in range(n_heads)]
        self.v = [[] for _ in range(n_heads)]

    def append(self, head: int, k, v):
        self.k[head].append(k)
        self.v[head].append(v)
        if len(self.k[head]) > self.window_size:
            self.k[head] = self.k[head][-self.window_size:]
            self.v[head] = self.v[head][-self.window_size:]

    def seq_len(self, head: int = 0) -> int:
        return len(self.k[head])

    def clear(self):
        self.k = [[] for _ in range(self.n_heads)]
        self.v = [[] for _ in range(self.n_heads)]

    def memory_bytes(self) -> int:
        # rough estimate: each float = 8 bytes
        total = 0
        for h in range(self.n_heads):
            total += sum(len(k) for k in self.k[h]) + sum(len(v) for v in self.v[h])
        return total * 8
