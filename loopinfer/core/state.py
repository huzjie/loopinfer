# -*- coding: utf-8 -*-
"""Deterministic RNG so every run is reproducible."""

from __future__ import annotations


def _splitmix64(seed: int) -> int:
    # SplitMix64: cheap, high-quality integer hash for seeding.
    seed = (seed + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = seed
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    return (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF


class Random:
    """A small, deterministic PRNG (xoshiro256**)."""

    def __init__(self, seed: int = 1337):
        self._s = [0] * 4
        for i in range(4):
            self._s[i] = _splitmix64(seed + i * 0x9E3779B97F4A7C15)

    @staticmethod
    def _rotl(x: int, k: int) -> int:
        return ((x << k) | (x >> (64 - k))) & 0xFFFFFFFFFFFFFFFF

    def next_u64(self) -> int:
        s = self._s
        result = (self._rotl(s[1] * 5, 7) * 9) & 0xFFFFFFFFFFFFFFFF
        t = (s[1] << 17) & 0xFFFFFFFFFFFFFFFF
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= t
        s[3] = self._rotl(s[3], 45)
        return result

    def random(self) -> float:
        # float in [0, 1)
        return self.next_u64() / float(1 << 64)

    def randint(self, lo: int, hi: int) -> int:
        # inclusive [lo, hi]
        if hi <= lo:
            return lo
        span = hi - lo + 1
        return lo + self.next_u64() % span

    def choice(self, seq):
        return seq[self.randint(0, len(seq) - 1)]

    def gauss(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        # Box-Muller
        u1 = max(self.random(), 1e-12)
        u2 = self.random()
        import math
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mu + sigma * z
