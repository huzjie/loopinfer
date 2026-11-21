# -*- coding: utf-8 -*-
"""RoPE utilities (optional): rotary position embedding for future extensions."""

from __future__ import annotations

import math


def rope_rotate(x, pos: int, base: float = 10000.0):
    """Apply rotary embedding to a vector (pairs of dims)."""
    out = [0.0] * len(x)
    d = len(x) // 2
    for i in range(d):
        theta = pos / (base ** (2 * i / len(x)))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i] = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
