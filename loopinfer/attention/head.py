# -*- coding: utf-8 -*-
"""Single-head scaled dot-product attention over cached K/V."""

from __future__ import annotations

import math


def softmax(values):
    m = max(values) if values else 0.0
    exps = [math.exp(v - m) for v in values]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


def head_attention(q, ks, vs):
    """q: [head_dim]; ks/vs: lists of [head_dim]. Returns [head_dim]."""
    hd = len(q)
    scale = math.sqrt(hd)
    scores = [sum(q[t] * k[t] for t in range(hd)) / scale for k in ks]
    probs = softmax(scores)
    out = [0.0] * hd
    for p, v in zip(probs, vs):
        for t in range(hd):
            out[t] += p * v[t]
    return out
