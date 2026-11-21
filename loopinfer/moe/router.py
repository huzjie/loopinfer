# -*- coding: utf-8 -*-
"""Top-k sparse routing: the heart of MoE. See README dry-block C."""

from __future__ import annotations

import math


def softmax(values):
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    s = sum(exps)
    return [e / s for e in exps]


def topk_indices(values, k):
    """Indices of the k largest values (stable, deterministic)."""
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    return order[:k]


def gate_logits(x, gate_w):
    """logits[e] = x @ gate_w[:, e]."""
    dim = len(x)
    return [sum(x[j] * gate_w[j][e] for j in range(dim)) for e in range(len(gate_w[0]))]


def route(x, gate_w, n_experts, top_k, experts, load_bal_coef=0.01):
    """Pure-function router mirroring README dry-block C.

    Returns (output, load_balance_loss).
    """
    logits = gate_logits(x, gate_w)
    topk_idx = topk_indices(logits, top_k)
    topk_vals = [logits[e] for e in topk_idx]
    weights = softmax(topk_vals)  # normalize ONLY within top-k

    out = None
    for k, e in enumerate(topk_idx):
        expert_out = experts[e](x)
        w = weights[k]
        out = [w * v for v in expert_out] if out is None else [o + w * v for o, v in zip(out, expert_out)]

    # load-balancing loss: n_experts * sum((frac - 1/n)^2), always >= 0
    counts = [0] * n_experts
    for e in topk_idx:
        counts[e] += 1
    n = n_experts
    lb = 0.0
    for c in counts:
        frac = c / top_k
        lb += (frac - 1.0 / n) ** 2
    lb *= load_bal_coef * n
    return out, lb
