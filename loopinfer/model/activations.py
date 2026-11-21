# -*- coding: utf-8 -*-
"""Activation functions (pure python)."""

from __future__ import annotations

import math


def relu(x):
    return x if x > 0.0 else 0.0


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def gelu(x):
    # tanh approximation
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def silu(x):
    return x * sigmoid(x)
