# -*- coding: utf-8 -*-
"""IEEE-style floating-point codec: 1 sign + E exp + M mantissa."""

from __future__ import annotations

import math


def encode_float(x, E, M):
    """Encode a float into (1 sign + E exp + M mantissa) bits."""
    if x == 0.0:
        return 0
    sign = 1 if x < 0 else 0
    ax = abs(x)
    e = int(math.floor(math.log2(ax)))
    bias = 2 ** (E - 1) - 1
    exp_field = e + bias
    max_exp = 2 ** E - 1
    if exp_field < 0:
        exp_field = 0
    elif exp_field > max_exp:
        exp_field = max_exp
    mantissa = ax / (2.0 ** e) - 1.0
    m_field = int(round(mantissa * (2 ** M)))
    if m_field == 2 ** M:
        m_field = 0
        exp_field = min(exp_field + 1, max_exp)
    return (sign << (E + M)) | (exp_field << M) | m_field


def decode_float(code, E, M):
    sign = (code >> (E + M)) & 1
    exp_field = (code >> M) & ((1 << E) - 1)
    m_field = code & ((1 << M) - 1)
    bias = 2 ** (E - 1) - 1
    e = exp_field - bias
    value = (1.0 + m_field / (2.0 ** M)) * (2.0 ** e)
    return -value if sign else value
