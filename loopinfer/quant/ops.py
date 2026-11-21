# -*- coding: utf-8 -*-
"""Block-wise quantize / dequantize with shared per-block scales."""

from __future__ import annotations

from loopinfer.quant.codec import encode_float, decode_float
from loopinfer.quant.config import QuantConfig


def quantize(values, config: QuantConfig):
    """Quantize a flat list of floats into (codes, scales)."""
    E, M = config.exp_bits, config.mant_bits
    bs = config.block_size
    codes = []
    scales = []
    n = len(values)
    for start in range(0, n, bs):
        block = values[start:start + bs]
        scale = max(abs(v) for v in block) or 1.0
        scales.append(scale)
        for v in block:
            codes.append(encode_float(v / scale, E, M))
    return codes, scales


def dequantize(codes, scales, config: QuantConfig):
    """Reconstruct floats from (codes, scales)."""
    E, M = config.exp_bits, config.mant_bits
    bs = config.block_size
    out = []
    idx = 0
    for scale in scales:
        for _ in range(bs):
            if idx >= len(codes):
                break
            out.append(decode_float(codes[idx], E, M) * scale)
            idx += 1
    return out


def roundtrip_error(values, config: QuantConfig):
    """Max abs error between original and quantize->dequantize."""
    codes, scales = quantize(values, config)
    restored = dequantize(codes, scales, config)
    return max(abs(a - b) for a, b in zip(values, restored))


def compression_ratio(original_floats: int, config: QuantConfig):
    """Storage ratio: original 32-bit floats vs quantized bits."""
    bits_original = original_floats * 32
    n_blocks = max(1, (original_floats + config.block_size - 1) // config.block_size)
    bits_quant = original_floats * config.total_bits + n_blocks * 32  # codes + per-block scale
    return bits_original / bits_quant
