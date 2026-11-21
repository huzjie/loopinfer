# -*- coding: utf-8 -*-
"""loopinfer.quant public API."""

from .config import QuantConfig
from .codec import encode_float, decode_float
from .ops import quantize, dequantize, roundtrip_error, compression_ratio

__all__ = ["QuantConfig", "encode_float", "decode_float", "quantize", "dequantize",
           "roundtrip_error", "compression_ratio"]
