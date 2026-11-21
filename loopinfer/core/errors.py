# -*- coding: utf-8 -*-
"""Exception hierarchy for loopinfer."""


class LoopInferError(Exception):
    """Base error for the runtime."""


class ConfigError(LoopInferError):
    """Invalid configuration."""


class ShapeError(LoopInferError):
    """Dimension mismatch."""


class QuantError(LoopInferError):
    """Quantization / dequantization failure."""


class DraftError(LoopInferError):
    """Speculative decoding failure."""


class TokenizerError(LoopInferError):
    """Tokenization failure."""
