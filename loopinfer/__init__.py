# -*- coding: utf-8 -*-
"""LoopInfer — looped transformer (looped MoE) inference runtime."""

__version__ = "1.0.0"

from .model import LoopedTransformer, build_model, ModelConfig
from .engine import InferenceEngine, EngineConfig
from .loop import LoopConfig, LoopScheduler, CrossPassMemory
from .moe import MoEConfig, MoELayer
from .drafter import DrafterConfig, SpeculativeDecoder
from .quant import QuantConfig
from .tokenizer import encode, decode

__all__ = [
    "__version__",
    "LoopedTransformer", "build_model", "ModelConfig",
    "InferenceEngine", "EngineConfig",
    "LoopConfig", "LoopScheduler", "CrossPassMemory",
    "MoEConfig", "MoELayer",
    "DrafterConfig", "SpeculativeDecoder",
    "QuantConfig",
    "encode", "decode",
]
