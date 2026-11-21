# -*- coding: utf-8 -*-
"""loopinfer.core public API."""

from .errors import LoopInferError, ConfigError, ShapeError, QuantError, DraftError, TokenizerError
from .events import EventBus, on, emit, clear
from .state import Random
from .tokens import Token, GenerationResult
from .metrics import Metrics
from .logger import Logger, get_logger
from .sampler import (
    softmax, argmax, get_sampler, list_samplers, register_sampler,
    GreedySampler, TemperatureSampler, TopKSampler, NucleusSampler,
)
from .registry import Registry, _MODELS, _SAMPLERS, _DRAFTERS
from .config import load_config, dump_config, ConfigError as ConfigLoadError

__all__ = [
    "LoopInferError", "ConfigError", "ShapeError", "QuantError", "DraftError", "TokenizerError",
    "EventBus", "on", "emit", "clear",
    "Random",
    "Token", "GenerationResult",
    "Metrics",
    "Logger", "get_logger",
    "softmax", "argmax", "get_sampler", "list_samplers", "register_sampler",
    "GreedySampler", "TemperatureSampler", "TopKSampler", "NucleusSampler",
    "Registry", "_MODELS", "_SAMPLERS", "_DRAFTERS",
    "load_config", "dump_config",
]
