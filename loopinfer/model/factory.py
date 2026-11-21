# -*- coding: utf-8 -*-
"""Model factory: build a LoopedTransformer from a config dict or file."""

from __future__ import annotations

from loopinfer.model.config import ModelConfig
from loopinfer.model.transformer import LoopedTransformer


def build_model(config=None, **overrides):
    """Build from a ModelConfig, a dict, or defaults (with optional overrides)."""
    if config is None:
        cfg = ModelConfig()
    elif isinstance(config, ModelConfig):
        cfg = config
    elif isinstance(config, dict):
        cfg = ModelConfig.from_dict(config)
    else:
        raise TypeError(f"unsupported config type: {type(config)}")
    if overrides:
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        cfg.validate()
    return LoopedTransformer(cfg)


def list_models():
    return ["loopinfer"]
