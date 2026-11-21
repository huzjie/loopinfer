# -*- coding: utf-8 -*-
"""Config loading: file (JSON/YAML) or environment defaults."""

from __future__ import annotations

from pathlib import Path

from loopinfer.core.config import load_config
from loopinfer.engine.config import EngineConfig


def load_engine_config(path=None, **overrides) -> EngineConfig:
    """Load EngineConfig from a JSON/YAML file, falling back to defaults."""
    if path:
        raw = load_config(path)
    else:
        raw = {}
    cfg = EngineConfig.from_dict(raw)
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        elif hasattr(cfg.model, k):
            setattr(cfg.model, k, v)
    return cfg


def default_config_paths():
    return ["config/loopinfer.example.json", "config/loopinfer.yaml"]
