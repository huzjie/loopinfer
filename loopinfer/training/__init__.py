# -*- coding: utf-8 -*-
"""loopinfer.training public API."""

from .data import SyntheticData
from .trainer import MockTrainer
from .metrics import skill_score

__all__ = ["SyntheticData", "MockTrainer", "skill_score"]
