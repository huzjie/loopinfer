# -*- coding: utf-8 -*-
"""Engine-level configuration aggregating model / drafter / quant sections."""

from __future__ import annotations

from dataclasses import dataclass, field

from loopinfer.model.config import ModelConfig
from loopinfer.drafter.config import DrafterConfig
from loopinfer.quant.config import QuantConfig


@dataclass
class EngineConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    drafter: DrafterConfig = field(default_factory=DrafterConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    sampler: str = "greedy"
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    max_tokens: int = 64

    def to_dict(self) -> dict:
        return {
            "model": self.model.to_dict(),
            "drafter": self.drafter.to_dict(),
            "quant": self.quant.to_dict(),
            "sampler": self.sampler,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EngineConfig":
        m = ModelConfig.from_dict(d.get("model", {}))
        dr = DrafterConfig.from_dict(d.get("drafter", {}))
        q = QuantConfig.from_dict(d.get("quant", {}))
        return cls(
            model=m, drafter=dr, quant=q,
            sampler=str(d.get("sampler", "greedy")),
            temperature=float(d.get("temperature", 1.0)),
            top_k=int(d.get("top_k", 50)),
            top_p=float(d.get("top_p", 0.9)),
            max_tokens=int(d.get("max_tokens", 64)),
        )
