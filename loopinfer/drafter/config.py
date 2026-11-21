# -*- coding: utf-8 -*-
"""Speculative decoding configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DrafterConfig:
    K: int = 4                    # how many tokens the drafter guesses per step
    draft_speedup: float = 3.0    # drafter is Nx faster per token than the main model
    enabled: bool = True

    def expected_speedup(self) -> float:
        """Theoretical speedup = K / (1 + K / draft_speedup)."""
        if not self.enabled:
            return 1.0
        return self.K / (1.0 + self.K / self.draft_speedup)

    def to_dict(self) -> dict:
        return {"K": self.K, "draft_speedup": self.draft_speedup, "enabled": self.enabled,
                "expected_speedup": self.expected_speedup()}

    @classmethod
    def from_dict(cls, d: dict) -> "DrafterConfig":
        cfg = cls(K=int(d.get("K", 4)), draft_speedup=float(d.get("draft_speedup", 3.0)),
                  enabled=bool(d.get("enabled", True)))
        return cfg
