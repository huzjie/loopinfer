# -*- coding: utf-8 -*-
"""Attention configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AttentionConfig:
    dim: int = 64
    n_heads: int = 4
    head_dim: int = 16
    window_size: int = 512       # sliding-window KV cache; set high for long context

    def validate(self):
        if self.dim != self.n_heads * self.head_dim:
            raise ValueError(f"dim ({self.dim}) != n_heads*head_dim ({self.n_heads * self.head_dim})")

    def to_dict(self) -> dict:
        return {"dim": self.dim, "n_heads": self.n_heads, "head_dim": self.head_dim, "window_size": self.window_size}

    @classmethod
    def from_dict(cls, d: dict) -> "AttentionConfig":
        cfg = cls(
            dim=int(d.get("dim", 64)),
            n_heads=int(d.get("n_heads", 4)),
            head_dim=int(d.get("head_dim", 16)),
            window_size=int(d.get("window_size", 512)),
        )
        cfg.validate()
        return cfg
