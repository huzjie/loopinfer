# -*- coding: utf-8 -*-
"""MoE configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MoEConfig:
    n_experts: int = 256
    top_k: int = 8
    dim: int = 64
    hidden_ffn: int = 128
    load_bal_coef: float = 0.01
    shared_expert: bool = True

    def validate(self):
        if not (1 <= self.top_k <= self.n_experts):
            raise ValueError(f"top_k must be in [1, n_experts]: top_k={self.top_k} experts={self.n_experts}")

    def to_dict(self) -> dict:
        return {
            "n_experts": self.n_experts,
            "top_k": self.top_k,
            "dim": self.dim,
            "hidden_ffn": self.hidden_ffn,
            "load_bal_coef": self.load_bal_coef,
            "shared_expert": self.shared_expert,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MoEConfig":
        cfg = cls(
            n_experts=int(d.get("n_experts", 256)),
            top_k=int(d.get("top_k", 8)),
            dim=int(d.get("dim", 64)),
            hidden_ffn=int(d.get("hidden_ffn", 128)),
            load_bal_coef=float(d.get("load_bal_coef", 0.01)),
            shared_expert=bool(d.get("shared_expert", True)),
        )
        cfg.validate()
        return cfg
