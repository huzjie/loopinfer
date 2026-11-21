# -*- coding: utf-8 -*-
"""Model configuration: everything needed to build the transformer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    # Reference scale (runs in milliseconds on CPU). Official Inkling 2 Mini values
    # (26.4B total / 256 experts / 1M context) are documented in README, not here.
    dim: int = 32
    n_layers: int = 24
    vocab_size: int = 256
    n_heads: int = 4
    head_dim: int = 8
    hidden_ffn: int = 64
    max_seq_len: int = 2048
    # loop block
    loop_start: int = 6
    loop_end: int = 18
    loop_passes: int = 3
    gate_init: float = 0.0
    # moe
    n_experts: int = 32
    top_k: int = 8
    load_bal_coef: float = 0.01
    use_moe_loop: bool = True
    window_size: int = 512

    @property
    def eff_depth(self) -> int:
        return self.loop_start + self.loop_passes * (self.loop_end - self.loop_start) + (self.n_layers - self.loop_end)

    def validate(self):
        if not (0 <= self.loop_start < self.loop_end <= self.n_layers):
            raise ValueError("invalid loop range")
        if self.n_heads * self.head_dim != self.dim:
            raise ValueError(f"n_heads*head_dim != dim ({self.n_heads}*{self.head_dim} != {self.dim})")
        if not (1 <= self.top_k <= self.n_experts):
            raise ValueError("top_k out of range")

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        known = set(cls.__dataclass_fields__.keys())
        cfg = cls(**{k: d[k] for k in known if k in d})
        cfg.validate()
        return cfg
