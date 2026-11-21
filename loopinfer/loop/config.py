# -*- coding: utf-8 -*-
"""Loop configuration: which layers are shared and how many passes they run."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LoopConfig:
    n_layers: int = 24
    loop_start: int = 6          # first looped layer index (inclusive)
    loop_end: int = 18           # last looped layer index (exclusive)
    loop_passes: int = 3         # how many times the loop block runs
    gate_init: float = 0.0       # initial value of the cross-pass message gate

    @property
    def n_loop_layers(self) -> int:
        return self.loop_end - self.loop_start

    @property
    def eff_depth(self) -> int:
        """Effective depth = prefix + passes*loop + suffix."""
        return self.loop_start + self.loop_passes * self.n_loop_layers + (self.n_layers - self.loop_end)

    def validate(self):
        if not (0 <= self.loop_start < self.loop_end <= self.n_layers):
            raise ValueError(f"invalid loop range: start={self.loop_start} end={self.loop_end} layers={self.n_layers}")
        if self.loop_passes < 1:
            raise ValueError("loop_passes must be >= 1")

    def to_dict(self) -> dict:
        return {
            "n_layers": self.n_layers,
            "loop_start": self.loop_start,
            "loop_end": self.loop_end,
            "loop_passes": self.loop_passes,
            "gate_init": self.gate_init,
            "eff_depth": self.eff_depth,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LoopConfig":
        cfg = cls(
            n_layers=int(d.get("n_layers", 24)),
            loop_start=int(d.get("loop_start", 6)),
            loop_end=int(d.get("loop_end", 18)),
            loop_passes=int(d.get("loop_passes", 3)),
            gate_init=float(d.get("gate_init", 0.0)),
        )
        cfg.validate()
        return cfg
