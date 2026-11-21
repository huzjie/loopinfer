# -*- coding: utf-8 -*-
"""Quantization configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QuantConfig:
    format: str = "mxfp8"        # "nvfp4" | "mxfp8"
    block_size: int = 16          # values sharing one scale

    @property
    def exp_bits(self) -> int:
        return 2 if self.format == "nvfp4" else 4

    @property
    def mant_bits(self) -> int:
        return 1 if self.format == "nvfp4" else 3

    @property
    def total_bits(self) -> int:
        return 1 + self.exp_bits + self.mant_bits

    def to_dict(self) -> dict:
        return {"format": self.format, "block_size": self.block_size,
                "exp_bits": self.exp_bits, "mant_bits": self.mant_bits, "total_bits": self.total_bits}

    @classmethod
    def from_dict(cls, d: dict) -> "QuantConfig":
        cfg = cls(format=str(d.get("format", "mxfp8")), block_size=int(d.get("block_size", 16)))
        if cfg.format not in ("nvfp4", "mxfp8"):
            raise ValueError(f"unsupported quant format: {cfg.format}")
        return cfg
