# -*- coding: utf-8 -*-
"""Token-level data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Token:
    id: int
    text: str = ""
    logprob: float = 0.0

    def __repr__(self):
        return f"Token({self.id}, {self.text!r})"


@dataclass
class GenerationResult:
    tokens: List[int] = field(default_factory=list)
    text: str = ""
    logprobs: List[float] = field(default_factory=list)
    loop_passes: int = 1
    eff_depth: int = 0

    def __repr__(self):
        return f"GenerationResult(tokens={len(self.tokens)}, text={self.text[:40]!r})"
