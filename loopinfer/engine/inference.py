# -*- coding: utf-8 -*-
"""Inference engine: ties model + drafter + quant + tokenizer together."""

from __future__ import annotations

from loopinfer.core.state import Random
from loopinfer.model.factory import build_model
from loopinfer.drafter.speculator import SpeculativeDecoder
from loopinfer.tokenizer import get_tokenizer
from loopinfer.engine.config import EngineConfig


class InferenceEngine:
    def __init__(self, config: EngineConfig = None, rng: Random = None):
        self.config = config or EngineConfig()
        rng = rng or Random(31)
        self.model = build_model(self.config.model)
        self.decoder = SpeculativeDecoder(self.model, self.config.drafter, rng)
        self.tokenizer = get_tokenizer("byte")

    def generate(self, text: str, max_tokens: int = None, sampler: str = None, **kw):
        max_tokens = max_tokens or self.config.max_tokens
        sampler = sampler or self.config.sampler
        prompt_ids = self.tokenizer.encode(text)
        self.model.reset()
        result = self.model.generate(prompt_ids, max_tokens=max_tokens, sampler=sampler,
                                     temperature=kw.get("temperature", self.config.temperature),
                                     top_k=kw.get("top_k", self.config.top_k),
                                     top_p=kw.get("top_p", self.config.top_p))
        result.text = self.tokenizer.decode(result.tokens)
        return result

    def speculate(self, first_token_id: int, rng: Random = None):
        """One speculative-decoding step (demo / benchmark)."""
        return self.decoder.decode_step(first_token_id, rng)

    def spec_speedup(self):
        return {
            "expected": self.decoder.expected_speedup(),
            "actual": self.decoder.metrics["speedup"],
            "acceptance_rate": self.decoder.acceptance_rate(),
        }

    def describe(self) -> dict:
        return {
            "model": self.config.model.to_dict(),
            "drafter": self.config.drafter.to_dict(),
            "quant": self.config.quant.to_dict(),
            "eff_depth": self.config.model.eff_depth,
            "n_params": self.model.n_params(),
            "spec_speedup": self.spec_speedup(),
        }
