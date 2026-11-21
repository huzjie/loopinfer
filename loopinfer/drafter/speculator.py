# -*- coding: utf-8 -*-
"""Speculative decoder: draft K tokens fast, then verify in one model pass."""

from __future__ import annotations

from loopinfer.core.state import Random
from loopinfer.core.sampler import GreedySampler
from loopinfer.drafter.config import DrafterConfig
from loopinfer.drafter.model import DraftModel
from loopinfer.drafter.verify import verify


class SpeculativeDecoder:
    def __init__(self, model, config: DrafterConfig = None, rng: Random = None):
        self.model = model
        self.config = config or DrafterConfig()
        rng = rng or Random(23)
        self.drafter = DraftModel(model.config.vocab_size, model.config.dim, rng)
        self.metrics = {"steps": 0, "accepted": 0, "drafted": 0, "speedup": 1.0}

    def decode_step(self, first_token_id: int, rng: Random = None):
        """One speculative step. Returns (accepted_count, draft_tokens)."""
        rng = rng or Random(29)
        sampler = GreedySampler(rng=rng)
        draft = []
        emb = self.model.embedding.vector(first_token_id)
        for _ in range(self.config.K):
            t = self.drafter.sample(emb, sampler)
            draft.append(t)
            emb = self.drafter.embed(t)

        log = {}
        accepted, _corrected = verify(self.model, first_token_id, draft, log)

        # actual speedup vs autoregressive: (accepted + 1) tokens for (model_steps) model passes
        model_steps = log["model_steps"]
        actual = (accepted + 1) / max(model_steps, 1)
        self.metrics["steps"] += 1
        self.metrics["accepted"] += accepted
        self.metrics["drafted"] += len(draft)
        self.metrics["speedup"] = actual
        return accepted, draft

    def expected_speedup(self):
        return self.config.expected_speedup()

    def acceptance_rate(self):
        if self.metrics["steps"] == 0:
            return 0.0
        return self.metrics["accepted"] / self.metrics["drafted"]
