# -*- coding: utf-8 -*-
"""The full looped transformer: embedding -> loop scheduler -> blocks -> lm head."""

from __future__ import annotations

from loopinfer.core.state import Random
from loopinfer.core.sampler import get_sampler
from loopinfer.model.config import ModelConfig
from loopinfer.model.embedding import Embedding
from loopinfer.model.block import TransformerBlock
from loopinfer.attention.config import AttentionConfig
from loopinfer.moe.config import MoEConfig
from loopinfer.loop.config import LoopConfig
from loopinfer.loop.scheduler import LoopScheduler


class LoopedTransformer:
    def __init__(self, config: ModelConfig, rng: Random = None):
        self.config = config
        rng = rng or Random(21)
        self.embedding = Embedding(config.vocab_size, config.dim, config.max_seq_len, rng)
        attn_cfg = AttentionConfig(dim=config.dim, n_heads=config.n_heads, head_dim=config.head_dim,
                                   window_size=config.window_size)
        moe_cfg = MoEConfig(n_experts=config.n_experts, top_k=config.top_k, dim=config.dim,
                            hidden_ffn=config.hidden_ffn, load_bal_coef=config.load_bal_coef)
        self.blocks = []
        for idx in range(config.n_layers):
            use_moe = config.use_moe_loop and (config.loop_start <= idx < config.loop_end)
            self.blocks.append(TransformerBlock(config.dim, attn_cfg, moe_cfg, use_moe, rng))

        loop_cfg = LoopConfig(n_layers=config.n_layers, loop_start=config.loop_start,
                              loop_end=config.loop_end, loop_passes=config.loop_passes,
                              gate_init=config.gate_init)
        self.loop = LoopScheduler(loop_cfg, config.dim, rng)

        self.lm_head = [[rng.gauss(0.0, 0.02) for _ in range(config.dim)] for _ in range(config.vocab_size)]
        self._pos = 0

    def _head(self, x):
        return [sum(w[j] * x[j] for j in range(self.config.dim)) for w in self.lm_head]

    def forward_token(self, token_id: int):
        """Run one token through the looped stack, return logits and the final hidden state."""
        x = self.embedding.forward(token_id, self._pos)
        x, _snap = self.loop.forward(x, self.blocks, return_summary=True)
        self._pos += 1
        logits = self._head(x)
        return logits, x

    def forward_prompt(self, token_ids):
        """Feed a whole prompt; returns logits for the last position."""
        logits = None
        for t in token_ids:
            logits, _ = self.forward_token(t)
        return logits

    def generate(self, prompt_ids, max_tokens: int = 32, sampler: str = "greedy", **sampler_kwargs):
        """Autoregressive generation."""
        from loopinfer.core.tokens import GenerationResult
        s = get_sampler(sampler, rng=Random(99), **sampler_kwargs)
        out_ids = []
        for t in prompt_ids:
            self.forward_token(t)
        last = prompt_ids[-1] if prompt_ids else 0
        for _ in range(max_tokens):
            logits, _ = self.forward_token(last)
            last = s.sample(logits)
            out_ids.append(last)
        return GenerationResult(tokens=out_ids, loop_passes=self.config.loop_passes,
                                eff_depth=self.config.eff_depth)

    def reset(self):
        self._pos = 0
        for b in self.blocks:
            b.reset_cache()
        self.loop.reset()

    def n_params(self) -> int:
        # rough parameter count for reporting
        n = 0
        n += self.config.vocab_size * self.config.dim          # embedding
        n += self.config.vocab_size * self.config.dim          # lm head
        for b in self.blocks:
            n += 4 * self.config.dim * self.config.dim          # attention q/k/v/o
            if b.use_moe:
                n += self.config.n_experts * 2 * self.config.hidden_ffn * self.config.dim  # experts
            else:
                n += 2 * self.config.hidden_ffn * self.config.dim
        return n
