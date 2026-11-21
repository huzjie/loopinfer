# -*- coding: utf-8 -*-
"""Transformer block: attention + (MoE | dense) FFN, pre-norm, residuals."""

from __future__ import annotations

from loopinfer.core.state import Random
from loopinfer.attention.multihead import MultiHeadAttention
from loopinfer.moe.layer import MoELayer
from loopinfer.model.layernorm import LayerNorm
from loopinfer.model.dense import DenseFFN


class TransformerBlock:
    def __init__(self, dim, attn_cfg, moe_cfg, use_moe: bool, rng: Random = None):
        self.dim = dim
        self.use_moe = use_moe
        rng = rng or Random(13)
        self.attn = MultiHeadAttention(attn_cfg, rng)
        self.ln1 = LayerNorm(dim)
        self.ln2 = LayerNorm(dim)
        if use_moe:
            self.ffn = MoELayer(moe_cfg, rng)
        else:
            self.ffn = DenseFFN(dim, moe_cfg.hidden_ffn, rng)

    def forward(self, x, return_summary=False):
        # attention sub-layer (pre-norm + residual)
        h = self.ln1.forward(x)
        h = self.attn.forward(h)
        x = [xi + hi for xi, hi in zip(x, h)]
        # ffn sub-layer (pre-norm + residual)
        h2 = self.ln2.forward(x)
        if self.use_moe:
            out, _lb = self.ffn.forward(h2)
        else:
            out = self.ffn.forward(h2)
        x = [xi + oi for xi, oi in zip(x, out)]
        if return_summary:
            return x, list(x)
        return x

    def __call__(self, x, return_summary=False):
        return self.forward(x, return_summary=return_summary)

    def reset_cache(self):
        self.attn.reset_cache()
