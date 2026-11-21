# -*- coding: utf-8 -*-
"""MoE layer: shared expert + top-k routed experts + load-balancing loss.

Experts are generated LAZILY: only the top-k experts activated per token are ever
materialized (and cached). Eagerly building n_experts x n_layers FFNs is the #1
cause of slow init in sparse MoE reference implementations.
"""

from __future__ import annotations

from loopinfer.core.state import Random
from loopinfer.moe.config import MoEConfig
from loopinfer.moe.experts import ExpertFFN
from loopinfer.moe.router import gate_logits, topk_indices, softmax


class MoELayer:
    def __init__(self, config: MoEConfig, rng: Random = None):
        self.config = config
        rng = rng or Random(11)
        self.dim = config.dim
        self.expert_seed = rng.randint(1, 1 << 30)
        self.gate_w = [[rng.gauss(0.0, 0.02) for _ in range(config.n_experts)] for _ in range(config.dim)]
        self._experts = {}
        self.shared = ExpertFFN(config.dim, config.hidden_ffn, rng) if config.shared_expert else None

    def _expert(self, e: int) -> ExpertFFN:
        if e not in self._experts:
            self._experts[e] = ExpertFFN(self.dim, self.config.hidden_ffn, Random(self.expert_seed + e))
        return self._experts[e]

    def forward(self, x):
        cfg = self.config
        logits = gate_logits(x, self.gate_w)
        topk_idx = topk_indices(logits, cfg.top_k)
        topk_vals = [logits[e] for e in topk_idx]
        weights = softmax(topk_vals)  # normalize ONLY within top-k

        out = None
        for k, e in enumerate(topk_idx):
            expert_out = self._expert(e).forward(x)
            w = weights[k]
            out = expert_out if out is None else [o + w * v for o, v in zip(out, expert_out)]

        if self.shared is not None:
            shared_out = self.shared.forward(x)
            out = [o + s for o, s in zip(out, shared_out)]

        # load-balancing loss: n_experts * sum((frac - 1/n)^2), always >= 0
        counts = [0] * cfg.n_experts
        for e in topk_idx:
            counts[e] += 1
        n = cfg.n_experts
        lb = 0.0
        for c in counts:
            frac = c / cfg.top_k
            lb += (frac - 1.0 / n) ** 2
        lb *= cfg.load_bal_coef * n
        return out, lb

    def n_activated(self):
        return self.config.top_k + (1 if self.shared is not None else 0)
