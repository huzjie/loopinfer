# -*- coding: utf-8 -*-
"""Loop scheduler: drives the three-segment forward with a shared loop block."""

from __future__ import annotations

from loopinfer.core.state import Random
from loopinfer.loop.config import LoopConfig
from loopinfer.loop.messaging import CrossPassMemory


class LoopScheduler:
    """Executes prefix -> (loop block x passes, with cross-pass memory) -> suffix."""

    def __init__(self, config: LoopConfig, dim: int, rng: Random = None):
        self.config = config
        self.dim = dim
        self.memory = CrossPassMemory(config.n_loop_layers, dim, gate_init=config.gate_init, rng=rng)

    def forward(self, x, layers, return_summary: bool = False):
        """layers: a list of n_layers callables `layer(x) -> list` or `layer(x, return_summary=True) -> (list, list)`."""
        cfg = self.config
        # prefix (non-looped)
        for idx in range(cfg.loop_start):
            x = layers[idx](x)

        # looped block, repeated cfg.loop_passes times
        for _pass in range(cfg.loop_passes):
            for slot, idx in enumerate(range(cfg.loop_start, cfg.loop_end)):
                x = self.memory.inject(x, slot)
                out = layers[idx](x, return_summary=True)
                if isinstance(out, tuple):
                    x, summary = out
                else:
                    x, summary = out, out
                self.memory.update(slot, summary)

        # suffix (non-looped)
        for idx in range(cfg.loop_end, cfg.n_layers):
            x = layers[idx](x)

        if return_summary:
            return x, self._state_snapshot()
        return x

    def _state_snapshot(self) -> dict:
        return {
            "eff_depth": self.config.eff_depth,
            "loop_passes": self.config.loop_passes,
            "gate": self.memory.gate,
            "memory_norm": sum(sum(v * v for v in m) for m in self.memory.memory) ** 0.5,
        }

    def reset(self):
        self.memory.reset()
