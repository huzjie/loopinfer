# -*- coding: utf-8 -*-
"""Cross-pass messaging: a per-loop-layer memory vector carried between passes.

This is what turns "running the same layer N times" into "recurrent depth":
pass k+1 sees the summary of pass k, so each pass refines the previous one
instead of blindly repeating. Analogue to an RNN hidden state, but layer-level.
"""

from __future__ import annotations

from loopinfer.core.state import Random


def _matvec(w, x):
    return [sum(wi * xi for wi, xi in zip(row, x)) for row in w]


def _mean(v):
    return sum(v) / len(v) if v else 0.0


class CrossPassMemory:
    """Holds one memory vector per looped layer and injects it into the next pass."""

    def __init__(self, n_loop_layers: int, dim: int, gate_init: float = 0.0, rng: Random = None):
        self.n_loop_layers = n_loop_layers
        self.dim = dim
        self.gate = gate_init
        rng = rng or Random(7)
        self.memory = [[0.0] * dim for _ in range(n_loop_layers)]
        # per-layer projection: memory(dim) -> injection(dim)
        self.proj = [[[rng.gauss(0.0, 0.02) for _ in range(dim)] for _ in range(dim)]
                     for _ in range(n_loop_layers)]
        # summary projection: layer output(dim) -> memory(dim), use a light residual
        self.summary_scale = [0.1] * n_loop_layers

    def inject(self, x, layer_slot: int):
        """x + gate * proj[slot] @ memory[slot]."""
        mem = self.memory[layer_slot]
        delta = _matvec(self.proj[layer_slot], mem)
        return [xi + self.gate * di for xi, di in zip(x, delta)]

    def update(self, layer_slot: int, layer_output):
        """Summarize the layer output into the next-pass memory (moving average)."""
        s = self.summary_scale[layer_slot]
        m = _mean(layer_output)
        self.memory[layer_slot] = [mi * (1.0 - s) + m * s for mi in self.memory[layer_slot]]

    def set_gate(self, value: float):
        self.gate = value

    def reset(self):
        self.memory = [[0.0] * self.dim for _ in range(self.n_loop_layers)]
