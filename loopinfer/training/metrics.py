# -*- coding: utf-8 -*-
"""Skill metric: top-1 accuracy on a task (memory-style deterministic mock)."""

from __future__ import annotations

from loopinfer.core.sampler import argmax


def skill_score(model, pairs):
    # model-based linear head accuracy (best-effort; the mock trainer's mem drives the
    # monotonic climb, this is the model-in-the-loop readout)
    correct = 0
    for x, y in pairs:
        ex = model.embedding.vector(x)
        logits = [sum(w[j] * ex[j] for j in range(len(ex))) for w in model.lm_head]
        if argmax(logits) == y:
            correct += 1
    return correct / len(pairs) if pairs else 0.0
