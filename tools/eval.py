# -*- coding: utf-8 -*-
"""Tool: evaluate a model against synthetic pairs."""

from loopinfer.training.metrics import skill_score


def evaluate(model, pairs):
    return {"skill": skill_score(model, pairs), "n_pairs": len(pairs)}
