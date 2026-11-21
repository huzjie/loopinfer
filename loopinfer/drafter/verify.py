# -*- coding: utf-8 -*-
"""Verify draft tokens against the main model, accept until first mismatch."""

from __future__ import annotations

from loopinfer.core.sampler import argmax


def verify(model, first_token_id: int, draft_tokens, log=None):
    """Run the main model over the draft and accept the matching prefix.

    Returns (accepted_count, corrected_first_token). `log` is an optional metrics dict to fill.
    """
    accepted = 0
    cur = first_token_id
    model_steps = 0
    for d in draft_tokens:
        logits, _ = model.forward_token(cur)
        model_steps += 1
        best = argmax(logits)
        if d == best:
            accepted += 1
            cur = d
        else:
            cur = best
            break
    if log is not None:
        log["model_steps"] = model_steps
        log["accepted"] = accepted
    return accepted, cur
