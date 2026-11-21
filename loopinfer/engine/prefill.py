# -*- coding: utf-8 -*-
"""Prefill: feed a prompt through the stack and cache K/V."""

from __future__ import annotations


def prefill(model, token_ids):
    """Feed prompt tokens, returning logits of the final position."""
    for t in token_ids:
        model.forward_token(t)
    return model
