# -*- coding: utf-8 -*-
"""Naive sequential batching (single model, iterate requests)."""

from __future__ import annotations


class BatchRunner:
    def __init__(self, engine):
        self.engine = engine

    def run(self, prompts, max_tokens: int = 32):
        out = []
        for p in prompts:
            r = self.engine.generate(p, max_tokens=max_tokens)
            out.append(r)
        return out
