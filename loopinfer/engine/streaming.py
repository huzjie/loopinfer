# -*- coding: utf-8 -*-
"""Streaming wrapper: yields tokens incrementally."""

from __future__ import annotations


def stream_generate(engine, text: str, max_tokens: int = 32):
    """Generator yielding token ids one by one (delegates to the engine)."""
    prompt_ids = engine.tokenizer.encode(text)
    engine.model.reset()
    for t in prompt_ids:
        engine.model.forward_token(t)
    last = prompt_ids[-1] if prompt_ids else 0
    from loopinfer.core.sampler import get_sampler
    from loopinfer.core.state import Random
    s = get_sampler(engine.config.sampler, rng=Random(99), temperature=engine.config.temperature,
                    top_k=engine.config.top_k, top_p=engine.config.top_p)
    for _ in range(max_tokens):
        logits, _ = engine.model.forward_token(last)
        last = s.sample(logits)
        yield last
