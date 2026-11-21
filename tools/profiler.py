# -*- coding: utf-8 -*-
"""Tool: time a single forward token."""

import time


def time_forward(model, token_id: int, n: int = 10):
    model.reset()
    t0 = time.perf_counter()
    for _ in range(n):
        model.forward_token(token_id)
    dt = time.perf_counter() - t0
    return {"per_token_ms": dt / n * 1000, "n": n}
