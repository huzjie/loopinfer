# -*- coding: utf-8 -*-
"""Tool: quick metrics summary for a generated result."""

from loopinfer.core.metrics import Metrics


def summarize_result(result):
    m = Metrics()
    m.add("tokens", len(result.tokens))
    m.set("loop_passes", result.loop_passes)
    m.set("eff_depth", result.eff_depth)
    return m.snapshot()
