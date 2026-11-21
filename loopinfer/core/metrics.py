# -*- coding: utf-8 -*-
"""Accumulating metrics for benchmarking and observability."""

from __future__ import annotations

from collections import defaultdict


class Metrics:
    def __init__(self):
        self._counters = defaultdict(float)
        self._gauges = {}

    def add(self, name: str, value: float = 1.0):
        self._counters[name] += value

    def set(self, name: str, value):
        self._gauges[name] = value

    def get(self, name: str, default: float = 0.0):
        if name in self._counters:
            return self._counters[name]
        if name in self._gauges:
            return self._gauges[name]
        return default

    def snapshot(self) -> dict:
        out = dict(self._counters)
        out.update({f"gauge:{k}": v for k, v in self._gauges.items()})
        return out

    def clear(self):
        self._counters.clear()
        self._gauges.clear()
