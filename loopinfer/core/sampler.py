# -*- coding: utf-8 -*-
"""Sampling strategies: greedy, temperature, top-k, top-p. All deterministic under a seeded RNG."""

from __future__ import annotations

import inspect
import math


def softmax(logits):
    # numerically stable softmax over a list of floats
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    s = sum(exps)
    return [e / s for e in exps]


def argmax(seq):
    best_i, best_v = 0, seq[0]
    for i, v in enumerate(seq):
        if v > best_v:
            best_i, best_v = i, v
    return best_i


class BaseSampler:
    kind = "base"

    def __init__(self, rng=None, temperature: float = 1.0):
        self.rng = rng
        self.temperature = temperature

    def sample(self, logits):
        raise NotImplementedError


class GreedySampler(BaseSampler):
    kind = "greedy"

    def __init__(self, rng=None, temperature: float = 1.0):
        super().__init__(rng=rng, temperature=temperature)

    def sample(self, logits):
        return argmax(logits)


class TemperatureSampler(BaseSampler):
    kind = "temperature"

    def __init__(self, rng=None, temperature: float = 1.0):
        super().__init__(rng=rng, temperature=temperature)

    def sample(self, logits):
        t = max(self.temperature, 1e-6)
        scaled = [v / t for v in logits]
        probs = softmax(scaled)
        r = self.rng.random() if self.rng else 0.5
        acc = 0.0
        for i, p in enumerate(probs):
            acc += p
            if r <= acc:
                return i
        return len(probs) - 1


class TopKSampler(BaseSampler):
    kind = "topk"

    def __init__(self, rng=None, temperature: float = 1.0, k: int = 50):
        super().__init__(rng=rng, temperature=temperature)
        self.k = k

    def sample(self, logits):
        k = min(max(self.k, 1), len(logits))
        idx = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)[:k]
        keep = [(-1e9 if i not in set(idx) else logits[i]) for i in range(len(logits))]
        return TemperatureSampler(rng=self.rng, temperature=self.temperature).sample(keep)


class NucleusSampler(BaseSampler):
    kind = "nucleus"

    def __init__(self, rng=None, temperature: float = 1.0, top_p: float = 0.9):
        super().__init__(rng=rng, temperature=temperature)
        self.top_p = top_p

    def sample(self, logits):
        t = max(self.temperature, 1e-6)
        probs = softmax([v / t for v in logits])
        order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
        acc = 0.0
        keep = set()
        for i in order:
            keep.add(i)
            acc += probs[i]
            if acc >= self.top_p:
                break
        r = self.rng.random() if self.rng else 0.5
        acc = 0.0
        norm = sum(probs[i] for i in keep) or 1.0
        for i in order:
            if i not in keep:
                continue
            acc += probs[i] / norm
            if r <= acc:
                return i
        return order[0]


_REGISTRY = {
    "greedy": GreedySampler,
    "temperature": TemperatureSampler,
    "topk": TopKSampler,
    "nucleus": NucleusSampler,
}


def register_sampler(name, cls):
    _REGISTRY[name] = cls


def list_samplers():
    return sorted(_REGISTRY.keys())


def get_sampler(kind: str = "greedy", **kwargs):
    cls = _REGISTRY.get(kind, GreedySampler)
    # filter kwargs to what the class actually accepts (avoid TypeError on extras)
    sig = inspect.signature(cls.__init__)
    accepted = {}
    for k, v in kwargs.items():
        if k in sig.parameters:
            accepted[k] = v
    return cls(**accepted)
