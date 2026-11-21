# -*- coding: utf-8 -*-
"""Speculative decoding demo: draft K tokens, verify, report speedup."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loopinfer import InferenceEngine, DrafterConfig
from loopinfer.core.state import Random


def main():
    engine = InferenceEngine()
    print(f"expected_speedup = {engine.decoder.expected_speedup():.2f}x")
    for seed in range(5):
        accepted, draft = engine.speculate(1, Random(seed))
        print(f"seed={seed} drafted={len(draft)} accepted={accepted} "
              f"actual_speedup={engine.spec_speedup()['actual']:.2f}x")
    print(f"acceptance_rate = {engine.spec_speedup()['acceptance_rate']:.2%}")


if __name__ == "__main__":
    main()
