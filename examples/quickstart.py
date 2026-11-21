# -*- coding: utf-8 -*-
"""Quickstart: build the default engine, run doctor-style smoke, generate text."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loopinfer import InferenceEngine


def main():
    engine = InferenceEngine()

    print("=== describe ===")
    info = engine.describe()
    print(f"eff_depth = {info['eff_depth']}")
    print(f"n_params = {info['n_params']}")
    print(f"spec_speedup = {info['spec_speedup']}")

    print("\n=== generate ===")
    r = engine.generate("循环 Transformer 的等效深度是怎么算出来的？", max_tokens=32)
    print(f"output: {r.text!r}")
    print(f"tokens={len(r.tokens)} loop_passes={r.loop_passes} eff_depth={r.eff_depth}")


if __name__ == "__main__":
    main()
