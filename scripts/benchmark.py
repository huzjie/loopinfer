# -*- coding: utf-8 -*-
"""Benchmark script: loop passes vs eff depth vs param count."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loopinfer import build_model, ModelConfig


def main():
    print("loopinfer benchmark")
    print(f"{'passes':>7} {'eff_depth':>9} {'n_params':>10}")
    for p in [1, 2, 3, 4, 6]:
        cfg = ModelConfig(n_layers=24, loop_start=6, loop_end=18, loop_passes=p)
        m = build_model(cfg)
        print(f"{p:>7} {cfg.eff_depth:>9} {m.n_params():>10}")


if __name__ == "__main__":
    main()
