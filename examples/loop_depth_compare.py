# -*- coding: utf-8 -*-
"""Compare recurrent depth: same params, different loop_passes."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loopinfer import build_model, ModelConfig


def main():
    print(f"{'passes':>7} {'eff_depth':>9} {'n_params':>9}")
    for p in [1, 2, 3, 4, 6]:
        cfg = ModelConfig(n_layers=24, loop_start=6, loop_end=18, loop_passes=p)
        model = build_model(cfg)
        print(f"{p:>7} {cfg.eff_depth:>9} {model.n_params():>9}")
    print("\n结论：深度随 passes 线性涨，参数量不变 —— 用算力换深度。")


if __name__ == "__main__":
    main()
