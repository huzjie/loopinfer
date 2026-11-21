# -*- coding: utf-8 -*-
"""Train the mock model and print skill progression."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loopinfer import build_model, ModelConfig
from loopinfer.training.data import SyntheticData
from loopinfer.training.trainer import MockTrainer


def main():
    cfg = ModelConfig(dim=16, n_heads=4, head_dim=4, hidden_ffn=32, vocab_size=64,
                      n_layers=8, loop_start=2, loop_end=6, loop_passes=2, n_experts=16, top_k=4)
    model = build_model(cfg)
    data = SyntheticData(vocab_size=64, n_train=48, n_test=32)
    t = MockTrainer(model, lr=0.5)
    n = len(data.train)
    for frac in [0.25, 0.5, 0.75, 1.0]:
        k = int(n * frac)
        t.train(data.train[:k], epochs=1)
        print(f"{frac:.0%} trained: skill={t.skill(data.test):.3f}")


if __name__ == "__main__":
    main()
