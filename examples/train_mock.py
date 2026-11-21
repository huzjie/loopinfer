# -*- coding: utf-8 -*-
"""Demo the deterministic trainable mock: skill climbs monotonically."""

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
    trainer = MockTrainer(model, lr=0.5)

    print(f"skill before training:        {trainer.skill(data.test):.3f}")
    n = len(data.train)
    for frac in [0.25, 0.5, 0.75, 1.0]:
        k = int(n * frac)
        trainer.train(data.train[:k], epochs=1)
        print(f"skill after {frac:.0%} of train:     {trainer.skill(data.test):.3f}")
    print("\nskill 单调趋 1 —— 不接真实模型也能验证「训练 -> 能力提升」链路。")


if __name__ == "__main__":
    main()
