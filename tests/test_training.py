# -*- coding: utf-8 -*-
import unittest

from loopinfer.model.config import ModelConfig
from loopinfer.model.factory import build_model
from loopinfer.training.data import SyntheticData
from loopinfer.training.trainer import MockTrainer


class TestTraining(unittest.TestCase):
    def test_skill_monotonic(self):
        cfg = ModelConfig(dim=16, n_heads=4, head_dim=4, hidden_ffn=32, vocab_size=64,
                          n_layers=8, loop_start=2, loop_end=6, loop_passes=2, n_experts=16, top_k=4)
        model = build_model(cfg)
        data = SyntheticData(vocab_size=64, n_train=32, n_test=16)
        trainer = MockTrainer(model, lr=0.1)
        s0 = trainer.skill(data.test)
        for _ in range(3):
            trainer.train(data.train, epochs=2)
        s1 = trainer.skill(data.test)
        self.assertGreaterEqual(s1, s0)


if __name__ == "__main__":
    unittest.main()
