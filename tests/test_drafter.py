# -*- coding: utf-8 -*-
import unittest

from loopinfer.drafter.config import DrafterConfig
from loopinfer.drafter.speculator import SpeculativeDecoder
from loopinfer.model.config import ModelConfig
from loopinfer.model.factory import build_model
from loopinfer.core.state import Random


class TestDrafter(unittest.TestCase):
    def test_expected_speedup(self):
        cfg = DrafterConfig(K=4, draft_speedup=3.0)
        self.assertAlmostEqual(cfg.expected_speedup(), 4.0 / (1.0 + 4.0 / 3.0), places=6)

    def test_decode_step(self):
        mc = ModelConfig(dim=16, n_heads=4, head_dim=4, hidden_ffn=32, vocab_size=64,
                         n_layers=12, loop_start=3, loop_end=9, loop_passes=2, n_experts=32, top_k=8)
        model = build_model(mc)
        dec = SpeculativeDecoder(model, DrafterConfig(K=4, draft_speedup=3.0), Random(1))
        accepted, draft = dec.decode_step(1, Random(2))
        self.assertGreaterEqual(accepted, 0)
        self.assertLessEqual(accepted, len(draft))
        self.assertEqual(len(draft), 4)


if __name__ == "__main__":
    unittest.main()
