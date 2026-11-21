# -*- coding: utf-8 -*-
import unittest

from loopinfer.model.config import ModelConfig
from loopinfer.model.factory import build_model
from loopinfer.model.layernorm import LayerNorm
from loopinfer.model.embedding import Embedding
from loopinfer.core.state import Random


class TestModel(unittest.TestCase):
    def _cfg(self):
        return ModelConfig(dim=16, n_heads=4, head_dim=4, hidden_ffn=32, vocab_size=64,
                           n_layers=12, loop_start=3, loop_end=9, loop_passes=2, n_experts=32, top_k=8)

    def test_forward_token(self):
        model = build_model(self._cfg())
        logits, x = model.forward_token(1)
        self.assertEqual(len(logits), 64)
        self.assertEqual(len(x), 16)

    def test_generate(self):
        model = build_model(self._cfg())
        r = model.generate([1, 2, 3], max_tokens=8, sampler="greedy")
        self.assertEqual(len(r.tokens), 8)

    def test_eff_depth(self):
        cfg = self._cfg()
        self.assertEqual(cfg.eff_depth, 3 + 2 * 6 + 3)


class TestLayers(unittest.TestCase):
    def test_layernorm(self):
        ln = LayerNorm(4)
        out = ln.forward([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(len(out), 4)

    def test_embedding(self):
        emb = Embedding(64, 16, 128, Random(1))
        v = emb.forward(1, 0)
        self.assertEqual(len(v), 16)


if __name__ == "__main__":
    unittest.main()
