# -*- coding: utf-8 -*-
import unittest

from loopinfer.moe.router import route, gate_logits, topk_indices, softmax
from loopinfer.moe.layer import MoELayer
from loopinfer.moe.config import MoEConfig
from loopinfer.moe.experts import ExpertFFN
from loopinfer.core.state import Random


class TestRouter(unittest.TestCase):
    def test_topk(self):
        vals = [3.0, 1.0, 5.0, 2.0]
        self.assertEqual(topk_indices(vals, 2), [2, 0])

    def test_softmax_sums_one(self):
        p = softmax([1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(p), 1.0, places=6)

    def test_route_shape(self):
        rng = Random(1)
        dim, n_experts, top_k = 8, 16, 4
        gate_w = [[rng.gauss() for _ in range(n_experts)] for _ in range(dim)]
        experts = [ExpertFFN(dim, 16, rng).forward for _ in range(n_experts)]
        out, lb = route([0.1] * dim, gate_w, n_experts, top_k, experts)
        self.assertEqual(len(out), dim)
        self.assertGreaterEqual(lb, 0.0)


class TestMoELayer(unittest.TestCase):
    def test_forward(self):
        cfg = MoEConfig(dim=8, hidden_ffn=16, n_experts=16, top_k=4)
        layer = MoELayer(cfg, Random(2))
        out, lb = layer.forward([0.1] * 8)
        self.assertEqual(len(out), 8)
        self.assertGreaterEqual(lb, 0.0)


if __name__ == "__main__":
    unittest.main()
