# -*- coding: utf-8 -*-
import unittest

from loopinfer.core.sampler import (get_sampler, list_samplers, argmax, softmax,
                                    GreedySampler, TopKSampler, NucleusSampler, TemperatureSampler)
from loopinfer.core.state import Random


class TestSampler(unittest.TestCase):
    def test_greedy(self):
        s = GreedySampler(rng=Random(1))
        self.assertEqual(s.sample([0.1, 0.9, 0.5]), 1)

    def test_topk(self):
        s = TopKSampler(rng=Random(2), k=2)
        idx = s.sample([0.1, 0.9, 0.5, 0.2])
        self.assertIn(idx, [1, 2])

    def test_list_samplers(self):
        self.assertGreaterEqual(len(list_samplers()), 4)

    def test_argmax(self):
        self.assertEqual(argmax([1.0, 3.0, 2.0]), 1)

    def test_softmax(self):
        p = softmax([0.0, 0.0])
        self.assertAlmostEqual(sum(p), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
