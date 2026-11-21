# -*- coding: utf-8 -*-
import unittest

from loopinfer.attention.head import head_attention
from loopinfer.attention.cache import KVCache
from loopinfer.attention.multihead import MultiHeadAttention
from loopinfer.attention.config import AttentionConfig
from loopinfer.core.state import Random


class TestHeadAttention(unittest.TestCase):
    def test_output_shape(self):
        q = [0.1, 0.2, 0.3, 0.4]
        ks = [[0.1] * 4, [0.2] * 4]
        vs = [[0.5] * 4, [0.6] * 4]
        out = head_attention(q, ks, vs)
        self.assertEqual(len(out), 4)


class TestKVCache(unittest.TestCase):
    def test_sliding_window(self):
        c = KVCache(2, window_size=3)
        for i in range(5):
            c.append(0, [float(i)] * 2, [float(i)] * 2)
            c.append(1, [float(i)] * 2, [float(i)] * 2)
        self.assertEqual(c.seq_len(0), 3)


class TestMultiHeadAttention(unittest.TestCase):
    def test_forward(self):
        cfg = AttentionConfig(dim=16, n_heads=4, head_dim=4, window_size=32)
        attn = MultiHeadAttention(cfg, Random(1))
        out = attn.forward([0.1] * 16)
        self.assertEqual(len(out), 16)


if __name__ == "__main__":
    unittest.main()
