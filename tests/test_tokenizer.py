# -*- coding: utf-8 -*-
import unittest

from loopinfer.tokenizer.vocab import ByteTokenizer
from loopinfer.tokenizer.bpe import BPETokenizer


class TestTokenizer(unittest.TestCase):
    def test_byte_roundtrip(self):
        tok = ByteTokenizer()
        text = "循环 Transformer"
        ids = tok.encode(text)
        self.assertEqual(tok.decode(ids), text)

    def test_bpe_train(self):
        bpe = BPETokenizer()
        bpe.train(["hello world hello", "world world"], num_merges=4)
        ids = bpe.encode("hello world")
        self.assertGreaterEqual(len(ids), 1)


if __name__ == "__main__":
    unittest.main()
