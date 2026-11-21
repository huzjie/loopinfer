# -*- coding: utf-8 -*-
import unittest

from loopinfer.engine.inference import InferenceEngine
from loopinfer.engine.config import EngineConfig


class TestEngine(unittest.TestCase):
    def test_generate(self):
        engine = InferenceEngine()
        r = engine.generate("循环", max_tokens=8)
        self.assertEqual(len(r.tokens), 8)
        self.assertTrue(r.text)

    def test_describe(self):
        engine = InferenceEngine()
        info = engine.describe()
        self.assertIn("eff_depth", info)
        self.assertEqual(info["eff_depth"], 48)


if __name__ == "__main__":
    unittest.main()
