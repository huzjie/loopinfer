# -*- coding: utf-8 -*-
import unittest

from loopinfer.core.state import Random
from loopinfer.core.metrics import Metrics
from loopinfer.core.sampler import softmax, argmax
from loopinfer.core.config import load_config, dump_config


class TestCore(unittest.TestCase):
    def test_rng_deterministic(self):
        a = Random(1)
        b = Random(1)
        self.assertEqual([a.random() for _ in range(5)], [b.random() for _ in range(5)])

    def test_metrics(self):
        m = Metrics()
        m.add("tokens", 3)
        m.set("depth", 48)
        self.assertEqual(m.get("tokens"), 3)
        self.assertEqual(m.get("depth"), 48)

    def test_config_roundtrip(self):
        import tempfile, os
        cfg = {"model": {"dim": 32}, "sampler": "greedy"}
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        with open(path, "w", encoding="utf-8") as f:
            f.write(dump_config(cfg))
        loaded = load_config(path)
        os.remove(path)
        self.assertEqual(loaded["model"]["dim"], 32)


if __name__ == "__main__":
    unittest.main()
