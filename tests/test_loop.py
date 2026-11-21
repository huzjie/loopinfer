# -*- coding: utf-8 -*-
import unittest

from loopinfer.loop.config import LoopConfig
from loopinfer.loop.messaging import CrossPassMemory
from loopinfer.loop.scheduler import LoopScheduler
from loopinfer.core.state import Random


class TestLoopConfig(unittest.TestCase):
    def test_eff_depth(self):
        cfg = LoopConfig(n_layers=24, loop_start=6, loop_end=18, loop_passes=3)
        self.assertEqual(cfg.eff_depth, 48)
        self.assertEqual(cfg.n_loop_layers, 12)

    def test_validate_bad_range(self):
        with self.assertRaises(ValueError):
            LoopConfig(n_layers=10, loop_start=8, loop_end=4).validate()


class TestCrossPassMemory(unittest.TestCase):
    def test_inject_update(self):
        mem = CrossPassMemory(2, 4, gate_init=0.5, rng=Random(1))
        x = [0.1, 0.2, 0.3, 0.4]
        y = mem.inject(x, 0)
        self.assertEqual(len(y), 4)
        mem.update(0, [0.5, 0.5, 0.5, 0.5])
        self.assertNotEqual(mem.memory[0], [0.0] * 4)


class TestLoopScheduler(unittest.TestCase):
    def test_forward(self):
        cfg = LoopConfig(n_layers=6, loop_start=2, loop_end=4, loop_passes=3)
        sched = LoopScheduler(cfg, 4, Random(2))

        def layer(x, return_summary=False):
            out = [v * 0.9 for v in x]
            return (out, out) if return_summary else out

        layers = [layer] * 6
        out, snap = sched.forward([0.1] * 4, layers, return_summary=True)
        self.assertEqual(len(out), 4)
        self.assertEqual(snap["eff_depth"], 2 + 3 * 2 + 2)
        self.assertEqual(snap["loop_passes"], 3)


if __name__ == "__main__":
    unittest.main()
