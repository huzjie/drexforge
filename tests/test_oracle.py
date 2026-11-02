# -*- coding: utf-8 -*-
import unittest

from drexforge.world.oracle import evidence, noise
from drexforge.world.corpus import make_batch


class TestOracle(unittest.TestCase):
    def test_evidence_correct_high(self):
        t = make_batch()[0]
        self.assertGreater(evidence(t)[t["correct"]], 0.9)

    def test_evidence_distractor_low(self):
        t = make_batch()[0]
        ev = evidence(t)
        for i in range(len(ev)):
            if i != t["correct"]:
                self.assertLess(ev[i], 0.1)

    def test_evidence_in_range(self):
        t = make_batch()[0]
        for v in evidence(t):
            self.assertTrue(0.0 <= v <= 1.0)

    def test_noise_in_range(self):
        t = make_batch()[0]
        for v in noise(t):
            self.assertTrue(0.0 <= v <= 1.0)

    def test_noise_deterministic(self):
        t = make_batch()[0]
        self.assertEqual(noise(t), noise(t))


if __name__ == "__main__":
    unittest.main()
