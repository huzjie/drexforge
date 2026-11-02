# -*- coding: utf-8 -*-
import unittest

from drexforge.core.model import DecisionModel


class TestDecisionModel(unittest.TestCase):
    def test_boost_zero_at_random(self):
        self.assertAlmostEqual(DecisionModel(0.5).boost(), 0.0)

    def test_boost_sign(self):
        self.assertAlmostEqual(DecisionModel(1.0).boost(), 1.0)
        self.assertAlmostEqual(DecisionModel(0.0).boost(), -1.0)

    def test_expert_picks_correct(self):
        d = DecisionModel(1.0).decide([0.95, 0.05, 0.05], [0.5, 0.5, 0.5])
        self.assertEqual(d["choice"], 0)

    def test_adversarial_picks_wrong(self):
        d = DecisionModel(0.0).decide([0.95, 0.05, 0.05], [0.5, 0.5, 0.5])
        self.assertNotEqual(d["choice"], 0)

    def test_random_equals_noise(self):
        logits = DecisionModel(0.5).logits([0.9, 0.1], [0.3, 0.7])
        self.assertAlmostEqual(logits[0], 0.3)
        self.assertAlmostEqual(logits[1], 0.7)

    def test_probs_sum_to_one(self):
        d = DecisionModel(0.8).decide([0.9, 0.1, 0.1], [0.5, 0.5, 0.5])
        self.assertAlmostEqual(sum(d["probs"]), 1.0, places=3)

    def test_expert_confidence_high(self):
        d = DecisionModel(1.0).decide([0.95, 0.05, 0.05], [0.5, 0.5, 0.5])
        self.assertGreater(d["confidence"], 0.9)

    def test_clamped(self):
        self.assertEqual(DecisionModel(2.0).skill, 1.0)
        self.assertEqual(DecisionModel(-1.0).skill, 0.0)


if __name__ == "__main__":
    unittest.main()
