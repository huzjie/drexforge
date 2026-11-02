# -*- coding: utf-8 -*-
import unittest

from drexforge.core.scorer import OptionScorer, embed_set, softmax, sigmoid


class TestScorer(unittest.TestCase):
    def test_embed_set_orthonormal(self):
        embeds = embed_set(["a", "b", "c"], 8)
        for i in range(3):
            for j in range(i + 1, 3):
                d = sum(x * y for x, y in zip(embeds[i], embeds[j]))
                self.assertAlmostEqual(d, 0.0, places=3)

    def test_choice_probs_sum_one(self):
        d = OptionScorer(8).score([0.1] * 8, ["x", "y", "z"], "choice")
        self.assertAlmostEqual(sum(d["probs"]), 1.0, places=3)

    def test_yesno_probs_sum_one(self):
        d = OptionScorer(8).score([0.1] * 8, ["yes", "no"], "yesno")
        self.assertAlmostEqual(sum(d["probs"]), 1.0, places=3)

    def test_ordinal_level_in_range(self):
        d = OptionScorer(8).score([0.1] * 8, ["1", "2", "3", "4", "5"], "ordinal")
        self.assertGreaterEqual(d["level"], 1)
        self.assertLessEqual(d["level"], 5)

    def test_softmax_peaks(self):
        p = softmax([0.0, 10.0])
        self.assertGreater(p[1], p[0])

    def test_no_text_generated(self):
        d = OptionScorer(8).score([0.1] * 8, ["a", "b"], "choice")
        self.assertIsInstance(d["choice"], int)          # index, not a string
        self.assertIn("label", d)                        # label is derived, not generated

    def test_sigmoid_monotonic(self):
        self.assertGreater(sigmoid(1.0), sigmoid(0.0))
        self.assertGreater(sigmoid(0.0), sigmoid(-1.0))


if __name__ == "__main__":
    unittest.main()
