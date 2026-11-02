# -*- coding: utf-8 -*-
import unittest

from drexforge.core.attention import FullAttention, softmax_scores


class TestAttention(unittest.TestCase):
    def test_softmax_sums_to_one(self):
        p = softmax_scores([1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(p), 1.0, places=6)

    def test_output_shape(self):
        layer = FullAttention(8, "s")
        outs = layer.forward([[0.1] * 8 for _ in range(6)])
        self.assertEqual(len(outs), 6)
        self.assertEqual(len(outs[0]), 8)

    def test_ops_quadratic(self):
        self.assertEqual(FullAttention.ops(10, 8), 10 * 10 * 8)

    def test_non_causal_full_context(self):
        layer = FullAttention(8, "s")
        a = [[0.1] * 8, [0.2] * 8, [0.3] * 8]
        b = [[0.1] * 8, [0.2] * 8, [0.3] * 8, [0.9] * 8]
        # adding a 4th token changes the FIRST token's output -> non-causal
        self.assertNotEqual(layer.forward(a)[0], layer.forward(b)[0])


if __name__ == "__main__":
    unittest.main()
