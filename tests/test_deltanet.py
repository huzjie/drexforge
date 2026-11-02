# -*- coding: utf-8 -*-
import unittest

from drexforge.core.deltanet import GatedDeltaNet, sigmoid, beta_span


class TestDeltaNet(unittest.TestCase):
    def test_output_shape(self):
        layer = GatedDeltaNet(8, "s")
        outs, S = layer.forward([[0.1] * 8 for _ in range(5)])
        self.assertEqual(len(outs), 5)
        self.assertEqual(len(outs[0]), 8)
        self.assertEqual(len(S), 8)

    def test_forget_gate_in_unit(self):
        layer = GatedDeltaNet(8, "s")
        x = [0.2, -0.3, 0.1, 0.0, 0.5, -0.1, 0.3, -0.2]
        self.assertTrue(0.0 < layer.forget_gate(x) < 1.0)

    def test_ops_linear(self):
        self.assertEqual(GatedDeltaNet.ops(10, 8), 10 * 8 * 8)

    def test_beta_span_positive(self):
        lo, hi = beta_span(8, 8)
        self.assertGreater(hi, 0.5)
        self.assertGreaterEqual(hi, lo)

    def test_sigmoid_bounds(self):
        self.assertAlmostEqual(sigmoid(0.0), 0.5)
        self.assertGreater(sigmoid(100.0), 0.99)
        self.assertLess(sigmoid(-100.0), 0.01)


if __name__ == "__main__":
    unittest.main()
