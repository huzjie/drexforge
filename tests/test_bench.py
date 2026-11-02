# -*- coding: utf-8 -*-
import unittest

from drexforge import bench


class TestBench(unittest.TestCase):
    def test_skill_monotonic(self):
        rows = bench.run_skill_table()
        self.assertGreater(rows[-1]["accuracy"], rows[0]["accuracy"])
        self.assertLess(rows[-1]["ordinal_mae"], rows[0]["ordinal_mae"])
        self.assertLess(rows[-1]["ece"], rows[0]["ece"])

    def test_expert_accuracy_one(self):
        rows = bench.run_skill_table()
        self.assertAlmostEqual(rows[-1]["accuracy"], 1.0)

    def test_confidence_rises(self):
        rows = bench.run_skill_table()
        self.assertGreater(rows[-1]["confidence"], rows[0]["confidence"])

    def test_ablation_ops_ordering(self):
        by = {r["mode"]: r["total_ops"] for r in bench.run_ablation()}
        self.assertLess(by["delta"], by["hybrid"])
        self.assertLess(by["hybrid"], by["full"])

    def test_ablation_layer_counts(self):
        abl = {r["mode"]: r for r in bench.run_ablation()}
        self.assertEqual(abl["hybrid"]["n_delta"], 9)
        self.assertEqual(abl["hybrid"]["n_full"], 3)
        self.assertEqual(abl["full"]["n_full"], 12)
        self.assertEqual(abl["delta"]["n_delta"], 12)


if __name__ == "__main__":
    unittest.main()
