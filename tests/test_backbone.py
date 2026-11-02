# -*- coding: utf-8 -*-
import unittest

from drexforge.core.backbone import Backbone
from drexforge.config import load_config


class TestBackbone(unittest.TestCase):
    def test_hybrid_composition(self):
        bb = Backbone(load_config())
        self.assertEqual(bb.n_delta, 9)
        self.assertEqual(bb.n_full, 3)

    def test_full_only(self):
        cfg = load_config()
        cfg["backbone"]["mode"] = "full"
        bb = Backbone(cfg)
        self.assertEqual(bb.n_full, 12)
        self.assertEqual(bb.n_delta, 0)

    def test_delta_only(self):
        cfg = load_config()
        cfg["backbone"]["mode"] = "delta"
        bb = Backbone(cfg)
        self.assertEqual(bb.n_delta, 12)
        self.assertEqual(bb.n_full, 0)

    def test_forward_pool_shape(self):
        cfg = load_config()
        bb = Backbone(cfg)
        h = bb.forward([[0.1] * cfg["backbone"]["dim"] for _ in range(8)])
        self.assertEqual(len(h), cfg["backbone"]["dim"])

    def test_full_every_fourth(self):
        bb = Backbone(load_config())
        kinds = [k for k, _ in bb.layers]
        self.assertEqual(kinds[3], "full")
        self.assertEqual(kinds[7], "full")
        self.assertEqual(kinds[11], "full")

    def test_ops_ordering(self):
        cfg = load_config()
        seq = cfg["backbone"]["seq_len"]
        delta = Backbone({**cfg, "backbone": {**cfg["backbone"], "mode": "delta"}})
        full = Backbone({**cfg, "backbone": {**cfg["backbone"], "mode": "full"}})
        self.assertLess(delta.total_ops(seq), full.total_ops(seq))


if __name__ == "__main__":
    unittest.main()
