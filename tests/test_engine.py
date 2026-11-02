# -*- coding: utf-8 -*-
import unittest

from drexforge.core.engine import Engine
from drexforge.config import load_config
from drexforge.world.corpus import make_batch


class TestEngine(unittest.TestCase):
    def test_run_all_tasks(self):
        eng = Engine(load_config())
        self.assertEqual(eng.run_all()["summary"]["tasks"], 12)

    def test_run_task_shape(self):
        eng = Engine(load_config())
        r = eng.run_task(make_batch()[0])
        self.assertIn("hit", r)
        self.assertIn("confidence", r)
        self.assertIn("direction_retention", r)

    def test_expert_all_correct(self):
        cfg = load_config()
        cfg["model"]["skill"] = 1.0
        eng = Engine(cfg)
        self.assertEqual(eng.run_all()["summary"]["accuracy"], 1.0)

    def test_retention_positive(self):
        eng = Engine(load_config())
        self.assertGreater(eng.run_all()["summary"]["mean_retention"], 0.5)

    def test_decide_by_id(self):
        eng = Engine(load_config())
        r = eng.decide_by_id("route_api", skill=1.0)
        self.assertIsNotNone(r)
        self.assertTrue(r["hit"])
        self.assertIsNone(eng.decide_by_id("nope"))

    def test_render_markdown(self):
        eng = Engine(load_config())
        md = eng.render_report(eng.run_all())
        self.assertIn("Drexforge", md)


if __name__ == "__main__":
    unittest.main()
