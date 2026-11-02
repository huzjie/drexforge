# -*- coding: utf-8 -*-
import unittest

from drexforge.training.trainer import SkillTrainer


class TestTrainer(unittest.TestCase):
    def test_skill_increases(self):
        t = SkillTrainer(0.5, 0.05)
        step = t.train_step()
        self.assertGreater(step["skill"], 0.5)

    def test_accuracy_improves(self):
        t = SkillTrainer(0.5, 0.5)
        step = t.train_step()
        self.assertGreaterEqual(step["accuracy_after"], step["accuracy_before"])


if __name__ == "__main__":
    unittest.main()
