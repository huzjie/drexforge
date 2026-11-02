# -*- coding: utf-8 -*-
import unittest

from drexforge.world.corpus import make_batch


class TestCorpus(unittest.TestCase):
    def test_batch_size(self):
        self.assertEqual(len(make_batch()), 12)

    def test_three_qtypes(self):
        batch = make_batch()
        self.assertEqual({t["qtype"] for t in batch}, {"choice", "yesno", "ordinal"})

    def test_context_unit_norm(self):
        t = make_batch()[0]
        for tok in t["context"]:
            self.assertAlmostEqual(sum(v * v for v in tok) ** 0.5, 1.0, places=3)

    def test_answer_dir_is_correct_embed(self):
        t = make_batch()[0]
        self.assertEqual(t["answer_dir"], t["embeds"][t["correct"]])

    def test_embeds_orthonormal(self):
        t = make_batch()[0]
        for i in range(len(t["embeds"])):
            for j in range(i + 1, len(t["embeds"])):
                d = sum(a * b for a, b in zip(t["embeds"][i], t["embeds"][j]))
                self.assertAlmostEqual(d, 0.0, places=3)

    def test_correct_in_range(self):
        t = make_batch()[0]
        self.assertGreaterEqual(t["correct"], 0)
        self.assertLess(t["correct"], len(t["options"]))

    def test_deterministic(self):
        self.assertEqual(make_batch()[0]["correct"], make_batch()[0]["correct"])


if __name__ == "__main__":
    unittest.main()
