# -*- coding: utf-8 -*-
import unittest

from drexforge.config import load_config, DEFAULTS


class TestConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = load_config()
        self.assertEqual(cfg["backbone"]["n_layers"], 12)
        self.assertEqual(cfg["backbone"]["full_every"], 4)
        self.assertEqual(cfg["backbone"]["mode"], "hybrid")

    def test_merge_recipe(self):
        import tempfile
        import os
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("model:\n  skill: 0.9\n")
            path = f.name
        try:
            cfg = load_config(path)
            self.assertEqual(cfg["model"]["skill"], 0.9)
            self.assertEqual(cfg["backbone"]["dim"], DEFAULTS["backbone"]["dim"])
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
