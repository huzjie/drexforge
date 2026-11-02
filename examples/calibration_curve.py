# -*- coding: utf-8 -*-
"""Demo: calibration -- confidence vs accuracy across skill levels.

Expected Calibration Error (ECE) measures how well confidence tracks accuracy.
At skill 0.5 the model is random-but-overconfident (high ECE); as skill rises
to 1.0, confidence and accuracy converge and ECE falls toward 0.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drexforge import bench


def main():
    rows = bench.run_skill_table()
    print("calibration demo\n")
    print("%-6s %-10s %-10s %-8s" % ("skill", "accuracy", "confidence", "ece"))
    for r in rows:
        print("%-6.2f %-10.3f %-10.3f %-8.3f" % (
            r["skill"], r["accuracy"], r["confidence"], r["ece"]))
    print()
    print("At skill 0.5 the model is random but overconfident (high ECE);")
    print("as skill -> 1.0, confidence and accuracy converge (ECE -> 0).")


if __name__ == "__main__":
    main()
