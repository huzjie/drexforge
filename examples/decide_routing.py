# -*- coding: utf-8 -*-
"""Demo: route a request to the right tool with a decision model (no text).

Each task returns a numeric decision -- an option index plus a probability
distribution -- instead of a generated sentence. The caller branches on the
index.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drexforge.config import load_config
from drexforge.core.engine import Engine


def main():
    cfg = load_config()
    cfg["model"]["skill"] = 0.9
    engine = Engine(cfg)
    print("routing demo (skill=0.9)\n")
    for tid in ("route_api", "tool_select", "triage_priority", "moder_route"):
        r = engine.decide_by_id(tid)
        print("%-16s -> %-12s (confidence=%.3f, hit=%s)" % (
            tid, r["decision_label"], r["confidence"], r["hit"]))


if __name__ == "__main__":
    main()
