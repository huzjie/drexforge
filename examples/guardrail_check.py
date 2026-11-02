# -*- coding: utf-8 -*-
"""Demo: policy / guardrail verdicts (yes/no) without generating text.

The verdict is a probability, so the caller sets its own threshold. Here we use
0.5 to turn p_yes into a crisp ALLOW / BLOCK.
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
    print("guardrail demo (skill=0.9)\n")
    for tid in ("guardrail_pii", "guardrail_block", "guardrail_allow", "guardrail_safe"):
        r = engine.decide_by_id(tid)
        verdict = "ALLOW" if r["decision_label"] == "yes" else "BLOCK"
        print("%-16s -> %-5s (p_yes confidence=%.3f, hit=%s)" % (
            tid, verdict, r["confidence"], r["hit"]))


if __name__ == "__main__":
    main()
