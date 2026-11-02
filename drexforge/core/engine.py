# -*- coding: utf-8 -*-
"""Engine: encode context -> score options -> decide (no text generation).

The pipeline mirrors a Drex Max-style decision model end to end:

1. ``backbone.forward(context)`` runs the hybrid DeltaNet + attention stack and
   pools a single representation ``h``.
2. ``scorer.raw_scores(h, options)`` aligns ``h`` with each option embedding --
   the honest, imperfect read of an untrained encoder.
3. ``model.decide(evidence, noise)`` blends the oracle's true relevance with the
   skill parameter to produce a calibrated option decision.

The backbone's real quality is captured by ``direction_retention = cos(h,
answer_dir)``: how well the encoder recovered the answer written into the
context. A good architecture preserves that direction; a bad one smears it.
"""
from __future__ import annotations

from .model import DecisionModel
from .backbone import Backbone
from .scorer import OptionScorer
from ..utils.linalg import cos, clamp
from ..world.corpus import make_batch
from ..world.oracle import evidence, noise


class Engine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dim = cfg["backbone"]["dim"]
        self.seq_len = cfg["backbone"]["seq_len"]
        self.backbone = Backbone(cfg)
        self.scorer = OptionScorer(self.dim, cfg["scorer"]["temperature"])
        self.model = DecisionModel(
            cfg["model"]["skill"],
            cfg["model"].get("decision_temperature", 6.0),
        )

    def run_task(self, task):
        """Encode + score + decide one task; return a numeric decision dict."""
        h = self.backbone.forward(task["context"])
        raw = self.scorer.raw_scores(h, task["options"])
        ev = evidence(task, self.scorer.temperature)
        ns = noise(task)
        dec = self.model.decide(ev, ns)
        correct = task["correct"]
        return {
            "id": task["id"],
            "qtype": task["qtype"],
            "options": task["options"],
            "correct": correct,
            "correct_label": task["options"][correct],
            "decision": dec["choice"],
            "decision_label": task["options"][dec["choice"]],
            "hit": dec["choice"] == correct,
            "confidence": round(dec["confidence"], 4),
            "entropy": round(dec["entropy"], 4),
            "direction_retention": round(cos(h, task["answer_dir"]), 4),
            "raw_scores": [round(s, 4) for s in raw],
        }

    def decide_by_id(self, task_id, skill=None):
        if skill is not None:
            self.model.skill = clamp(skill, 0.0, 1.0)
        for t in make_batch(self.dim, self.seq_len):
            if t["id"] == task_id:
                return self.run_task(t)
        return None

    def run_all(self):
        rows = [self.run_task(t) for t in make_batch(self.dim, self.seq_len)]
        n = len(rows)
        hits = sum(1 for r in rows if r["hit"])
        return {
            "rows": rows,
            "summary": {
                "tasks": n,
                "hits": hits,
                "accuracy": round(hits / n, 4),
                "mean_confidence": round(sum(r["confidence"] for r in rows) / n, 4),
                "mean_entropy": round(sum(r["entropy"] for r in rows) / n, 4),
                "mean_retention": round(sum(r["direction_retention"] for r in rows) / n, 4),
            },
        }

    def render_report(self, result):
        if self.cfg["report"]["format"] == "json":
            import json
            return json.dumps(result, ensure_ascii=False, indent=2)
        rows = result["rows"]
        s = result["summary"]
        lines = [
            "# Drexforge decision report",
            "",
            "| task | qtype | correct | decision | hit | confidence | retention |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            lines.append("| %s | %s | %s | %s | %s | %.3f | %.3f |" % (
                r["id"], r["qtype"], r["correct_label"], r["decision_label"],
                "yes" if r["hit"] else "no", r["confidence"], r["direction_retention"]))
        lines += [
            "",
            "Summary: %d tasks, %.1f%% accurate, mean confidence %.3f"
            % (s["tasks"], s["accuracy"] * 100, s["mean_confidence"]),
        ]
        return "\n".join(lines)
