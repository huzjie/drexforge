# -*- coding: utf-8 -*-
"""Reproducible benchmark: one monotonic axis (skill) + one ablation axis.

Axis A -- model skill (0.5 random -> 1.0 expert):
- accuracy         (up)   fraction of decisions correct
- choice_acc       (up)   routing / tool-selection accuracy
- precision/recall (up)   guardrail (yes/no) precision & recall
- fpr              (down) guardrail false-positive rate
- ordinal_mae      (down) mean absolute error of 1..5 scores
- ece              (down) expected calibration error (confidence vs accuracy)
- confidence       (up)   decision confidence
- entropy          (down) decision entropy

Axis B -- backbone variant (delta | hybrid | full), the "why 48 DeltaNet + 16
full attention" ablation: linear-time DeltaNet layers vs quadratic full
attention vs the hybrid that places full attention every 4th layer.
"""
from __future__ import annotations

from .core.engine import Engine
from .core.backbone import Backbone
from .config import load_config
from .world.corpus import make_batch
from .utils.linalg import cos

SKILLS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def _ece(rows, nbins=5):
    bins = [[] for _ in range(nbins)]
    for r in rows:
        b = min(nbins - 1, int(r["confidence"] * nbins))
        bins[b].append(r)
    total = len(rows)
    ece = 0.0
    for b in bins:
        if not b:
            continue
        conf = sum(r["confidence"] for r in b) / len(b)
        acc = sum(1 for r in b if r["hit"]) / len(b)
        ece += (len(b) / total) * abs(conf - acc)
    return ece


def _metrics(skill, dim=8, seq_len=8):
    cfg = load_config()
    cfg["model"]["skill"] = skill
    engine = Engine(cfg)
    result = engine.run_all()
    rows = result["rows"]
    n = len(rows)
    acc = sum(1 for r in rows if r["hit"]) / n
    choice = [r for r in rows if r["qtype"] == "choice"]
    yesno = [r for r in rows if r["qtype"] == "yesno"]
    ordinal = [r for r in rows if r["qtype"] == "ordinal"]
    choice_acc = sum(1 for r in choice if r["hit"]) / len(choice)
    tp = fp = tn = fn = 0
    for r in yesno:
        pos_pred = r["decision"] == 0
        pos_true = r["correct"] == 0
        if pos_pred and pos_true:
            tp += 1
        elif pos_pred:
            fp += 1
        elif pos_true:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    mae = sum(abs(r["decision"] - r["correct"]) for r in ordinal) / len(ordinal)
    ordinal_acc = sum(1 for r in ordinal if r["hit"]) / len(ordinal)
    return {
        "accuracy": round(acc, 3),
        "choice_acc": round(choice_acc, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "fpr": round(fpr, 3),
        "ordinal_mae": round(mae, 3),
        "ordinal_acc": round(ordinal_acc, 3),
        "ece": round(_ece(rows), 3),
        "confidence": round(sum(r["confidence"] for r in rows) / n, 3),
        "entropy": round(sum(r["entropy"] for r in rows) / n, 3),
    }


def _cfg_dims():
    cfg = load_config()
    return cfg["backbone"]["dim"], cfg["backbone"]["seq_len"]


def run_skill_table(dim=None, seq_len=None):
    if dim is None or seq_len is None:
        d, s = _cfg_dims()
        dim = dim or d
        seq_len = seq_len or s
    rows = []
    for s in SKILLS:
        m = _metrics(s, dim, seq_len)
        m["skill"] = s
        rows.append(m)
    return rows


def run_ablation(dim=None, seq_len=None):
    if dim is None or seq_len is None:
        d, s = _cfg_dims()
        dim = dim or d
        seq_len = seq_len or s
    batch = make_batch(dim, seq_len)
    rows = []
    for mode in ("delta", "hybrid", "full"):
        cfg = load_config()
        cfg["backbone"]["mode"] = mode
        bb = Backbone(cfg)
        rets = [cos(bb.forward(t["context"]), t["answer_dir"]) for t in batch]
        rows.append({
            "mode": mode,
            "n_delta": bb.n_delta,
            "n_full": bb.n_full,
            "total_ops": bb.total_ops(seq_len),
            "mean_retention": round(sum(rets) / len(rets), 4),
        })
    return rows


def summarize(rows):
    """Endpoint deltas (skill 0.5 -> 1.0) for the README."""
    first, last = rows[0], rows[-1]
    return {
        "accuracy": (first["accuracy"], last["accuracy"]),
        "choice_acc": (first["choice_acc"], last["choice_acc"]),
        "precision": (first["precision"], last["precision"]),
        "recall": (first["recall"], last["recall"]),
        "fpr": (first["fpr"], last["fpr"]),
        "ordinal_mae": (first["ordinal_mae"], last["ordinal_mae"]),
        "ece": (first["ece"], last["ece"]),
        "confidence": (first["confidence"], last["confidence"]),
        "entropy": (first["entropy"], last["entropy"]),
    }
