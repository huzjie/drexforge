# -*- coding: utf-8 -*-
"""Oracle: ground-truth relevance per option, plus per-task decision noise.

The oracle exposes the "true relevance" a perfect decision model would output
for every option. Because the corpus writes the answer into the context's
dominant direction and orthogonalizes the option embeddings, relevance is
exactly ``sigmoid(T)`` for the correct option and ``sigmoid(-T)`` for every
distractor -- a clean, deterministic margin that makes the skill benchmark
monotonic and provable rather than asserted.
"""
from __future__ import annotations

from ..utils.linalg import dot
from ..utils.rng import stable_float
from ..core.scorer import sigmoid
from .corpus import make_batch


def evidence(task, temperature=6.0):
    """True relevance of each option: correct ~1.0, distractors ~0.0."""
    return [sigmoid(temperature * (2.0 * dot(task["answer_dir"], task["embeds"][i]) - 1.0))
            for i in range(len(task["options"]))]


def noise(task):
    """Deterministic per-option pseudo-noise in [0, 1]."""
    return [stable_float("noise:%s:%d" % (task["id"], i))
            for i in range(len(task["options"]))]


def correct(task):
    return task["correct"]


def make_batch_alias(dim=8, seq_len=8):
    """Re-export the corpus batch (single entry point for callers)."""
    return make_batch(dim, seq_len)
