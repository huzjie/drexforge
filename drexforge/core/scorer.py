# -*- coding: utf-8 -*-
"""Option scoring head: one forward pass -> probabilities, never text.

The defining trait of a decision model is that it does NOT generate text. Given
a pooled backbone representation `h` and a set of named options, it returns a
probability for every option in a single pass, then you take argmax / threshold
/ scale. Three question types are supported:

- choice  : softmax over named options (distribution sums to 1)
- yesno   : sigmoid over the "yes" logit (no is the complement)
- ordinal : map a bounded scalar to an integer level

Shared helpers softmax / entropy / logsumexp live here so the decision model
and the scorer stay in one import graph (no cycles).
"""
from __future__ import annotations

import math

from ..utils.linalg import dot, normalize, norm2
from ..utils.rng import stable_vector

_EPS = 1e-9


def softmax(logits):
    m = max(logits)
    ex = [math.exp(v - m) for v in logits]
    s = sum(ex) + _EPS
    return [e / s for e in ex]


def entropy(probs):
    return -sum(p * math.log(p + _EPS) for p in probs)


def logsumexp(logits):
    m = max(logits)
    return m + math.log(sum(math.exp(v - m) for v in logits) + _EPS)


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def embed_set(names, dim):
    """Orthogonalized option embeddings (Gram-Schmidt), deterministic.

    Orthogonalizing options guarantees a clean margin: the correct option's
    embedding has cosine ~1 with the answer direction while every distractor
    has cosine exactly 0, so argmax over evidence is provably correct.
    """
    vecs = [stable_vector("opt:" + name, dim, -1.0, 1.0) for name in names]
    out = []
    for v in vecs:
        w = list(v)
        for u in out:
            c = dot(w, u)
            w = [wi - c * ui for wi, ui in zip(w, u)]
        n = norm2(w)
        if n < 1e-9:
            w = list(v)
            n = norm2(w)
        out.append([wi / n for wi in w])
    return out


class OptionScorer:
    """Scores options by aligning the pooled vector h with each option embedding."""

    def __init__(self, dim, temperature=6.0):
        self.dim = dim
        self.temperature = temperature

    def embed(self, name):
        return embed_set([name], self.dim)[0]

    def raw_scores(self, h, options):
        hn = normalize(h)
        embeds = embed_set(options, self.dim)
        return [dot(hn, e) for e in embeds]

    def score(self, h, options, qtype="choice"):
        """Return a decision dict for a question of type `qtype`.

        No text is generated anywhere: the result is probabilities + argmax +
        confidence + entropy, which downstream code consumes numerically.
        """
        scores = self.raw_scores(h, options)
        if qtype == "choice":
            probs = softmax([self.temperature * s for s in scores])
            best = max(range(len(options)), key=lambda i: scores[i])
            return {
                "qtype": "choice",
                "options": options,
                "scores": [round(s, 4) for s in scores],
                "probs": [round(p, 4) for p in probs],
                "choice": best,
                "label": options[best],
                "confidence": probs[best],
                "entropy": round(entropy(probs), 4),
            }
        if qtype == "yesno":
            # single "yes" logit; "no" is the complement
            p_yes = sigmoid(self.temperature * scores[0])
            best = 0 if p_yes >= 0.5 else 1
            probs = [p_yes, 1.0 - p_yes]
            return {
                "qtype": "yesno",
                "options": options,
                "scores": [round(s, 4) for s in scores],
                "probs": [round(p, 4) for p in probs],
                "choice": best,
                "label": options[best],
                "confidence": probs[best],
                "entropy": round(entropy(probs), 4),
            }
        if qtype == "ordinal":
            # bounded scalar mapped to the nearest integer level
            s = scores[0]                      # in [-1, 1]
            scaled = (s + 1.0) / 2.0           # -> [0, 1]
            n = len(options)
            raw = scaled * (n - 1) + 1.0       # -> [1, n]
            level = int(round(min(n, max(1, raw))))
            return {
                "qtype": "ordinal",
                "options": options,
                "scores": [round(s, 4)],
                "level": level,
                "label": options[level - 1],
                "confidence": round(min(1.0, max(0.0, 1.0 - abs(raw - level))), 4),
                "entropy": round(entropy([1.0]), 4),
            }
        raise ValueError("unknown qtype: %s" % qtype)
