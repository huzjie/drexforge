# -*- coding: utf-8 -*-
"""Full (non-causal) attention: precise token-to-token mixing.

Drex Max places a full-attention layer every fourth layer of its backbone. The
other three-quarters are Gated DeltaNet (linear-time). Full attention is *non-
causal*: every token attends to every other token in the sequence, which gives
precise pairwise relationships at the cost of an O(N^2 * d) attention matrix.

This module is the full-attention side, implemented with plain Python lists.
"""
from __future__ import annotations

import math

from ..utils.rng import stable_matrix
from ..utils.linalg import normalize, add


def softmax_scores(scores):
    """Numerically stable softmax over a row of scores."""
    m = max(scores)
    ex = [math.exp(s - m) for s in scores]
    s = sum(ex)
    return [e / s for e in ex]


class FullAttention:
    def __init__(self, dim, seed):
        self.dim = dim
        self.seed = seed
        self.W_q = stable_matrix("%s:W_q" % seed, dim, dim, -0.25, 0.25)
        self.W_k = stable_matrix("%s:W_k" % seed, dim, dim, -0.25, 0.25)
        self.W_v = stable_matrix("%s:W_v" % seed, dim, dim, -0.25, 0.25)

    def _proj(self, W, x):
        return [sum(W[i][j] * x[i] for i in range(len(x))) for j in range(len(W[0]))]

    def forward(self, tokens):
        """Non-causal attention over the whole sequence; return outputs."""
        Q = [self._proj(self.W_q, x) for x in tokens]
        K = [self._proj(self.W_k, x) for x in tokens]
        V = [self._proj(self.W_v, x) for x in tokens]
        d = self.dim
        scale = 1.0 / math.sqrt(d)
        outs = []
        for i in range(len(tokens)):
            scores = [scale * sum(Q[i][k] * K[j][k] for k in range(d)) for j in range(len(tokens))]
            w = softmax_scores(scores)
            o = [0.0] * d
            for j in range(len(tokens)):
                for k in range(d):
                    o[k] += w[j] * V[j][k]
            outs.append(normalize(add(tokens[i], o)))
        return outs

    @staticmethod
    def ops(seq_len, dim):
        """Approximate scalar op count: quadratic in sequence length."""
        return seq_len * seq_len * dim
