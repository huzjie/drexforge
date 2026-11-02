# -*- coding: utf-8 -*-
"""Gated DeltaNet linear attention: a recurrent, linear-time sequence encoder.

Drex Max's backbone interleaves Gated DeltaNet layers (linear-time recurrent)
with full attention layers. This module is the DeltaNet side: a matrix *state*
updated by a decay-gated delta rule and read out by a linear-attention query,
so processing N tokens costs O(N * d^2) instead of the O(N^2 * d) of full
attention. That is precisely what keeps a 262144-token context tractable on
one machine.

State update (per token t)::

    S_t = beta_t * S_{t-1} + outer(v_t, k_t)     # decay-gated write
    o_t = matvec(S_t, q_t)                       # linear read

where::

    k_t, q_t, v_t = W_k x_t, W_q x_t, W_v x_t   # projections
    beta_t        = sigmoid(W_beta . x_t + b)    # forget gate in (0, 1)

Properties that make it a DeltaNet-style recurrence:

- beta -> 1 : cumulative memory (S accumulates every token, long context)
- beta -> 0 : memoryless      (only the current token is written)
- linear time : no N x N attention matrix is ever materialized

The delta rule's value is the *forget gate beta*: it decides, per token, how
much of the past state survives, so the model can keep long-range structure
while still erasing stale content. Everything below is pure-Python vector math
(no numpy / torch), seeded deterministically.
"""
from __future__ import annotations

import math

from ..utils.linalg import matvec, clamp, normalize, add
from ..utils.rng import stable_vector


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def outer(a, b):
    """Outer product a (x) b -> matrix[i][j] = a[i] * b[j]."""
    return [[av * bj for bj in b] for av in a]


class GatedDeltaNet:
    """A single Gated DeltaNet layer over a `dim`-dimensional token space."""

    def __init__(self, dim, seed, decay_bias=2.0):
        self.dim = dim
        self.seed = seed
        self.decay_bias = decay_bias
        # deterministic projections (W^T x means we index [j][i] to go in->out)
        self.W_k = self._mat("W_k", dim, dim)
        self.W_q = self._mat("W_q", dim, dim)
        self.W_v = self._mat("W_v", dim, dim)
        self.W_beta = stable_vector("%s:W_beta" % seed, dim, -1.0, 1.0)
        self.b_beta = 0.0

    def _mat(self, name, rows, cols):
        from ..utils.rng import stable_matrix
        return stable_matrix("%s:%s" % (self.seed, name), rows, cols, -0.25, 0.25)

    def _proj(self, W, x):
        """W^T x : out[j] = sum_i W[i][j] * x[i]  (i in, j out)."""
        return [sum(W[i][j] * x[i] for i in range(len(x))) for j in range(len(W[0]))]

    def forget_gate(self, x):
        return sigmoid(sum(w * v for w, v in zip(self.W_beta, x)) + self.b_beta)

    def forward(self, tokens):
        """Encode a token sequence; return (residual outputs, final_state).

        A residual + normalization keeps the dominant input direction intact so
        the encoder *recovers* the answer written into the context rather than
        scrambling it.
        """
        d = self.dim
        S = [[0.0] * d for _ in range(d)]
        outputs = []
        for x in tokens:
            k = self._proj(self.W_k, x)
            q = self._proj(self.W_q, x)
            v = self._proj(self.W_v, x)
            beta = self.forget_gate(x)
            # decay-gated write: S = beta * S + v (x) k
            S = [[beta * S[i][j] + v[i] * k[j] for j in range(d)] for i in range(d)]
            o = matvec(S, q)
            outputs.append(normalize(add(x, o)))
        return outputs, S

    @staticmethod
    def ops(seq_len, dim):
        """Approximate scalar op count: linear in sequence length."""
        return seq_len * dim * dim


def beta_span(seq_len, dim, decay_bias=2.0, seed="span"):
    """Return (min_beta, max_beta) actually produced over a sample sequence.

    With a positive decay_bias the forget gate sits near 1 (cumulative memory);
    this helper exposes the realized span so the "forgetting" behaviour is
    measurable rather than asserted.
    """
    layer = GatedDeltaNet(dim, seed, decay_bias=decay_bias)
    tokens = [stable_vector("t%d:%s" % (i, seed), dim, -0.5, 0.5) for i in range(seq_len)]
    betas = [layer.forget_gate(x) for x in tokens]
    return min(betas), max(betas)
