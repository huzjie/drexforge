# -*- coding: utf-8 -*-
"""Hybrid backbone: Gated DeltaNet + full attention, full layer every 4th.

This is the core architectural decision of a Drex Max-style decision model: a
*dense* stack where most layers are linear-time Gated DeltaNet (long context,
cheap) and every fourth layer is full attention (precise pairwise mixing).
The reference default is 12 layers = 9 DeltaNet + 3 full attention, keeping the
"full every fourth" cadence (layers 4, 8, 12).

The final representation `h` is a pooled vector over the last layer's outputs
(mean or last), which the option scorer consumes in a single forward pass.
"""
from __future__ import annotations

from .deltanet import GatedDeltaNet
from .attention import FullAttention


class Backbone:
    def __init__(self, cfg, mode=None):
        self.dim = cfg["backbone"]["dim"]
        self.n_layers = cfg["backbone"]["n_layers"]
        self.full_every = cfg["backbone"]["full_every"]
        self.pool = cfg["backbone"].get("pool", "mean")
        mode = mode or cfg["backbone"].get("mode", "hybrid")
        self.layers = []
        for i in range(self.n_layers):
            seed = "backbone:L%d" % i
            if mode == "delta":
                kind = "delta"
            elif mode == "full":
                kind = "full"
            else:
                kind = "full" if (i + 1) % self.full_every == 0 else "delta"
            if kind == "full":
                self.layers.append(("full", FullAttention(self.dim, seed)))
            else:
                self.layers.append(("delta", GatedDeltaNet(self.dim, seed)))

    @property
    def n_delta(self):
        return sum(1 for k, _ in self.layers if k == "delta")

    @property
    def n_full(self):
        return sum(1 for k, _ in self.layers if k == "full")

    def forward(self, tokens):
        """Run tokens through every layer; return the pooled final vector."""
        h = tokens
        for _kind, layer in self.layers:
            if _kind == "delta":
                h, _state = layer.forward(h)
            else:
                h = layer.forward(h)
        if self.pool == "last":
            return h[-1]
        d = self.dim
        return [sum(row[k] for row in h) / len(h) for k in range(d)]

    def total_ops(self, seq_len):
        """Total scalar ops, separated by delta (linear) vs full (quadratic)."""
        d = seq_len * self.dim * self.dim
        f = seq_len * seq_len * self.dim
        return self.n_delta * d + self.n_full * f
