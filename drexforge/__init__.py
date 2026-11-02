# -*- coding: utf-8 -*-
"""drexforge: a zero-dependency reference implementation of a direct-scoring
decision model (Drex Max style).

When Nace AI open-sourced Drex Max v1, the headline was the *decision model*
paradigm itself: instead of generating text, the model reads a context (text,
JSON or images) plus a set of named options and returns a probability for every
option in a single forward pass. You get argmax / threshold / scale directly --
no sampling, no temperature over tokens, no decode loop.

drexforge implements the two ideas that make that paradigm work:

1. **Hybrid backbone (Gated DeltaNet + full attention)** -- a dense stack where
   most layers are linear-time Gated DeltaNet (recurrent, long-context) and
   every fourth layer is full attention (precise pairwise mixing). 48 DeltaNet
   + 16 full layers is what keeps a 262144-token window tractable.
2. **Option scoring head (no text generation)** -- one pooled representation is
   aligned against each option embedding, softmaxed, and returned as a
   probability distribution, with three question types: choice / yesno /
   ordinal.

The "model" is a trainable, skill-parametrized scorer (deterministic, seeded,
monotonic in skill) layered over real vector math, so the whole encode ->
score -> decide pipeline runs end-to-end on a laptop in seconds, on the Python
standard library only.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
