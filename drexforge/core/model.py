# -*- coding: utf-8 -*-
"""The skill-parametrized decision model (the reference stand-in for Drex Max).

It turns a *skill* (0.0 = adversarial, 0.5 = random, 1.0 = expert) and an
oracle *evidence* vector into a deterministic option decision, so the whole
pipeline is reproducible and the benchmark is monotonic in skill.

The key formula (same scale for signal and noise):

    boost    = (skill - 0.5) * 2                  # -1 .. +1, zero at random
    logit_i  = 0.5 + boost*(evidence_i - 0.5) + (1-|boost|)*(noise_i - 0.5)

- skill = 0.5  -> boost = 0     -> logits = noise        (coin flip)
- skill = 1.0  -> boost = +1    -> logits = evidence     (perfect)
- skill = 0.0  -> boost = -1    -> logits = 1 - evidence (anti-correlated)

`evidence_i` is the true relevance of option i (correct option near 1.0,
distractors near 0.5), produced honestly by running the backbone + scorer over
the task context. A real model does not "know" it; it approximates it, and
training raises that fidelity. We collapse that into one scalar `skill`.
"""
from __future__ import annotations

from ..utils.linalg import clamp
from .scorer import softmax, entropy


class DecisionModel:
    def __init__(self, skill=0.5, temperature=6.0):
        self.skill = clamp(skill, 0.0, 1.0)
        self.temperature = temperature

    def boost(self):
        return (self.skill - 0.5) * 2.0

    def logits(self, evidence, noise):
        b = self.boost()
        return [clamp(0.5 + b * (e - 0.5) + (1.0 - abs(b)) * (n - 0.5), 0.0, 1.0)
                for e, n in zip(evidence, noise)]

    def decide(self, evidence, noise):
        """Return the decision for one task: choice index + probs + confidence.

        A decision temperature sharpens the softmax so that, at skill 1.0,
        the correct option concentrates nearly all probability mass (calibrated
        confidence ~1.0). argmax is temperature-invariant (monotonic scaling).
        """
        logits = self.logits(evidence, noise)
        probs = softmax([self.temperature * v for v in logits])
        best = max(range(len(logits)), key=lambda i: logits[i])
        return {
            "logits": [round(v, 4) for v in logits],
            "probs": [round(p, 4) for p in probs],
            "choice": best,
            "confidence": probs[best],
            "entropy": round(entropy(probs), 4),
        }
