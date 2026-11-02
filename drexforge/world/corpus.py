# -*- coding: utf-8 -*-
"""Ground-truth decision corpus: contexts that encode a single answer.

Every task writes its answer into the context as the DOMINANT direction: the
context tokens are small perturbations of the correct option's embedding
direction. A perfect encoder therefore recovers the answer direction, and the
option scorer reads the right option off in one forward pass. Orthogonalizing
the option embeddings guarantees the correct option's relevance is ~1 while
every distractor sits at ~0, so the benchmark "more skill -> more correct" is
measured, not asserted.

Three question types mirror a Drex Max-style decision API:

- choice  : pick the right tool / agent / priority (argmax over options)
- yesno   : a policy / guardrail verdict (yes vs no)
- ordinal : a bounded 1..5 score
"""
from __future__ import annotations

from ..utils.rng import stable_int, stable_vector
from ..utils.linalg import normalize, add
from ..core.scorer import embed_set

CHOICE = [
    ("route_api", ["search", "python_exec", "browse", "calculator"]),
    ("tool_select", ["read_file", "run_test", "git_commit", "http_fetch"]),
    ("triage_priority", ["critical", "high", "medium", "low"]),
    ("moder_route", ["finance", "legal", "support", "engineering"]),
]
YESNO = [
    ("guardrail_pii", ["yes", "no"]),
    ("guardrail_block", ["yes", "no"]),
    ("guardrail_allow", ["yes", "no"]),
    ("guardrail_safe", ["yes", "no"]),
]
ORDINAL = [
    ("risk_score", ["1", "2", "3", "4", "5"]),
    ("relevance_score", ["1", "2", "3", "4", "5"]),
    ("confidence_score", ["1", "2", "3", "4", "5"]),
    ("severity_score", ["1", "2", "3", "4", "5"]),
]


def make_batch(dim=8, seq_len=8):
    """Build the full deterministic task list."""
    tasks = []
    for qtype, entries in (("choice", CHOICE), ("yesno", YESNO), ("ordinal", ORDINAL)):
        for tid, options in entries:
            n = len(options)
            correct = stable_int("correct:" + tid, 0, n - 1)
            embeds = embed_set(options, dim)
            answer_dir = embeds[correct]
            context = []
            for j in range(seq_len):
                noise = stable_vector("ctx:%s:%d" % (tid, j), dim, -0.3, 0.3)
                context.append(normalize(add(answer_dir, noise)))
            tasks.append({
                "id": tid,
                "qtype": qtype,
                "options": options,
                "correct": correct,
                "context": context,
                "answer_dir": answer_dir,
                "embeds": embeds,
            })
    return tasks


def task_ids(dim=8, seq_len=8):
    return [t["id"] for t in make_batch(dim, seq_len)]
