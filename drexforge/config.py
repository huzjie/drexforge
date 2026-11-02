# -*- coding: utf-8 -*-
"""Runtime configuration with layered defaults (code default < recipe yaml < cli)."""
from __future__ import annotations

import copy

from .utils.yamlish import yaml_loads

# Defaults are deliberately small so a reference run stays interactive
# (<10s on a laptop) while still exercising every code path. The real Drex Max
# uses 64 layers (48 DeltaNet + 16 full) over a 262144-token window; the
# reference keeps the same cadence (full attention every 4th layer) at a
# laptop-friendly size.
DEFAULTS = {
    "backbone": {
        "dim": 8,                # token / hidden dimension
        "n_layers": 12,          # total layers -> 9 delta + 3 full (every 4th)
        "full_every": 4,         # place full attention every N-th layer
        "seq_len": 32,           # context tokens per task (long enough to show
                                 # the quadratic vs linear cost gap)
        "pool": "mean",          # mean | last (how to pool the final layer)
        "mode": "hybrid",        # hybrid | delta | full (ablation)
    },
    "scorer": {
        "temperature": 6.0,      # evidence sharpness: sigmoid(T * (2*cos - 1))
    },
    "model": {
        "skill": 0.5,            # 0.0 (adversarial) .. 1.0 (expert); trainable
        "decision_temperature": 6.0,  # softmax sharpness -> calibrated confidence
    },
    "engine": {},
    "report": {
        "format": "markdown",    # markdown | json
        "out_dir": "reports",
    },
}


def _merge(base, override):
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(recipe_path=None):
    """Return the merged config dict. A recipe (yaml subset) overrides defaults."""
    cfg = copy.deepcopy(DEFAULTS)
    if recipe_path:
        with open(recipe_path, "r", encoding="utf-8") as f:
            raw = yaml_loads(f.read())
        cfg = _merge(cfg, raw)
    return cfg
