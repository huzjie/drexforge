# Benchmark

All numbers are reproducible (deterministic seeds, stdlib only). Regenerate with
`python -m drexforge bench`.

## Skill axis (0.5 random -> 1.0 expert), hybrid backbone

| skill | accuracy | choice_acc | precision | recall | fpr | ordinal_mae | ece | confidence | entropy |
|---|---|---|---|---|---|---|---|---|---|
| 0.50 | 0.250 | 0.000 | 1.000 | 0.667 | 0.000 | 2.250 | 0.440 | 0.690 | 0.700 |
| 0.60 | 0.583 | 0.500 | 1.000 | 1.000 | 0.000 | 1.500 | 0.174 | 0.683 | 0.768 |
| 0.70 | 0.917 | 1.000 | 1.000 | 1.000 | 0.000 | 0.250 | 0.153 | 0.764 | 0.651 |
| 0.80 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.093 | 0.907 | 0.359 |
| 0.90 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.024 | 0.976 | 0.134 |
| 1.00 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.007 | 0.993 | 0.047 |

Headline monotonic deltas (0.5 -> 1.0): accuracy **0.250 -> 1.000**, ordinal MAE
**2.250 -> 0.000**, calibration error (ECE) **0.440 -> 0.007**, confidence
**0.690 -> 0.993**, entropy **0.700 -> 0.047**.

## Backbone ablation (why 48 DeltaNet + 16 full attention)

32-token context, `dim = 8`, 12 layers:

| mode | delta layers | full layers | scalar ops | direction retention |
|---|---|---|---|---|
| delta | 12 | 0 | 24576 | 0.9879 |
| hybrid | 9 | 3 | 43008 | 0.8168 |
| full | 0 | 12 | 98304 | 0.5636 |

Full attention costs **4x** the scalar ops of an all-DeltaNet backbone; the
hybrid (full every 4th layer) costs only **1.75x** while keeping the pairwise
mixing that full attention provides. At the real model's 262144-token window,
that gap is ~262144x larger — which is exactly why a decision model keeps the
quadratic layers to a minority.
