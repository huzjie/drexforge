# Architecture

```
                          context (token sequence)
                                   │
                     ┌─────────────▼─────────────┐
                     │        Backbone           │
                     │  9x Gated DeltaNet (lin)   │
                     │  + 3x Full Attention (quad)│  full every 4th layer
                     │  residual + normalize      │
                     └─────────────┬─────────────┘
                                   │  pooled h
                     ┌─────────────▼─────────────┐
                     │       OptionScorer        │
                     │  align h with each option  │
                     │  embedding -> raw scores   │
                     └─────────────┬─────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
   ┌──────▼──────┐          ┌──────▼──────┐          ┌──────▼──────┐
   │   Oracle    │          │ DecisionModel│          │   Corpus    │
   │ true relevance        │ skill-blended │          │ answer-in-  │
   │ evidence(i) │          │   decide()    │          │ context     │
   └─────────────┘          └──────┬──────┘          └─────────────┘
                                   │
                        numeric decision (argmax + probs +
                        confidence + entropy)  -- no text
```

## Modules

- **Backbone** (`core/backbone.py`) — composes `n_layers` blocks: Gated DeltaNet
  everywhere except every 4th layer, which is full attention. Pools the final
  layer to one vector `h`.
- **GatedDeltaNet** (`core/deltanet.py`) — linear-time recurrent layer: a matrix
  state `S` updated by a decay-gated delta rule, read out linearly.
- **FullAttention** (`core/attention.py`) — non-causal attention over the whole
  sequence (O(n^2 d)).
- **OptionScorer** (`core/scorer.py`) — aligns `h` with orthonormal option
  embeddings; supports choice / yesno / ordinal question types.
- **DecisionModel** (`core/model.py`) — skill-parametrized decision: blends the
  oracle evidence with a noise term, then softmaxes (with a decision temperature)
  to a calibrated probability distribution.
- **Corpus / Oracle** (`world/`) — synthetic tasks whose context encodes the
  answer as the dominant direction; the oracle exposes the true relevance of
  each option (correct ≈ 1, distractor ≈ 0).
- **Engine** (`core/engine.py`) — ties it together: encode -> score -> decide,
  and reports `direction_retention = cos(h, answer_dir)` as the honest measure
  of how well the backbone recovered the answer.
