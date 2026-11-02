# Gated DeltaNet: linear-time recurrent attention

The core trick that lets a decision model read a 262144-token window on one
machine is *not* attention over all pairs — that is O(n^2 d). It is a recurrent
layer that updates a fixed-size state once per token, so cost is O(n d^2).

## The update rule

```
S_t = beta_t * S_{t-1} + outer(v_t, k_t)     # decay-gated write
o_t = S_t @ q_t                              # linear read
```

where `k, q, v` are projections of the input token `x_t`, and `beta_t` is a
forget gate in `(0, 1)`.

- `beta -> 1` : cumulative memory — every token accumulates into `S`.
- `beta -> 0` : memoryless — only the current token is written.
- linear time  : `S` is a `d x d` matrix, updated once per token. No `n x n`
  attention matrix is ever materialized.

## Reusable implementation notes

1. **Gate the decay, not the write.** A scalar `beta` that multiplies the whole
   previous state is cheap and enough to model "forget this much of the past".
2. **Delta rule = erase-then-write.** The useful property over a plain RNN is
   that the state is a *linear* combination of `outer(v_t, k_t)` terms, so the
   read `S @ q` reduces to `sum_t (k_t . q) v_t` — exactly linear attention.
3. **Residual + normalize.** To keep the dominant input direction intact across
   many layers (so the encoder *recovers* the answer instead of scrambling it),
   return `normalize(x_t + o_t)` rather than raw `o_t`.
4. **Measure forgetting, don't assert it.** Expose the realized `min/max` of
   `beta_t` (see `beta_span`) so "the gate forgets" is a measurement.

See `drexforge/core/deltanet.py`.
