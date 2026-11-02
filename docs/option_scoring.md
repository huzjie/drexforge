# Option scoring: one forward pass, no text

The defining trait of a decision model is that it never generates a sequence.
It takes a context and a set of named options and returns a probability for each
option in a single pass.

## Three question types

| type | math | decision |
|---|---|---|
| choice | softmax over option logits | argmax |
| yesno | sigmoid over the "yes" logit (no = complement) | `p_yes >= 0.5` |
| ordinal | bounded scalar mapped to nearest integer level | `round(level)` |

## How the reference does it

1. Pool the backbone output to one vector `h`.
2. Embed each option name to an **orthonormal** vector (Gram-Schmidt over the
   seeded option embeddings).
3. `logit_i = cos(h, embed_i)`; sharpen with a temperature.

Orthonormalizing options is the trick that makes the margin clean: the correct
option's embedding has cosine ~1 with the answer direction, every distractor has
cosine exactly 0, so argmax is provably correct.

## Reusable prompt/format template

Decision models are usually fed a typed question. A clean request shape:

```json
{
  "context": "<the document / ticket / state to read>",
  "question": {
    "type": "choice",
    "text": "Which tool should handle this request?",
    "options": ["search", "python_exec", "browse", "calculator"]
  }
}
```

The model returns `{choice, probability, logprobs}` — no string. Code that calls
it branches on the index, not on parsed text, which is what makes decision models
cheap, deterministic, and easy to put inside a hot loop (agent routing, ticket
triage, guardrails, tool selection).

See `drexforge/core/scorer.py` and `prompts/`.
