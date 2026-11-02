# Guardrail prompt — yes/no policy verdicts

Guardrails are the canonical decision-model use case: cheap, deterministic,
runnable on every message in a hot loop.

```text
You are a policy guardrail. Decide whether the content VIOLATES the policy.

Policy: never emit personally identifiable information (PII).

Content: {content}

Return: {"choice": 0|1, "probs": [p_violation, p_ok]}   # 0 = violation, 1 = ok
```

## Reuse points

- Frame the verdict as a probability, not a boolean, so the caller can set its
  own threshold (block if `p_violation >= 0.8`, escalate if `0.5..0.8`).
- Put the *policy* in the prompt, keep the *content* as a separate field — this
  keeps the guardrail auditable and lets you swap policies without retraining.
- Run several narrow guardrails (PII, safety, tone) instead of one "is this ok"
  catch-all; each returns a clean probability and failures are easier to debug.
