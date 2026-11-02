# Choice prompt — closed-set selection

The workhorse question type: pick one from a closed set, return a full
distribution so downstream code can threshold on confidence.

```text
Given the context, choose ONE option and give a probability for every option.

Context: {context}
Options: {option_list}

Return ONLY: {"choice": <index>, "probs": [...]}  (probs sum to 1)
```

## Reuse points

- Always ask for the full distribution, not just the argmax. The runner-up
  probability is your uncertainty signal and drives fallback / human escalation.
- Keep the option set small (<= ~8); decision models score a closed set in one
  pass, so more options cost more embedding columns, not more decode steps.
- If two options are near-tied (top prob < 0.5), treat it as "needs escalation"
  rather than committing to the argmax.
