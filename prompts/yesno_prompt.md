# Yes/no prompt — binary decisions

Yes/no is a two-option choice under the hood, but it deserves its own template
because the semantics (a threshold) differ from "pick the best".

```text
Answer YES or NO only, as a probability.

Question: {question}
Context: {context}

Return: {"choice": 0|1, "probs": [p_yes, p_no]}   # 0 = yes, 1 = no
```

## Reuse points

- `p_yes` and `p_no` are complements — only one number is really free. Use
  `sigmoid` on a single logit rather than softmax over two.
- Threshold in the CALLER, not in the prompt: `yes if p_yes >= 0.5`. This keeps
  the decision reversible and lets you tune precision/recall by moving the bar.
- For asymmetric costs (blocking wrongly is worse than missing), expose the raw
  `p_yes` and let the caller apply the cost matrix.
