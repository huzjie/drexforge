# Ordinal prompt — bounded 1..N scores

For severity, relevance, or confidence scoring. The output is an integer level
plus the distribution, not a float.

```text
Score the item on a 1..5 scale. 1 = trivial, 5 = critical.

Item: {item}
Context: {context}

Return: {"level": <1..5 int>, "probs": [p1, p2, p3, p4, p5]}
```

## Reuse points

- Ordinal beats "rate 1-10" when the scale is bounded and labeled — the model
  scores a closed set of levels instead of regressing a float.
- Return the full distribution, then derive: `E[level] = sum(k * p_k)` for a
  weighted score, or `argmax` for a crisp label.
- For risk/severity, consider two questions instead of one: "is this critical?"
  (yes/no) + "what level?" (ordinal). The yes/no gate is usually more reliable
  than the tail of a 1..5 scale.
