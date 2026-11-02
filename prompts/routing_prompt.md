# Routing prompt — agent / tool selection

Use when a coordinator must hand a request to one of a fixed set of agents or
tools. The decision model picks an index, so the router branches on an integer.

```text
Route the request to exactly one tool. Do not explain.

Context: {request}
Options: search | python_exec | browse | calculator

Return: {"choice": <index>, "probs": [<p for each option>]}
```

## Reuse points

- Provide the option list twice: once as labels (for the model) and once as a
  fixed index order (for your code). The model must not reorder options.
- If a request spans tools, force a single top pick — a probability distribution
  already encodes the runner-up, so you get "fallback" for free.
- For routing, an ordinal *ranking* is often better than a single choice: ask for
  `probs` and use the top-k as a retry order.
