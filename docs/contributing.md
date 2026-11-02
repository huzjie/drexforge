# Contributing

1. Keep it zero-dependency (stdlib only).
2. Everything deterministic: use `utils/rng.py` seeded helpers.
3. Benchmarks must be monotonic in skill; add a test if you add a metric.
4. The option scorer must never produce text — only probabilities and indices.
5. Run `python -m unittest discover -s tests -t .` before contributing.
