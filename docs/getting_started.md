# Getting Started

drexforge is a zero-dependency (stdlib-only) reference implementation of a
**direct-scoring decision model** in the style of Nace AI's Drex Max: it reads a
context plus a set of named options and returns a probability for every option
in one forward pass — no text generation, no sampling, no decode loop.

## Quick start

```bash
python -m drexforge doctor     # smoke: every decision correct at skill 1.0
python -m drexforge bench      # monotonic skill benchmark + backbone ablation
python -m drexforge decide route_api --skill 0.9   # single numeric decision
python -m drexforge run        # run all tasks, render a report
python -m drexforge train      # monotonic skill-training loop
```

## Serve over HTTP

```bash
python -m drexforge serve --port 8431
curl http://127.0.0.1:8431/health
curl http://127.0.0.1:8431/v1/tasks
curl -X POST http://127.0.0.1:8431/v1/decide -d '{"task_id":"route_api","skill":0.9}'
```

## Run the examples

```bash
python examples/decide_routing.py
python examples/guardrail_check.py
python examples/calibration_curve.py
```

No pip install required — everything imports from the Python standard library.
