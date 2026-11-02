# Examples

Runnable, dependency-free demonstrations of drexforge's decision model.

| Script | What it shows |
|---|---|
| `decide_routing.py` | Route requests to the right tool — argmax over options, no text. |
| `guardrail_check.py` | Yes/no policy verdicts with a confidence threshold. |
| `calibration_curve.py` | Confidence vs accuracy across skill (ECE falls to ~0). |

Run from the repo root:

```bash
python examples/decide_routing.py
python examples/guardrail_check.py
python examples/calibration_curve.py
```
