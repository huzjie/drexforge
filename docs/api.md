# HTTP API

Standard-library HTTP server, no external dependency. Every response is numeric
JSON — a decision model returns probabilities and indices, never text.

| Method | Path | Description |
|---|---|---|
| GET | `/health` | `{"ok": true, "name": "drexforge", ...}` |
| GET | `/v1/info` | backbone composition (n_delta/n_full/full_every/mode) + skill |
| GET | `/v1/tasks` | list of task ids, question types, and options |
| POST | `/v1/decide` | `{"task_id": "...", "skill": 0.9}` -> numeric decision |
| GET | `/v1/bench` | skill table + backbone ablation |

Start with `python -m drexforge serve --port 8431`.

Example decision response:

```json
{
  "id": "route_api",
  "qtype": "choice",
  "options": ["search", "python_exec", "browse", "calculator"],
  "decision": 0,
  "decision_label": "search",
  "confidence": 0.993,
  "entropy": 0.047,
  "hit": true
}
```
