# -*- coding: utf-8 -*-
"""Standard-library HTTP API (no external server dependency).

Endpoints (all numeric JSON -- a decision model returns probabilities, never text):
- GET  /health          -> {"ok": true, "name": "drexforge"}
- GET  /v1/info         -> backbone composition + skill
- GET  /v1/tasks        -> list of task ids / options / question types
- POST /v1/decide       -> {"task_id": "...", "skill": 0.9} -> numeric decision
- GET  /v1/bench        -> skill table + backbone ablation
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from ..core.engine import Engine
from .. import bench as benchmod
from ..utils.linalg import clamp
from ..world.corpus import make_batch


def make_handler(cfg):
    engine = Engine(cfg)
    tasks = {t["id"]: t for t in make_batch(cfg["backbone"]["dim"], cfg["backbone"]["seq_len"])}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"ok": True, "name": "drexforge", "version": "0.1.0"})
            elif self.path == "/v1/info":
                self._send(200, {
                    "name": "drexforge", "version": "0.1.0",
                    "backbone": {
                        "n_delta": engine.backbone.n_delta,
                        "n_full": engine.backbone.n_full,
                        "full_every": cfg["backbone"]["full_every"],
                        "mode": cfg["backbone"]["mode"],
                        "dim": cfg["backbone"]["dim"],
                    },
                    "skill": cfg["model"]["skill"],
                })
            elif self.path == "/v1/tasks":
                self._send(200, {
                    "tasks": [{"id": t["id"], "qtype": t["qtype"], "options": t["options"]}
                              for t in tasks.values()],
                })
            elif self.path == "/v1/bench":
                table = benchmod.run_skill_table(cfg["backbone"]["dim"], cfg["backbone"]["seq_len"])
                ablation = benchmod.run_ablation(cfg["backbone"]["dim"], cfg["backbone"]["seq_len"])
                self._send(200, {
                    "skill_table": table,
                    "skill_summary": benchmod.summarize(table),
                    "ablation": ablation,
                })
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path == "/v1/decide":
                ln = int(self.headers.get("Content-Length", 0))
                data = json.loads(self.rfile.read(ln).decode("utf-8"))
                tid = data.get("task_id")
                skill = data.get("skill")
                if skill is not None:
                    engine.model.skill = clamp(float(skill), 0.0, 1.0)
                t = tasks.get(tid)
                if t is None:
                    self._send(404, {"error": "unknown task_id"})
                    return
                self._send(200, engine.run_task(t))
            else:
                self._send(404, {"error": "not found"})

        def log_message(self, *args):
            pass

    return Handler


def serve(host="127.0.0.1", port=8431, cfg=None):
    if cfg is None:
        from ..config import load_config
        cfg = load_config()
    handler = make_handler(cfg)
    srv = HTTPServer((host, port), handler)
    print("[serve] drexforge listening on http://%s:%d" % (host, port))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
