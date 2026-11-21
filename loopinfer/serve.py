# -*- coding: utf-8 -*-
"""OpenAI-compatible HTTP server (minimal, zero-dependency)."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from loopinfer.engine.inference import InferenceEngine
from loopinfer.config_loader import load_engine_config


def run_server(port: int = 8080, config=None):
    engine = InferenceEngine(load_engine_config(config))

    class Handler(BaseHTTPRequestHandler):
        def _json(self, code, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/v1/models":
                self._json(200, {"object": "list", "data": [{"id": "loopinfer", "object": "model"}]})
            elif self.path == "/health":
                self._json(200, {"status": "ok"})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path == "/v1/completions":
                length = int(self.headers.get("Content-Length", 0))
                data = json.loads(self.rfile.read(length).decode("utf-8"))
                prompt = data.get("prompt", "")
                max_tokens = data.get("max_tokens", 32)
                r = engine.generate(prompt, max_tokens=max_tokens)
                self._json(200, {
                    "id": "cmpl-loopinfer",
                    "object": "text_completion",
                    "choices": [{"text": r.text, "index": 0, "finish_reason": "length"}],
                })
            else:
                self._json(404, {"error": "not found"})

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"loopinfer serve: http://127.0.0.1:{port}  (OpenAI-compatible)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
