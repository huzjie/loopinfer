# -*- coding: utf-8 -*-
"""Tool: minimal HTTP client for the serve endpoint."""

import json
import urllib.request


def complete(base_url, prompt, max_tokens=32):
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions",
        data=json.dumps({"prompt": prompt, "max_tokens": max_tokens}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))
