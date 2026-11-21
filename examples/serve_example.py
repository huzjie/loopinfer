# -*- coding: utf-8 -*-
"""Example: start the HTTP server."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loopinfer.serve import run_server


if __name__ == "__main__":
    print("start http://127.0.0.1:8080  (OpenAI-compatible)")
    run_server(8080)
