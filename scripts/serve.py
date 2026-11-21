# -*- coding: utf-8 -*-
"""Start the OpenAI-compatible HTTP server."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loopinfer.serve import run_server


if __name__ == "__main__":
    run_server(8080)
