# -*- coding: utf-8 -*-
"""Dump the resolved default config to a JSON file."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loopinfer.engine.config import EngineConfig


def main():
    cfg = EngineConfig()
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "config", "resolved.default.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, ensure_ascii=False, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
