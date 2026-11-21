# -*- coding: utf-8 -*-
"""Minimal leveled logger (zero-dependency)."""

from __future__ import annotations

import sys
import time

LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}


class Logger:
    def __init__(self, name: str = "loopinfer", level: str = "INFO"):
        self.name = name
        self.level = LEVELS.get(level.upper(), 20)

    def _log(self, lvl: str, msg: str):
        if LEVELS.get(lvl, 20) >= self.level:
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] [{lvl:5}] {self.name}: {msg}", file=sys.stderr)

    def debug(self, msg: str):
        self._log("DEBUG", msg)

    def info(self, msg: str):
        self._log("INFO", msg)

    def warn(self, msg: str):
        self._log("WARN", msg)

    def error(self, msg: str):
        self._log("ERROR", msg)


_default = Logger()


def get_logger(name: str = "loopinfer") -> Logger:
    return Logger(name)
