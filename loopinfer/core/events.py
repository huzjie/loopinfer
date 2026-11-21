# -*- coding: utf-8 -*-
"""Lightweight event emitter for instrumentation hooks."""

from __future__ import annotations


class EventBus:
    def __init__(self):
        self._handlers = {}

    def on(self, event: str, handler):
        self._handlers.setdefault(event, []).append(handler)
        return handler

    def emit(self, event: str, **payload):
        for handler in self._handlers.get(event, []):
            handler(payload)
        return payload

    def clear(self):
        self._handlers.clear()


_bus = EventBus()

on = _bus.on
emit = _bus.emit
clear = _bus.clear
