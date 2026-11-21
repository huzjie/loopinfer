# -*- coding: utf-8 -*-
"""Generic component registry."""

from __future__ import annotations


class Registry:
    def __init__(self, name: str):
        self.name = name
        self._items = {}

    def register(self, key: str, item):
        self._items[key] = item
        return item

    def get(self, key: str, default=None):
        return self._items.get(key, default)

    def keys(self):
        return sorted(self._items.keys())

    def __contains__(self, key: str):
        return key in self._items

    def __len__(self):
        return len(self._items)

    def __repr__(self):
        return f"Registry({self.name}, {len(self._items)} items)"


_MODELS = Registry("models")
_SAMPLERS = Registry("samplers")
_DRAFTERS = Registry("drafters")
