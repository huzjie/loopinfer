# -*- coding: utf-8 -*-
"""Conversation session state."""

from __future__ import annotations


class Session:
    def __init__(self, engine, session_id: str = "default"):
        self.engine = engine
        self.session_id = session_id
        self.history = []

    def chat(self, text: str, max_tokens: int = 32):
        r = self.engine.generate(text, max_tokens=max_tokens)
        self.history.append({"role": "user", "content": text})
        self.history.append({"role": "assistant", "content": r.text})
        return r

    def transcript(self):
        return "\n".join(f"{h['role']}: {h['content']}" for h in self.history)
