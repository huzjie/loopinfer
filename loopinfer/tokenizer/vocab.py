# -*- coding: utf-8 -*-
"""Byte-level tokenizer: every UTF-8 byte is a token, vocab_size = 256."""

from __future__ import annotations


class ByteTokenizer:
    vocab_size = 256
    name = "byte"

    def encode(self, text: str):
        return list(text.encode("utf-8"))

    def decode(self, tokens):
        return bytes(t for t in tokens if 0 <= t < 256).decode("utf-8", errors="replace")

    def encode_single(self, ch: str):
        return list(ch.encode("utf-8"))
