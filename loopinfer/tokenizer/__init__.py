# -*- coding: utf-8 -*-
"""loopinfer.tokenizer public API."""

from .vocab import ByteTokenizer
from .bpe import BPETokenizer

_default = ByteTokenizer()


def get_tokenizer(name: str = "byte"):
    if name == "byte":
        return ByteTokenizer()
    if name == "bpe":
        return BPETokenizer()
    return ByteTokenizer()


def encode(text: str):
    return _default.encode(text)


def decode(tokens):
    return _default.decode(tokens)


__all__ = ["ByteTokenizer", "BPETokenizer", "get_tokenizer", "encode", "decode"]
