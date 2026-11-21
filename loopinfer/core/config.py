# -*- coding: utf-8 -*-
"""Minimal config loading: JSON + a tiny recursive-descent YAML subset."""

from __future__ import annotations

import json
from pathlib import Path


class ConfigError(Exception):
    pass


def _strip_comment(line: str) -> str:
    # strip inline comments, but respect quoted strings naively (no quotes in keys/values we use)
    in_q = False
    q = None
    for i, ch in enumerate(line):
        if ch in ('"', "'"):
            if not in_q:
                in_q, q = True, ch
            elif ch == q:
                in_q, q = False, None
        elif ch == "#" and not in_q:
            return line[:i].rstrip()
    return line.rstrip()


def _parse_scalar(s: str):
    s = s.strip()
    if s == "":
        return None
    if s == "true":
        return True
    if s == "false":
        return False
    if s == "null" or s == "~":
        return None
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    if s.startswith("'") and s.endswith("'"):
        return s[1:-1]
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _parse_yaml(text: str):
    lines = [ln for ln in text.splitlines()]
    # strip comments (line-level, conservative)
    lines = [_strip_comment(ln) for ln in lines]
    lines = [ln.rstrip() for ln in lines]
    # drop blank lines while keeping line numbers stable via index map
    idx_of = []
    cleaned = []
    for i, ln in enumerate(lines):
        if ln.strip() == "" or ln.lstrip().startswith("---"):
            continue
        cleaned.append(ln)
        idx_of.append(i)

    root = {}
    stack = [(root, -1)]

    def indent_of(ln: str) -> int:
        return len(ln) - len(ln.lstrip(" "))

    i = 0
    n = len(cleaned)
    while i < n:
        ln = cleaned[i]
        ind = indent_of(ln)
        content = ln.strip()
        if content.startswith("- "):
            # list item
            item_text = content[2:].strip()
            while stack and stack[-1][1] >= ind:
                stack.pop()
            parent = stack[-1][0]
            if not isinstance(parent, list):
                # a sequence: create a list container
                # parent is a dict; need a key -> handled below, but here assume we
                # are building a list under a key. We keep a sentinel.
                raise ConfigError("unexpected list item at: " + content)
            if ":" in item_text and not (item_text.startswith('"') or item_text.startswith("'")):
                k, _, v = item_text.partition(":")
                d = {}
                if v.strip():
                    d[k.strip()] = _parse_scalar(v)
                parent.append(d)
                stack.append((d, ind))
                # subsequent deeper keys append to d
                j = i + 1
                while j < n and indent_of(cleaned[j]) > ind:
                    ln2 = cleaned[j]
                    if ln2.strip().startswith("- "):
                        break
                    if ":" in ln2:
                        k2, _, v2 = ln2.partition(":")
                        d[k2.strip()] = _parse_scalar(v2)
                    j += 1
                i = j
                continue
            parent.append(_parse_scalar(item_text))
            i += 1
            continue

        if ":" not in content:
            raise ConfigError("cannot parse line: " + content)
        key, _, val = content.partition(":")
        key = key.strip()
        val = val.strip()
        while stack and stack[-1][1] >= ind:
            stack.pop()
        parent = stack[-1][0]
        if val == "":
            # nested mapping or list; look ahead
            if i + 1 < n and cleaned[i + 1].strip().startswith("- "):
                child = []
                parent[key] = child
                stack.append((child, ind))
                i += 1
                continue
            child = {}
            parent[key] = child
            stack.append((child, ind))
        else:
            parent[key] = _parse_scalar(val)
        i += 1

    return root


def load_config(path) -> dict:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config not found: {path}")
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        return json.loads(text)
    return _parse_yaml(text)


def dump_config(cfg: dict) -> str:
    return json.dumps(cfg, ensure_ascii=False, indent=2)
