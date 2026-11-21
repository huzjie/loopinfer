# -*- coding: utf-8 -*-
"""Minimal byte-pair encoding (BPE) for reference. Default runtime uses byte-level."""

from __future__ import annotations

from collections import Counter


class BPETokenizer:
    def __init__(self, vocab_size: int = 512):
        self.vocab_size = vocab_size
        self.merges = {}   # pair -> new id
        self.vocab = {bytes([i]): i for i in range(256)}

    def _get_stats(self, ids):
        return Counter(zip(ids, ids[1:]))

    def train(self, corpus, num_merges: int = 256):
        ids_list = [list(t.encode("utf-8")) for t in corpus]
        for _ in range(num_merges):
            stats = Counter()
            for ids in ids_list:
                stats.update(self._get_stats(ids))
            if not stats:
                break
            pair = stats.most_common(1)[0][0]
            new_id = 256 + len(self.merges)
            self.merges[pair] = new_id
            new_ids_list = []
            for ids in ids_list:
                new_ids = []
                i = 0
                while i < len(ids):
                    if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
                        new_ids.append(new_id)
                        i += 2
                    else:
                        new_ids.append(ids[i])
                        i += 1
                new_ids_list.append(new_ids)
            ids_list = new_ids_list
        return self

    def encode(self, text: str):
        ids = list(text.encode("utf-8"))
        # iteratively apply merges
        while len(ids) >= 2:
            best = None
            best_pair = None
            for i in range(len(ids) - 1):
                p = (ids[i], ids[i + 1])
                if p in self.merges:
                    rank = self.merges[p]
                    if best is None or rank < best:
                        best = rank
                        best_pair = p
            if best_pair is None:
                break
            new_ids = []
            i = 0
            while i < len(ids):
                if i + 1 < len(ids) and (ids[i], ids[i + 1]) == best_pair:
                    new_ids.append(best)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids
        return ids

    def decode(self, ids):
        # decode back to bytes via reverse mapping (best-effort)
        out = bytearray()
        for t in ids:
            if t < 256:
                out.append(t)
        return out.decode("utf-8", errors="replace")
