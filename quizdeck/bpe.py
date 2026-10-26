"""Lightweight Byte-Pair Encoding learner + segmenter.

Used by the quiz engine to split words into subword morphemes so that distractors
can be chosen by shared roots/suffixes (e.g. tokenization ~ tokenizer ~ tokenizing).
"""
from __future__ import annotations

import re
from collections import Counter
from itertools import chain

EOW = "</w>"


def _merge(symbols: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    out, i = [], 0
    while i < len(symbols):
        if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
            out.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


class BPE:
    def __init__(self, num_merges: int = 200, max_words: int = 4000):
        self.num_merges = num_merges
        self.max_words = max_words
        self.merges: list[tuple[str, str]] = []
        self._rank: dict[tuple[str, str], int] = {}
        self._cache: dict[str, tuple[str, ...]] = {}

    def fit(self, word_freqs: dict[str, int]) -> "BPE":
        top = Counter(word_freqs).most_common(self.max_words)
        vocab = {tuple(w) + (EOW,): f for w, f in top}
        for _ in range(self.num_merges):
            pairs: Counter = Counter()
            for sym, f in vocab.items():
                for a, b in zip(sym, sym[1:]):
                    pairs[(a, b)] += f
            if not pairs:
                break
            best, count = pairs.most_common(1)[0]
            if count < 2:
                break
            self.merges.append(best)
            vocab = {_merge(sym, best): f for sym, f in vocab.items()}
        self._rank = {p: i for i, p in enumerate(self.merges)}
        self._cache.clear()
        return self

    def segment(self, word: str) -> tuple[str, ...]:
        word = word.lower()
        if word in self._cache:
            return self._cache[word]
        sym = list(word) + [EOW]
        while len(sym) > 1:
            cands = [(self._rank[p], i) for i, p in enumerate(zip(sym, sym[1:])) if p in self._rank]
            if not cands:
                break
            _, i = min(cands)
            sym[i:i + 2] = [sym[i] + sym[i + 1]]
        self._cache[word] = tuple(sym)
        return self._cache[word]

    def similarity(self, a: str, b: str) -> float:
        """Subword overlap (Jaccard) plus a bonus for a shared leading root."""
        wa, wb = re.findall(r"[a-z]+", a.lower()), re.findall(r"[a-z]+", b.lower())
        if not wa or not wb:
            return 0.0
        sa = set(chain.from_iterable(self.segment(w) for w in wa))
        sb = set(chain.from_iterable(self.segment(w) for w in wb))
        jac = len(sa & sb) / len(sa | sb)
        ca, cb = "".join(wa), "".join(wb)
        k = 0
        while k < min(len(ca), len(cb)) and ca[k] == cb[k]:
            k += 1
        return jac + 0.5 * k / max(len(ca), len(cb))
