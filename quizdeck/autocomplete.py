"""Module 4 - Autocomplete Assistant.

Bigram/Trigram language models (nltk.lm) with Laplace or Lidstone smoothing,
trained on the slide sentences. Provides next-word / word-completion predictions
and perplexity (PP = 2^H) for user-typed notes.
"""
from __future__ import annotations

import math
import re
import threading
from statistics import median

from nltk.lm import Laplace, Lidstone
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.util import ngrams, pad_sequence

from .analyzer import Corpus

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
SPECIAL = {"<s>", "</s>", "<UNK>"}


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower().replace("-", " "))


def _split_sentences(text: str) -> list[str]:
    return [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p]


class AutocompleteAssistant:
    def __init__(self, corpus: Corpus):
        self.sentences = [t for t in (tokenize(s.text) for s in corpus.sents) if len(t) >= 3]
        self._models: dict[tuple, dict] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ models
    def _build(self, n: int, smoothing: str, gamma: float):
        train, vocab = padded_everygram_pipeline(n, self.sentences)
        lm = Laplace(n) if smoothing == "laplace" else Lidstone(gamma, n)
        lm.fit(train, vocab)
        return lm

    def _entry(self, n: int, smoothing: str, gamma: float) -> dict:
        key = (n, smoothing, round(gamma, 4) if smoothing == "lidstone" else 1.0)
        with self._lock:
            if key not in self._models:
                lm = self._build(n, smoothing, gamma)
                entry = {"lm": lm, "bi": lm if n == 2 else None}
                if n != 2:
                    entry["bi"] = self._build(2, smoothing, gamma)
                entry["ref"] = self._reference(lm, n)
                self._models[key] = entry
            return self._models[key]

    def _grams(self, toks: list[str], n: int, finished: bool):
        if finished:
            padded = pad_sequence(toks, n, pad_left=True, left_pad_symbol="<s>",
                                  pad_right=True, right_pad_symbol="</s>")
        else:
            padded = pad_sequence(toks, n, pad_left=True, left_pad_symbol="<s>", pad_right=False)
        return list(ngrams(padded, n))

    def _reference(self, lm, n: int) -> dict:
        """Perplexity spread of the training sentences themselves (best-case baseline)."""
        pps = []
        for toks in self.sentences[:300]:
            g = self._grams(toks, n, True)
            if g:
                pps.append(lm.perplexity(g))
        pps.sort()
        if not pps:
            return {"median": 1.0, "p90": 1.0}
        return {"median": float(median(pps)), "p90": float(pps[int(0.9 * (len(pps) - 1))])}

    # ------------------------------------------------------------- predictions
    @staticmethod
    def _observed(lm, ctx: tuple) -> list[tuple[str, int]]:
        order = len(ctx) + 1
        if order == 1:
            fd = lm.counts.unigrams
        else:
            cfd = lm.counts[order]
            fd = cfd[ctx] if ctx in cfd else None
        if not fd:
            return []
        return [(w, c) for w, c in fd.most_common() if w not in SPECIAL]

    def predict_next_words(self, context_tokens: list[str], lm, bi, n: int, top_k: int = 5) -> list[dict]:
        """Top-k next tokens with smoothed probabilities (backs off to lower orders if sparse)."""
        out: list[dict] = []
        seen: set[str] = set()

        def take(model, order: int, source: str):
            padded = (["<s>"] * (order - 1) + context_tokens)[-(order - 1):] if order > 1 else []
            ctx = tuple(model.vocab.lookup(padded)) if padded else ()
            for w, _ in self._observed(model, ctx):
                if w in seen:
                    continue
                seen.add(w)
                out.append({"word": w, "prob": float(model.score(w, ctx or None)), "source": source})
                if len(out) >= top_k:
                    return

        take(lm, n, f"{'bigram' if n == 2 else 'trigram'}")
        if len(out) < top_k and n > 2:
            take(bi, 2, "bigram backoff")
        if len(out) < top_k:
            take(bi, 1, "most frequent")
        return out[:top_k]

    def complete_prefix(self, context_tokens: list[str], partial: str, lm, n: int, top_k: int) -> list[dict]:
        padded = (["<s>"] * (n - 1) + context_tokens)[-(n - 1):]
        ctx = tuple(lm.vocab.lookup(padded))
        cands = [w for w in lm.vocab if w.startswith(partial) and w != partial and w not in SPECIAL]
        scored = [{"word": w, "prob": float(lm.score(w, ctx)), "source": "completion"} for w in cands]
        scored.sort(key=lambda d: -d["prob"])
        return scored[:top_k]

    # ---------------------------------------------------------------- analysis
    def analyze(self, text: str, n: int = 2, smoothing: str = "laplace",
                gamma: float = 0.1, top_k: int = 5) -> dict:
        n = 3 if n == 3 else 2
        smoothing = "lidstone" if smoothing == "lidstone" else "laplace"
        gamma = min(max(float(gamma), 0.001), 1.0)
        top_k = min(max(int(top_k), 1), 10)
        entry = self._entry(n, smoothing, gamma)
        lm, bi, ref = entry["lm"], entry["bi"], entry["ref"]

        result: dict = {
            "n": n, "smoothing": smoothing, "gamma": gamma if smoothing == "lidstone" else 1.0,
            "vocab_size": len(lm.vocab), "reference": ref,
            "mode": "idle", "predictions": [], "perplexity": None,
        }
        text = text or ""
        segments = [s for s in _split_sentences(text) if tokenize(s)]
        ends_sentence = bool(re.search(r"[.!?]\s*$", text))
        trailing_space = text[-1:].isspace() if text else False

        # ---- predictions for the sentence currently being typed
        if text.strip():
            last = [] if ends_sentence or not segments else tokenize(segments[-1])
            if last and not trailing_space and not ends_sentence:
                result["mode"] = "completing"
                result["prefix"] = last[-1]
                result["predictions"] = self.complete_prefix(last[:-1], last[-1], lm, n, top_k)
            else:
                result["mode"] = "next"
                result["predictions"] = self.predict_next_words(last, lm, bi, n, top_k)

        # ---- perplexity over everything typed
        grams, token_rows, oov = [], [], 0
        for i, seg in enumerate(segments):
            toks = tokenize(seg)
            finished = (i < len(segments) - 1) or ends_sentence
            for g in self._grams(toks, n, finished):
                grams.append(g)
                w = g[-1]
                if w == "</s>":
                    continue
                known = lm.vocab.lookup(w) != "<UNK>"
                oov += 0 if known else 1
                p = float(lm.score(w, tuple(lm.vocab.lookup(list(g[:-1])))))
                token_rows.append({"token": w, "prob": p, "surprisal": -math.log2(p) if p > 0 else 30.0,
                                   "oov": not known})
        if grams:
            h = float(lm.entropy(grams))
            pp = float(2 ** h)
            ratio = pp / max(ref["p90"], 1e-9)
            if ratio <= 1.0:
                verdict = ("good", "Low uncertainty - strong alignment with the lecture slides.")
            elif ratio <= 3.0:
                verdict = ("mid", "Moderate uncertainty - partly matches the slide wording.")
            else:
                verdict = ("bad", "High uncertainty - this reads differently from the slides.")
            result["perplexity"] = {
                "value": pp, "cross_entropy": h, "tokens": len(token_rows), "oov": oov,
                "level": verdict[0], "verdict": verdict[1],
            }
            result["tokens"] = token_rows[:400]
        return result
