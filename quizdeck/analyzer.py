"""Module 2 - Linguistic Analyzer.

spaCy (en_core_web_sm) supplies lemmas, POS tags, entities and sentence
boundaries; scikit-learn's TfidfVectorizer supplies term importance weights.
"""
from __future__ import annotations

import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

from .ingestion import IngestResult

ENT_LABELS = {
    "ORG", "PERSON", "GPE", "DATE", "MONEY", "PRODUCT", "EVENT",
    "NORP", "LAW", "WORK_OF_ART", "FAC", "LOC",
}
WORD_RE = r"(?u)\b[a-zA-Z]{3,}\b"

_nlp = None
_lock = threading.Lock()


def get_nlp():
    global _nlp
    with _lock:
        if _nlp is None:
            try:
                _nlp = spacy.load("en_core_web_sm")
            except OSError as exc:  # model not installed
                raise RuntimeError(
                    "spaCy model 'en_core_web_sm' is missing. "
                    "Run: python -m spacy download en_core_web_sm"
                ) from exc
        return _nlp


@dataclass
class Tok:
    text: str
    lemma: str
    pos: str
    is_stop: bool
    is_alpha: bool
    ws: str


@dataclass
class Sent:
    id: int
    slide: int
    text: str
    toks: list[Tok]
    ents: list[tuple[int, int, str]]    # (start_tok, end_tok_exclusive, label)
    chunks: list[tuple[int, int]]       # trimmed noun chunks
    has_verb: bool

    def span_text(self, a: int, b: int) -> str:
        return "".join(t.text + t.ws for t in self.toks[a:b]).strip()


@dataclass
class Corpus:
    sents: list[Sent]
    slides: dict[int, str]
    titles: dict[int, str]
    topics: dict[int, str]
    term_df: pd.DataFrame                       # term, tfidf, doc_freq, pos, lemma
    scores: dict[str, float]                    # word -> normalised TF-IDF (0..1)
    pos_of: dict[str, str]
    lemma_of: dict[str, str]
    n_pages: int
    skipped: list[int]
    by_slide: dict[int, list[int]] = field(default_factory=dict)  # slide -> sentence ids

    def content_score(self, s: Sent, a: int = 0, b: int | None = None) -> float:
        toks = s.toks[a:b]
        vals = [self.scores.get(t.text.lower(), 0.0) for t in toks if t.is_alpha and not t.is_stop]
        return float(np.mean(vals)) if vals else 0.0

    def entities(self) -> list[dict]:
        seen: dict[tuple[str, str], dict] = {}
        for s in self.sents:
            for a, b, label in s.ents:
                if label not in ENT_LABELS:
                    continue
                text = s.span_text(a, b)
                key = (text.lower(), label)
                if key not in seen:
                    seen[key] = {"text": text, "label": label, "count": 0, "slides": set()}
                seen[key]["count"] += 1
                seen[key]["slides"].add(s.slide)
        out = []
        for e in seen.values():
            e["slides"] = sorted(e["slides"])
            out.append(e)
        return sorted(out, key=lambda e: (-e["count"], e["text"].lower()))

    def overview(self, n_terms: int = 18, n_ents: int = 24) -> dict:
        nounish = self.term_df[self.term_df["pos"].isin(["NOUN", "PROPN", "ADJ"])]
        top = nounish.head(n_terms)
        ents = self.entities()
        return {
            "stats": {
                "pages": self.n_pages,
                "slides_with_text": len(self.slides),
                "sentences": len(self.sents),
                "vocabulary": int(len(self.term_df)),
                "entities": len(ents),
            },
            "skipped_pages": self.skipped,
            "key_terms": [
                {"term": r.term, "score": round(float(r.tfidf), 3), "pos": r.pos}
                for r in top.itertuples()
            ],
            "entities": [
                {"text": e["text"], "label": e["label"], "count": e["count"]}
                for e in ents[:n_ents]
            ],
        }


def analyze(ing: IngestResult) -> Corpus:
    nlp = get_nlp()
    items = [(slide, unit) for slide, units in ing.units.items() for unit in units]
    docs = nlp.pipe([u for _, u in items], batch_size=64)

    sents: list[Sent] = []
    for (slide, _), doc in zip(items, docs):
        for sp in doc.sents:
            toks_raw = [t for t in sp if not t.is_space]
            if sum(t.is_alpha for t in toks_raw) < 4:
                continue
            base = sp.start
            # indices must be relative to the filtered token list; spaces are rare so
            # we rebuild a mapping from doc index -> filtered index.
            idx = {t.i: k for k, t in enumerate(toks_raw)}
            toks = [Tok(t.text, t.lemma_.lower(), t.pos_, t.is_stop, t.is_alpha, t.whitespace_) for t in toks_raw]
            ents = []
            for e in sp.ents:
                if e.start in idx and (e.end - 1) in idx:
                    ents.append((idx[e.start], idx[e.end - 1] + 1, e.label_))
            chunks = []
            for nc in sp.noun_chunks:
                a, b = nc.start, nc.end
                while a < b and (doc[a].pos_ in {"DET", "PRON", "PUNCT"} or doc[a].is_stop):
                    a += 1
                if b - a >= 1 and a in idx and (b - 1) in idx:
                    chunks.append((idx[a], idx[b - 1] + 1))
            has_verb = any(t.pos_ in {"VERB", "AUX"} for t in toks_raw)
            text = "".join(t.text + t.ws for t in toks).strip()
            sents.append(Sent(len(sents), slide, text, toks, ents, chunks, has_verb))

    if not sents:
        raise ValueError("The PDF contained too little readable sentence text to build a quiz from.")

    # --- TF-IDF over slide sentences ---------------------------------------
    vec = TfidfVectorizer(stop_words="english", token_pattern=WORD_RE, sublinear_tf=True, lowercase=True)
    try:
        X = vec.fit_transform([s.text for s in sents])
    except ValueError:
        raise ValueError("The PDF contained too little readable text to analyse.")
    raw = np.asarray(X.sum(axis=0)).ravel()
    norm = raw / raw.max() if raw.max() > 0 else raw
    vocab = vec.get_feature_names_out()
    doc_freq = np.asarray((X > 0).sum(axis=0)).ravel()
    scores = {w: float(v) for w, v in zip(vocab, norm)}

    # majority POS + lemma for every vocabulary word
    pos_votes: dict[str, Counter] = defaultdict(Counter)
    lemma_votes: dict[str, Counter] = defaultdict(Counter)
    for s in sents:
        for t in s.toks:
            if t.is_alpha:
                w = t.text.lower()
                pos_votes[w][t.pos] += 1
                lemma_votes[w][t.lemma] += 1
    pos_of = {w: c.most_common(1)[0][0] for w, c in pos_votes.items()}
    lemma_of = {w: c.most_common(1)[0][0] for w, c in lemma_votes.items()}

    term_df = (
        pd.DataFrame(
            {
                "term": vocab,
                "tfidf": norm,
                "doc_freq": doc_freq,
                "pos": [pos_of.get(w, "X") for w in vocab],
                "lemma": [lemma_of.get(w, w) for w in vocab],
            }
        )
        .sort_values("tfidf", ascending=False)
        .reset_index(drop=True)
    )

    by_slide: dict[int, list[int]] = defaultdict(list)
    for s in sents:
        by_slide[s.slide].append(s.id)

    corpus = Corpus(
        sents=sents, slides=ing.slides, titles=ing.titles, topics={}, term_df=term_df,
        scores=scores, pos_of=pos_of, lemma_of=lemma_of, n_pages=ing.n_pages,
        skipped=ing.skipped, by_slide=dict(by_slide),
    )

    # slide topics: title line if present, else the slide's strongest noun
    for slide, ids in corpus.by_slide.items():
        title = ing.titles.get(slide)
        if title:
            corpus.topics[slide] = title[:60]
            continue
        best, best_s = None, -1.0
        for i in ids:
            for t in sents[i].toks:
                w = t.text.lower()
                if t.pos in {"NOUN", "PROPN"} and scores.get(w, 0) > best_s:
                    best, best_s = w, scores[w]
        corpus.topics[slide] = best.capitalize() if best else f"Slide {slide}"
    return corpus
