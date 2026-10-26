"""Module 3 - Quiz Engine.

1. Rank sentences by mean TF-IDF density + entity presence.
2. Pick a masking target (entities > proper nouns > high-TF-IDF nouns/adjectives).
3. Build 3 distractors whose BPE subword segments / root overlap the answer.
"""
from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass

from .analyzer import ENT_LABELS, Corpus, Sent
from .bpe import BPE

BLANK = "_____"
NOUNISH = {"NOUN", "PROPN"}


@dataclass
class Cand:
    a: int
    b: int
    text: str
    kind: str       # "entity" | "term"
    label: str      # entity label or POS
    score: float


@dataclass
class Question:
    id: str
    slide: int
    topic: str
    prompt: str
    options: list[str]
    answer_idx: int
    answer: str
    explanation: str
    kind: str
    score: float

    def public(self, number: int, total: int) -> dict:
        return {
            "id": self.id, "number": number, "total": total, "slide": self.slide,
            "topic": self.topic, "prompt": self.prompt, "options": self.options,
        }


def _match_case(word: str, like: str) -> str:
    if len(like) > 1 and like.isupper():
        return word.upper()
    if like[:1].isupper():
        return word[:1].upper() + word[1:]
    return word.lower()


class QuizEngine:
    def __init__(self, corpus: Corpus):
        self.c = corpus
        freq = Counter(
            t.text.lower() for s in corpus.sents for t in s.toks if t.is_alpha and len(t.text) >= 3
        )
        self.bpe = BPE().fit(freq)
        self.word_pool: dict[str, list[str]] = {"NOUN": [], "ADJ": []}
        for r in corpus.term_df.itertuples():
            if len(r.term) < 4:
                continue
            if r.pos in NOUNISH:
                self.word_pool["NOUN"].append(r.term)
            elif r.pos == "ADJ":
                self.word_pool["ADJ"].append(r.term)
        self.entity_pool: dict[str, dict[str, str]] = {}
        for e in corpus.entities():
            self.entity_pool.setdefault(e["label"], {})[e["text"].lower()] = e["text"]

    # ------------------------------------------------------------------ targets
    def _candidates(self, s: Sent) -> list[Cand]:
        out: list[Cand] = []
        covered: set[int] = set()
        for a, b, label in s.ents:
            if label not in ENT_LABELS:
                continue
            covered.update(range(a, b))
            out.append(Cand(a, b, s.span_text(a, b), "entity", label, self.c.content_score(s, a, b) + 0.25))
        for i, t in enumerate(s.toks):
            if i in covered or not t.is_alpha or t.is_stop or len(t.text) < 4:
                continue
            if t.pos not in {"NOUN", "PROPN", "ADJ"}:
                continue
            sc = self.c.scores.get(t.text.lower(), 0.0)
            if t.pos == "PROPN":
                sc += 0.10
            elif t.pos == "ADJ":
                sc *= 0.8
            out.append(Cand(i, i + 1, t.text, "term", t.pos, sc))
        return sorted(out, key=lambda c: -c.score)

    def _sentence_score(self, s: Sent) -> float:
        n_alpha = sum(t.is_alpha for t in s.toks)
        if not s.has_verb or n_alpha < 7 or n_alpha > 40:
            return 0.0
        n_ent = sum(1 for _, _, l in s.ents if l in ENT_LABELS)
        return self.c.content_score(s) + 0.05 * min(n_ent, 3)

    # -------------------------------------------------------------- distractors
    def _distractors(self, cand: Cand, s: Sent, k: int = 3) -> list[str]:
        target = cand.text.lower()
        sent_words = {t.text.lower() for t in s.toks}
        sent_lower = s.text.lower()
        ranked: list[tuple[float, str]] = []

        if cand.kind == "entity":
            pool = self.entity_pool.get(cand.label, {})
            for low, orig in pool.items():
                if low == target or low in target or target in low or low in sent_lower:
                    continue
                ranked.append((self.bpe.similarity(target, low), orig))
        else:
            tl = self.c.lemma_of.get(target, target)
            pool = self.word_pool["ADJ" if cand.label == "ADJ" else "NOUN"]
            for w in pool:
                if w == target or w in sent_words or self.c.lemma_of.get(w, w) == tl:
                    continue
                sim = self.bpe.similarity(target, w) + 0.25 * self.c.scores.get(w, 0.0)
                ranked.append((sim, _match_case(w, cand.text)))

        ranked.sort(key=lambda x: -x[0])
        picked, seen = [], set()
        for _, w in ranked:
            if w.lower() not in seen:
                seen.add(w.lower())
                picked.append(w)
            if len(picked) == k:
                break
        return picked if len(picked) == k else []

    # --------------------------------------------------------------- generation
    def _build_prompt(self, s: Sent, cand: Cand) -> str:
        mask: set[int] = set(range(cand.a, cand.b))
        if cand.kind == "term":
            mask |= {i for i, t in enumerate(s.toks) if t.text.lower() == cand.text.lower()}
        parts, i = [], 0
        while i < len(s.toks):
            if i in mask:
                j = i
                while j + 1 < len(s.toks) and (j + 1) in mask:
                    j += 1
                parts.append(BLANK + s.toks[j].ws)
                i = j + 1
            else:
                parts.append(s.toks[i].text + s.toks[i].ws)
                i += 1
        return "".join(parts).strip()

    def generate(self, n: int = 10, seed: int | None = None) -> list[Question]:
        rng = random.Random(seed)
        ranked = sorted(((self._sentence_score(s), s) for s in self.c.sents), key=lambda x: -x[0])
        ranked = [(sc, s) for sc, s in ranked if sc > 0]
        pool_size = min(len(ranked), max(n * 6, 40))
        head = ranked[:pool_size]
        rng.shuffle(head)                      # fresh sets on "new quiz" while staying high-value
        head.sort(key=lambda x: -x[0] + rng.random() * 0.08)

        questions: list[Question] = []
        used_targets: set[str] = set()
        used_prompts: set[str] = set()
        per_slide: Counter = Counter()

        for cap in (2, 99):
            for sc, s in head + ranked[pool_size:]:
                if len(questions) >= n:
                    break
                if per_slide[s.slide] >= cap or s.text.lower() in used_prompts:
                    continue
                for cand in self._candidates(s):
                    key = cand.text.lower()
                    if key in used_targets:
                        continue
                    wrong = self._distractors(cand, s)
                    if not wrong:
                        continue
                    options = wrong + [cand.text]
                    rng.shuffle(options)
                    if cand.kind == "entity":
                        why = (f"'{cand.text}' is a named entity ({cand.label}) in this sentence "
                               f"(TF-IDF weight {min(cand.score, 1.0):.2f}).")
                    else:
                        why = (f"'{cand.text}' was identified as a primary keyword "
                               f"(TF-IDF: {self.c.scores.get(key, 0.0):.2f}).")
                    questions.append(Question(
                        id=f"q{len(questions) + 1}", slide=s.slide,
                        topic=self.c.topics.get(s.slide, f"Slide {s.slide}"),
                        prompt=self._build_prompt(s, cand), options=options,
                        answer_idx=options.index(cand.text), answer=cand.text,
                        explanation=why, kind=cand.kind, score=cand.score,
                    ))
                    used_targets.add(key)
                    used_prompts.add(s.text.lower())
                    per_slide[s.slide] += 1
                    break
            if len(questions) >= n:
                break
        return questions
