"""Module 1 - Document Ingestion.

pypdf extracts raw slide text; regex cleaning removes page numbers, URLs,
bullet glyphs, repeated headers/footers and non-ASCII noise.
Output: {slide_number: clean_slide_text} plus per-slide "units" (one per bullet/line).

Note: original casing is preserved here because spaCy's POS/NER models depend on
it. Lower-casing is applied later, where it is actually needed (TF-IDF, N-gram LM).
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import BinaryIO

from pypdf import PdfReader

MAX_PAGES = 200

_BULLETS = re.compile(
    "[\u2022\u25CF\u25AA\u25AB\u25E6\u25A0\u25A1\u2023\u2043\u27A2\u27A4\u2192"
    "\uf0b7\uf0a7\uf0d8\u2713\u2756\u25BA\u25B6]"
)
_URL = re.compile(r"(https?://\S+|www\.\S+)", re.I)
_PAGE_NO = re.compile(r"^\s*(?:slide|page)?\s*\d{1,3}\s*(?:/|of)?\s*\d{0,3}\s*$", re.I)
_PUNCT = {
    ord(k): v
    for k, v in {
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": " - ", "\u2212": "-", "\u00a0": " ",
        "\u2026": "...", "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
        "\ufb03": "ffi", "\ufb04": "ffl",
    }.items()
}


@dataclass
class IngestResult:
    slides: dict[int, str] = field(default_factory=dict)       # slide -> clean text
    units: dict[int, list[str]] = field(default_factory=dict)  # slide -> clean lines
    titles: dict[int, str] = field(default_factory=dict)       # slide -> title (if any)
    n_pages: int = 0
    skipped: list[int] = field(default_factory=list)           # pages with no text


def _page_lines(raw: str) -> list[str]:
    raw = raw.translate(_PUNCT)
    raw = _BULLETS.sub("\n", raw)
    raw = _URL.sub(" ", raw)
    raw = raw.encode("ascii", "ignore").decode()
    lines = [re.sub(r"^[-*>~]\s+", "", re.sub(r"\s+", " ", ln).strip()) for ln in raw.splitlines()]
    return [ln for ln in lines if ln and not _PAGE_NO.match(ln)]


def _signature(line: str) -> str:
    return re.sub(r"\d+", "#", line.lower())


def _merge_wrapped(lines: list[str]) -> list[str]:
    """Re-join lines that were wrapped by the PDF layout."""
    out: list[str] = []
    for ln in lines:
        if out and ln[0].islower():
            prev = out[-1]
            if prev.endswith("-") and len(prev) > 1 and prev[-2].isalpha():
                out[-1] = prev[:-1] + ln
            else:
                out[-1] = prev + " " + ln
        elif out and out[-1].endswith(","):
            out[-1] += " " + ln
        else:
            out.append(ln)
    return out


def ingest_pdf(fileobj: BinaryIO) -> IngestResult:
    reader = PdfReader(fileobj)
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            raise ValueError("This PDF is password-protected. Remove the password and try again.")
    pages = reader.pages[:MAX_PAGES]
    per_page = []
    for page in pages:
        try:
            per_page.append(_page_lines(page.extract_text() or ""))
        except Exception:
            per_page.append([])

    # Drop running headers/footers: lines repeated on most slides.
    n = len(per_page)
    if n >= 4:
        sig_count = Counter(sig for lines in per_page for sig in {_signature(l) for l in lines})
        repeated = {s for s, c in sig_count.items() if c / n >= 0.6}
        per_page = [[l for l in lines if _signature(l) not in repeated] for lines in per_page]

    res = IngestResult(n_pages=len(reader.pages))
    for i, lines in enumerate(per_page, start=1):
        units = _merge_wrapped(lines)
        if not units:
            res.skipped.append(i)
            continue
        if len(units) > 1 and len(units[0].split()) <= 10 and not units[0].endswith((".", "?", "!")):
            res.titles[i] = units[0].rstrip(":")
        res.units[i] = units
        res.slides[i] = "\n".join(units)

    if not res.slides:
        raise ValueError(
            "No selectable text was found. Scanned or image-only PDFs aren't supported - "
            "export the slides as a text-based PDF and try again."
        )
    return res
