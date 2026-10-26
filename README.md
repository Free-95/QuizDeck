# QuizDeck

Lecture-slide PDF to cloze quiz and an N-gram notes assistant.
Backend: Flask + pypdf + spaCy + scikit-learn + NLTK. Frontend: plain HTML/CSS/JS (no build step).

## Modules (as in the project spec)
| Module | File |
|---|---|
| 1 Document ingestion | `quizdeck/ingestion.py` |
| 2 Linguistic analyzer (spaCy POS/NER + TF-IDF) | `quizdeck/analyzer.py` |
| 3 Quiz engine (cloze + BPE distractors) | `quizdeck/quiz_engine.py`, `quizdeck/bpe.py` |
| 4 Autocomplete assistant (nltk.lm, perplexity) | `quizdeck/autocomplete.py` |
| 5 Dashboard | `app.py`, `templates/`, `static/` |

## Run locally
```bash
python -m venv .venv 
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python app.py            # http://localhost:5000
```

## Deploy on Render
1. Push this folder to a GitHub repo.
2. Render dashboard: New > Blueprint, pick the repo (it reads `render.yaml`).
   Or New > Web Service with:
   - Build: `pip install -r requirements.txt && python -m spacy download en_core_web_sm`
   - Start: `gunicorn app:app --workers 1 --threads 4 --timeout 180 --bind 0.0.0.0:$PORT`
   - Env var `PYTHON_VERSION` = `3.11.9`
3. Keep `--workers 1`: uploaded decks live in server memory (20 most recent, 3 h TTL).
   A second worker would not see the first worker's sessions.

Notes: the free plan has 512 MB RAM and sleeps when idle (first request after a sleep is slow).
spaCy's small model plus sklearn/pandas fits, but large decks (100+ slides) may be tight.

## Design notes
- Original casing is kept through ingestion and spaCy (POS/NER need it); lower-casing happens
  in TF-IDF and the N-gram model, where it belongs.
- Blanks are `_____`; answer options are shuffled; answers are checked server-side.
- Autocomplete completes a half-typed word, or predicts the next word after a space,
  backing off from trigram to bigram to unigram when the context is unseen.
- Perplexity verdict is relative to how the model scores the slides' own sentences.
