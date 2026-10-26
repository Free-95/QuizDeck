"""QuizDeck - Module 5: web dashboard backend (Flask JSON API + static single-page UI)."""
from __future__ import annotations

import io
import threading
import time
import uuid
from collections import OrderedDict

from flask import Flask, jsonify, render_template, request

from quizdeck.analyzer import analyze, get_nlp
from quizdeck.autocomplete import AutocompleteAssistant
from quizdeck.ingestion import ingest_pdf
from quizdeck.quiz_engine import QuizEngine

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB uploads

SESSION_TTL = 3 * 3600
MAX_SESSIONS = 20


class Session:
    def __init__(self, filename, corpus):
        self.id = uuid.uuid4().hex
        self.filename = filename
        self.corpus = corpus
        self.engine = QuizEngine(corpus)
        self.assistant = AutocompleteAssistant(corpus)
        self.quiz: dict = {}
        self.touched = time.time()


_sessions: "OrderedDict[str, Session]" = OrderedDict()
_lock = threading.Lock()


def _store(sess: Session):
    with _lock:
        now = time.time()
        for sid in [k for k, v in _sessions.items() if now - v.touched > SESSION_TTL]:
            _sessions.pop(sid, None)
        _sessions[sess.id] = sess
        while len(_sessions) > MAX_SESSIONS:
            _sessions.popitem(last=False)


def _get(sid: str) -> Session | None:
    with _lock:
        sess = _sessions.get(sid)
        if sess:
            sess.touched = time.time()
            _sessions.move_to_end(sid)
        return sess


def _err(msg: str, code: int = 400):
    return jsonify({"error": msg}), code


def _need(sid: str):
    sess = _get(sid)
    if not sess:
        return None, _err("This session has expired. Upload the PDF again to continue.", 404)
    return sess, None


# warm the spaCy model so the first upload isn't slow
threading.Thread(target=lambda: get_nlp(), daemon=True).start()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/healthz")
def healthz():
    return "ok"


@app.post("/api/upload")
def upload():
    f = request.files.get("file")
    if not f or not f.filename:
        return _err("Choose a PDF file to upload.")
    data = f.read()
    if not f.filename.lower().endswith(".pdf") or not data.startswith(b"%PDF"):
        return _err("That doesn't look like a PDF. Upload a .pdf lecture deck.", 415)
    try:
        ing = ingest_pdf(io.BytesIO(data))
        corpus = analyze(ing)
        sess = Session(f.filename, corpus)
    except ValueError as exc:
        return _err(str(exc), 422)
    except RuntimeError as exc:
        return _err(str(exc), 500)
    except Exception:
        app.logger.exception("processing failed")
        return _err("Something went wrong while reading this PDF. Try a different export of the deck.", 500)
    _store(sess)
    return jsonify({"session_id": sess.id, "filename": sess.filename, **corpus.overview()})


@app.get("/api/session/<sid>/overview")
def overview(sid):
    sess, err = _need(sid)
    if err:
        return err
    return jsonify({"session_id": sess.id, "filename": sess.filename, **sess.corpus.overview()})


@app.post("/api/session/<sid>/quiz")
def quiz(sid):
    sess, err = _need(sid)
    if err:
        return err
    body = request.get_json(silent=True) or {}
    n = min(max(int(body.get("n", 10)), 3), 25)
    questions = sess.engine.generate(n=n, seed=int(time.time()))
    if not questions:
        return _err("Couldn't build questions from this deck. It may have too few full sentences.", 422)
    sess.quiz = {q.id: q for q in questions}
    total = len(questions)
    return jsonify({"total": total, "questions": [q.public(i + 1, total) for i, q in enumerate(questions)]})


@app.post("/api/session/<sid>/quiz/answer")
def answer(sid):
    sess, err = _need(sid)
    if err:
        return err
    body = request.get_json(silent=True) or {}
    q = sess.quiz.get(body.get("qid"))
    if not q:
        return _err("Unknown question. Start a new quiz.", 404)
    try:
        choice = int(body.get("choice"))
    except (TypeError, ValueError):
        return _err("Pick one of the options.")
    return jsonify({
        "correct": choice == q.answer_idx, "answer_idx": q.answer_idx,
        "answer": q.answer, "explanation": q.explanation,
    })


@app.post("/api/session/<sid>/lm")
def lm(sid):
    sess, err = _need(sid)
    if err:
        return err
    b = request.get_json(silent=True) or {}
    try:
        res = sess.assistant.analyze(
            text=str(b.get("text", ""))[:4000], n=int(b.get("n", 2)),
            smoothing=str(b.get("smoothing", "laplace")),
            gamma=float(b.get("gamma", 0.1)), top_k=int(b.get("top_k", 5)),
        )
    except (TypeError, ValueError):
        return _err("Invalid language-model settings.")
    return jsonify(res)


@app.errorhandler(413)
def too_big(_):
    return _err("That file is over the 25 MB limit.", 413)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
