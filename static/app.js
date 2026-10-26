/* QuizDeck front end. All PDF-derived text is inserted with textContent / text nodes (never innerHTML). */
(() => {
  'use strict';

  const $ = (s, r = document) => r.querySelector(s);
  const h = (tag, props = {}, ...kids) => {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(props || {})) {
      if (v === false || v == null) continue;
      if (k === 'class') e.className = v;
      else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v === true ? '' : v);
    }
    for (const kid of kids.flat()) {
      if (kid === false || kid == null) continue;
      e.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    }
    return e;
  };
  const clear = (el) => { while (el.firstChild) el.removeChild(el.firstChild); return el; };
  const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
  const fmt = (n) => Number(n).toLocaleString('en-US');

  const keyHandlers = {};
  const state = { sid: null, overview: null, quiz: null, lm: { n: 2, smoothing: 'laplace', gamma: 0.1 } };

  /* ───────── helpers ───────── */
  async function api(path, opts = {}) {
    const res = await fetch(path, opts);
    let data = null;
    try { data = await res.json(); } catch (_) { /* non-JSON */ }
    if (!res.ok) {
      const err = new Error((data && data.error) || 'Request failed. Please try again.');
      err.status = res.status;
      throw err;
    }
    return data;
  }
  const post = (path, body, extra = {}) => api(path, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), ...extra,
  });

  let toastTimer;
  function toast(msg) {
    const t = $('#toast');
    t.textContent = msg; t.classList.add('show');
    clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove('show'), 3800);
  }
  function handleErr(e) {
    if (e.name === 'AbortError') return;
    if (e.status === 404 && state.sid) { resetToUpload(); showUploadError(e.message); return; }
    toast(e.message || 'Something went wrong.');
  }
  const store = {
    get: (k) => { try { return sessionStorage.getItem(k); } catch (_) { return null; } },
    set: (k, v) => { try { v == null ? sessionStorage.removeItem(k) : sessionStorage.setItem(k, v); } catch (_) {} },
  };

  /* ───────── theme ───────── */
  $('#btn-theme').addEventListener('click', () => {
    const root = document.documentElement;
    const dark = root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
    root.dataset.theme = dark ? 'light' : 'dark';
    try { localStorage.setItem('qd-theme', root.dataset.theme); } catch (_) {}
  });

  /* ───────── hero demo ───────── */
  $('#demo-opts').addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    const ok = b.dataset.v === 'subword';
    [...$('#demo-opts').children].forEach((x) => x.classList.remove('ok', 'no'));
    b.classList.add(ok ? 'ok' : 'no');
    const blank = $('#demo-blank');
    if (ok) { blank.textContent = 'subword'; blank.classList.add('filled'); }
    $('#demo-note').textContent = ok
      ? "Right. 'subword' was the top-weighted keyword in that sentence."
      : 'Not that one. Look at what BPE actually merges.';
  });

  /* ───────── upload ───────── */
  const STEPS = ['Reading slides and cleaning text', 'Tagging parts of speech and entities', 'Weighting terms with TF-IDF', 'Learning subword pieces for answer choices'];
  let stepTimer;
  const drop = $('#drop'), fileInput = $('#file');

  function showUploadError(msg) { const e = $('#upload-error'); e.textContent = msg; e.hidden = !msg; }
  drop.addEventListener('click', () => { if (!drop.classList.contains('busy')) fileInput.click(); });
  drop.addEventListener('keydown', (e) => { if ((e.key === 'Enter' || e.key === ' ') && !drop.classList.contains('busy')) { e.preventDefault(); fileInput.click(); } });
  ['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
  drop.addEventListener('drop', (e) => { const f = e.dataTransfer.files[0]; if (f) upload(f); });
  fileInput.addEventListener('change', () => { if (fileInput.files[0]) upload(fileInput.files[0]); fileInput.value = ''; });

  function startSteps(name) {
    $('#busy-file').textContent = name;
    const ol = clear($('#steps'));
    STEPS.forEach((s) => ol.append(h('li', {}, s)));
    let i = 0;
    const paint = () => [...ol.children].forEach((li, k) => { li.className = k < i ? 'done' : k === i ? 'active' : ''; });
    paint();
    stepTimer = setInterval(() => { if (i < STEPS.length - 1) { i++; paint(); } }, 2200);
    return () => { clearInterval(stepTimer); [...ol.children].forEach((li) => { li.className = 'done'; }); };
  }

  async function upload(file) {
    showUploadError('');
    if (!/\.pdf$/i.test(file.name)) { showUploadError("That isn't a PDF. Choose a .pdf lecture deck."); return; }
    if (file.size > 25 * 1024 * 1024) { showUploadError('That file is over the 25 MB limit.'); return; }
    drop.classList.add('busy'); $('#drop-idle').hidden = true; $('#drop-busy').hidden = false;
    const finish = startSteps(file.name);
    try {
      const fd = new FormData(); fd.append('file', file);
      const data = await api('/api/upload', { method: 'POST', body: fd });
      finish();
      await new Promise((r) => setTimeout(r, 450));
      enterWorkspace(data);
    } catch (e) {
      clearInterval(stepTimer);
      showUploadError(e.message);
    } finally {
      drop.classList.remove('busy'); $('#drop-idle').hidden = false; $('#drop-busy').hidden = true;
    }
  }

  function resetToUpload() {
    state.sid = null; state.overview = null; state.quiz = null;
    store.set('qd-sid', null);
    $('#view-work').hidden = true; $('#view-upload').hidden = false; $('#btn-new').hidden = true;
    ['quiz', 'lm'].forEach((k) => { built[k] = false; clear($('#panel-' + k)); });
    window.scrollTo({ top: 0 });
  }
  $('#btn-new').addEventListener('click', resetToUpload);

  /* ───────── workspace + tabs ───────── */
  const built = { overview: false, quiz: false, lm: false };
  const TABS = ['overview', 'quiz', 'lm'];

  function enterWorkspace(data) {
    state.sid = data.session_id; state.overview = data; store.set('qd-sid', data.session_id);
    state.quiz = null;
    TABS.forEach((k) => { built[k] = false; clear($('#panel-' + k)); });
    $('#view-upload').hidden = true; $('#view-work').hidden = false; $('#btn-new').hidden = false;
    $('#deck-name').textContent = data.filename;
    const s = data.stats, st = clear($('#stats'));
    [['Slides', s.slides_with_text], ['Sentences', s.sentences], ['Vocabulary', s.vocabulary], ['Entities', s.entities]]
      .forEach(([k, v]) => st.append(h('div', {}, h('dt', {}, k), h('dd', {}, fmt(v)))));
    selectTab('overview');
    window.scrollTo({ top: 0 });
  }

  function selectTab(name) {
    TABS.forEach((k) => {
      const on = k === name;
      $('#tab-' + k).setAttribute('aria-selected', String(on));
      $('#tab-' + k).tabIndex = on ? 0 : -1;
      $('#panel-' + k).hidden = !on;
    });
    state.tab = name;
    if (!built[name]) { built[name] = true; ({ overview: renderOverview, quiz: renderQuiz, lm: renderLM })[name](); }
  }
  TABS.forEach((k, i) => {
    $('#tab-' + k).addEventListener('click', () => selectTab(k));
    $('#tab-' + k).addEventListener('keydown', (e) => {
      if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
        const next = TABS[(i + (e.key === 'ArrowRight' ? 1 : TABS.length - 1)) % TABS.length];
        selectTab(next); $('#tab-' + next).focus();
      }
    });
  });

  /* ───────── overview ───────── */
  function renderOverview() {
    const p = clear($('#panel-overview')), o = state.overview;
    const terms = h('ul', { class: 'terms' }, o.key_terms.map((t) =>
      h('li', {}, h('span', { class: 't', title: t.term }, t.term),
        h('span', { class: 'bar' }, h('i', { style: `width:${Math.max(4, t.score * 100)}%` })),
        h('span', { class: 'v' }, t.score.toFixed(2)))));
    const groups = {};
    o.entities.forEach((e) => (groups[e.label] = groups[e.label] || []).push(e));
    const ents = h('div', {}, Object.keys(groups).length
      ? Object.entries(groups).map(([label, list]) => h('div', { class: 'ent-group' }, h('h4', {}, label),
        h('div', { class: 'chips' }, list.map((e) => h('span', { class: 'chip' }, e.text, e.count > 1 && h('small', {}, '\u00d7' + e.count))))))
      : h('p', { class: 'sub', style: 'margin-top:14px' }, 'No named entities were detected in this deck. Quizzes will focus on key terms.'));

    p.append(h('div', { class: 'grid-2' },
      h('div', { class: 'card' }, h('h3', {}, 'Key terms'),
        h('p', { class: 'sub' }, 'Ranked by TF-IDF weight across the deck. Longer bars show terms that matter most.'), terms),
      h('div', { class: 'card' }, h('h3', {}, 'Named entities'),
        h('p', { class: 'sub' }, 'People, organisations, places and dates found by spaCy.'), ents)));
    if (o.skipped_pages && o.skipped_pages.length) {
      const n = o.skipped_pages.length;
      p.append(h('p', { class: 'note' }, `${n} page${n > 1 ? 's' : ''} had no selectable text (likely images) and ${n > 1 ? 'were' : 'was'} skipped.`));
    }
    p.append(h('div', { class: 'next-steps' },
      h('button', { onclick: () => selectTab('quiz') }, h('strong', {}, 'Take a quiz'), h('span', {}, 'Fill in the blanks on the sentences that carry the most weight.')),
      h('button', { onclick: () => selectTab('lm') }, h('strong', {}, 'Check your notes'), h('span', {}, 'Get next-word suggestions and a perplexity score against the deck.'))));
  }

  /* ───────── quiz ───────── */
  const LETTERS = 'ABCD';
  function renderQuiz() {
    const p = clear($('#panel-quiz'));
    const wrap = h('div', { class: 'quiz-wrap' });
    p.append(wrap);
    state.quizLen = state.quizLen || 10;
    showSetup();

    function showSetup() {
      state.quiz = null; clear(wrap);
      const seg = h('div', { class: 'seg', role: 'group', 'aria-label': 'Number of questions' },
        [5, 10, 15, 20].map((n) => h('button', {
          type: 'button', 'aria-pressed': String(n === state.quizLen),
          onclick: (e) => { state.quizLen = n; [...seg.children].forEach((b) => b.setAttribute('aria-pressed', String(b === e.currentTarget))); },
        }, String(n))));
      const go = h('button', { class: 'btn', onclick: start }, 'Start quiz');
      wrap.append(h('div', { class: 'card setup' },
        h('h3', {}, 'Fill-in-the-blank quiz'),
        h('p', { class: 'sub' }, 'Each question hides one key term from a high-value slide sentence. Wrong options share word pieces with the answer, so guessing by shape won\'t work.'),
        h('div', { class: 'field' }, h('span', { class: 'lbl' }, 'Questions'), seg), go));
    }

    async function start() {
      clear(wrap).append(h('div', { class: 'card' }, h('p', { class: 'sub' }, 'Building your questions...')));
      try {
        const d = await post(`/api/session/${state.sid}/quiz`, { n: state.quizLen });
        state.quiz = { qs: d.questions, i: 0, score: 0, results: [], locked: false };
        showQuestion();
      } catch (e) { handleErr(e); showSetup(); }
    }

    function promptNode(text, fill) {
      const parts = text.split('_____'), out = h('p', { class: 'qtext' });
      parts.forEach((part, k) => {
        out.append(part);
        if (k < parts.length - 1) out.append(h('span', { class: 'blank' + (fill ? ' filled' : ''), id: 'blank-' + k }, fill || '_____'));
      });
      return out;
    }

    function showQuestion() {
      const qz = state.quiz, q = qz.qs[qz.i]; qz.locked = false; clear(wrap);
      const opts = h('div', { class: 'opts', role: 'group', 'aria-label': 'Answer options' }, q.options.map((o, k) =>
        h('button', { class: 'opt', type: 'button', onclick: () => choose(k) }, h('b', {}, LETTERS[k]), h('span', {}, o))));
      const fb = h('div', { 'aria-live': 'polite' });
      wrap.append(h('div', { class: 'card' },
        h('div', { class: 'qhead' }, h('span', {}, `Question `, h('b', {}, q.number), ` of ${q.total}`),
          h('span', {}, `Slide ${q.slide} \u00b7 ${q.topic}`)),
        h('div', { class: 'progress', role: 'progressbar', 'aria-valuemin': 0, 'aria-valuemax': q.total, 'aria-valuenow': qz.i },
          h('i', { style: `width:${(qz.i / q.total) * 100}%` })),
        promptNode(q.prompt), opts, fb,
        h('div', { class: 'kbd', style: 'margin-top:16px' }, 'Tip: press 1\u20134 to answer, Enter for next.')));
      qz.el = { opts, fb, q };
    }

    async function choose(k) {
      const qz = state.quiz; if (!qz || qz.locked) return;
      qz.locked = true;
      const { opts, fb, q } = qz.el;
      [...opts.children].forEach((b) => { b.disabled = true; });
      try {
        const r = await post(`/api/session/${state.sid}/quiz/answer`, { qid: q.id, choice: k });
        [...opts.children].forEach((b, idx) => {
          if (idx === r.answer_idx) b.classList.add('ok');
          else if (idx === k) b.classList.add('no');
          else b.classList.add('dim');
        });
        if (r.correct) qz.score++;
        qz.results.push({ q, picked: k, correct: r.correct, answer: r.answer, explanation: r.explanation });
        const blank = $('#blank-0', wrap); if (blank) { blank.textContent = r.answer; blank.classList.add('filled'); }
        const last = qz.i === qz.qs.length - 1;
        clear(fb).append(
          h('div', { class: 'feedback ' + (r.correct ? 'ok' : 'no') },
            h('strong', {}, r.correct ? 'Correct' : `Not quite. The answer is ${r.answer}.`), r.explanation),
          h('div', { class: 'qfoot' }, h('span', { class: 'kbd' }, `Score so far: ${qz.score} / ${qz.results.length}`),
            h('button', { class: 'btn', id: 'btn-next', onclick: next }, last ? 'See results' : 'Next question')));
        $('#btn-next', wrap).focus({ preventScroll: true });
      } catch (e) { qz.locked = false; [...opts.children].forEach((b) => { b.disabled = false; }); handleErr(e); }
    }

    function next() {
      const qz = state.quiz; if (!qz) return;
      if (qz.i < qz.qs.length - 1) { qz.i++; showQuestion(); } else showResult();
    }

    function showResult() {
      const qz = state.quiz, total = qz.qs.length, pct = Math.round((qz.score / total) * 100);
      const C = 2 * Math.PI * 54;
      clear(wrap);
      const ring = h('div', { class: 'ring', role: 'img', 'aria-label': `${pct} percent` },
        (() => {
          const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg'); svg.setAttribute('viewBox', '0 0 120 120');
          const mk = (cls, extra) => { const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle'); c.setAttribute('class', cls); c.setAttribute('cx', 60); c.setAttribute('cy', 60); c.setAttribute('r', 54); Object.entries(extra || {}).forEach(([a, b]) => c.setAttribute(a, b)); return c; };
          const fg = mk('fg', { 'stroke-dasharray': C, 'stroke-dashoffset': C });
          svg.append(mk('bg'), fg);
          requestAnimationFrame(() => requestAnimationFrame(() => fg.setAttribute('stroke-dashoffset', C * (1 - qz.score / total))));
          return svg;
        })(), h('span', {}, pct + '%'));
      const msg = pct >= 80 ? 'Strong grasp of this deck.' : pct >= 50 ? 'Solid start. A few terms to revisit.' : 'Worth another pass through the slides.';
      const missed = qz.results.filter((r) => !r.correct);
      wrap.append(
        h('div', { class: 'card result' }, ring,
          h('div', {}, h('h3', {}, `${qz.score} of ${total} correct`), h('p', { class: 'sub' }, msg),
            h('div', { class: 'actions' },
              h('button', { class: 'btn', onclick: start }, 'New quiz'),
              h('button', { class: 'btn ghost', onclick: showSetup }, 'Change length'),))),
        missed.length ? h('div', { class: 'review' }, h('h4', {}, 'To review'), missed.map((r) =>
          h('div', { class: 'review-item' },
            (() => { const p2 = promptNode(r.q.prompt, r.answer); p2.className = ''; return p2; })(),
            h('p', { class: 'sub' }, `Slide ${r.q.slide} \u00b7 ${r.q.topic}. You chose ${r.q.options[r.picked]}. ${r.explanation}`)))) : '');
    }

    if (keyHandlers.quiz) document.removeEventListener('keydown', keyHandlers.quiz);
    keyHandlers.quiz = (e) => {
      if (state.tab !== 'quiz' || !state.quiz || e.metaKey || e.ctrlKey || e.altKey) return;
      if (/^(input|textarea)$/i.test(document.activeElement.tagName)) return;
      const qz = state.quiz; if (!qz.el || qz.i >= qz.qs.length) return;
      if (!qz.locked) {
        const k = '1234'.indexOf(e.key) >= 0 && e.key ? '1234'.indexOf(e.key) : 'abcd'.indexOf(e.key.toLowerCase());
        if (e.key.length === 1 && k >= 0 && k < qz.qs[qz.i].options.length) choose(k);
      } else if (e.key === 'Enter' && document.activeElement.id !== 'btn-next') {
        if ($('#btn-next', wrap)) { e.preventDefault(); next(); }
      }
    };
    document.addEventListener('keydown', keyHandlers.quiz);
  }

  /* ───────── notes assistant (N-gram LM) ───────── */
  function renderLM() {
    const p = clear($('#panel-lm')), S = state.lm;
    const area = h('textarea', { class: 'editor', id: 'lm-text', rows: 7, spellcheck: 'false', 'aria-label': 'Your notes',
      placeholder: 'Start typing notes in your own words, for example: "byte pair encoding is a"' });
    const sugg = h('div', { class: 'sugg', 'aria-live': 'polite' });
    const tokens = h('div', { class: 'tokens' });
    const legend = h('p', { class: 'legend', hidden: true }, 'Each word is shaded by how surprising it was to the model. Dashed outline means the word never appears in the deck.');

    const seg = (label, key, opts) => {
      const box = h('div', { class: 'seg', role: 'group', 'aria-label': label }, opts.map(([val, text]) =>
        h('button', { type: 'button', 'aria-pressed': String(S[key] === val), onclick: () => {
          S[key] = val; [...box.children].forEach((b, i) => b.setAttribute('aria-pressed', String(opts[i][0] === val))); syncGamma(); run(); } }, text)));
      return h('div', { class: 'field' }, h('span', { class: 'lbl' }, label), box);
    };
    const gammaVal = h('output', {}, S.gamma.toFixed(2));
    const gamma = h('input', { type: 'range', min: 0.01, max: 1, step: 0.01, value: S.gamma, 'aria-label': 'Lidstone gamma',
      oninput: (e) => { S.gamma = parseFloat(e.target.value); gammaVal.textContent = S.gamma.toFixed(2); runSoon(); } });
    const gammaField = h('div', { class: 'field', hidden: true }, h('label', {}, 'Add-m value (gamma): ', gammaVal), gamma);
    const syncGamma = () => { gammaField.hidden = S.smoothing !== 'lidstone'; };

    const ppNum = h('div', { class: 'pp-num' }, '\u2013');
    const verdict = h('p', { class: 'pp-verdict' }, 'Type a sentence to see how closely it follows the slides.');
    const pin = h('i', { class: 'pin', style: 'left:0%', hidden: true });
    const ref = h('i', { class: 'ref', hidden: true });
    const kv = h('dl', { class: 'kv' });

    p.append(h('div', { class: 'grid-2' },
      h('div', { class: 'card' },
        h('h3', {}, 'Your notes'),
        h('p', { class: 'sub', style: 'margin-bottom:12px' }, 'Suggestions come from a language model trained only on this deck. Press Tab to accept the first one.'),
        area, sugg, tokens, legend),
      h('div', { class: 'stack' },
        h('div', { class: 'card settings' }, h('h3', {}, 'Language model'),
          seg('Order', 'n', [[2, 'Bigram'], [3, 'Trigram']]),
          seg('Smoothing', 'smoothing', [['laplace', 'Laplace (add-1)'], ['lidstone', 'Lidstone (add-m)']]),
          gammaField),
        h('div', { class: 'card' }, h('h3', {}, 'Perplexity'),
          h('p', { class: 'sub', style: 'margin-bottom:10px' }, 'PP = 2^H. Lower means the notes sound more like the slides.'),
          ppNum, verdict, h('div', { class: 'meter', 'aria-hidden': 'true' }, pin, ref),
          h('div', { class: 'meter-cap' }, h('span', {}, 'Matches slides'), h('span', {}, 'Guessing at random')), kv))));

    let timer, ctrl, last = null;
    const runSoon = () => { clearTimeout(timer); timer = setTimeout(run, 220); };

    async function run() {
      if (ctrl) ctrl.abort(); ctrl = new AbortController();
      const text = area.value;
      try {
        const r = await post(`/api/session/${state.sid}/lm`, { text, n: S.n, smoothing: S.smoothing, gamma: S.gamma, top_k: 5 }, { signal: ctrl.signal });
        last = r; paintSugg(r); paintPP(r);
      } catch (e) { handleErr(e); }
    }

    function paintSugg(r) {
      clear(sugg);
      if (!r.predictions.length) { return; }
      const max = Math.max(...r.predictions.map((x) => x.prob)) || 1;
      sugg.append(h('span', { class: 'sugg-label' }, r.mode === 'completing' ? `Finish "${r.prefix}"` : 'Next word'));
      r.predictions.forEach((x) => sugg.append(h('button', { class: 'sg', type: 'button', onclick: () => accept(x.word),
        title: x.source },
        h('strong', {}, x.word), h('em', {}, (x.prob * 100).toFixed(x.prob < 0.1 ? 1 : 0) + '%' + (x.source.includes('backoff') || x.source === 'most frequent' ? ' \u00b7 backoff' : '')),
        h('i', { style: `width:${Math.max(6, (x.prob / max) * 100)}%` }))));
    }

    function paintPP(r) {
      clear(tokens); clear(kv); legend.hidden = true;
      const pp = r.perplexity;
      if (!pp) { ppNum.textContent = '\u2013'; verdict.className = 'pp-verdict'; verdict.textContent = 'Type a sentence to see how closely it follows the slides.'; pin.hidden = true; ref.hidden = true; return; }
      ppNum.textContent = pp.value >= 1000 ? fmt(Math.round(pp.value)) : pp.value.toFixed(2);
      verdict.className = 'pp-verdict ' + pp.level; verdict.textContent = pp.verdict;
      const lnV = Math.log(Math.max(r.vocab_size, 3));
      pin.hidden = false; pin.style.left = clamp((Math.log(Math.max(pp.value, 1)) / lnV) * 100, 1, 99) + '%';
      ref.hidden = false; ref.style.left = clamp((Math.log(Math.max(r.reference.p90, 1)) / lnV) * 100, 1, 99) + '%';
      [['Cross-entropy (H)', pp.cross_entropy.toFixed(3) + ' bits'], ['Words scored', pp.tokens], ['Unseen words', pp.oov], ['Vocabulary', fmt(r.vocab_size)]]
        .forEach(([k, v]) => kv.append(h('dt', {}, k), h('dd', {}, String(v))));
      (r.tokens || []).forEach((t) => tokens.append(h('span', { class: 'tk' + (t.oov ? ' oov' : ''), style: `--s:${clamp(t.surprisal / 12, 0, 1).toFixed(2)}`,
        title: `p = ${t.prob.toExponential(2)} \u00b7 ${t.surprisal.toFixed(1)} bits` }, t.token)));
      legend.hidden = !(r.tokens && r.tokens.length);
    }

    function accept(word) {
      let v = area.value;
      if (last && last.mode === 'completing') v = v.replace(/[A-Za-z0-9']+$/, '');
      else if (v && !/\s$/.test(v)) v += ' ';
      area.value = v + word + ' ';
      area.focus(); area.setSelectionRange(area.value.length, area.value.length);
      run();
    }

    area.addEventListener('input', runSoon);
    area.addEventListener('keydown', (e) => {
      if (e.key === 'Tab' && !e.shiftKey && last && last.predictions.length && area.selectionStart === area.value.length && area.value.trim()) {
        e.preventDefault(); accept(last.predictions[0].word);
      }
    });
    syncGamma();
  }

  /* ───────── boot: restore session after a refresh ───────── */
  (async () => {
    const sid = store.get('qd-sid');
    if (!sid) return;
    try { enterWorkspace(await api(`/api/session/${sid}/overview`)); }
    catch (_) { store.set('qd-sid', null); }
  })();
})();
