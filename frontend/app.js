/* ═══════════════════════════════════════════════════════════
   EduExtract AI — Frontend Application Logic
   Single-page app communicating with Flask REST API
   ═══════════════════════════════════════════════════════════ */

const API = "";  // same origin

/* ── Utilities ─────────────────────────────────────────── */
const $ = id => document.getElementById(id);
const h = (tag, cls, html) => {
  const el = document.createElement(tag);
  if (cls)  el.className = cls;
  if (html !== undefined) el.innerHTML = html;
  return el;
};

function toast(msg, type = "info", dur = 3200) {
  const t = h("div", `toast ${type}`, msg);
  $("toastContainer").appendChild(t);
  setTimeout(() => t.remove(), dur);
}

function showLoading(v) {
  $("loadingOverlay").classList.toggle("hidden", !v);
}

async function api(path, opts = {}) {
  const res = await fetch(API + path, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || res.statusText);
  }
  return res.json();
}

async function apiRaw(path, opts = {}) {
  const res = await fetch(API + path, opts);
  if (!res.ok) throw new Error(res.statusText);
  return res;
}

function fmtDate(s) {
  return s ? new Date(s).toLocaleString() : "";
}

/* ── State ──────────────────────────────────────────────── */
const State = {
  currentDoc: null,
  docs: [],
  quiz: null,     // { quiz_id, questions, doc_id }
  quizAnswers: {},
};

/* ── Navigation ─────────────────────────────────────────── */
const App = {
  navigate(view) {
    document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
    document.querySelectorAll(".nav-item").forEach(n => n.classList.remove("active"));
    const vEl = $(`view-${view}`);
    if (vEl) vEl.classList.add("active");
    const nEl = document.querySelector(`.nav-item[data-view="${view}"]`);
    if (nEl) nEl.classList.add("active");
    $("topbarTitle").textContent = { home:"Home", upload:"Upload PDF", notes:"Smart Notes",
      quiz:"Quiz Generator", lab:"NLP Lab", history:"Quiz History" }[view] || view;
    if (view === "home")    App.loadHome();
    if (view === "history") History.load();
  },

  async loadHome() {
    try {
      State.docs = await api("/api/documents");
      renderDocList($("homeDocList"), State.docs);
      refreshDocSelect(State.docs);
    } catch(e) { console.error(e); }
  },

  async setActiveDoc(docId) {
    if (!docId) return;
    try {
      showLoading(true);
      const doc = await api(`/api/documents/${docId}`);
      State.currentDoc = doc;
      $("activeDocTitle").textContent = doc.title || doc.filename;
      $("activeDocMeta").textContent = `${doc.page_count} pages · ${doc.sentence_count} sentences`;
      $("globalDocSelect").value = docId;
    } catch(e) { toast(e.message, "error"); }
    finally { showLoading(false); }
  }
};

function renderDocList(container, docs) {
  container.innerHTML = "";
  if (!docs.length) {
    container.innerHTML = `<div class="empty-state">No documents uploaded yet. Go to Upload to get started.</div>`;
    return;
  }
  docs.slice(0, 8).forEach(d => {
    const el = h("div", "doc-entry");
    el.innerHTML = `
      <div class="de-icon">📄</div>
      <div class="de-info">
        <div class="de-title">${escHtml(d.title || d.filename)}</div>
        <div class="de-meta">${d.page_count} pages · ${fmtDate(d.created_at)}</div>
      </div>`;
    el.addEventListener("click", () => {
      App.setActiveDoc(d.id);
      App.navigate("notes");
    });
    container.appendChild(el);
  });
}

function refreshDocSelect(docs) {
  const sel = $("globalDocSelect");
  sel.innerHTML = `<option value="">— Select Document —</option>`;
  docs.forEach(d => {
    const opt = document.createElement("option");
    opt.value = d.id;
    opt.textContent = (d.title || d.filename).substring(0, 50);
    sel.appendChild(opt);
  });
  if (State.currentDoc) sel.value = State.currentDoc.document_id;
}

function escHtml(s) {
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

/* ── Upload ─────────────────────────────────────────────── */
const Upload = {
  init() {
    const zone  = $("uploadZone");
    const input = $("fileInput");

    $("browseLink").addEventListener("click", () => input.click());
    zone.addEventListener("click",     () => input.click());
    zone.addEventListener("dragover",  e => { e.preventDefault(); zone.classList.add("drag-over"); });
    zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
    zone.addEventListener("drop",      e => { e.preventDefault(); zone.classList.remove("drag-over"); Upload.handle(e.dataTransfer.files[0]); });
    input.addEventListener("change",   () => Upload.handle(input.files[0]));
  },

  async handle(file) {
    if (!file || !file.name.toLowerCase().endsWith(".pdf")) {
      toast("Please select a valid PDF file.", "error"); return;
    }
    const prog = $("uploadProgress");
    const bar  = $("progressBar");
    const lbl  = $("progressLabel");
    $("uploadResult").classList.add("hidden");
    prog.classList.remove("hidden");
    bar.style.width = "15%";
    lbl.textContent = "Reading file…";

    const fd = new FormData();
    fd.append("file", file);
    try {
      bar.style.width = "40%"; lbl.textContent = "Extracting pages & running NLP…";
      const doc = await api("/api/documents/upload", { method:"POST", body: fd });
      bar.style.width = "100%"; lbl.textContent = "Done!";

      State.currentDoc = doc;
      await App.setActiveDoc(doc.document_id);
      State.docs = await api("/api/documents");
      refreshDocSelect(State.docs);

      Upload.showResult(doc);
      toast(`"${doc.title}" processed — ${doc.page_count} pages extracted!`, "success");
    } catch(e) {
      toast(e.message, "error");
    } finally {
      setTimeout(() => prog.classList.add("hidden"), 1200);
    }
  },

  showResult(doc) {
    $("urTitle").textContent = doc.title || doc.filename;
    const meta = $("urMeta");
    const items = [
      ["Pages",  doc.page_count],
      ["Sentences", doc.sentence_count],
      ["Tokens", doc.token_count.toLocaleString()],
      ["OCR", doc.ocr_used ? "Used" : "Not needed"],
    ];
    meta.innerHTML = items.map(([l,v]) =>
      `<div class="meta-item"><div class="mi-val">${v}</div><div class="mi-label">${l}</div></div>`
    ).join("");
    $("uploadResult").classList.remove("hidden");
  }
};

/* ── Notes ──────────────────────────────────────────────── */
const Notes = {
  lastResult: null,
  lastMethod: "tfidf",

  init() {
    $("sentSlider").addEventListener("input", () => {
      $("sentSliderVal").textContent = $("sentSlider").value;
    });

    document.querySelectorAll(".tg-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".tg-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        Notes.lastMethod = btn.dataset.method;
      });
    });

    $("genNotesBtn").addEventListener("click", Notes.generate);
    $("dlMd").addEventListener("click", () => Notes.download("markdown"));
    $("dlTxt").addEventListener("click", () => Notes.download("txt"));
  },

  async generate() {
    const doc = State.currentDoc;
    if (!doc) { toast("Select a document first.", "error"); return; }

    const method = Notes.lastMethod;
    const maxS   = parseInt($("sentSlider").value);
    const out    = $("notesOutput");
    out.classList.add("hidden");
    out.innerHTML = `<div class="spinner"><div class="spin-ring"></div><p>Generating ${method === "compare" ? "comparison" : method.toUpperCase()} summary…</p></div>`;
    out.classList.remove("hidden");

    try {
      const res = await api(`/api/documents/${doc.document_id}/notes`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ method, max_sentences: maxS })
      });
      Notes.lastResult = res;
      Notes.lastMethod = method;
      Notes.render(res, method);
      $("notesDownloadBar").classList.remove("hidden");
    } catch(e) { toast(e.message, "error"); out.innerHTML = `<p class="text-danger">${e.message}</p>`; }
  },

  render(res, method) {
    const out = $("notesOutput");
    out.innerHTML = "";

    if (method === "compare") {
      Notes.renderCompare(out, res);
    } else {
      Notes.renderSingle(out, res, method);
    }
  },

  renderSingle(container, res, method) {
    // Stats
    const statsEl = h("div", "stats-bar");
    [
      [res.summary_sentence_count, "Summary Sentences"],
      [res.summary_word_count,     "Summary Words"],
      [`${res.compression_ratio_pct}%`, "Compression"],
      [res.pages_referenced?.join(", ") || "—", "Pages Cited"],
    ].forEach(([v,l]) => {
      const c = h("div", "sb-item");
      c.innerHTML = `<div class="sb-val">${v}</div><div class="sb-label">${l}</div>`;
      statsEl.appendChild(c);
    });
    container.appendChild(statsEl);

    // Sentences
    const col = h("div");
    const heading = h("h3", "", `📌 ${method === "textrank" ? "TextRank" : "TF-IDF"} Key Points`);
    heading.style.marginBottom = "12px";
    col.appendChild(heading);

    (res.sentences || []).forEach((s, i) => {
      const card = h("div", "sentence-card");
      card.innerHTML = `
        <span class="sc-num">${i+1}</span>
        <div class="sc-text">${escHtml(s.text)}</div>
        <div class="sc-meta">
          <span class="sc-badge page">Page ${s.page_number}</span>
          <span class="sc-badge">Score: ${(s.score || 0).toFixed(4)}</span>
          <span class="sc-badge">${escHtml(s.method || method)}</span>
        </div>`;
      col.appendChild(card);
    });
    container.appendChild(col);

    // Keywords
    const kwSection = h("div");
    kwSection.style.marginTop = "28px";
    kwSection.innerHTML = `<h3 style="margin-bottom:12px;">🏷️ Top Keywords</h3>`;
    const kwWrap = h("div", "kw-chips");
    ((res.keywords?.tfidf || []).slice(0, 18)).forEach(k => {
      const chip = h("span", "kw-chip");
      chip.textContent = `${k.keyword} (${k.score.toFixed(3)})`;
      chip.title = `Pages: ${k.pages?.join(", ") || "?"}\n${k.evidence_sentences?.[0] || ""}`;
      kwWrap.appendChild(chip);
    });
    kwSection.appendChild(kwWrap);
    container.appendChild(kwSection);

    // Entities
    const entCats = res.entities || {};
    if (Object.keys(entCats).length) {
      const entSection = h("div");
      entSection.style.marginTop = "24px";
      entSection.innerHTML = `<h3 style="margin-bottom:12px;">👥 Named Entities</h3>`;
      Object.entries(entCats).slice(0, 6).forEach(([cat, list]) => {
        if (!list.length) return;
        const grp = h("div", "entity-group");
        grp.innerHTML = `<div class="eg-title">${escHtml(cat)}</div>`;
        const chips = h("div", "eg-chips");
        list.slice(0, 8).forEach(e => {
          const c = h("span", "eg-chip", escHtml(e.text));
          c.title = `Occurrences: ${e.count} · Pages: ${e.pages?.join(", ")}`;
          chips.appendChild(c);
        });
        grp.appendChild(chips);
        entSection.appendChild(grp);
      });
      container.appendChild(entSection);
    }
  },

  renderCompare(container, res) {
    const header = h("div", "stats-bar");
    header.innerHTML = `
      <div class="sb-item"><div class="sb-val">${res.sentence_overlap_count}</div><div class="sb-label">Shared Sentences</div></div>
      <div class="sb-item"><div class="sb-val">${res.sentence_overlap_percentage}%</div><div class="sb-label">Algorithm Overlap</div></div>`;
    container.appendChild(header);

    const grid = h("div", "notes-compare");

    ["tfidf", "textrank"].forEach(m => {
      const col = h("div", "notes-col");
      const title = m === "tfidf" ? "📊 TF-IDF" : "🕸️ TextRank";
      const data  = res[m];
      col.innerHTML = `<h3>${title} — ${data.compression_ratio_pct}% compression · ${data.summary_word_count} words</h3>`;
      (data.sentences || []).forEach((s, i) => {
        const card = h("div", "sentence-card");
        card.innerHTML = `
          <span class="sc-num">${i+1}</span>
          <div class="sc-text">${escHtml(s.text)}</div>
          <div class="sc-meta">
            <span class="sc-badge page">Page ${s.page_number}</span>
            <span class="sc-badge">Score: ${(s.score || 0).toFixed(4)}</span>
          </div>`;
        col.appendChild(card);
      });
      grid.appendChild(col);
    });
    container.appendChild(grid);
  },

  async download(fmt) {
    const doc = State.currentDoc;
    if (!doc) return;
    const method = ["tfidf","textrank"].includes(Notes.lastMethod) ? Notes.lastMethod : "tfidf";
    try {
      const res = await apiRaw(`/api/documents/${doc.document_id}/export/notes`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ method, format: fmt })
      });
      const blob = await res.blob();
      const url  = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url; a.download = `notes.${fmt === "markdown" ? "md" : "txt"}`;
      a.click(); URL.revokeObjectURL(url);
    } catch(e) { toast(e.message, "error"); }
  }
};

/* ── Quiz ───────────────────────────────────────────────── */
const Quiz = {
  init() {
    $("generateQuizBtn").addEventListener("click", Quiz.generate);
  },

  async generate() {
    const doc = State.currentDoc;
    if (!doc) { toast("Select a document first.", "error"); return; }

    const mcq   = parseInt($("cfgMcq").value)   || 0;
    const fitb  = parseInt($("cfgFitb").value)  || 0;
    const short = parseInt($("cfgShort").value) || 0;
    if (mcq + fitb + short === 0) { toast("Configure at least one question type.", "error"); return; }

    $("quizForm").classList.add("hidden");
    $("quizResults").classList.add("hidden");
    $("quizSpinner").classList.remove("hidden");

    try {
      const quiz = await api(`/api/documents/${doc.document_id}/quiz`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mcq_count: mcq, fitb_count: fitb, short_count: short })
      });
      if (!quiz.questions?.length) {
        toast("No questions could be generated. Try a longer document.", "warn"); return;
      }
      State.quiz = { quiz_id: quiz.quiz_id, questions: quiz.questions, doc_id: doc.document_id };
      State.quizAnswers = {};
      Quiz.render(quiz.questions, quiz.validation_results?.summary);
      toast(`${quiz.questions.length} validated questions ready!`, "success");
    } catch(e) {
      toast(e.message, "error");
    } finally {
      $("quizSpinner").classList.add("hidden");
    }
  },

  render(questions, valSummary) {
    const form = $("quizForm");
    form.innerHTML = "";

    // Validation banner
    if (valSummary) {
      const banner = h("div", "card", `
        <span style="color:var(--success);font-weight:700;">✓ ${valSummary.validated_count} validated</span>
        &nbsp; <span style="color:var(--text-3);">· ${valSummary.review_count} need review · ${valSummary.rejected_count} filtered out</span>
      `);
      banner.style.marginBottom = "16px";
      form.appendChild(banner);
    }

    questions.forEach((q, i) => {
      const block = h("div", "q-block");
      const typeLabel = { mcq: "Multiple Choice", fill_in_the_blank: "Fill in the Blank", short_answer: "Short Answer" }[q.question_type] || q.question_type;
      block.innerHTML = `<div class="q-number">Q${i+1} · ${typeLabel}</div><div class="q-text">${escHtml(q.question)}</div>`;

      if ((q.question_type === "mcq" || q.question_type === "fill_in_the_blank") && q.options?.length) {
        const optList = h("div", "q-options");
        q.options.forEach(opt => {
          const label = h("label", "q-option");
          const radio = document.createElement("input");
          radio.type  = "radio";
          radio.name  = q.question_id;
          radio.value = opt;
          radio.addEventListener("change", () => {
            block.querySelectorAll(".q-option").forEach(o => o.classList.remove("selected"));
            label.classList.add("selected");
            State.quizAnswers[q.question_id] = opt;
          });
          label.appendChild(radio);
          label.appendChild(document.createTextNode(opt));
          optList.appendChild(label);
        });
        block.appendChild(optList);
      } else {
        const inp = h("input");
        inp.type  = "text";
        inp.className = "q-text-input";
        inp.placeholder = "Type your answer here…";
        inp.addEventListener("input", () => { State.quizAnswers[q.question_id] = inp.value; });
        block.appendChild(inp);
      }

      const src = h("div", "q-source", `📖 Source: Page ${q.source_page || "?"} · <em>${escHtml((q.generation_method || "").replace(/_/g," "))}</em>`);
      block.appendChild(src);
      form.appendChild(block);
    });

    const submitBar = h("div", "quiz-submit-bar");
    const submitBtn = h("button", "btn btn-primary", "Submit Quiz →");
    submitBtn.addEventListener("click", Quiz.submit);
    submitBar.appendChild(submitBtn);
    form.appendChild(submitBar);

    form.classList.remove("hidden");
  },

  async submit() {
    const { quiz_id, questions, doc_id } = State.quiz || {};
    if (!quiz_id) return;

    try {
      showLoading(true);
      const result = await api(`/api/quiz/${quiz_id}/submit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ doc_id, questions, answers: State.quizAnswers })
      });
      Quiz.renderResults(result);
    } catch(e) {
      toast(e.message, "error");
    } finally {
      showLoading(false);
    }
  },

  renderResults(result) {
    $("quizForm").classList.add("hidden");
    const el = $("quizResults");
    el.innerHTML = "";

    const pct = result.percentage;
    const pctClass = pct >= 75 ? "text-success" : pct >= 50 ? "text-warn" : "text-danger";

    const header = h("div", "results-header");
    header.innerHTML = `
      <div class="score-big ${pctClass}">${result.score}/${result.total_score}</div>
      <div class="score-pct ${pctClass}">${pct}% · ${pct >= 60 ? "Passed ✅" : "Needs more practice ⚠️"}</div>`;
    el.appendChild(header);

    (result.detailed_results || []).forEach((r, i) => {
      const block = h("div", `r-block ${r.is_correct ? "correct" : "wrong"}`);
      block.innerHTML = `
        <div class="r-status">${r.is_correct ? "✓ Correct" : "✗ Incorrect"}</div>
        <div class="r-question">Q${i+1}: ${escHtml(r.question)}</div>
        <div class="r-answers">
          <div>
            <div class="r-ans-label">Your Answer</div>
            <div class="r-ans-val">${escHtml(r.user_answer || "(no answer)")}</div>
          </div>
          <div>
            <div class="r-ans-label">Correct Answer</div>
            <div class="r-ans-val">${escHtml(r.correct_answer)}</div>
          </div>
        </div>
        ${r.explanation ? `<div class="r-evidence">📖 ${escHtml(r.explanation)}</div>` : ""}`;
      el.appendChild(block);
    });

    const retryBtn = h("button", "btn btn-secondary", "⟳ Try Again");
    retryBtn.style.marginTop = "16px";
    retryBtn.addEventListener("click", () => {
      $("quizResults").classList.add("hidden");
      Quiz.render(State.quiz.questions);
      State.quizAnswers = {};
    });
    el.appendChild(retryBtn);
    el.classList.remove("hidden");
  }
};

/* ── NLP Lab ─────────────────────────────────────────────── */
const Lab = {
  nlpData: null,

  init() {
    document.querySelectorAll(".lab-tab").forEach(tab => {
      tab.addEventListener("click", () => {
        document.querySelectorAll(".lab-tab").forEach(t => t.classList.remove("active"));
        tab.classList.add("active");
        const which = tab.dataset.lab;
        document.querySelectorAll(".lab-panel").forEach(p => p.classList.add("hidden"));
        const panel = {tokens: $("labTokens"), tfidf: $("labTfidf"), textrank: $("labTextrank"), decoder: $("labDecoder")}[which];
        if (panel) panel.classList.remove("hidden");
        if (which === "tfidf" || which === "textrank") Lab.renderScoredSents(which);
      });
    });

    $("loadNlpBtn").addEventListener("click", Lab.load);
    Lab.initDecoder();
  },

  async load() {
    const doc = State.currentDoc;
    if (!doc) { toast("Select a document first.", "error"); return; }
    try {
      showLoading(true);
      const data = await api(`/api/documents/${doc.document_id}/nlp`);
      Lab.nlpData = data;
      Lab.renderTokens(data.preprocessing);
      toast("NLP analysis loaded.", "success");
    } catch(e) {
      toast(e.message, "error");
    } finally {
      showLoading(false);
    }
  },

  renderTokens(pre) {
    const c = $("labTokensContent");
    c.innerHTML = "";

    const row = h("div", "stat-row");
    [["Token Count", pre.token_count], ["Unique Tokens", pre.unique_token_count],
     ["After Filter", pre.filtered_token_count], ["Stopwords", pre.stopwords_removed]
    ].forEach(([l,v]) => {
      row.innerHTML += `<div class="stat-chip"><div class="sv">${v?.toLocaleString() ?? "—"}</div><div class="sl">${l}</div></div>`;
    });
    c.appendChild(row);

    // Vocab table
    const maxFreq = (pre.top_vocabulary?.[0]?.[1] ?? 1) || 1;
    const tbl = h("table", "vocab-table");
    tbl.innerHTML = `<thead><tr><th>#</th><th>Lemma</th><th>Frequency</th><th>Bar</th></tr></thead>`;
    const tbody = h("tbody");
    (pre.top_vocabulary || []).forEach(([w, cnt], i) => {
      const tr = h("tr");
      const pct = Math.round((cnt / maxFreq) * 100);
      tr.innerHTML = `<td>${i+1}</td><td class="mono">${escHtml(w)}</td><td>${cnt}</td>
        <td><div class="bar-cell"><div class="mini-bar" style="width:${pct}px;"></div>${cnt}</div></td>`;
      tbody.appendChild(tr);
    });
    tbl.appendChild(tbody);
    c.appendChild(h("h4", "", "Top Vocabulary Lemmas"));
    c.appendChild(tbl);

    // POS Table
    if (pre.pos_tags?.length) {
      const pt = h("table", "pos-table");
      pt.innerHTML = `<thead><tr><th>Token</th><th>Lemma</th><th>POS</th></tr></thead>`;
      const ptb = h("tbody");
      pre.pos_tags.forEach(tag => {
        ptb.innerHTML += `<tr><td class="mono">${escHtml(tag.token)}</td><td class="mono">${escHtml(tag.lemma)}</td><td>${escHtml(tag.pos)}</td></tr>`;
      });
      pt.appendChild(ptb);
      c.appendChild(h("h4", "", "Part-of-Speech Tags (Sample)"));
      c.appendChild(pt);
    }

    c.classList.remove("hidden");
  },

  renderScoredSents(which) {
    const data = Lab.nlpData;
    const panel = which === "tfidf" ? $("labTfidf") : $("labTextrank");
    panel.innerHTML = "";
    if (!data) {
      panel.innerHTML = `<p style="color:var(--text-3)">Load NLP analysis first (Tokens tab).</p>`;
      return;
    }
    const rows = which === "tfidf" ? data.tfidf_sentences : data.textrank_sentences;
    const title = which === "tfidf" ? "TF-IDF Sentence Scores" : "TextRank (PageRank) Sentence Scores";
    const maxScore = Math.max(...rows.map(r => r.score), 0.001);
    const tbl = h("table", "score-table");
    tbl.innerHTML = `<thead><tr><th>#</th><th>Page</th><th>Score</th><th>Weight</th><th>Sentence Preview</th></tr></thead>`;
    const tbody = h("tbody");
    rows.forEach((r, i) => {
      const pct = Math.round((r.score / maxScore) * 80);
      tbody.innerHTML += `<tr>
        <td>${i+1}</td>
        <td><span class="sc-badge page">p.${r.page}</span></td>
        <td class="mono">${r.score.toFixed(5)}</td>
        <td><div class="bar-cell"><div class="mini-bar" style="width:${pct}px;"></div></div></td>
        <td>${escHtml(r.text)}</td>
      </tr>`;
    });
    tbl.appendChild(tbody);
    panel.innerHTML = `<h3 style="margin-bottom:16px;">${title}</h3>`;
    panel.appendChild(tbl);
  },

  initDecoder() {
    const linkRange = (rangeId, labelId, fmt) => {
      const r = $(rangeId), l = $(labelId);
      r.addEventListener("input", () => { l.textContent = fmt ? fmt(parseFloat(r.value)) : r.value; });
    };
    linkRange("dcTemp","dcTempVal", v => v.toFixed(1));
    linkRange("dcK",   "dcKVal");
    linkRange("dcP",   "dcPVal",  v => v.toFixed(2));
    linkRange("dcB",   "dcBVal");
    linkRange("dcM",   "dcMVal");

    $("runDecoderBtn").addEventListener("click", Lab.runDecoder);
  },

  async runDecoder() {
    const payload = {
      strategy:    $("dcStrategy").value,
      temperature: parseFloat($("dcTemp").value),
      top_k:       parseInt($("dcK").value),
      top_p:       parseFloat($("dcP").value),
      beam_width:  parseInt($("dcB").value),
      max_tokens:  parseInt($("dcM").value),
      seed: 42,
    };
    try {
      showLoading(true);
      const res = await api("/api/nlp/decode", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      Lab.renderDecoderOutput(res, payload);
    } catch(e) {
      toast(e.message, "error");
    } finally {
      showLoading(false);
    }
  },

  renderDecoderOutput(res, cfg) {
    const out = $("decoderOutput");
    out.innerHTML = "";

    out.innerHTML += `<h4>Generated Text</h4>
      <div class="generated-text">${escHtml(res.generated_text || "(empty)")}</div>`;

    out.innerHTML += `<h4>Token Sequence</h4>`;
    const chips = h("div", "token-chips");
    (res.tokens || []).forEach(t => chips.innerHTML += `<span class="token-chip">${escHtml(t)}</span>`);
    out.appendChild(chips);

    // Logit distribution table
    if (res.logit_table?.length) {
      out.innerHTML += `<h4 style="margin-top:16px;">Next-Token Distribution (Step 1)</h4>`;
      const maxP = Math.max(...res.logit_table.map(r => r.probability));
      const tbl = h("table", "score-table");
      tbl.innerHTML = `<thead><tr><th>Token</th><th>Raw Logit</th><th>Scaled (T=${cfg.temperature})</th><th>P(token)</th><th>Distribution</th></tr></thead>`;
      const tbody = h("tbody");
      res.logit_table.forEach(row => {
        const pct = Math.round((row.probability / maxP) * 80);
        tbody.innerHTML += `<tr>
          <td class="mono">${escHtml(row.token)}</td>
          <td class="mono">${row.logit.toFixed(3)}</td>
          <td class="mono">${row.scaled_logit.toFixed(3)}</td>
          <td class="mono">${(row.probability * 100).toFixed(2)}%</td>
          <td><div class="bar-cell"><div class="mini-bar" style="width:${pct}px;"></div></div></td>
        </tr>`;
      });
      tbl.appendChild(tbody);
      out.appendChild(tbl);
    }

    // Insight
    out.innerHTML += `<div class="insight-box" style="margin-top:16px;">
      <strong>Strategy Note:</strong> ${escHtml(res.notes || "")}
    </div>`;

    out.classList.remove("hidden");
  }
};

/* ── History ─────────────────────────────────────────────── */
const History = {
  async load() {
    try {
      const attempts = await api("/api/quiz/history");
      History.render(attempts);
    } catch(e) { toast(e.message, "error"); }
  },

  render(attempts) {
    const c = $("historyContent");
    if (!attempts.length) {
      c.innerHTML = `<div class="empty-state">No quiz attempts recorded yet.</div>`;
      return;
    }

    const tbl = h("table", "history-table");
    tbl.innerHTML = `<thead><tr>
      <th>Quiz</th><th>Document</th><th>Score</th><th>Result</th><th>Date</th>
    </tr></thead>`;
    const tbody = h("tbody");
    attempts.forEach(a => {
      const pct = a.percentage;
      const cls = pct >= 75 ? "pass" : pct >= 50 ? "warn" : "fail";
      tbody.innerHTML += `<tr>
        <td>${escHtml(a.quiz_title || "Quiz")}</td>
        <td>${escHtml(a.doc_filename || "—")}</td>
        <td>${a.score} / ${a.total_score}</td>
        <td><span class="pct-badge ${cls}">${pct}%</span></td>
        <td>${fmtDate(a.attempted_at)}</td>
      </tr>`;
    });
    tbl.appendChild(tbody);
    c.innerHTML = "";
    c.appendChild(tbl);
  }
};

/* ── Boot ───────────────────────────────────────────────── */
document.addEventListener("DOMContentLoaded", () => {
  // Sidebar nav clicks
  document.querySelectorAll(".nav-item").forEach(n => {
    n.addEventListener("click", e => { e.preventDefault(); App.navigate(n.dataset.view); });
  });

  // Burger (mobile)
  $("burgerBtn").addEventListener("click", () => {
    $("sidebar").classList.toggle("open");
  });

  // Global doc selector
  $("globalDocSelect").addEventListener("change", async function() {
    if (this.value) await App.setActiveDoc(this.value);
  });

  // Init modules
  Upload.init();
  Notes.init();
  Quiz.init();
  Lab.init();

  // Boot to home
  App.navigate("home");
});
