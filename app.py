<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RefManager</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

  :root {
    --bg: #0e0e0e;
    --surface: #161616;
    --border: #2a2a2a;
    --accent: #c8f06e;
    --accent-dim: #8aaa40;
    --text: #e8e8e8;
    --text-dim: #888;
    --text-faint: #444;
    --red: #ff5f5f;
    --mono: 'IBM Plex Mono', monospace;
    --sans: 'IBM Plex Sans', sans-serif;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 14px;
    line-height: 1.5;
    min-height: 100vh;
  }

  /* LAYOUT */
  .app {
    display: grid;
    grid-template-rows: auto 1fr;
    height: 100vh;
    overflow: hidden;
  }

  header {
    padding: 14px 24px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 16px;
    background: var(--surface);
  }

  header .logo {
    font-family: var(--mono);
    font-size: 13px;
    font-weight: 600;
    color: var(--accent);
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }

  header .tagline {
    color: var(--text-dim);
    font-size: 12px;
    font-family: var(--mono);
  }

  header .stats {
    margin-left: auto;
    display: flex;
    gap: 20px;
  }

  .stat {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 1px;
  }

  .stat-val {
    font-family: var(--mono);
    font-size: 16px;
    font-weight: 600;
    color: var(--accent);
    line-height: 1;
  }

  .stat-label {
    font-size: 10px;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  /* MAIN BODY */
  .main {
    display: grid;
    grid-template-columns: 380px 1fr;
    overflow: hidden;
  }

  /* LEFT PANEL */
  .left-panel {
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    background: var(--surface);
  }

  .panel-header {
    padding: 14px 18px 10px;
    border-bottom: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-dim);
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .panel-header span { color: var(--accent); }

  textarea {
    flex: 1;
    resize: none;
    background: transparent;
    border: none;
    outline: none;
    color: var(--text);
    font-family: var(--mono);
    font-size: 11.5px;
    line-height: 1.6;
    padding: 14px 18px;
    overflow-y: auto;
  }

  textarea::placeholder { color: var(--text-faint); }

  textarea::-webkit-scrollbar { width: 4px; }
  textarea::-webkit-scrollbar-track { background: transparent; }
  textarea::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

  /* OPTIONS ROW */
  .options-row {
    padding: 12px 18px;
    border-top: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .toggle-group {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }

  .toggle {
    display: flex;
    align-items: center;
    gap: 6px;
    cursor: pointer;
    user-select: none;
  }

  .toggle input { display: none; }

  .toggle-pill {
    width: 28px;
    height: 15px;
    background: var(--border);
    border-radius: 8px;
    position: relative;
    transition: background 0.2s;
  }

  .toggle-pill::after {
    content: '';
    position: absolute;
    width: 11px;
    height: 11px;
    background: var(--text-dim);
    border-radius: 50%;
    top: 2px;
    left: 2px;
    transition: left 0.2s, background 0.2s;
  }

  .toggle input:checked ~ .toggle-pill { background: var(--accent-dim); }
  .toggle input:checked ~ .toggle-pill::after { left: 15px; background: var(--accent); }

  .toggle-label {
    font-size: 11px;
    color: var(--text-dim);
    font-family: var(--mono);
  }

  .file-row {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .file-label {
    font-size: 11px;
    color: var(--text-dim);
    font-family: var(--mono);
    white-space: nowrap;
  }

  .file-input-wrap {
    flex: 1;
    position: relative;
    overflow: hidden;
  }

  .file-input-wrap input[type="file"] {
    position: absolute;
    opacity: 0;
    width: 100%;
    height: 100%;
    cursor: pointer;
  }

  .file-display {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text-faint);
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 4px 8px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    cursor: pointer;
  }

  .file-display.has-file { color: var(--accent); border-color: var(--accent-dim); }

  /* ACTION BUTTONS */
  .action-row {
    padding: 12px 18px;
    border-top: 1px solid var(--border);
    display: flex;
    gap: 8px;
  }

  .btn {
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    border: none;
    border-radius: 3px;
    cursor: pointer;
    padding: 8px 14px;
    transition: all 0.15s;
  }

  .btn-primary {
    background: var(--accent);
    color: #0e0e0e;
    flex: 1;
  }

  .btn-primary:hover { background: #d8ff7e; }
  .btn-primary:active { transform: scale(0.98); }
  .btn-primary:disabled { background: var(--border); color: var(--text-faint); cursor: not-allowed; }

  .btn-ghost {
    background: transparent;
    color: var(--text-dim);
    border: 1px solid var(--border);
  }

  .btn-ghost:hover { border-color: var(--text-dim); color: var(--text); }

  .btn-danger {
    background: transparent;
    color: var(--red);
    border: 1px solid #3a1a1a;
  }

  .btn-danger:hover { background: #1e0a0a; border-color: var(--red); }

  /* RIGHT PANEL */
  .right-panel {
    display: flex;
    flex-direction: column;
    overflow: hidden;
    background: var(--bg);
  }

  .right-toolbar {
    padding: 10px 18px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 10px;
    flex-shrink: 0;
  }

  .right-toolbar .panel-label {
    font-family: var(--mono);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-dim);
  }

  .right-toolbar .count-badge {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--accent);
    background: rgba(200,240,110,0.08);
    border: 1px solid rgba(200,240,110,0.2);
    border-radius: 10px;
    padding: 2px 8px;
  }

  .toolbar-right {
    margin-left: auto;
    display: flex;
    gap: 8px;
  }

  /* REFERENCES LIST */
  .refs-list {
    flex: 1;
    overflow-y: auto;
    padding: 0;
  }

  .refs-list::-webkit-scrollbar { width: 4px; }
  .refs-list::-webkit-scrollbar-track { background: transparent; }
  .refs-list::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

  .ref-row {
    display: grid;
    grid-template-columns: 48px 100px 1fr 130px 140px;
    gap: 0;
    border-bottom: 1px solid var(--border);
    align-items: center;
    transition: background 0.1s;
    cursor: pointer;
    min-height: 38px;
  }

  .ref-row:hover { background: rgba(255,255,255,0.025); }
  .ref-row.expanded { background: rgba(200,240,110,0.04); }

  .ref-cell {
    padding: 8px 12px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: 12px;
    border-right: 1px solid var(--border);
  }

  .ref-cell:last-child { border-right: none; }

  .cell-idx {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text-faint);
    text-align: center;
  }

  .cell-year {
    font-family: var(--mono);
    font-size: 12px;
    color: var(--accent);
    font-weight: 500;
  }

  .cell-title {
    color: var(--text);
    font-size: 12px;
  }

  .cell-journal {
    color: var(--text-dim);
    font-size: 11px;
    font-style: italic;
  }

  .cell-key {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text-faint);
  }

  /* COL HEADERS */
  .refs-header {
    display: grid;
    grid-template-columns: 48px 100px 1fr 130px 140px;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
    position: sticky;
    top: 0;
    z-index: 10;
    flex-shrink: 0;
  }

  .col-head {
    padding: 6px 12px;
    font-family: var(--mono);
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-faint);
    border-right: 1px solid var(--border);
  }

  .col-head:last-child { border-right: none; }

  /* EXPANDED BIB ROW */
  .bibtex-row {
    display: none;
    border-bottom: 1px solid var(--border);
    background: rgba(200,240,110,0.03);
  }

  .bibtex-row.open { display: block; }

  .bibtex-content {
    padding: 10px 14px;
    font-family: var(--mono);
    font-size: 10.5px;
    color: var(--text-dim);
    white-space: pre-wrap;
    line-height: 1.6;
    border-left: 2px solid var(--accent-dim);
    margin: 0 0 0 60px;
    position: relative;
  }

  .copy-inline-btn {
    position: absolute;
    top: 8px;
    right: 10px;
    font-family: var(--mono);
    font-size: 10px;
    background: var(--border);
    color: var(--text-dim);
    border: none;
    border-radius: 2px;
    padding: 3px 8px;
    cursor: pointer;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    transition: all 0.15s;
  }

  .copy-inline-btn:hover { background: var(--accent-dim); color: #0e0e0e; }
  .copy-inline-btn.copied { background: var(--accent); color: #0e0e0e; }

  /* EMPTY STATE */
  .empty-state {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: var(--text-faint);
    gap: 8px;
    font-family: var(--mono);
    font-size: 12px;
  }

  .empty-icon { font-size: 32px; opacity: 0.3; }

  /* LOADING */
  .loading-bar {
    height: 2px;
    background: var(--border);
    display: none;
    overflow: hidden;
    flex-shrink: 0;
  }

  .loading-bar.active { display: block; }

  .loading-fill {
    height: 100%;
    background: var(--accent);
    animation: load-sweep 1.4s infinite ease-in-out;
  }

  @keyframes load-sweep {
    0% { width: 0; margin-left: 0; }
    50% { width: 60%; margin-left: 20%; }
    100% { width: 0; margin-left: 100%; }
  }

  /* TOAST */
  .toast {
    position: fixed;
    bottom: 24px;
    right: 24px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 10px 16px;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text);
    opacity: 0;
    transform: translateY(8px);
    transition: all 0.2s;
    z-index: 100;
    pointer-events: none;
  }

  .toast.show { opacity: 1; transform: translateY(0); }
  .toast.success { border-color: var(--accent-dim); color: var(--accent); }
  .toast.error { border-color: var(--red); color: var(--red); }

  /* MODE TABS */
  .mode-tabs {
    display: flex;
    gap: 0;
    border-bottom: 1px solid var(--border);
  }

  .mode-tab {
    font-family: var(--mono);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 8px 14px;
    cursor: pointer;
    color: var(--text-dim);
    border: none;
    background: transparent;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
    transition: all 0.15s;
  }

  .mode-tab.active { color: var(--accent); border-bottom-color: var(--accent); }
  .mode-tab:hover:not(.active) { color: var(--text); }

  .author-chip {
    display: inline-block;
    font-size: 11px;
    color: var(--text-dim);
  }
</style>
</head>
<body>
<div class="app">
  <header>
    <div class="logo">RefManager</div>
    <div class="tagline">// bibtex pipeline</div>
    <div class="stats">
      <div class="stat">
        <div class="stat-val" id="stat-total">0</div>
        <div class="stat-label">refs</div>
      </div>
      <div class="stat">
        <div class="stat-val" id="stat-enriched">—</div>
        <div class="stat-label">enriched</div>
      </div>
    </div>
  </header>

  <div class="main">
    <!-- LEFT: INPUT PANEL -->
    <div class="left-panel">
      <div class="mode-tabs">
        <button class="mode-tab active" data-mode="bibtex">BibTeX</button>
        <button class="mode-tab" data-mode="title">Titles</button>
      </div>

      <div class="panel-header">
        Input <span id="char-count">0 chars</span>
      </div>

      <textarea id="bibtex-input" placeholder="@article{smith2024,
  author = {Smith, John},
  title  = {Deep Learning for Energy Systems},
  journal = {Applied Energy},
  year   = {2024},
  doi    = {10.1000/xyz}
}

Paste one or more BibTeX entries..."></textarea>

      <div class="options-row">
        <div class="toggle-group">
          <label class="toggle">
            <input type="checkbox" id="opt-enrich">
            <div class="toggle-pill"></div>
            <span class="toggle-label">Crossref</span>
          </label>
          <label class="toggle">
            <input type="checkbox" id="opt-abbreviate">
            <div class="toggle-pill"></div>
            <span class="toggle-label">Abbreviate</span>
          </label>
          <label class="toggle">
            <input type="checkbox" id="opt-protect">
            <div class="toggle-pill"></div>
            <span class="toggle-label">Protect caps</span>
          </label>
          <label class="toggle">
            <input type="checkbox" id="opt-save">
            <div class="toggle-pill"></div>
            <span class="toggle-label">Save to DB</span>
          </label>
        </div>

        <div class="file-row">
          <span class="file-label">.tex file</span>
          <div class="file-input-wrap">
            <input type="file" id="latex-file" accept=".tex">
            <div class="file-display" id="file-display">no file selected</div>
          </div>
        </div>
      </div>

      <div class="action-row">
        <button class="btn btn-primary" id="process-btn" onclick="processRefs()">Process</button>
        <button class="btn btn-ghost" onclick="clearInput()" title="Clear input">↺</button>
        <button class="btn btn-danger" onclick="clearAll()" title="Clear all results">✕</button>
      </div>
    </div>

    <!-- RIGHT: RESULTS PANEL -->
    <div class="right-panel">
      <div class="loading-bar" id="loading-bar">
        <div class="loading-fill"></div>
      </div>

      <div class="right-toolbar">
        <span class="panel-label">References</span>
        <span class="count-badge" id="ref-count">0 entries</span>
        <div class="toolbar-right">
          <button class="btn btn-ghost" onclick="copyAll()" id="copy-all-btn" style="display:none">Copy all BibTeX</button>
          <button class="btn btn-ghost" onclick="exportBib()" id="export-btn" style="display:none">Export .bib</button>
        </div>
      </div>

      <div class="refs-header" id="refs-header" style="display:none">
        <div class="col-head">#</div>
        <div class="col-head">Year / Author</div>
        <div class="col-head">Title</div>
        <div class="col-head">Journal</div>
        <div class="col-head">BibKey</div>
      </div>

      <div class="refs-list" id="refs-list">
        <div class="empty-state" id="empty-state">
          <div class="empty-icon">⌗</div>
          <div>paste bibtex and press process</div>
        </div>
      </div>
    </div>
  </div>
</div>

<div class="toast" id="toast"></div>

<script>
let processedData = [];
let currentMode = 'bibtex';

// MODE TABS
document.querySelectorAll('.mode-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.mode-tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    currentMode = tab.dataset.mode;
    const ta = document.getElementById('bibtex-input');
    if (currentMode === 'title') {
      ta.placeholder = 'Deep Learning for Energy Systems\nNeural Networks for Battery Management\n...\n\nOne title per line';
    } else {
      ta.placeholder = '@article{smith2024,\n  author = {Smith, John},\n  title  = {Deep Learning for Energy Systems},\n  journal = {Applied Energy},\n  year   = {2024},\n  doi    = {10.1000/xyz}\n}\n\nPaste one or more BibTeX entries...';
    }
  });
});

// CHAR COUNT
document.getElementById('bibtex-input').addEventListener('input', function() {
  const len = this.value.length;
  document.getElementById('char-count').textContent = len.toLocaleString() + ' chars';
});

// FILE LABEL
document.getElementById('latex-file').addEventListener('change', function() {
  const display = document.getElementById('file-display');
  if (this.files.length > 0) {
    display.textContent = this.files[0].name;
    display.classList.add('has-file');
  } else {
    display.textContent = 'no file selected';
    display.classList.remove('has-file');
  }
});

function showToast(msg, type = 'success') {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = 'toast ' + type + ' show';
  setTimeout(() => t.classList.remove('show'), 2800);
}

function setLoading(v) {
  document.getElementById('loading-bar').classList.toggle('active', v);
  document.getElementById('process-btn').disabled = v;
  document.getElementById('process-btn').textContent = v ? 'Processing…' : 'Process';
}

async function processRefs() {
  const content = document.getElementById('bibtex-input').value.trim();
  if (!content) { showToast('No input provided', 'error'); return; }

  setLoading(true);

  const formData = new FormData();
  formData.append('bibtex_content', content);
  formData.append('input_mode', currentMode);
  formData.append('enrich', document.getElementById('opt-enrich').checked);
  formData.append('abbreviate', document.getElementById('opt-abbreviate').checked);
  formData.append('protect', document.getElementById('opt-protect').checked);
  formData.append('save_to_db', document.getElementById('opt-save').checked);

  const latexFile = document.getElementById('latex-file').files[0];
  if (latexFile) formData.append('latex_file', latexFile);

  try {
    const res = await fetch('/api/process', { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok || !data.success) {
      showToast(data.error || 'Processing failed', 'error');
      setLoading(false);
      return;
    }

    processedData = data.full_data || data.data || [];
    renderRefs(processedData);
    document.getElementById('stat-total').textContent = processedData.length;

    const enriched = processedData.filter(r => r.Title_Similarity > 80).length;
    document.getElementById('stat-enriched').textContent =
      document.getElementById('opt-enrich').checked ? enriched : '—';

    showToast(`Processed ${processedData.length} references`);
  } catch (e) {
    showToast('Network error: ' + e.message, 'error');
  }
  setLoading(false);
}

function getFirstAuthor(authors) {
  if (!authors) return '—';
  const parts = authors.split(/,|and/);
  const first = parts[0].trim();
  const nameParts = first.split(/\s+/);
  return nameParts[nameParts.length - 1];
}

function getBestBib(row) {
  return row.Crossref_BibTeX_Protected
    || row.Crossref_BibTeX_Abbrev
    || row.Crossref_BibTeX_LocalKey
    || row.Crossref_BibTeX
    || row.BibTeX
    || '';
}

function renderRefs(refs) {
  const list = document.getElementById('refs-list');
  const header = document.getElementById('refs-header');
  const empty = document.getElementById('empty-state');
  const copyBtn = document.getElementById('copy-all-btn');
  const exportBtn = document.getElementById('export-btn');
  document.getElementById('ref-count').textContent = refs.length + ' entries';

  if (!refs.length) {
    list.innerHTML = '';
    list.appendChild(empty);
    header.style.display = 'none';
    copyBtn.style.display = 'none';
    exportBtn.style.display = 'none';
    return;
  }

  header.style.display = 'grid';
  copyBtn.style.display = 'inline-block';
  exportBtn.style.display = 'inline-block';

  list.innerHTML = refs.map((r, i) => {
    const key = r.Key || r.Reference || '—';
    const year = r.Year || '—';
    const author = getFirstAuthor(r.Authors);
    const title = r.Title || '—';
    const journal = r['Journal/Booktitle'] || r.Journal_Booktitle || '—';
    const bib = getBestBib(r);

    return `
      <div class="ref-row" onclick="toggleBib(${i})" id="row-${i}">
        <div class="ref-cell cell-idx">${i + 1}</div>
        <div class="ref-cell cell-year"><div>${year}</div><div class="author-chip">${author}</div></div>
        <div class="ref-cell cell-title">${escHtml(title)}</div>
        <div class="ref-cell cell-journal">${escHtml(journal)}</div>
        <div class="ref-cell cell-key">${escHtml(key)}</div>
      </div>
      <div class="bibtex-row" id="bib-${i}">
        <div class="bibtex-content" id="bib-content-${i}">${escHtml(bib)}<button class="copy-inline-btn" id="copy-btn-${i}" onclick="copyBib(event,${i})">copy</button></div>
      </div>`;
  }).join('');
}

function toggleBib(i) {
  const row = document.getElementById(`row-${i}`);
  const bib = document.getElementById(`bib-${i}`);
  const isOpen = bib.classList.contains('open');
  // close all others
  document.querySelectorAll('.bibtex-row.open').forEach(el => {
    el.classList.remove('open');
    const idx = el.id.replace('bib-', '');
    document.getElementById(`row-${idx}`).classList.remove('expanded');
  });
  if (!isOpen) {
    bib.classList.add('open');
    row.classList.add('expanded');
  }
}

function copyBib(e, i) {
  e.stopPropagation();
  const bib = getBestBib(processedData[i]);
  navigator.clipboard.writeText(bib).then(() => {
    const btn = document.getElementById(`copy-btn-${i}`);
    btn.textContent = 'copied!';
    btn.classList.add('copied');
    setTimeout(() => { btn.textContent = 'copy'; btn.classList.remove('copied'); }, 1500);
  });
}

function copyAll() {
  const all = processedData.map(r => getBestBib(r)).filter(Boolean).join('\n\n');
  navigator.clipboard.writeText(all).then(() => showToast('Copied ' + processedData.length + ' entries'));
}

function exportBib() {
  const all = processedData.map(r => getBestBib(r)).filter(Boolean).join('\n\n');
  const blob = new Blob([all], { type: 'text/plain' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'references.bib';
  a.click();
  showToast('Exported references.bib');
}

function clearInput() {
  document.getElementById('bibtex-input').value = '';
  document.getElementById('char-count').textContent = '0 chars';
}

function clearAll() {
  processedData = [];
  renderRefs([]);
  document.getElementById('stat-total').textContent = '0';
  document.getElementById('stat-enriched').textContent = '—';
  const empty = document.getElementById('empty-state');
  document.getElementById('refs-list').appendChild(empty);
  showToast('Cleared');
}

function escHtml(s) {
  if (!s) return '';
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
</script>
</body>
</html>