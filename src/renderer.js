/* ═══════════════════════════════════════════════════════════════════════════════
   AI CORE POLICY CHANGER — renderer.js
   All UI logic, state management, and event wiring.
   ═══════════════════════════════════════════════════════════════════════════════ */

'use strict';

// ─── App State ────────────────────────────────────────────────────────────────

const State = {
  filePath:   null,
  fileSize:   0,
  parsed:     null,    // result of GGUFParser.parse()
  hexOffset:  0,
  hexRows:    64,      // rows per hex page
  settings: {
    scanlines:  true,
    matrix:     true,
    glitch:     true,
    vignette:   true,
    rainColor:  'red',
    hexRows:    64,
    maxHeader:  32,    // MiB
  },
};

// ─── Logging ──────────────────────────────────────────────────────────────────

const LOG_LEVELS = { sys:0, info:1, ok:2, warn:3, error:4, debug:5 };

function log (level, ...args) {
  const msg  = args.join(' ');
  const ts   = new Date().toTimeString().slice(0,8);
  const term = document.getElementById('terminal');
  const auto = document.getElementById('term-autoscroll')?.checked ?? true;

  const line = document.createElement('div');
  line.className = `term-line ${level}`;
  line.innerHTML =
    `<span class="term-ts">${ts}</span>` +
    `<span class="term-tag">${level.toUpperCase()}</span>` +
    `<span class="term-msg">${escHtml(msg)}</span>`;
  term.appendChild(line);

  if (auto) term.scrollTop = term.scrollHeight;

  const countEl = document.getElementById('term-count-label');
  if (countEl) countEl.textContent = `${term.querySelectorAll('.term-line').length} lines`;

  // Mirror to console for DevTools
  const conFn = level === 'error' ? console.error : level === 'warn' ? console.warn : console.log;
  conFn(`[${level.toUpperCase()}]`, msg);
}

// ─── Toast notifications ──────────────────────────────────────────────────────

function toast (message, type = 'info', duration = 3500) {
  const icons = { ok:'✔', error:'✘', warn:'⚠', info:'ℹ' };
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.innerHTML = `<span>${icons[type] ?? ''}</span><span>${escHtml(message)}</span>`;
  document.getElementById('toast-container').appendChild(el);
  setTimeout(() => {
    el.classList.add('leaving');
    el.addEventListener('animationend', () => el.remove(), { once: true });
  }, duration);
}

// ─── Modal ────────────────────────────────────────────────────────────────────

function showModal ({ title, body, buttons = [] }) {
  document.getElementById('modal-head').textContent = title;
  document.getElementById('modal-body').innerHTML = body;
  const foot = document.getElementById('modal-foot');
  foot.innerHTML = '';
  for (const btn of buttons) {
    const b = document.createElement('button');
    b.className = `btn ${btn.cls || 'btn-secondary'}`;
    b.textContent = btn.label;
    b.onclick = () => { closeModal(); btn.action?.(); };
    foot.appendChild(b);
  }
  document.getElementById('modal-backdrop').style.display = 'flex';
}

function closeModal () {
  document.getElementById('modal-backdrop').style.display = 'none';
}

// ─── Status bar ───────────────────────────────────────────────────────────────

function setStatus (mode, fileLabel) {
  const el = document.getElementById('sb-mode');
  if (el) el.textContent = mode || 'STANDBY';
  if (fileLabel !== undefined) {
    const f = document.getElementById('sb-file');
    if (f) f.textContent = fileLabel || 'NO FILE';
  }
}

function setTitleStatus (text, state) {
  const dot  = document.getElementById('tb-status-dot');
  const span = document.getElementById('tb-status-text');
  if (dot)  { dot.className = `tb-status-dot ${state || ''}`; }
  if (span) span.textContent = text || 'STANDBY';
}

function setProgress (pct) {
  const bar  = document.getElementById('sb-progress');
  const fill = document.getElementById('sb-progress-fill');
  if (pct === null) { bar.style.display = 'none'; return; }
  bar.style.display = 'block';
  fill.style.width  = `${Math.max(0, Math.min(100, pct))}%`;
}

// ─── Panel switching ──────────────────────────────────────────────────────────

function switchPanel (name) {
  document.querySelectorAll('.nav-item').forEach(el => {
    el.classList.toggle('active', el.dataset.panel === name);
    el.setAttribute('aria-selected', el.dataset.panel === name);
  });
  document.querySelectorAll('.panel').forEach(el => {
    el.classList.toggle('active', el.id === `panel-${name}`);
  });
  setStatus(name.toUpperCase());
}

// ─── Stats update ─────────────────────────────────────────────────────────────

function refreshStats () {
  const p = State.parsed;
  document.getElementById('stat-files').textContent    = State.filePath ? '1' : '0';
  document.getElementById('stat-meta').textContent     = p ? p.metadata.size : '0';
  document.getElementById('stat-tensors').textContent  = p ? p.tensors.length : '0';
  const modCount = p ? [...p.metadata.values()].filter(e => e.modified).length : 0;
  document.getElementById('stat-modified').textContent = modCount;
  document.getElementById('stat-size').textContent     = State.fileSize ? formatBytes(State.fileSize) : '—';
  document.getElementById('stat-ver').textContent      = p ? `v${p.version}` : '—';
  document.getElementById('forge-empty') && refreshForge();
}

// ─── GGUF Load / Parse ────────────────────────────────────────────────────────

async function loadGGUF (filePath) {
  log('info', `Loading: ${filePath}`);
  setTitleStatus('DECRYPTING…', 'active');
  setStatus('DECRYPTING', baseName(filePath));

  showParseProgress(true, 'READING FILE HEADER…', 10);

  const maxBytes = State.settings.maxHeader * 1024 * 1024;
  const statsRes = await window.api.getFileStats(filePath);
  if (!statsRes.success) { return loadError(statsRes.error); }

  State.fileSize = statsRes.size;

  showParseProgress(true, 'PARSING GGUF HEADER…', 40);
  const readRes = await window.api.readFileHead(filePath, maxBytes);
  if (!readRes.success) { return loadError(readRes.error); }

  showParseProgress(true, 'BUILDING METADATA MAP…', 75);

  let parsed;
  try {
    const parser = new GGUFParser(readRes.data);
    parsed = parser.parse();
  } catch (err) {
    return loadError(`Parse error: ${err.message}`);
  }

  showParseProgress(true, 'INDEXING TENSORS…', 95);

  State.filePath = filePath;
  State.parsed   = parsed;
  State.hexOffset = 0;

  showParseProgress(true, 'COMPLETE', 100);
  setTimeout(() => showParseProgress(false), 600);

  log('ok', `Parsed GGUF v${parsed.version} — ${parsed.metadata.size} metadata keys, ${parsed.tensors.length} tensors`);
  log('info', `File size: ${formatBytes(State.fileSize)} | Header end: 0x${parsed.headerEnd.toString(16).toUpperCase()} | Data offset: 0x${parsed.dataOffset.toString(16).toUpperCase()}`);

  setTitleStatus('ACTIVE', 'ok');
  setStatus('READY', baseName(filePath));

  renderCoreFileInfo();
  renderPoliciesTable();
  renderTensorsTable();
  renderHex();
  refreshStats();
  refreshForge();
  toast(`Loaded: ${baseName(filePath)}`, 'ok');
}

function loadError (msg) {
  log('error', msg);
  showParseProgress(false);
  setTitleStatus('ERROR', 'warn');
  toast(msg, 'error', 5000);
}

function showParseProgress (visible, label, pct) {
  const el   = document.getElementById('parse-progress');
  const lbl  = document.getElementById('parse-progress-label');
  const fill = document.getElementById('parse-progress-fill');
  el.style.display = visible ? 'block' : 'none';
  if (visible) {
    if (label) lbl.textContent = label;
    if (pct !== undefined) fill.style.width = `${pct}%`;
  }
  setProgress(visible ? pct : null);
}

// ─── Core panel: file info ────────────────────────────────────────────────────

function renderCoreFileInfo () {
  const p   = State.parsed;
  const el  = document.getElementById('core-file-info');
  if (!p) { el.innerHTML = `<div class="no-file">NO FILE LOADED</div><div class="no-file-sub">DRAG &amp; DROP A .GGUF FILE ANYWHERE<br>OR USE THE DECRYPT PANEL</div>`; return; }

  const arch = p.metadata.get('general.architecture')?.value ?? '—';
  const name = p.metadata.get('general.name')?.value ?? baseName(State.filePath);
  const rows = [
    ['NAME',        name],
    ['ARCHITECTURE',arch],
    ['FILE',        baseName(State.filePath)],
    ['SIZE',        formatBytes(State.fileSize)],
    ['GGUF VER',    `v${p.version}`],
    ['TENSORS',     p.tensors.length.toLocaleString()],
    ['META KEYS',   p.metadata.size.toLocaleString()],
    ['DATA OFFSET', `0x${p.dataOffset.toString(16).toUpperCase()}`],
  ];
  el.innerHTML = rows.map(([k,v]) =>
    `<div class="parse-info-row"><span>${k}</span><span>${escHtml(String(v))}</span></div>`
  ).join('');
}

// ─── Policies panel ───────────────────────────────────────────────────────────

let _policySortKey = 'key';
let _policySortAsc = true;

function renderPoliciesTable (filter = '') {
  const p = State.parsed;
  const emptyMsg = document.getElementById('policies-empty');
  const wrap     = document.getElementById('policies-table-wrap');
  if (!p) { emptyMsg.style.display = 'flex'; wrap.style.display = 'none'; return; }
  emptyMsg.style.display = 'none';
  wrap.style.display = 'block';

  const q    = filter.toLowerCase();
  const rows = [...p.metadata.values()]
    .filter(e => !q || e.key.toLowerCase().includes(q) || String(e.value).toLowerCase().includes(q))
    .sort((a, b) => {
      let va = a[_policySortKey], vb = b[_policySortKey];
      if (typeof va === 'string') va = va.toLowerCase();
      if (typeof vb === 'string') vb = vb.toLowerCase();
      return _policySortAsc ? (va < vb ? -1 : va > vb ? 1 : 0) : (va > vb ? -1 : va < vb ? 1 : 0);
    });

  const tbody = document.getElementById('policies-tbody');
  tbody.innerHTML = rows.map(e => {
    const isArr  = e.valueType === 9;
    const canEdit = !isArr;
    const dispVal = valueToDisplayString(e);
    const tagCls  = e.valueType === 8 ? 'tag-str' : (e.valueType >= 10 || (e.valueType >= 0 && e.valueType <= 6)) ? 'tag-num' : e.valueType === 7 ? 'tag-bool' : e.valueType === 9 ? 'tag-arr' : 'tag-num';
    return `<tr class="${e.modified ? 'modified' : ''}" data-key="${escAttr(e.key)}">
      <td title="${escAttr(e.key)}">${escHtml(e.key)}</td>
      <td><span class="tag ${tagCls}">${escHtml(e.typeName)}</span></td>
      <td class="val-cell ${canEdit ? 'editable' : ''}" data-key="${escAttr(e.key)}" title="${escAttr(dispVal)}">${escHtml(dispVal)}</td>
      <td>${e.modified ? '<span class="tag tag-mod">MODIFIED</span>' : '<span class="tag tag-ok">ORIGINAL</span>'}</td>
      <td>
        ${canEdit ? `<button class="btn btn-sm btn-secondary" onclick="editPolicyInline('${escAttr(e.key)}')">EDIT</button>` : ''}
        <button class="btn btn-sm btn-danger" onclick="deletePolicy('${escAttr(e.key)}')">DEL</button>
      </td>
    </tr>`;
  }).join('');
}

function editPolicyInline (key) {
  const p = State.parsed;
  if (!p) return;
  const entry = p.metadata.get(key);
  if (!entry) return;

  const td = document.querySelector(`td.val-cell[data-key="${CSS.escape(key)}"]`);
  if (!td) return;
  const cur = entry.valueType === 8 ? entry.value : valueToDisplayString(entry);
  td.innerHTML = `<input class="val-input" value="${escAttr(String(cur))}" autofocus>`;
  const inp = td.querySelector('input');
  inp.focus(); inp.select();
  inp.addEventListener('blur',    () => commitPolicyEdit(key, inp.value));
  inp.addEventListener('keydown', e => {
    if (e.key === 'Enter')  commitPolicyEdit(key, inp.value);
    if (e.key === 'Escape') renderPoliciesTable(document.getElementById('policy-search').value);
  });
}

function commitPolicyEdit (key, rawValue) {
  const p = State.parsed;
  if (!p) return;
  const entry = p.metadata.get(key);
  if (!entry) return;

  let coerced;
  try { coerced = coerceValue(entry.valueType, rawValue); }
  catch (err) { toast(`Invalid value: ${err.message}`, 'error'); renderPoliciesTable(document.getElementById('policy-search').value); return; }

  const changed = String(coerced) !== String(entry.value);
  entry.value    = coerced;
  entry.modified = changed || entry.modified;
  if (changed) log('info', `Policy edited: ${key} = ${coerced}`);
  refreshStats();
  renderPoliciesTable(document.getElementById('policy-search').value);
}

function deletePolicy (key) {
  const p = State.parsed;
  if (!p) return;
  showModal({
    title: 'DELETE POLICY',
    body:  `<div class="modal-label">Remove key: <b style="color:var(--cyan)">${escHtml(key)}</b>?<br>This cannot be undone.</div>`,
    buttons: [
      { label: 'CANCEL', cls: 'btn-secondary' },
      { label: 'DELETE', cls: 'btn-danger', action () {
          p.metadata.delete(key);
          p.metaCount = p.metadata.size;
          log('warn', `Policy deleted: ${key}`);
          refreshStats();
          renderPoliciesTable(document.getElementById('policy-search').value);
          toast(`Deleted: ${key}`, 'warn');
        }
      },
    ],
  });
}

function showAddPolicyModal () {
  showModal({
    title: 'ADD POLICY KEY',
    body: `
      <div class="modal-group">
        <div class="modal-label">KEY NAME</div>
        <input class="modal-input" id="new-key" placeholder="e.g. general.name" spellcheck="false">
      </div>
      <div class="modal-group">
        <div class="modal-label">TYPE</div>
        <select class="modal-input" id="new-type">
          <option value="8">STRING</option>
          <option value="4">UINT32</option>
          <option value="10">UINT64</option>
          <option value="6">FLOAT32</option>
          <option value="7">BOOL</option>
          <option value="5">INT32</option>
        </select>
      </div>
      <div class="modal-group">
        <div class="modal-label">VALUE</div>
        <input class="modal-input" id="new-val" placeholder="Value" spellcheck="false">
      </div>`,
    buttons: [
      { label: 'CANCEL',  cls: 'btn-secondary' },
      { label: 'ADD KEY', cls: 'btn-primary', action () {
          const key  = document.getElementById('new-key').value.trim();
          const type = parseInt(document.getElementById('new-type').value, 10);
          const raw  = document.getElementById('new-val').value;
          if (!key) { toast('Key name is required', 'error'); return; }
          if (!State.parsed) return;
          let coerced;
          try { coerced = coerceValue(type, raw); } catch (e) { toast(`Invalid value: ${e.message}`, 'error'); return; }
          State.parsed.metadata.set(key, {
            key, valueType: type,
            typeName: GGUFTypeName[type] ?? '?',
            value: coerced, modified: true,
          });
          State.parsed.metaCount = State.parsed.metadata.size;
          log('ok', `Policy added: ${key} = ${coerced}`);
          refreshStats();
          renderPoliciesTable(document.getElementById('policy-search').value);
          toast(`Added: ${key}`, 'ok');
        }
      },
    ],
  });
}

function exportPoliciesJson () {
  const p = State.parsed;
  if (!p) { toast('No file loaded', 'warn'); return; }
  const obj = {};
  for (const [key, entry] of p.metadata) {
    obj[key] = { type: entry.typeName, value: typeof entry.value === 'bigint' ? entry.value.toString() : entry.value, modified: entry.modified };
  }
  const json = JSON.stringify(obj, null, 2);
  const defaultName = (baseName(State.filePath) || 'metadata').replace(/\.gguf$/i, '') + '_metadata.json';
  window.api.saveJsonDialog(defaultName).then(res => {
    if (!res.canceled && res.filePath) {
      window.api.writeJson(res.filePath, json).then(r => {
        if (r.success) { log('ok', `Exported JSON: ${res.filePath}`); toast('Exported!', 'ok'); }
        else toast(r.error, 'error');
      });
    }
  });
}

function resetPolicies () {
  const p = State.parsed;
  if (!p) return;
  showModal({
    title: 'RESET ALL MODIFICATIONS',
    body: '<div class="modal-label">Revert ALL modified policy values to their original parsed values?</div>',
    buttons: [
      { label: 'CANCEL', cls: 'btn-secondary' },
      { label: 'RESET ALL', cls: 'btn-danger', action () {
          // Re-parse from disk
          if (State.filePath) { loadGGUF(State.filePath); }
          toast('All changes reverted', 'warn');
        }
      },
    ],
  });
}

// ─── Tensors panel ────────────────────────────────────────────────────────────

let _tensorSortKey = 'name';
let _tensorSortAsc = true;

function renderTensorsTable (filter = '', typeFilter = '') {
  const p = State.parsed;
  const emptyMsg = document.getElementById('tensors-empty');
  const wrap     = document.getElementById('tensors-table-wrap');
  if (!p) { emptyMsg.style.display = 'flex'; wrap.style.display = 'none'; return; }
  emptyMsg.style.display = 'none';
  wrap.style.display = 'block';

  // Populate type filter if needed
  const sel = document.getElementById('tensor-type-filter');
  if (sel.options.length === 1) {
    const types = [...new Set(p.tensors.map(t => t.typeName))].sort();
    types.forEach(t => { const o = document.createElement('option'); o.value = t; o.textContent = t; sel.appendChild(o); });
  }

  const q = filter.toLowerCase();
  let rows = p.tensors.filter(t =>
    (!q || t.name.toLowerCase().includes(q)) &&
    (!typeFilter || t.typeName === typeFilter)
  );

  rows = rows.slice().sort((a, b) => {
    let va = a[_tensorSortKey] ?? '';
    let vb = b[_tensorSortKey] ?? '';
    if (typeof va === 'bigint') va = Number(va);
    if (typeof vb === 'bigint') vb = Number(vb);
    if (typeof va === 'string') va = va.toLowerCase();
    if (typeof vb === 'string') vb = vb.toLowerCase();
    return _tensorSortAsc ? (va < vb ? -1 : va > vb ? 1 : 0) : (va > vb ? -1 : va < vb ? 1 : 0);
  });

  document.getElementById('tensor-count-label').textContent = `${rows.length} / ${p.tensors.length} tensors`;

  const tbody = document.getElementById('tensors-tbody');
  tbody.innerHTML = rows.map((t, i) => {
    const dims = t.dims.map(d => d.toString()).join(' × ');
    const elems = t.numElements.toString();
    const off  = `0x${t.offset.toString(16).toUpperCase()}`;
    return `<tr style="cursor:pointer" onclick="showTensorDrawer(${p.tensors.indexOf(t)})">
      <td style="color:var(--text-dim)">${p.tensors.indexOf(t)}</td>
      <td title="${escAttr(t.name)}">${escHtml(t.name)}</td>
      <td><span class="tag tag-num">${escHtml(t.typeName)}</span></td>
      <td>${escHtml(dims)}</td>
      <td style="color:var(--cyan)">${escHtml(elems)}</td>
      <td style="color:var(--text-dim)">${off}</td>
    </tr>`;
  }).join('');
}

function showTensorDrawer (idx) {
  const p = State.parsed;
  if (!p) return;
  const t = p.tensors[idx];
  if (!t) return;

  document.getElementById('drawer-name').textContent = t.name;

  const rows = [
    ['INDEX',    idx],
    ['NAME',     t.name],
    ['TYPE',     t.typeName],
    ['GGML TYPE ID', t.ggmlType],
    ['DIMENSIONS', t.dims.join(' × ')],
    ['N DIMS',   t.dims.length],
    ['ELEMENTS', t.numElements.toString()],
    ['OFFSET',   `0x${t.offset.toString(16).toUpperCase()}`],
  ];

  document.getElementById('drawer-body').innerHTML = rows.map(([k,v]) =>
    `<div class="parse-info-row"><span>${k}</span><span style="color:var(--cyan)">${escHtml(String(v))}</span></div>`
  ).join('');

  document.getElementById('tensor-drawer').style.display = 'block';
  document.getElementById('sb-offset').textContent = `OFF: 0x${t.offset.toString(16).toUpperCase().padStart(8,'0')}`;
}

// ─── Hex viewer ───────────────────────────────────────────────────────────────

const HEX_BYTES_PER_ROW = 16;

async function renderHex (offset) {
  if (offset !== undefined) State.hexOffset = offset;
  const p = State.parsed;
  const emptyEl = document.getElementById('hex-empty');
  const container = document.getElementById('hex-container');

  if (!State.filePath) { emptyEl.style.display = 'flex'; container.style.display = 'none'; return; }
  emptyEl.style.display = 'none';
  container.style.display = 'flex';

  const rows    = State.settings.hexRows;
  const chunkSz = rows * HEX_BYTES_PER_ROW;
  const off     = Math.max(0, State.hexOffset);

  const res = await window.api.readFileChunk(State.filePath, off, chunkSz);
  if (!res.success) { log('error', `Hex read error: ${res.error}`); return; }

  const bytes = res.data;
  const body  = document.getElementById('hex-body');
  const html  = [];

  for (let r = 0; r < rows; r++) {
    const rowOff = r * HEX_BYTES_PER_ROW;
    if (rowOff >= bytes.length) break;
    const absOff = off + rowOff;

    let byteStr = '', asciiStr = '';
    for (let c = 0; c < HEX_BYTES_PER_ROW; c++) {
      if (c === 8) byteStr += '<span class="hex-gap"></span>';
      const idx = rowOff + c;
      if (idx < bytes.length) {
        const b = bytes[idx];
        const cls = b === 0 ? 'null-byte' : (b >= 0x20 && b < 0x7F) ? 'ascii-byte' : (b > 0x7F) ? 'high-byte' : '';
        byteStr += `<span class="hex-byte ${cls}">${b.toString(16).toUpperCase().padStart(2,'0')}</span>`;
        asciiStr += (b >= 0x20 && b < 0x7F) ? escHtml(String.fromCharCode(b)) : '·';
      } else {
        byteStr += '<span class="hex-byte" style="opacity:0">  </span>';
        asciiStr += ' ';
      }
    }

    html.push(`<div class="hex-row">
      <span class="hex-off">${absOff.toString(16).toUpperCase().padStart(8,'0')}</span>
      <span class="hex-bytes-col">${byteStr}</span>
      <span class="hex-ascii-text">${asciiStr}</span>
    </div>`);
  }

  body.innerHTML = html.join('');

  const totalPages = Math.ceil(State.fileSize / chunkSz);
  const curPage    = Math.floor(off / chunkSz) + 1;
  document.getElementById('hex-page-info').textContent =
    `Page ${curPage} / ${totalPages} — Offset 0x${off.toString(16).toUpperCase()} — ${formatBytes(State.fileSize)} total`;
  document.getElementById('hex-offset-input').value = off.toString(16).toUpperCase();
  document.getElementById('sb-offset').textContent  = `OFF: 0x${off.toString(16).toUpperCase().padStart(8,'0')}`;
}

// ─── Forge ────────────────────────────────────────────────────────────────────

function refreshForge () {
  const p         = State.parsed;
  const emptyEl   = document.getElementById('forge-empty');
  const contentEl = document.getElementById('forge-content');
  if (!p) { emptyEl.style.display = 'flex'; contentEl.style.display = 'none'; return; }
  emptyEl.style.display = 'none';
  contentEl.style.display = 'block';

  const modified = [...p.metadata.values()].filter(e => e.modified);
  const summary  = document.getElementById('forge-summary');
  if (modified.length === 0) {
    summary.innerHTML = '<div style="color:var(--text-dim);font-size:12px">No modifications — output will be identical to input.</div>';
  } else {
    summary.innerHTML = `<div style="margin-bottom:8px;color:var(--orange)">${modified.length} key(s) modified:</div>` +
      modified.map(e => `<div class="parse-info-row"><span style="color:var(--cyan)">${escHtml(e.key)}</span><span style="color:var(--orange)">${escHtml(valueToDisplayString(e))}</span></div>`).join('');
  }

  const defaultOut = State.filePath
    ? State.filePath.replace(/\.gguf$/i, '') + '_modified.gguf'
    : 'output_modified.gguf';
  const outPath = document.getElementById('forge-out-path');
  if (!outPath.value) outPath.value = defaultOut;

  document.getElementById('btn-forge-rebuild').disabled = !State.filePath;
}

async function startForge () {
  const p       = State.parsed;
  const outPath = document.getElementById('forge-out-path').value.trim();
  if (!p)       { toast('No file loaded', 'warn'); return; }
  if (!outPath) { toast('Select an output path first', 'warn'); return; }
  if (outPath === State.filePath) { toast('Output path must differ from input', 'error'); return; }

  const btn = document.getElementById('btn-forge-rebuild');
  btn.disabled = true;
  showForgeProgress(true, 'BUILDING HEADER…', 10);
  forgeLog('info', 'Forge started');
  forgeLog('info', `Input:  ${State.filePath}`);
  forgeLog('info', `Output: ${outPath}`);

  let headerBytes;
  try {
    const writer  = new GGUFWriter(p);
    headerBytes   = writer.build();
    forgeLog('ok', `Header built — ${formatBytes(headerBytes.length)}`);
  } catch (err) {
    forgeLog('err', `Header build failed: ${err.message}`);
    btn.disabled = false;
    showForgeProgress(false);
    toast(`Build failed: ${err.message}`, 'error');
    return;
  }

  showForgeProgress(true, 'STREAMING TENSOR DATA…', 40);

  const res = await window.api.rebuildGguf(
    State.filePath, outPath,
    Array.from(headerBytes),
    p.dataOffset
  );

  if (!res.success) {
    forgeLog('err', `Rebuild failed: ${res.error}`);
    btn.disabled = false;
    showForgeProgress(false);
    toast(`Rebuild failed: ${res.error}`, 'error');
    return;
  }

  showForgeProgress(true, 'VERIFYING MAGIC…', 90);

  if (document.getElementById('forge-verify').checked) {
    const verify = await window.api.readFileChunk(outPath, 0, 4);
    if (verify.success && verify.data.length === 4) {
      const magic = String.fromCharCode(...verify.data);
      if (magic === 'GGUF') {
        forgeLog('ok', `Magic verified: GGUF ✔`);
      } else {
        forgeLog('warn', `Magic unexpected: ${magic}`);
      }
    }
  }

  showForgeProgress(true, 'COMPLETE', 100);
  setTimeout(() => showForgeProgress(false), 800);

  forgeLog('ok', `Done — ${formatBytes(res.bytesWritten)} written`);
  btn.disabled = false;
  toast('Forge complete!', 'ok');
  log('ok', `Forge complete → ${outPath}`);
}

function showForgeProgress (visible, label, pct) {
  const el   = document.getElementById('forge-progress');
  const lbl  = document.getElementById('forge-progress-label');
  const fill = document.getElementById('forge-progress-fill');
  el.style.display = visible ? 'block' : 'none';
  if (visible) {
    if (label) lbl.textContent = label;
    if (pct !== undefined) fill.style.width = `${pct}%`;
  }
  setProgress(visible ? pct : null);
}

function forgeLog (level, msg) {
  const div = document.getElementById('forge-log');
  const line = document.createElement('div');
  const ts = new Date().toTimeString().slice(0,8);
  line.className = `forge-log-line ${level}`;
  line.textContent = `[${ts}] ${msg}`;
  div.appendChild(line);
  div.scrollTop = div.scrollHeight;
}

// ─── Decode panel info ────────────────────────────────────────────────────────

function renderDecryptResult () {
  const p = State.parsed;
  if (!p) return;
  const pr = document.getElementById('parse-result');
  pr.style.display = 'grid';

  const arch = p.metadata.get('general.architecture')?.value ?? '—';

  document.getElementById('parse-info').innerHTML = [
    ['FILE',         baseName(State.filePath)],
    ['SIZE',         formatBytes(State.fileSize)],
    ['GGUF VERSION', `v${p.version}`],
    ['TENSORS',      p.tensorCount.toLocaleString()],
    ['META KEYS',    p.metaCount.toLocaleString()],
    ['HEADER END',   `0x${p.headerEnd.toString(16).toUpperCase()}`],
    ['DATA OFFSET',  `0x${p.dataOffset.toString(16).toUpperCase()}`],
    ['ALIGNMENT',    `${p.alignment} bytes`],
  ].map(([k,v]) =>
    `<div class="parse-info-row"><span>${k}</span><span>${escHtml(String(v))}</span></div>`
  ).join('');

  // Architecture fingerprint
  const archKeys = [...p.metadata.keys()].filter(k => k.startsWith(arch + '.') || k.startsWith('general.'));
  document.getElementById('arch-info').innerHTML = archKeys.slice(0, 20).map(k => {
    const e = p.metadata.get(k);
    return `<div class="parse-info-row"><span style="color:var(--text-dim)">${escHtml(k)}</span><span style="color:var(--cyan)">${escHtml(valueToDisplayString(e))}</span></div>`;
  }).join('') + (archKeys.length > 20 ? `<div style="color:var(--text-dim);font-size:10px;padding-top:6px">… and ${archKeys.length - 20} more</div>` : '');
}

// ─── Matrix Rain ──────────────────────────────────────────────────────────────

const RAIN_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@#$%^&*()アイウエオカキクケコサシスセソ';

function initMatrixRain () {
  const canvas = document.getElementById('matrix-canvas');
  const ctx    = canvas.getContext('2d');
  let cols, drops, animId;

  const COLORS = {
    red:    ['#cc0000','#ff4444','#880000'],
    green:  ['#00cc00','#44ff44','#008800'],
    purple: ['#7700cc','#bb44ff','#440088'],
    cyan:   ['#00cccc','#44ffff','#008888'],
  };

  function resize () {
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
    cols  = Math.ceil(canvas.width / 14);
    drops = Array(cols).fill(1);
  }

  function frame () {
    if (!State.settings.matrix) { ctx.clearRect(0,0,canvas.width,canvas.height); animId = requestAnimationFrame(frame); return; }
    const palette = COLORS[State.settings.rainColor] || COLORS.red;
    ctx.fillStyle = 'rgba(8,8,8,0.05)';
    ctx.fillRect(0,0,canvas.width,canvas.height);
    ctx.font = '13px Courier New';
    for (let i = 0; i < cols; i++) {
      const ch  = RAIN_CHARS[Math.floor(Math.random() * RAIN_CHARS.length)];
      const x   = i * 14;
      const y   = drops[i] * 14;
      const col = palette[Math.floor(Math.random() * palette.length)];
      ctx.fillStyle = col;
      ctx.fillText(ch, x, y);
      if (y > canvas.height && Math.random() > 0.975) drops[i] = 0;
      drops[i]++;
    }
    animId = requestAnimationFrame(frame);
  }

  window.addEventListener('resize', resize);
  resize();
  frame();
}

// ─── Clock ────────────────────────────────────────────────────────────────────

function startClock () {
  function tick () {
    const el = document.getElementById('sb-clock');
    if (el) el.textContent = new Date().toTimeString().slice(0,8);
  }
  tick();
  setInterval(tick, 1000);
}

// ─── Settings ─────────────────────────────────────────────────────────────────

function initSettings () {
  const link = (id, prop, onChange) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener('change', () => {
      State.settings[prop] = el.type === 'checkbox' ? el.checked : el.value;
      if (onChange) onChange(State.settings[prop]);
      log('debug', `Setting changed: ${prop} = ${State.settings[prop]}`);
    });
  };

  link('s-scanlines', 'scanlines', v => { document.querySelector('.scanlines').style.opacity = v ? '1' : '0'; });
  link('s-matrix',    'matrix',    () => {});
  link('s-glitch',    'glitch',    v => { document.querySelectorAll('.glitch').forEach(el => el.style.setProperty('--glitch', v ? '' : 'none')); });
  link('s-vignette',  'vignette',  v => { document.querySelector('.vignette').style.opacity = v ? '1' : '0'; });
  link('s-rain-color','rainColor', () => {});
  link('s-hex-rows',  'hexRows',   v => { State.settings.hexRows = parseInt(v,10); State.hexRows = parseInt(v,10); });
  link('s-max-header','maxHeader', v => { State.settings.maxHeader = parseInt(v,10); });
}

// ─── Drag & drop (global) ─────────────────────────────────────────────────────

function initDragDrop () {
  document.addEventListener('dragover', e => { e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; });
  document.addEventListener('drop', e => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) loadGGUF(file.path);
  });

  const dz = document.getElementById('drop-zone');
  dz.addEventListener('dragenter', () => dz.classList.add('drag-over'));
  dz.addEventListener('dragleave', () => dz.classList.remove('drag-over'));
  dz.addEventListener('drop', e => {
    e.preventDefault();
    dz.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) loadGGUF(file.path);
  });
  dz.addEventListener('click', () => document.getElementById('btn-browse').click());
}

// ─── Utilities ────────────────────────────────────────────────────────────────

function escHtml (s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}
function escAttr (s) { return escHtml(s).replace(/\n/g,'&#10;'); }
function baseName (p) { return p ? p.split(/[\\/]/).pop() : ''; }
function formatBytes (n) {
  if (n < 1024)                return `${n} B`;
  if (n < 1048576)             return `${(n/1024).toFixed(2)} KiB`;
  if (n < 1073741824)          return `${(n/1048576).toFixed(2)} MiB`;
  return `${(n/1073741824).toFixed(2)} GiB`;
}

function coerceValue (type, raw) {
  switch (type) {
    case 0:  { const v = parseInt(raw,10);  if (isNaN(v)||v<0||v>255)  throw new Error('Expected UINT8 (0–255)');  return v; }
    case 1:  { const v = parseInt(raw,10);  if (isNaN(v)||v<-128||v>127) throw new Error('Expected INT8');         return v; }
    case 2:  { const v = parseInt(raw,10);  if (isNaN(v)||v<0)          throw new Error('Expected UINT16');        return v; }
    case 3:  { const v = parseInt(raw,10);  if (isNaN(v))               throw new Error('Expected INT16');         return v; }
    case 4:  { const v = parseInt(raw,10);  if (isNaN(v)||v<0)          throw new Error('Expected UINT32');        return v; }
    case 5:  { const v = parseInt(raw,10);  if (isNaN(v))               throw new Error('Expected INT32');         return v; }
    case 6:  { const v = parseFloat(raw);   if (isNaN(v))               throw new Error('Expected FLOAT32');       return v; }
    case 7:  return raw === 'true' || raw === '1' || raw.toLowerCase() === 'true';
    case 8:  return String(raw);
    case 10: try { return BigInt(raw.trim()); } catch { throw new Error('Expected UINT64 integer'); }
    case 11: try { return BigInt(raw.trim()); } catch { throw new Error('Expected INT64 integer'); }
    case 12: { const v = parseFloat(raw);   if (isNaN(v))               throw new Error('Expected FLOAT64');       return v; }
    default: return raw;
  }
}

// ─── Boot ─────────────────────────────────────────────────────────────────────

async function boot () {
  log('sys', 'AI CORE POLICY CHANGER — NIGHTMARE EDITION — INITIALISING');

  initMatrixRain();
  startClock();
  initSettings();
  initDragDrop();

  // Version info
  const versions = await window.api.getVersions();
  document.getElementById('si-platform').textContent  = versions.platform;
  document.getElementById('si-arch').textContent      = versions.arch;
  document.getElementById('si-node').textContent      = versions.node;
  document.getElementById('si-electron').textContent  = versions.electron;
  document.getElementById('si-chrome').textContent    = versions.chrome;
  log('info', `Electron ${versions.electron} | Node ${versions.node} | Chrome ${versions.chrome} | ${versions.platform}/${versions.arch}`);

  // Window controls
  document.getElementById('btn-minimize').onclick = () => window.api.minimize();
  document.getElementById('btn-maximize').onclick = () => window.api.maximize();
  document.getElementById('btn-close').onclick    = () => window.api.close();

  window.api.onWinState(s => {
    document.getElementById('btn-maximize').textContent = s === 'maximized' ? '❐' : '□';
  });

  // Nav
  document.querySelectorAll('.nav-item').forEach(el => {
    el.addEventListener('click', () => {
      switchPanel(el.dataset.panel);
      if (el.dataset.panel === 'hex') renderHex();
      if (el.dataset.panel === 'forge') refreshForge();
    });
  });

  // Quick actions
  document.getElementById('qa-load').onclick     = () => switchPanel('decrypt');
  document.getElementById('qa-policies').onclick = () => switchPanel('policies');
  document.getElementById('qa-tensors').onclick  = () => switchPanel('tensors');
  document.getElementById('qa-forge').onclick    = () => switchPanel('forge');

  // Decrypt panel
  document.getElementById('btn-browse').onclick = async () => {
    const res = await window.api.openFileDialog();
    if (!res.canceled && res.filePaths[0]) {
      await loadGGUF(res.filePaths[0]);
      renderDecryptResult();
    }
  };

  // Policies toolbar
  document.getElementById('policy-search').addEventListener('input', e => renderPoliciesTable(e.target.value));
  document.getElementById('btn-add-policy').onclick    = showAddPolicyModal;
  document.getElementById('btn-export-json').onclick   = exportPoliciesJson;
  document.getElementById('btn-reset-policies').onclick = resetPolicies;

  // Policy table sort
  document.querySelectorAll('#policies-table th[data-sort]').forEach(th => {
    th.addEventListener('click', () => {
      if (_policySortKey === th.dataset.sort) { _policySortAsc = !_policySortAsc; }
      else { _policySortKey = th.dataset.sort; _policySortAsc = true; }
      renderPoliciesTable(document.getElementById('policy-search').value);
    });
  });

  // Tensors toolbar
  document.getElementById('tensor-search').addEventListener('input', e =>
    renderTensorsTable(e.target.value, document.getElementById('tensor-type-filter').value));
  document.getElementById('tensor-type-filter').addEventListener('change', e =>
    renderTensorsTable(document.getElementById('tensor-search').value, e.target.value));
  document.getElementById('drawer-close').onclick = () => {
    document.getElementById('tensor-drawer').style.display = 'none';
  };

  // Hex navigation
  document.getElementById('btn-hex-goto').onclick = () => {
    const raw = document.getElementById('hex-offset-input').value.replace(/^0x/i,'');
    const off = parseInt(raw, 16);
    if (!isNaN(off)) renderHex(off);
  };
  document.getElementById('btn-hex-start').onclick = () => renderHex(0);
  document.getElementById('btn-hex-end').onclick   = () => {
    if (!State.fileSize) return;
    const rows = State.settings.hexRows;
    const off  = Math.max(0, State.fileSize - rows * HEX_BYTES_PER_ROW);
    renderHex(off);
  };
  document.getElementById('btn-hex-prev').onclick  = () => {
    const step = State.settings.hexRows * HEX_BYTES_PER_ROW;
    renderHex(Math.max(0, State.hexOffset - step));
  };
  document.getElementById('btn-hex-next').onclick  = () => {
    const step = State.settings.hexRows * HEX_BYTES_PER_ROW;
    renderHex(Math.min(State.fileSize - 1, State.hexOffset + step));
  };
  document.getElementById('hex-offset-input').addEventListener('keydown', e => {
    if (e.key === 'Enter') document.getElementById('btn-hex-goto').click();
  });

  // Forge
  document.getElementById('btn-forge-pick').onclick = async () => {
    const defaultName = State.filePath
      ? State.filePath.replace(/\.gguf$/i,'') + '_modified.gguf'
      : 'output_modified.gguf';
    const res = await window.api.saveFileDialog(defaultName);
    if (!res.canceled && res.filePath) {
      document.getElementById('forge-out-path').value = res.filePath;
    }
  };
  document.getElementById('btn-forge-rebuild').onclick = startForge;

  // Terminal
  document.getElementById('btn-term-clear').onclick = () => {
    document.getElementById('terminal').innerHTML = '';
    document.getElementById('term-count-label').textContent = '0 lines';
  };
  document.getElementById('btn-term-copy').onclick = () => {
    const lines = [...document.querySelectorAll('.term-msg')].map(el => el.textContent);
    navigator.clipboard.writeText(lines.join('\n')).then(() => toast('Copied!', 'ok'));
  };

  // Modal close on backdrop click
  document.getElementById('modal-backdrop').addEventListener('click', e => {
    if (e.target === document.getElementById('modal-backdrop')) closeModal();
  });

  log('ok', 'Boot complete — awaiting your command, nightmare operator');
  setStatus('READY');
  setTitleStatus('STANDBY', '');
}

// Entry point
document.addEventListener('DOMContentLoaded', boot);
