// AI Remove - thsky90 — web UI
import * as E from './engine.js';
import * as M from './meta.js';

const $ = id => document.getElementById(id);
const IMG_EXT = ['png', 'jpg', 'jpeg', 'jfif', 'webp', 'gif', 'bmp'];
const VID_EXT = ['mp4', 'mov', 'm4v', 'webm', 'avi', 'mkv', 'wmv'];
const ext = n => (n.split('.').pop() || '').toLowerCase();

const state = {
  kind: 'image',
  tabs: { image: { items: [], cur: null, rel: null }, video: { items: [], cur: null, rel: null } },
  outDir: null,
  busy: false,
};
let V = null; // video module (lazy)
const tab = () => state.tabs[state.kind];

// ---------------------------------------------------------------------------
// logging & progress
function log(msg) { const l = $('log'); l.textContent += msg + '\n'; l.scrollTop = l.scrollHeight; }
function progress(f, text, cls = '') {
  const pr = $('prog');
  pr.style.setProperty('--pw', pr.clientWidth + 'px');
  pr.className = 'prog ' + cls;
  $('progBar').style.width = `${Math.round(Math.max(0, Math.min(1, f)) * 100)}%`;
  $('progText').textContent = text; $('progText2').textContent = text;
}
function setDetect(text, cls = '') { const d = $('detect'); d.textContent = text; d.className = 'detect ' + cls; }

// ---------------------------------------------------------------------------
// canvas view with editable square box
const view = {
  cv: $('cv'), before: null, after: null, showAfter: false, box: null, editable: false, ok: true,
  scale: 1, ox: 0, oy: 0, fitted: true, drag: null,
  setImage(img, keep = false) {
    const same = this.before && img && this.before.width === img.width && this.before.height === img.height;
    this.before = img ? toBitmapCanvas(img) : null; this.after = null;
    $('empty').style.display = img ? 'none' : 'grid';
    if (!(keep && same)) this.fit(); else this.draw();
  },
  setAfter(img) { this.after = img ? toBitmapCanvas(img) : null; this.draw(); },
  size() { const r = this.cv.getBoundingClientRect(); return [r.width, r.height]; },
  fit() {
    this.fitted = true;
    if (!this.before) { this.draw(); return; }
    const [w, h] = this.size();
    this.scale = Math.min((w - 16) / this.before.width, (h - 16) / this.before.height);
    this.ox = (w - this.before.width * this.scale) / 2; this.oy = (h - this.before.height * this.scale) / 2;
    this.draw();
  },
  zoomBox(f = 3.5) {
    if (!this.box || !this.before) return;
    const [x, y, s] = this.box, [w, h] = this.size();
    this.scale = Math.min(w, h) / (s * f);
    this.ox = w / 2 - (x + s / 2) * this.scale; this.oy = h / 2 - (y + s / 2) * this.scale;
    this.fitted = false; this.draw();
  },
  toImg(px, py) { return [(px - this.ox) / this.scale, (py - this.oy) / this.scale]; },
  draw() {
    const dpr = window.devicePixelRatio || 1, [w, h] = this.size();
    if (this.cv.width !== Math.round(w * dpr) || this.cv.height !== Math.round(h * dpr)) { this.cv.width = Math.round(w * dpr); this.cv.height = Math.round(h * dpr); }
    const g = this.cv.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = '#1E1F24'; g.fillRect(0, 0, w, h);
    const img = this.showAfter && this.after ? this.after : this.before;
    if (!img) return;
    g.imageSmoothingEnabled = this.scale < 2;
    g.drawImage(img, this.ox, this.oy, img.width * this.scale, img.height * this.scale);
    if (this.box && (!this.showAfter || this.editable)) {
      const [x, y, s] = this.box, X = x * this.scale + this.ox, Y = y * this.scale + this.oy, S = s * this.scale;
      const col = this.editable ? '#38BDF8' : (this.ok ? '#22C55E' : '#F59E0B');
      g.strokeStyle = col; g.lineWidth = 2; g.strokeRect(X, Y, S, S);
      g.setLineDash([4, 3]); g.lineWidth = 1; g.beginPath();
      const cx = X + S / 2, cy = Y + S / 2, r = S / 2;
      g.moveTo(cx, cy - r); g.quadraticCurveTo(cx, cy, cx + r, cy); g.quadraticCurveTo(cx, cy, cx, cy + r);
      g.quadraticCurveTo(cx, cy, cx - r, cy); g.quadraticCurveTo(cx, cy, cx, cy - r); g.stroke(); g.setLineDash([]);
      if (this.editable) { g.fillStyle = col; for (const [a, b] of [[X, Y], [X + S, Y], [X, Y + S], [X + S, Y + S]]) g.fillRect(a - 4, b - 4, 9, 9); }
    }
    g.fillStyle = 'rgba(0,0,0,.6)'; g.fillRect(8, 8, 120, 22); g.fillStyle = '#E5E7EB'; g.font = '12px system-ui';
    g.textAlign = 'center'; g.fillText(this.showAfter && this.after ? 'SAU KHI XỬ LÝ' : 'ẢNH GỐC', 68, 23);
  },
  hit(px, py) {
    if (!this.box) return null;
    const [x, y, s] = this.box, X = x * this.scale + this.ox, Y = y * this.scale + this.oy, S = s * this.scale, h = 10;
    const c = { tl: [X, Y], tr: [X + S, Y], bl: [X, Y + S], br: [X + S, Y + S] };
    for (const k in c) if (Math.abs(px - c[k][0]) <= h && Math.abs(py - c[k][1]) <= h) return k;
    if (px >= X && px <= X + S && py >= Y && py <= Y + S) return 'move';
    return null;
  },
};

function toBitmapCanvas(img) {
  const c = document.createElement('canvas'); c.width = img.width; c.height = img.height;
  const id = img instanceof ImageData ? img : new ImageData(img.data, img.width, img.height);
  c.getContext('2d').putImageData(id, 0, 0); return c;
}

view.cv.addEventListener('contextmenu', e => e.preventDefault());
view.cv.addEventListener('wheel', e => {
  if (!view.before) return; e.preventDefault();
  const r = view.cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
  const [ix, iy] = view.toImg(px, py), f = e.deltaY < 0 ? 1.2 : 1 / 1.2;
  view.scale = Math.max(0.02, Math.min(40, view.scale * f)); view.ox = px - ix * view.scale; view.oy = py - iy * view.scale;
  view.fitted = false; view.draw();
}, { passive: false });
view.cv.addEventListener('pointerdown', e => {
  if (!view.before) return;
  view.cv.setPointerCapture(e.pointerId);
  const r = view.cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
  if (e.button !== 0 || !view.editable) { view.drag = { k: 'pan', px, py, ox: view.ox, oy: view.oy }; return; }
  let k = view.hit(px, py);
  const [ix, iy] = view.toImg(px, py);
  if (!k) { view.box = [ix, iy, 4]; k = 'br'; }
  view.drag = { k, ix, iy, box: [...view.box] };
});
view.cv.addEventListener('pointermove', e => {
  const r = view.cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
  const d = view.drag;
  if (!d) {
    if (view.editable) {
      const k = view.hit(px, py);
      view.cv.style.cursor = { move: 'move', tl: 'nwse-resize', br: 'nwse-resize', tr: 'nesw-resize', bl: 'nesw-resize' }[k] || 'crosshair';
    } else view.cv.style.cursor = 'grab';
    return;
  }
  if (d.k === 'pan') { view.ox = d.ox + px - d.px; view.oy = d.oy + py - d.py; view.fitted = false; view.draw(); return; }
  const [ix, iy] = view.toImg(px, py), [x, y, s] = d.box;
  let nb;
  if (d.k === 'move') nb = [x + ix - d.ix, y + iy - d.iy, s];
  else if (d.k === 'br') nb = [x, y, Math.max(6, Math.max(ix - x, iy - y))];
  else if (d.k === 'tl') { const ax = x + s, ay = y + s, ns = Math.max(6, Math.max(ax - ix, ay - iy)); nb = [ax - ns, ay - ns, ns]; }
  else if (d.k === 'tr') { const ax = x, ay = y + s, ns = Math.max(6, Math.max(ix - ax, ay - iy)); nb = [ax, ay - ns, ns]; }
  else { const ax = x + s, ay = y, ns = Math.max(6, Math.max(ax - ix, iy - ay)); nb = [ax - ns, ay, ns]; }
  view.box = nb; view.draw(); boxChanged(nb);
});
view.cv.addEventListener('pointerup', () => { view.drag = null; });
window.addEventListener('resize', () => (view.fitted ? view.fit() : view.draw()));
window.addEventListener('keydown', e => {
  if (!view.editable || !view.box || ['INPUT', 'SELECT'].includes(document.activeElement.tagName)) return;
  const st = e.shiftKey ? 0.25 : 1; let [x, y, s] = view.box;
  if (e.key === 'ArrowLeft') x -= st; else if (e.key === 'ArrowRight') x += st;
  else if (e.key === 'ArrowUp') y -= st; else if (e.key === 'ArrowDown') y += st;
  else if (e.key === '+' || e.key === '=') { x -= st / 2; y -= st / 2; s += st; }
  else if (e.key === '-') { x += st / 2; y += st / 2; s = Math.max(6, s - st); } else return;
  e.preventDefault(); view.box = [x, y, s]; view.draw(); boxChanged(view.box);
});

// ---------------------------------------------------------------------------
// list
function addFiles(files) {
  let n = 0;
  for (const f of files) {
    const e = ext(f.name);
    const kind = IMG_EXT.includes(e) ? 'image' : VID_EXT.includes(e) ? 'video' : (['heic', 'heif'].includes(e) ? 'heic' : null);
    if (kind === 'heic') { log(`✖ ${f.name}: trình duyệt chưa đọc được HEIC – hãy dùng bản Windows.`); continue; }
    if (!kind) continue;
    const t = state.tabs[kind];
    const rel = f.webkitRelativePath || f.name;
    if (t.items.some(it => it.rel === rel && it.file.size === f.size)) continue;
    t.items.push({ file: f, rel, kind, status: 'Chờ xử lý', manualBox: null, geo: null, size: null, checked: true });
    n++;
    if (kind !== state.kind && !tab().items.length) switchTab(kind);
  }
  renderList(); updateCounts();
  if (tab().cur === null && tab().items.length) select(0);
  if (n) log(`Đã thêm ${n} tệp`);
}

function renderList() {
  const ul = $('list'); ul.innerHTML = '';
  tab().items.forEach((it, i) => {
    const li = document.createElement('li');
    if (i === tab().cur) li.className = 'sel';
    const cb = document.createElement('input'); cb.type = 'checkbox'; cb.checked = !!it.checked;
    cb.title = 'Chọn để xử lý / tải về';
    cb.onclick = e => { e.stopPropagation(); it.checked = cb.checked; updateSel(); };
    const name = document.createElement('span'); name.className = 'nm'; name.textContent = it.rel; name.title = it.rel;
    const st = document.createElement('span'); st.className = 'st ' + (it.cls || '');
    if (it.url) {
      const a = document.createElement('a'); a.href = it.url; a.download = it.outName; a.textContent = it.status + ' ⤓';
      a.title = 'Tải tệp này'; a.onclick = e => e.stopPropagation(); st.appendChild(a);
    } else st.textContent = it.status;
    li.append(cb, name, st); li.onclick = () => select(i); ul.appendChild(li);
  });
  updateSel();
}

function updateSel() {
  const items = tab().items, n = items.length, k = items.filter(x => x.checked).length;
  const ca = $('checkAll');
  ca.checked = n > 0 && k === n; ca.indeterminate = k > 0 && k < n;
  $('selInfo').textContent = n ? `Đã chọn ${k}/${n}` : '';
  if (!state.busy) {
    $('zipSel').disabled = !items.some(x => x.checked && x.blob);
    $('zipAll').disabled = !items.some(x => x.blob);
  }
  const what = state.kind === 'image' ? 'ảnh' : 'video';
  $('start').textContent = n && k < n ? `Xử lý ${k} ${what} đã chọn` : `Bắt đầu xử lý ${what}`;
}
function updateCounts() {
  const a = state.tabs.image.items.length, b = state.tabs.video.items.length;
  $('cntImage').textContent = a ? `(${a})` : ''; $('cntVideo').textContent = b ? `(${b})` : '';
}

function switchTab(kind) {
  state.kind = kind;
  document.querySelectorAll('.tab').forEach(b => b.classList.toggle('active', b.dataset.kind === kind));
  $('listTitle').textContent = kind === 'image' ? 'Danh sách ảnh' : 'Danh sách video';
  $('vrow').hidden = kind !== 'video';
  $('fmtHint').textContent = kind === 'image'
    ? 'Kéo thả tệp vào trang. Định dạng: PNG, JPG, JPEG, JFIF, WEBP, GIF (khung đầu), BMP.'
    : 'Định dạng: MP4, MOV, WEBM, AVI, MKV, WMV. Kết quả xuất MP4 (H.264). Video dài nên dùng bản Windows.';
  $('pickFiles').accept = kind === 'image' ? IMG_EXT.map(e => '.' + e).join(',') : VID_EXT.map(e => '.' + e).join(',');
  renderList(); view.setImage(null); view.box = null; setDetect('');
  if (tab().cur !== null && tab().items[tab().cur]) select(tab().cur);
}

// ---------------------------------------------------------------------------
// preview
async function decodeImage(file) {
  const bmp = await createImageBitmap(file);
  const c = new OffscreenCanvas(bmp.width, bmp.height), g = c.getContext('2d', { willReadFrequently: true });
  g.drawImage(bmp, 0, 0);
  return g.getImageData(0, 0, bmp.width, bmp.height);
}

async function ensureVideo(it) {
  if (!V) V = await import('./video.js');
  const ffm = await V.getFFmpeg(t => progress(0, t));
  if (!it.ffName) {
    it.ffName = `in_${Math.random().toString(36).slice(2)}.${ext(it.file.name)}`;
    await ffm.writeFile(it.ffName, new Uint8Array(await it.file.arrayBuffer()));
    it.info = await V.probe(it.ffName);
  }
  return it.info;
}

let selToken = 0;
async function select(i) {
  const t = tab(); t.cur = i; renderList();
  const it = t.items[i]; if (!it) return;
  const token = ++selToken;
  setView('before'); setDetect('Đang tải…');
  try {
    let img;
    if (it.kind === 'image') img = await decodeImage(it.file);
    else {
      const info = await ensureVideo(it);
      const tt = Math.min(info.duration * ($('seek').value / 1000), Math.max(0, info.duration - 0.1));
      $('tlabel').textContent = tt.toFixed(1) + ' s';
      img = await V.grabFrameRGBA(it.ffName, info, tt);
    }
    if (token !== selToken) return;
    it.size = [img.width, img.height]; it.preview = img;
    view.setImage(img); progress(0, 'Sẵn sàng');
    resolveRel(it);
    if (manual() && it.manualBox) { showBox(it.manualBox, true); setDetect('Khung thủ công'); }
    await autoDetect(it, token);
    if (manual() && !it.manualBox) initManual(it);
  } catch (err) { console.error(err); setDetect('Lỗi: ' + err.message, 'warn'); }
}

async function autoDetect(it, token) {
  if (!it.geo) {
    setDetect('Đang dò logo Gemini…');
    await new Promise(r => setTimeout(r, 10));
    if (it.kind === 'image') {
      const g = E.grayFromRGBA(it.preview.data, it.preview.width, it.preview.height);
      const [score, s, x, y] = E.coarseDetect(g, it.preview.width, it.preview.height);
      it.geo = (!s || score < 0.3) ? { found: false, x, y, size: s, score }
        : E.fitGeometry([g], [x, y, s], 255, it.preview.width, it.preview.height, score);
    } else {
      it.geo = await V.detect(it.ffName, it.info, { onStatus: t => setDetect(t) });
    }
  }
  if (token !== selToken) return;
  const g = it.geo;
  if (g.found) {
    setDetect(`Đã tìm thấy logo tại (${g.x.toFixed(1)}, ${g.y.toFixed(1)}), cỡ ${g.size.toFixed(1)} px – độ khớp ${Math.round((g.score || 0) * 100)}%, độ mờ ${g.strength.toFixed(2)}${g.calib ? ', đã hiệu chuẩn theo video' : ''}.`, 'ok');
    if (!(manual() && it.manualBox)) showBox([g.x, g.y, g.size], true);
  } else {
    setDetect('Không chắc có logo Gemini ở góc phải-dưới. Nếu vẫn thấy ngôi sao, bật "Định vị thủ công" và kéo khung vào đúng chỗ.', 'warn');
    if (!(manual() && it.manualBox)) { view.box = null; view.draw(); }
  }
}

function showBox(b, ok) {
  view.box = [...b]; view.ok = ok; view.draw();
  $('bx').value = b[0].toFixed(1); $('by').value = b[1].toFixed(1); $('bs').value = b[2].toFixed(1);
}

const manual = () => document.querySelector('input[name=mode]:checked').value === 'manual';
function initManual(it) {
  if (!it.size) return;
  const [W, H] = it.size;
  let b;
  if (it.geo && it.geo.size) b = [it.geo.x, it.geo.y, it.geo.size];
  else if (tab().rel) b = fromRel(tab().rel, W, H);
  else { const s = Math.min(W, H) <= 1100 ? 48 * Math.min(W, H) / 1024 : 96 * Math.min(W, H) / 1536; b = [W - 3 * s, H - 3 * s, s]; }
  it.manualBox = b; showBox(b, true); view.zoomBox(5);
}
function boxChanged(b) {
  const it = tab().items[tab().cur]; if (!it) return;
  it.manualBox = [...b]; showBox(b, true); view.setAfter(null); setView('before');
}
const toRel = ([x, y, s], W, H) => { const r = Math.min(W, H); return [(W - x) / r, (H - y) / r, s / r]; };
const fromRel = ([a, b, c], W, H) => { const r = Math.min(W, H); return [W - a * r, H - b * r, c * r]; };
function resolveRel(it) { if (it.manualBox && it.manualBox.rel && it.size) it.manualBox = fromRel(it.manualBox.rel, ...it.size); }

document.querySelectorAll('input[name=mode]').forEach(r => r.addEventListener('change', () => {
  const m = manual(); view.editable = m;
  document.querySelector('.ctl.man').classList.toggle('off', !m);
  const it = tab().items[tab().cur];
  if (m && it) { if (!it.manualBox) initManual(it); else { showBox(it.manualBox, true); view.zoomBox(5); } }
  if (!m && it) { it.manualBox = null; if (it.geo && it.geo.found) showBox([it.geo.x, it.geo.y, it.geo.size], true); else { view.box = null; view.draw(); } }
  view.setAfter(null); setView('before');
}));
['bx', 'by', 'bs'].forEach(id => $(id).addEventListener('change', () => {
  if (!manual()) return;
  const b = [+$('bx').value, +$('by').value, Math.max(6, +$('bs').value)];
  view.box = b; view.draw(); boxChanged(b);
}));

function setView(which) {
  view.showAfter = which === 'after';
  $('vBefore').classList.toggle('on', which !== 'after'); $('vAfter').classList.toggle('on', which === 'after');
  view.draw();
}
$('vBefore').onclick = () => setView('before');
$('vAfter').onclick = () => { if (!view.after) makePreview(); else setView('after'); };
$('preview').onclick = () => makePreview();
$('zoomBox').onclick = () => view.zoomBox(); $('fitView').onclick = () => view.fit();
$('method').onchange = () => { view.setAfter(null); setView('before'); };

function currentOpts(it) {
  const o = { method: $('method').value, mode: 'auto' };
  if (it.manualBox) { o.mode = 'manual'; o.box = it.manualBox; }
  return o;
}

async function makePreview() {
  const it = tab().items[tab().cur]; if (!it || !it.preview) return;
  $('preview').disabled = true; $('preview').textContent = 'Đang xử lý…';
  await new Promise(r => setTimeout(r, 10));
  try {
    const o = currentOpts(it);
    let out;
    if (it.kind === 'image') {
      out = (o.mode === 'auto' && o.method === 'alpha' && it.geo) ? runImageWithGeo(it.preview, it.geo) : E.processImage(it.preview, o).img;
    } else {
      const geo = await videoGeo(it, o);
      const tt = Math.min(it.info.duration * ($('seek').value / 1000), Math.max(0, it.info.duration - 0.1));
      out = await V.previewAfter(it.ffName, it.info, geo, tt, o.method);
    }
    view.setAfter(out); setView('after');
  } catch (err) { console.error(err); setDetect('Lỗi xem trước: ' + err.message, 'warn'); }
  $('preview').disabled = false; $('preview').textContent = 'Xem kết quả';
}

function runImageWithGeo(img, geo) {
  if (!geo.found) return { width: img.width, height: img.height, data: new Uint8ClampedArray(img.data) };
  return E.processImage(img, { mode: 'manual', box: [geo.x, geo.y, geo.size], method: 'alpha' }).img;
}

async function videoGeo(it, o) {
  if (o.mode === 'manual') {
    if (o.method === 'inpaint') return { x: o.box[0], y: o.box[1], size: o.box[2], found: false };
    const key = o.box.map(v => v.toFixed(2)).join(',');
    if (it.manualGeoKey !== key) { it.manualGeo = await V.detect(it.ffName, it.info, { manualBox: o.box, onStatus: t => setDetect(t) }); it.manualGeoKey = key; }
    return it.manualGeo;
  }
  return it.geo || (it.geo = await V.detect(it.ffName, it.info, { onStatus: t => setDetect(t) }));
}

$('snap').onclick = async () => {
  const it = tab().items[tab().cur]; if (!it || !it.manualBox) return;
  setDetect('Đang khớp ngôi sao trong khung…');
  await new Promise(r => setTimeout(r, 10));
  let g;
  if (it.kind === 'image') {
    const gr = E.grayFromRGBA(it.preview.data, it.preview.width, it.preview.height);
    const [score, s, x, y] = E.coarseDetect(gr, it.preview.width, it.preview.height, it.manualBox);
    g = s ? E.fitGeometry([gr], [x, y, s], 255, it.preview.width, it.preview.height, score) : { found: false };
  } else g = await V.detect(it.ffName, it.info, { manualBox: it.manualBox, onStatus: t => setDetect(t) });
  if (g.found) { it.manualBox = [g.x, g.y, g.size]; showBox(it.manualBox, true); setDetect(`Đã khớp: (${g.x.toFixed(2)}, ${g.y.toFixed(2)}), cỡ ${g.size.toFixed(2)} px.`, 'ok'); }
  else setDetect('Không khớp được ngôi sao trong khung – khi xử lý sẽ vá vùng theo đúng khung này.', 'warn');
};
$('applyAll').onclick = () => {
  const it = tab().items[tab().cur]; if (!it || !it.manualBox || !it.size) return;
  const rel = toRel(it.manualBox, ...it.size); tab().rel = rel; let n = 0;
  for (const o of tab().items) if (o !== it) { o.manualBox = { rel }; o.status = 'Thủ công (theo mẫu)'; n++; }
  renderList(); log(`Đã áp dụng khung cho ${n} tệp khác`);
};
$('seek').addEventListener('change', () => { const it = tab().items[tab().cur]; if (it && it.kind === 'video') select(tab().cur); });
$('seek').addEventListener('input', () => { const it = tab().items[tab().cur]; if (it && it.info) $('tlabel').textContent = (it.info.duration * $('seek').value / 1000).toFixed(1) + ' s'; });

// ---------------------------------------------------------------------------
// output
$('pickOut').onclick = async () => {
  if (!window.showDirectoryPicker) { alert('Trình duyệt này không hỗ trợ chọn thư mục lưu. Kết quả sẽ được tải về (từng tệp hoặc .zip). Dùng Chrome/Edge để lưu thẳng vào thư mục.'); return; }
  try { state.outDir = await window.showDirectoryPicker({ mode: 'readwrite' }); $('outName').textContent = 'Lưu vào: ' + state.outDir.name; }
  catch (e) { /* cancelled */ }
};

async function saveOut(it, blob, name) {
  if (state.outDir) {
    const parts = name.split('/'); let dir = state.outDir;
    for (const p of parts.slice(0, -1)) dir = await dir.getDirectoryHandle(p, { create: true });
    const fh = await dir.getFileHandle(parts[parts.length - 1], { create: true });
    const w = await fh.createWritable(); await w.write(blob); await w.close();
  }
  if (it.url) URL.revokeObjectURL(it.url);
  it.blob = blob; it.outName = name.split('/').pop(); it.outPath = name; it.url = URL.createObjectURL(blob);
}

const MIME = { png: 'image/png', jpg: 'image/jpeg', jpeg: 'image/jpeg', jfif: 'image/jpeg', webp: 'image/webp' };
async function encodeImage(img, srcFile, strip) {
  let e = ext(srcFile.name); if (!MIME[e]) e = 'png';
  const c = new OffscreenCanvas(img.width, img.height);
  c.getContext('2d').putImageData(new ImageData(img.data, img.width, img.height), 0, 0);
  let blob = await c.convertToBlob({ type: MIME[e], quality: 0.95 });
  const orig = new Uint8Array(await srcFile.arrayBuffer());
  let bytes = new Uint8Array(await blob.arrayBuffer());
  if (MIME[e] === 'image/jpeg') bytes = M.rebuildJpeg(bytes, orig, !strip);
  else if (e === 'png' && ext(srcFile.name) === 'png') bytes = M.rebuildPng(bytes, orig, !strip);
  return { blob: new Blob([bytes], { type: MIME[e] }), ext: e, orig, bytes };
}

$('start').onclick = async () => {
  if (state.busy) return;
  const t = tab();
  if (!t.items.length) { alert('Danh sách đang trống.'); return; }
  const jobs = t.items.filter(x => x.checked);
  if (!jobs.length) { alert('Chưa tích chọn tệp nào. Hãy tích vào ô bên trái tên tệp (hoặc "Chọn tất cả").'); return; }
  state.busy = true; state.cancel = false;
  $('start').disabled = true; $('stop').disabled = false; $('zipAll').disabled = true; $('zipSel').disabled = true;
  document.querySelectorAll('.tab').forEach(b => (b.disabled = true));
  const strip = $('strip').checked, method = $('method').value;
  log(`— Bắt đầu ${jobs.length} tệp`);
  for (const it of jobs) { it.status = 'Chờ xử lý'; it.cls = ''; }
  renderList();
  let ok = 0, fail = 0, stopped = false;
  for (let i = 0; i < jobs.length; i++) {
    if (state.cancel) { stopped = true; break; }
    const it = jobs[i];
    it.status = 'Đang xử lý…'; it.cls = ''; renderList();
    progress(i / jobs.length, `${i + 1}/${jobs.length}: ${it.rel}`);
    await new Promise(r => setTimeout(r, 10));
    try {
      if (it.manualBox && it.manualBox.rel) {
        if (!it.size) { if (it.kind === 'image') { const im = await decodeImage(it.file); it.size = [im.width, im.height]; } else { const inf = await ensureVideo(it); it.size = [inf.width, inf.height]; } }
        resolveRel(it);
      }
      const o = { method, mode: it.manualBox ? 'manual' : 'auto', box: it.manualBox };
      let geo;
      const base = it.rel.replace(/\.[^.]+$/, '');
      if (it.kind === 'image') {
        const img = it.preview && it.preview.width ? it.preview : await decodeImage(it.file);
        let res;
        if (o.mode === 'auto' && method === 'alpha' && it.geo) res = { img: runImageWithGeo(img, it.geo), geo: it.geo };
        else res = E.processImage(img, o);
        geo = res.geo;
        if (state.cancel) throw new Error('__cancel__');
        const enc = await encodeImage(res.img, it.file, strip);
        await saveOut(it, enc.blob, `${base}.${enc.ext}`);
        const before = M.scan(enc.orig), after = M.scan(enc.bytes);
        log(`✔ ${it.rel}: ${geo.found ? 'đã gỡ logo' : (o.mode === 'manual' ? 'đã vá vùng' : 'không thấy logo')}${strip ? ` | metadata: ${before.join(', ') || 'không có'} → ${after.join(', ') || 'sạch'}` : ''}`);
        it.preview = null;
      } else {
        const info = await ensureVideo(it);
        geo = await videoGeo(it, o);
        if (!geo.found && o.mode !== 'manual') { log(`• ${it.rel}: không thấy logo, bỏ qua`); it.status = 'Không thấy logo'; it.cls = 'warn'; renderList(); continue; }
        const data = await V.processVideo(it.ffName, info, geo, { method, strip, onProgress: (f, s) => progress((i + f) / jobs.length, `${i + 1}/${jobs.length}: ${s}`) });
        await saveOut(it, new Blob([data], { type: 'video/mp4' }), `${base}.mp4`);
        log(`✔ ${it.rel}: ${geo.found ? 'đã gỡ logo' : 'đã vá vùng'} → ${base}.mp4`);
      }
      it.status = geo.found ? 'Đã gỡ logo' : (o.mode === 'manual' ? 'Đã vá vùng' : 'Không thấy logo');
      it.cls = geo.found || o.mode === 'manual' ? 'ok' : 'warn'; ok++;
    } catch (err) {
      if (state.cancel) { it.status = 'Đã dừng'; it.cls = 'warn'; stopped = true; renderList(); break; }
      console.error(err); it.status = 'Lỗi'; it.cls = 'err'; fail++; log(`✖ ${it.rel}: ${err.message}`);
    }
    renderList();
  }
  if (stopped) {
    progress(0, `Đã ngưng: ${ok} tệp xong trước khi dừng. Bấm "Bắt đầu" để chạy lại từ đầu.`, 'stopped');
    log(`— Đã ngưng tiến trình (${ok} xong, ${fail} lỗi)`);
  } else {
    progress(1, `Hoàn tất: ${ok} thành công, ${fail} lỗi${state.outDir ? '' : ' – bấm vào trạng thái để tải từng tệp, hoặc dùng nút tải bên trên'}`, 'done');
    log(`— Hoàn tất: ${ok} thành công, ${fail} lỗi`);
  }
  state.busy = false; state.cancel = false;
  $('start').disabled = false; $('stop').disabled = true;
  document.querySelectorAll('.tab').forEach(b => (b.disabled = false));
  updateSel();
};

$('stop').onclick = () => {
  if (!state.busy) return;
  state.cancel = true;
  $('stop').disabled = true;
  progress(0, 'Đang ngưng…', 'stopped');
  if (V && V.abort && state.kind === 'video') {
    V.abort();                                   // stops ffmpeg immediately
    state.tabs.video.items.forEach(x => { x.ffName = null; });  // its virtual files are gone
  }
};

async function downloadItems(items, zipName) {
  if (!items.length) return;
  if (items.length === 1) {
    const a = document.createElement('a'); a.href = items[0].url; a.download = items[0].outName; a.click(); return;
  }
  const zip = new window.JSZip();
  for (const it of items) zip.file(it.outPath, it.blob);
  progress(0, 'Đang nén .zip…');
  const blob = await zip.generateAsync({ type: 'blob' }, m => progress(m.percent / 100, 'Đang nén .zip…'));
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = zipName; a.click();
  progress(1, `Đã tạo ${zipName} (${items.length} tệp)`, 'done');
}

$('zipSel').onclick = () => {
  const items = tab().items.filter(x => x.checked && x.blob);
  if (!items.length) { alert('Các mục đã chọn chưa có kết quả. Hãy xử lý trước.'); return; }
  downloadItems(items, state.kind === 'image' ? 'AI-Remove-anh-da-chon.zip' : 'AI-Remove-video-da-chon.zip');
};

$('zipAll').onclick = () => {
  const items = tab().items.filter(x => x.blob);
  downloadItems(items, state.kind === 'image' ? 'AI-Remove-anh.zip' : 'AI-Remove-video.zip');
};

$('checkAll').onchange = () => {
  const v = $('checkAll').checked;
  tab().items.forEach(x => { x.checked = v; });
  renderList();
};

// ---------------------------------------------------------------------------
$('pickFiles').onchange = e => { addFiles([...e.target.files]); e.target.value = ''; };
$('pickDir').onchange = e => { addFiles([...e.target.files]); e.target.value = ''; };
$('clear').onclick = () => { if (state.busy) return; const t = tab(); t.items.forEach(it => it.url && URL.revokeObjectURL(it.url)); t.items = []; t.cur = null; renderList(); updateCounts(); view.setImage(null); view.box = null; setDetect(''); };
document.querySelectorAll('.tab').forEach(b => b.onclick = () => switchTab(b.dataset.kind));
document.addEventListener('dragover', e => e.preventDefault());
document.addEventListener('drop', async e => {
  e.preventDefault();
  const files = [];
  const walk = async (entry, path) => {
    if (entry.isFile) await new Promise(r => entry.file(f => { try { Object.defineProperty(f, 'webkitRelativePath', { value: path + f.name }); } catch (_) {} files.push(f); r(); }, r));
    else if (entry.isDirectory) { const rd = entry.createReader(); let batch; do { batch = await new Promise(r => rd.readEntries(r, () => r([]))); for (const en of batch) await walk(en, path + entry.name + '/'); } while (batch.length); }
  };
  const its = [...(e.dataTransfer.items || [])];
  if (its.length && its[0].webkitGetAsEntry) { for (const i of its) { const en = i.webkitGetAsEntry(); if (en) await walk(en, ''); } }
  else files.push(...e.dataTransfer.files);
  addFiles(files);
});

(async () => {
  try { await E.loadTemplates('assets/'); } catch (err) { setDetect('Không tải được dữ liệu mẫu: ' + err.message, 'warn'); }
  document.querySelector('.ctl.man').classList.add('off');
  switchTab('image');
  // GitHub Pages: https://<user>.github.io/<repo>/  ->  releases of that repository
  const h = location.hostname, seg = location.pathname.split('/').filter(Boolean);
  if (h.endsWith('github.io')) $('dlLink').href = `https://github.com/${h.split('.')[0]}/${seg[0] || ''}/releases/latest`;
  window.__airemove = { state, E }; // for debugging/tests
})();
