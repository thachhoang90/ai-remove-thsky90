// =============================================================================
// Tab "Chỉnh sửa ảnh" — điều khiển giao diện (bản web)
//
// Mô hình dữ liệu:
//   S.hist[S.idx]   ảnh "đã chốt" (sau các thao tác đã áp dụng) — dùng cho Undo/Redo.
//   S.adj           thông số ĐANG XEM TRƯỚC (bộ lọc, sáng, tương phản, bão hòa, sắc nét,
//                   mịn da) — chưa ghi vào ảnh cho tới khi bấm "Áp dụng".
//   S.logo + S.wm   lớp logo xem trước, luôn vẽ trên cùng.
// Kéo thanh trượt chỉ tính lại bản thu nhỏ (cạnh dài ≤ 1200 px) => phản hồi tức thì;
// "Áp dụng" mới tính trên ảnh độ phân giải đầy đủ và đẩy vào lịch sử.
// =============================================================================
import * as O from './ops.js';
import { attachScrollbars } from '../scrollbars.js';

const $ = id => document.getElementById(id);
const PREVIEW_MAX = 1200;
const MAX_STEPS = 30;
const IMG_EXT = ['jpg', 'jpeg', 'jfif', 'png', 'webp', 'bmp', 'gif'];
const ext = n => (n.split('.').pop() || '').toLowerCase();
const tick = () => new Promise(r => setTimeout(r, 16));

const S = {
  name: '', ext: 'png', origSize: 0, original: null,
  hist: [], idx: -1,                       // ngăn xếp lịch sử: [{img, label}]
  adj: O.defaultAdjustments(), wm: O.defaultWatermark(),
  logo: null, logoUrl: null,
  prev: null, prevK: 1, prevKey: -1,       // ảnh xem trước (thu nhỏ) của trạng thái hiện tại
  cropKey: null, cropRatio: null,
  mask: null,                              // mặt nạ cọ vẽ ở độ phân giải thật
  bulk: [], bulkCancel: false, bulkRunning: false,
  busy: false,
};
const cur = () => (S.idx >= 0 ? S.hist[S.idx].img : null);

function status(msg) { $('edStatus').textContent = msg; }

// =============================================================================
// Vùng hiển thị: phóng to/thu nhỏ, khung cắt, cọ vẽ (tọa độ theo ảnh THẬT)
// =============================================================================
const V = {
  cv: $('edCanvas'), img: null, W: 0, H: 0, mode: 'view', scale: 1, ox: 0, oy: 0, fitted: true,
  crop: null, drag: null, cursor: null, brush: 40, maskCanvas: null, tag: '',
  set(img, W, H) {
    const changed = W !== this.W || H !== this.H;
    this.img = img; this.W = W; this.H = H;
    $('edEmpty').style.display = img ? 'none' : 'grid';
    if (changed || this.fitted) this.fit(); else this.draw();
  },
  size() { const r = this.cv.getBoundingClientRect(); return [r.width, r.height]; },
  fit() {
    this.fitted = true;
    if (!this.W) { this.draw(); return; }
    const [w, h] = this.size();
    this.scale = Math.min((w - 24) / this.W, (h - 24) / this.H);
    this.ox = (w - this.W * this.scale) / 2; this.oy = (h - this.H * this.scale) / 2;
    this.draw();
  },
  actual() {
    if (!this.W) return;
    const [w, h] = this.size(), cx = (w / 2 - this.ox) / this.scale, cy = (h / 2 - this.oy) / this.scale;
    this.scale = 1; this.ox = w / 2 - cx; this.oy = h / 2 - cy; this.fitted = false; this.draw();
  },
  toImg(px, py) { return [(px - this.ox) / this.scale, (py - this.oy) / this.scale]; },
  maskOverlay() {
    if (!S.mask) return null;
    if (!this.maskCanvas) {
      const dw = Math.min(this.W, 1200), dh = Math.max(1, Math.round(this.H * dw / this.W));
      const c = new OffscreenCanvas(dw, dh), g = c.getContext('2d'), id = g.createImageData(dw, dh);
      let any = false;
      for (let y = 0; y < dh; y++) {
        const sy = Math.min(this.H - 1, Math.floor((y + 0.5) * this.H / dh));
        for (let x = 0; x < dw; x++) {
          const sx = Math.min(this.W - 1, Math.floor((x + 0.5) * this.W / dw));
          if (S.mask[sy * this.W + sx]) { const p = 4 * (y * dw + x); id.data[p] = 255; id.data[p + 1] = 64; id.data[p + 2] = 64; id.data[p + 3] = 120; any = true; }
        }
      }
      g.putImageData(id, 0, 0);
      this.maskCanvas = any ? c : 'none';
    }
    return this.maskCanvas === 'none' ? null : this.maskCanvas;
  },
  draw() {
    const dpr = window.devicePixelRatio || 1, [w, h] = this.size();
    if (this.cv.width !== Math.round(w * dpr) || this.cv.height !== Math.round(h * dpr)) { this.cv.width = Math.round(w * dpr); this.cv.height = Math.round(h * dpr); }
    const g = this.cv.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = '#0E1014'; g.fillRect(0, 0, w, h);
    if (!this.img) return;
    const X = this.ox, Y = this.oy, TW = this.W * this.scale, TH = this.H * this.scale;
    g.imageSmoothingEnabled = TW / this.img.width < 2.5; g.imageSmoothingQuality = 'high';
    g.drawImage(this.img, X, Y, TW, TH);
    if (this.mode === 'brush') {
      const ov = this.maskOverlay(); if (ov) g.drawImage(ov, X, Y, TW, TH);
      if (this.cursor) { g.strokeStyle = 'rgba(255,255,255,.9)'; g.lineWidth = 1.5; g.beginPath(); g.arc(this.cursor[0], this.cursor[1], this.brush / 2 * this.scale, 0, Math.PI * 2); g.stroke(); }
    }
    if (this.mode === 'crop' && this.crop) {
      const [cx, cy, cw, ch] = this.crop, rx = cx * this.scale + X, ry = cy * this.scale + Y, rw = cw * this.scale, rh = ch * this.scale;
      g.fillStyle = 'rgba(0,0,0,.6)';
      g.fillRect(X, Y, TW, ry - Y); g.fillRect(X, ry + rh, TW, Y + TH - ry - rh); g.fillRect(X, ry, rx - X, rh); g.fillRect(rx + rw, ry, X + TW - rx - rw, rh);
      g.strokeStyle = '#fff'; g.lineWidth = 1.5; g.strokeRect(rx, ry, rw, rh);
      g.strokeStyle = 'rgba(255,255,255,.35)'; g.lineWidth = 1; g.beginPath();
      for (const t of [1 / 3, 2 / 3]) { g.moveTo(rx + rw * t, ry); g.lineTo(rx + rw * t, ry + rh); g.moveTo(rx, ry + rh * t); g.lineTo(rx + rw, ry + rh * t); }
      g.stroke();
      g.fillStyle = '#38BDF8';
      for (const [hx, hy] of Object.values(this.handles(rx, ry, rw, rh))) g.fillRect(hx - 5, hy - 5, 10, 10);
      g.fillStyle = 'rgba(0,0,0,.7)'; g.fillRect(rx, ry - 24, 130, 22);
      g.fillStyle = '#fff'; g.font = '12px system-ui'; g.textAlign = 'center';
      g.fillText(`${Math.round(cw)} × ${Math.round(ch)} px`, rx + 65, ry - 9);
    }
    if (this.tag) { g.fillStyle = 'rgba(0,0,0,.65)'; g.fillRect(10, 10, 130, 24); g.fillStyle = '#E5E7EB'; g.font = '12px system-ui'; g.textAlign = 'center'; g.fillText(this.tag, 75, 26); }
  },
  handles(rx, ry, rw, rh) {
    const h = { tl: [rx, ry], tr: [rx + rw, ry], bl: [rx, ry + rh], br: [rx + rw, ry + rh] };
    if (!S.cropRatio) Object.assign(h, { t: [rx + rw / 2, ry], b: [rx + rw / 2, ry + rh], l: [rx, ry + rh / 2], r: [rx + rw, ry + rh / 2] });
    return h;
  },
  cropHit(px, py) {
    if (!this.crop) return null;
    const [cx, cy, cw, ch] = this.crop, rx = cx * this.scale + this.ox, ry = cy * this.scale + this.oy, rw = cw * this.scale, rh = ch * this.scale;
    for (const [k, [hx, hy]] of Object.entries(this.handles(rx, ry, rw, rh))) if (Math.abs(px - hx) <= 12 && Math.abs(py - hy) <= 12) return k;
    return (px >= rx && px <= rx + rw && py >= ry && py <= ry + rh) ? 'move' : null;
  },
  // Di chuyển / co giãn khung cắt, giữ trong ảnh và giữ tỉ lệ nếu có
  resizeCrop(hit, dx, dy, [x, y, w, h]) {
    const W = this.W, H = this.H, ratio = S.cropRatio, minw = 16;
    if (hit === 'move') return [Math.min(Math.max(0, x + dx), W - w), Math.min(Math.max(0, y + dy), H - h), w, h];
    let L = x, T = y, R = x + w, B = y + h;
    const hl = hit.includes('l'), hr = hit.includes('r'), ht = hit.startsWith('t'), hb = hit.startsWith('b');
    if (hl) L = Math.min(R - minw, Math.max(0, L + dx));
    if (hr) R = Math.max(L + minw, Math.min(W, R + dx));
    if (ht) T = Math.min(B - minw, Math.max(0, T + dy));
    if (hb) B = Math.max(T + minw, Math.min(H, B + dy));
    let nw = R - L, nh = B - T;
    if (ratio) {
      if (Math.abs(dx) >= Math.abs(dy)) nh = nw / ratio; else nw = nh * ratio;
      const ax = hl ? R : L, ay = ht ? B : T, maxw = hl ? ax : W - ax, maxh = ht ? ay : H - ay;
      if (nw > maxw) { nw = maxw; nh = nw / ratio; }
      if (nh > maxh) { nh = maxh; nw = nh * ratio; }
      L = hl ? ax - nw : ax; T = ht ? ay - nh : ay;
    }
    return [L, T, nw, nh];
  },
};

// Tô/tẩy nét cọ tròn lên mặt nạ ở độ phân giải thật
function paintStroke(x0, y0, x1, y1, erase) {
  if (!S.mask) return;
  const W = V.W, H = V.H, r = Math.max(1, V.brush / 2), r2 = r * r, val = erase ? 0 : 1;
  const steps = Math.max(1, Math.ceil(Math.hypot(x1 - x0, y1 - y0) / Math.max(1, r / 2)));
  for (let s = 0; s <= steps; s++) {
    const cx = x0 + (x1 - x0) * s / steps, cy = y0 + (y1 - y0) * s / steps;
    for (let y = Math.max(0, Math.floor(cy - r)); y <= Math.min(H - 1, Math.ceil(cy + r)); y++)
      for (let x = Math.max(0, Math.floor(cx - r)); x <= Math.min(W - 1, Math.ceil(cx + r)); x++)
        if ((x - cx) ** 2 + (y - cy) ** 2 <= r2) S.mask[y * W + x] = val;
  }
  V.maskCanvas = null; V.draw();
}

V.cv.addEventListener('contextmenu', e => e.preventDefault());
// v2.3: thanh trượt ngang + dọc quanh khung ảnh (dùng khi phóng to)
attachScrollbars($('edStage'), V, () => (V.W ? [V.W, V.H] : null));

V.cv.addEventListener('wheel', e => {
  if (!V.img) return; e.preventDefault();
  const r = V.cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top, [ix, iy] = V.toImg(px, py);
  V.scale = Math.max(0.02, Math.min(32, V.scale * (e.deltaY < 0 ? 1.15 : 1 / 1.15)));
  V.ox = px - ix * V.scale; V.oy = py - iy * V.scale; V.fitted = false; V.draw();
}, { passive: false });
V.cv.addEventListener('pointerdown', e => {
  if (!V.img) return;
  V.cv.setPointerCapture(e.pointerId);
  const r = V.cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
  const pan = e.button === 1 || (V.mode !== 'brush' && e.button === 2);
  if (V.mode === 'view' || pan) { V.drag = { k: 'pan', px, py, ox: V.ox, oy: V.oy }; return; }
  if (V.mode === 'crop') { const hit = V.cropHit(px, py); if (hit) { const [ix, iy] = V.toImg(px, py); V.drag = { k: 'crop', hit, ix, iy, orig: [...V.crop] }; } return; }
  if (V.mode === 'brush') { const [ix, iy] = V.toImg(px, py), erase = e.button === 2; V.drag = { k: 'brush', erase, last: [ix, iy] }; paintStroke(ix, iy, ix, iy, erase); }
});
V.cv.addEventListener('pointermove', e => {
  const r = V.cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top, d = V.drag;
  if (V.mode === 'brush') { V.cursor = [px, py]; if (!d) V.draw(); }
  if (!d) {
    if (V.mode === 'crop') V.cv.style.cursor = { move: 'move', tl: 'nwse-resize', br: 'nwse-resize', tr: 'nesw-resize', bl: 'nesw-resize', t: 'ns-resize', b: 'ns-resize', l: 'ew-resize', r: 'ew-resize' }[V.cropHit(px, py)] || 'default';
    else V.cv.style.cursor = V.mode === 'brush' ? 'crosshair' : 'grab';
    return;
  }
  if (d.k === 'pan') { V.ox = d.ox + px - d.px; V.oy = d.oy + py - d.py; V.fitted = false; V.draw(); }
  else if (d.k === 'crop') { const [ix, iy] = V.toImg(px, py); V.crop = V.resizeCrop(d.hit, ix - d.ix, iy - d.iy, d.orig); V.draw(); }
  else if (d.k === 'brush') { const [ix, iy] = V.toImg(px, py); paintStroke(d.last[0], d.last[1], ix, iy, d.erase); d.last = [ix, iy]; }
});
V.cv.addEventListener('pointerup', () => { if (V.drag && V.drag.k === 'brush') updateButtons(); V.drag = null; });
V.cv.addEventListener('pointerleave', () => { V.cursor = null; if (V.mode === 'brush') V.draw(); });
window.addEventListener('resize', () => { if (!$('editor').hidden) { V.fitted ? V.fit() : V.draw(); drawSocial(); } });

// =============================================================================
// Lịch sử thao tác (Undo/Redo) — MODULE 5
// =============================================================================
function pushState(img, label) {
  S.hist.splice(S.idx + 1);                 // bỏ các trạng thái "tương lai" sau khi đã Undo
  S.hist.push({ img, label });
  if (S.hist.length > MAX_STEPS) S.hist.splice(1, 1);   // giữ ảnh gốc ở vị trí đầu
  S.idx = S.hist.length - 1;
  afterHistory();
}
function afterHistory() {
  S.mask = null; V.maskCanvas = null;
  if (V.mode === 'brush') setBrush(false);
  if (V.mode === 'crop') cancelCrop();
  renderHistory(); refreshPreview(true); updateButtons();
}
function renderHistory() {
  const ol = $('edHist'); ol.innerHTML = '';
  S.hist.forEach((h, i) => {
    const li = document.createElement('li'); li.textContent = h.label;
    li.className = i === S.idx ? 'cur' : i > S.idx ? 'future' : '';
    li.onclick = () => { resetAdj(true); S.idx = i; afterHistory(); status(`Quay lại bước ${i + 1}: ${h.label}`); };
    ol.appendChild(li);
  });
}
function undo() {
  if (!cur()) return;
  if (!O.isIdentity(S.adj)) { resetAdj(); status('Đã hủy các điều chỉnh chưa áp dụng.'); return; }
  if (S.idx > 0) { S.idx--; afterHistory(); status('Hoàn tác: ' + S.hist[S.idx + 1].label); }
}
function redo() { if (S.idx < S.hist.length - 1) { S.idx++; afterHistory(); status('Làm lại: ' + S.hist[S.idx].label); } }

function updateButtons() {
  const has = !!cur();
  $('edUndo').disabled = $('edUndo2').disabled = !(has && (S.idx > 0 || !O.isIdentity(S.adj)));
  $('edRedo').disabled = $('edRedo2').disabled = !(has && S.idx < S.hist.length - 1);
  $('edCompare').disabled = !has;
  $('edCropApply').disabled = $('edCropCancel').disabled = V.mode !== 'crop';
  $('edWmApply').disabled = !(has && S.logo);
  $('edRemove').disabled = !(has && S.mask && S.mask.some(v => v));
}

// =============================================================================
// Mở ảnh & xem trước thời gian thực
// =============================================================================
async function openFile(file) {
  if (!IMG_EXT.includes(ext(file.name))) { status('Định dạng chưa hỗ trợ trên bản web: ' + file.name); return; }
  status('Đang mở ảnh…');
  try {
    const bmp = await createImageBitmap(file), c = new OffscreenCanvas(bmp.width, bmp.height);
    const g = c.getContext('2d', { willReadFrequently: true }); g.drawImage(bmp, 0, 0);
    const id = g.getImageData(0, 0, bmp.width, bmp.height);
    const img = { width: id.width, height: id.height, data: id.data };
    S.name = file.name; S.ext = ext(file.name); S.origSize = file.size; S.original = img;
    S.hist = [{ img, label: 'Mở ảnh' }]; S.idx = 0; S.prevKey = -1;
    resetAdj(true); V.fitted = true;
    afterHistory();
    $('edInfo').textContent = `${file.name} — ${img.width}×${img.height} px — ${Math.round(file.size / 1024).toLocaleString('vi-VN')} KB`;
    status('Đã mở ảnh. Chọn công cụ ở cột bên trái.');
  } catch (err) { status('Không mở được ảnh: ' + err.message); }
}

function previewBase() {
  if (S.prevKey !== S.idx || !S.prev || S.prevSrc !== cur()) {
    [S.prev, S.prevK] = O.resizeLongSide(cur(), PREVIEW_MAX);
    S.prevKey = S.idx; S.prevSrc = cur();
    renderThumbs();
  }
  return [S.prev, S.prevK];
}

let rafId = 0;
function schedule() { if (!rafId) rafId = requestAnimationFrame(() => { rafId = 0; refreshPreview(); }); updateButtons(); }

function renderWith(img, scale, withLogo = true) {
  let out = O.renderAdjustments(img, S.adj, scale);
  if (withLogo && S.logo && $('edWmShow').checked) out = O.composeWatermark(out, S.logo, S.wm);
  return out;
}

let lastPreview = null;
function refreshPreview() {
  if (!cur()) { V.set(null, 0, 0); return; }
  const [p, k] = previewBase();
  const out = renderWith(p, k);
  lastPreview = O.toCanvas(out);
  V.tag = '';
  V.set(lastPreview, cur().width, cur().height);
  drawSocial();
}

// =============================================================================
// MODULE 3 — Bộ lọc & điều chỉnh
// =============================================================================
function buildFilters() {
  const box = $('edFilters');
  for (const [key, label] of O.FILTERS) {
    const b = document.createElement('button'); b.dataset.key = key;
    const c = document.createElement('canvas'); c.width = 120; c.height = 72;
    b.append(c, document.createTextNode(label));
    b.onclick = () => { S.adj.filter = key; markFilter(); schedule(); };
    box.appendChild(b);
  }
  markFilter();
}
function markFilter() { document.querySelectorAll('#edFilters button').forEach(b => b.classList.toggle('on', b.dataset.key === S.adj.filter)); }
function renderThumbs() {
  if (!cur()) return;
  const [th] = O.resizeLongSide(S.prev || cur(), 120);
  document.querySelectorAll('#edFilters button').forEach(b => {
    const c = b.querySelector('canvas'), out = O.applyFilter(th, b.dataset.key, 100), g = c.getContext('2d');
    c.width = out.width; c.height = out.height; g.putImageData(new ImageData(out.data, out.width, out.height), 0, 0);
  });
}

const SLIDERS = { edBright: 'brightness', edContrast: 'contrast', edSat: 'saturation', edSharp: 'sharpness', edSkin: 'skin', edFAmount: 'filterAmount' };
function bindSlider(id, onValue, fmt = v => v) {
  const el = $(id), out = el.parentElement.querySelector('b'), lab = el.parentElement.querySelector('span');
  const def = +el.value;
  el.addEventListener('input', () => { out.textContent = fmt(+el.value); onValue(+el.value); });
  if (lab) { lab.title = 'Nhấp đúp để đặt lại'; lab.ondblclick = () => { el.value = def; el.dispatchEvent(new Event('input')); }; }
}
function setSlider(id, v, fmt = x => x) { const el = $(id); el.value = v; el.parentElement.querySelector('b').textContent = fmt(v); }

function resetAdj(silent = false) {
  S.adj = O.defaultAdjustments();
  for (const id of ['edBright', 'edContrast', 'edSat', 'edSharp', 'edSkin']) setSlider(id, 0);
  setSlider('edFAmount', 100, v => v + '%');
  markFilter();
  if (!silent) schedule();
}

function adjDesc() {
  const a = S.adj, parts = [];
  if (a.filter !== 'none' && a.filterAmount > 0) parts.push(`${O.FILTERS.find(f => f[0] === a.filter)[1]} ${a.filterAmount}%`);
  if (a.brightness) parts.push(`sáng ${a.brightness > 0 ? '+' : ''}${a.brightness}`);
  if (a.contrast) parts.push(`tương phản ${a.contrast > 0 ? '+' : ''}${a.contrast}`);
  if (a.saturation) parts.push(`bão hòa ${a.saturation > 0 ? '+' : ''}${a.saturation}`);
  if (a.sharpness) parts.push(`sắc nét ${a.sharpness}`);
  if (a.skin) parts.push(`mịn da ${a.skin}`);
  return parts.join(', ');
}

/** Tính thông số đang xem trước trên ảnh độ phân giải đầy đủ rồi ghi vào lịch sử. */
async function commitAdj(label, quiet = false) {
  if (!cur() || O.isIdentity(S.adj)) { if (!quiet) status('Không có thay đổi nào để áp dụng.'); return false; }
  status('Đang áp dụng trên ảnh gốc…'); await tick();
  const desc = adjDesc(), out = O.renderAdjustments(cur(), S.adj, 1);
  resetAdj(true);
  pushState(out, desc ? `${label}: ${desc}` : label);
  if (!quiet) status(`Đã áp dụng: ${label.toLowerCase()}.`);
  return true;
}

// =============================================================================
// MODULE 4 — Làm đẹp & AI
// =============================================================================
async function autoEnhance() {
  if (!cur()) return;
  await commitAdj('Điều chỉnh', true);
  status('Đang tối ưu…'); await tick();
  const lv = +$('edAutoStrength').value;
  const [out, notes] = O.autoEnhance(cur(), lv);
  pushState(out, `Tối ưu một chạm (${lv}%)`);
  status('Tối ưu một chạm: ' + notes);
}

function setBrush(on) {
  if (on && !cur()) return;
  if (on) {
    if (V.mode === 'crop') cancelCrop();
    if (!S.mask || S.mask.length !== V.W * V.H) S.mask = new Uint8Array(V.W * V.H);
    V.mode = 'brush'; V.brush = +$('edBrushSize').value;
    status('Tô lên vật thể cần xóa (chuột phải để tẩy), rồi bấm "Xóa vùng đã tô".');
  } else if (V.mode === 'brush') V.mode = 'view';
  $('edBrush').textContent = on ? 'Tắt cọ vẽ' : 'Bật cọ vẽ';
  $('edBrush').classList.toggle('on', on);
  V.draw(); updateButtons();
}

async function removeObjects() {
  if (!cur() || !S.mask || !S.mask.some(v => v)) { status('Hãy bật cọ vẽ và tô lên vật thể cần xóa trước.'); return; }
  const mask = S.mask;
  await commitAdj('Điều chỉnh', true);
  status('Đang xóa vật thể…'); await tick();
  const out = O.removeObjects(cur(), mask);
  pushState(out, 'Xóa vật thể');
  status('Đã xóa vật thể và lấp vùng trống (thuật toán Telea).');
}

// =============================================================================
// MODULE 1 — Cắt ảnh, xuất file, xem trước mạng xã hội
// =============================================================================
function buildPresets() {
  const box = $('edPresets');
  for (const [key, label, ratio, size] of O.CROP_PRESETS) {
    const b = document.createElement('button'); b.className = 'eb'; b.textContent = label; b.dataset.key = key;
    b.title = size ? `${label} – xuất ${size[0]}×${size[1]} px` : 'Cắt tự do';
    b.onclick = () => startCrop(key);
    box.appendChild(b);
  }
}
function startCrop(key) {
  if (!cur()) return;
  setBrush(false);
  const [, label, ratio] = O.CROP_PRESETS.find(p => p[0] === key);
  S.cropKey = key; S.cropRatio = ratio;
  V.crop = O.maxRectForRatio(cur().width, cur().height, ratio); V.mode = 'crop';
  document.querySelectorAll('#edPresets .eb').forEach(b => b.classList.toggle('on', b.dataset.key === key));
  V.draw(); updateButtons();
  status(`Khung cắt: ${label}. Chỉnh khung rồi bấm "Áp dụng cắt".`);
}
function cancelCrop() {
  if (V.mode === 'crop') V.mode = 'view';
  V.crop = null; S.cropKey = null;
  document.querySelectorAll('#edPresets .eb').forEach(b => b.classList.remove('on'));
  V.draw(); updateButtons();
}
async function applyCrop() {
  if (V.mode !== 'crop' || !V.crop) return;
  const rect = V.crop, [, label, , size] = O.CROP_PRESETS.find(p => p[0] === S.cropKey);
  await commitAdj('Điều chỉnh', true);
  const out = O.cropImage(cur(), rect, $('edResize').checked ? size : null);
  V.fitted = true; cancelCrop();
  pushState(out, `Cắt ${label}`);
  status(`Đã cắt: ${out.width}×${out.height} px.`);
}

async function exportImage() {
  if (!cur()) { status('Chưa có ảnh để xuất.'); return; }
  let fmt = $('edFmt').value;
  if (fmt === 'keep') fmt = { jpeg: 'jpg', jfif: 'jpg' }[S.ext] || (['jpg', 'png', 'webp'].includes(S.ext) ? S.ext : 'png');
  status('Đang xuất ảnh…'); await tick();
  try {
    const img = renderWith(cur(), 1);
    let blob, info;
    if ($('edSmart').checked) {
      [blob, info] = await O.smartCompress(img, fmt, $('edPlatform').value, (q, s) => status(`Nén thông minh: thử chất lượng ${q} → SSIM ${s.toFixed(4)}`));
    } else {
      blob = await O.encode(img, fmt, +$('edQuality').value);
      info = { width: img.width, height: img.height, size: blob.size, quality: fmt === 'png' ? null : +$('edQuality').value };
    }
    const name = S.name.replace(/\.[^.]+$/, '') + '_edit.' + fmt;
    await saveBlob(blob, name, fmt);
    let msg = `Đã xuất ${name} — ${info.width}×${info.height} px, ${Math.round(info.size / 1024).toLocaleString('vi-VN')} KB (ảnh gốc ${Math.round(S.origSize / 1024).toLocaleString('vi-VN')} KB)`;
    if (info.quality) msg += `, chất lượng ${info.quality}`;
    if (info.ssim && info.ssim < 1) msg += `, SSIM ${info.ssim.toFixed(3)}`;
    status(msg);
  } catch (err) { if (err.name !== 'AbortError') status('Không xuất được ảnh: ' + err.message); }
}

async function saveBlob(blob, name, fmt) {
  if (window.showSaveFilePicker) {
    try {
      const h = await window.showSaveFilePicker({ suggestedName: name, types: [{ description: fmt.toUpperCase(), accept: { [blob.type]: ['.' + fmt] } }] });
      const w = await h.createWritable(); await w.write(blob); await w.close(); return;
    } catch (err) { if (err.name === 'AbortError') throw err; }
  }
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 60000);
}

// ---- Xem trước mạng xã hội (mô phỏng bố cục, không dùng giao diện thật của nền tảng)
function drawSocial() {
  const on = $('edSocial').checked, sc = $('edSocialCanvas');
  sc.hidden = !on; V.cv.hidden = on;
  if (!on) { V.draw(); return; }
  const dpr = window.devicePixelRatio || 1, r = sc.getBoundingClientRect();
  sc.width = Math.round(r.width * dpr); sc.height = Math.round(r.height * dpr);
  const g = sc.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0);
  const Wv = r.width, Hv = r.height;
  g.fillStyle = '#0E1014'; g.fillRect(0, 0, Wv, Hv);
  let H = Hv - 40, W = H * 9 / 19.5;
  if (W > Wv - 40) { W = Wv - 40; H = W * 19.5 / 9; }
  const px = (Wv - W) / 2, py = (Hv - H) / 2;
  const rr = (x, y, w, h, rad) => { g.beginPath(); g.moveTo(x + rad, y); g.arcTo(x + w, y, x + w, y + h, rad); g.arcTo(x + w, y + h, x, y + h, rad); g.arcTo(x, y + h, x, y, rad); g.arcTo(x, y, x + w, y, rad); g.closePath(); };
  rr(px, py, W, H, 34); g.fillStyle = '#000'; g.fill(); g.strokeStyle = '#3A3F4B'; g.lineWidth = 3; g.stroke();
  const sx = px + 10, sy = py + 10, sw = W - 20, sh = H - 20;
  g.save(); rr(sx, sy, sw, sh, 26); g.clip();
  const fb = document.querySelector('input[name=edSocialMode]:checked').value === 'facebook';
  g.fillStyle = fb ? '#fff' : '#000'; g.fillRect(sx, sy, sw, sh);
  const img = lastPreview;
  const cover = (dx, dy, dw, dh, srcRect) => {
    if (!img) return;
    let [ix, iy, iw, ih] = srcRect || [0, 0, img.width, img.height];
    const tr = dw / dh;
    if (iw / ih > tr) { const nw = ih * tr; ix += (iw - nw) / 2; iw = nw; } else { const nh = iw / tr; iy += (ih - nh) / 2; ih = nh; }
    g.drawImage(img, ix, iy, iw, ih, dx, dy, dw, dh);
  };
  const ph = (x, y, w, h, i) => { const cols = [['#334155', '#1E293B'], ['#3F3F46', '#27272A'], ['#1F2937', '#111827'], ['#374151', '#1F2937']][i % 4]; const gr = g.createLinearGradient(x, y, x + w, y + h); gr.addColorStop(0, cols[0]); gr.addColorStop(1, cols[1]); g.fillStyle = gr; g.fillRect(x, y, w, h); };
  g.textAlign = 'left';
  if (!fb) {
    g.fillStyle = '#fff'; g.font = 'bold 15px system-ui'; g.fillText('ten_cua_ban', sx + 16, sy + 46);
    g.fillStyle = '#4F46E5'; g.beginPath(); g.arc(sx + 48, sy + 98, 32, 0, Math.PI * 2); g.fill();
    g.strokeStyle = '#E5E7EB'; g.lineWidth = 2; g.stroke();
    [['9', 'bài viết'], ['1.250', 'theo dõi'], ['180', 'đang theo dõi']].forEach(([num, lab], i) => {
      const cx = sx + 96 + i * (sw - 112) / 3 + (sw - 112) / 6;
      g.textAlign = 'center'; g.fillStyle = '#fff'; g.font = 'bold 14px system-ui'; g.fillText(num, cx, sy + 92);
      g.fillStyle = '#A1A1AA'; g.font = '10px system-ui'; g.fillText(lab, cx, sy + 110);
    });
    g.textAlign = 'left'; g.fillStyle = '#fff'; g.font = '12px system-ui'; g.fillText('Tên hiển thị', sx + 16, sy + 152);
    g.fillStyle = '#A1A1AA'; g.fillText('Mô tả ngắn về trang của bạn', sx + 16, sy + 170);
    const bw = (sw - 40) / 2;
    ['Chỉnh sửa', 'Chia sẻ'].forEach((t, i) => { g.fillStyle = '#27272A'; rr(sx + 16 + i * (bw + 8), sy + 184, bw, 28, 8); g.fill(); g.fillStyle = '#fff'; g.textAlign = 'center'; g.fillText(t, sx + 16 + i * (bw + 8) + bw / 2, sy + 202); });
    const top = sy + 226, gap = 2, cw = (sw - 2 * gap) / 3, ch = cw / ($('edSquare').checked ? 1 : 3 / 4), puzzle = $('edPuzzle').checked;
    for (let i = 0; i < 12; i++) {
      const rI = Math.floor(i / 3), cI = i % 3, x = sx + cI * (cw + gap), y = top + rI * (ch + gap);
      if (y > sy + sh) break;
      if (img && puzzle && i < 9) {
        const totW = 3 * cw + 2 * gap, totH = 3 * ch + 2 * gap, k = Math.max(totW / img.width, totH / img.height);
        const ox = (img.width * k - totW) / 2, oy = (img.height * k - totH) / 2;
        g.drawImage(img, (cI * (cw + gap) + ox) / k, (rI * (ch + gap) + oy) / k, cw / k, ch / k, x, y, cw, ch);
      } else if (img && i === 0) cover(x, y, cw, ch);
      else ph(x, y, cw, ch, i);
    }
  } else {
    g.fillStyle = '#111827'; g.font = 'bold 18px system-ui'; g.fillText('Bảng tin', sx + 14, sy + 42);
    g.fillStyle = '#E5E7EB'; g.fillRect(sx, sy + 58, sw, 8);
    const cy = sy + 76;
    g.fillStyle = '#4F46E5'; g.beginPath(); g.arc(sx + 32, cy + 20, 20, 0, Math.PI * 2); g.fill();
    g.fillStyle = '#111827'; g.font = 'bold 13px system-ui'; g.fillText('Trang của bạn', sx + 60, cy + 16);
    g.fillStyle = '#6B7280'; g.font = '11px system-ui'; g.fillText('Vừa xong · Công khai', sx + 60, cy + 34);
    g.fillStyle = '#111827'; g.font = '12px system-ui'; g.fillText('Nội dung bài viết của bạn…', sx + 12, cy + 64);
    let top = cy + 76;
    if (img) { const ar = img.width / img.height, shown = Math.max(ar, 4 / 5), ih = sw / shown; cover(sx, top, sw, ih); top += ih; }
    else { ph(sx, top, sw, sw, 0); top += sw; }
    g.fillStyle = '#6B7280'; g.font = '11px system-ui'; g.fillText('Bạn và 248 người khác', sx + 12, top + 20);
    g.textAlign = 'right'; g.fillText('32 bình luận', sx + sw - 12, top + 20);
    g.fillStyle = '#E5E7EB'; g.fillRect(sx + 12, top + 30, sw - 24, 1);
    g.textAlign = 'center'; g.fillStyle = '#4B5563'; g.font = 'bold 12px system-ui';
    ['Thích', 'Bình luận', 'Chia sẻ'].forEach((t, i) => g.fillText(t, sx + i * sw / 3 + sw / 6, top + 50));
  }
  g.restore();
  g.textAlign = 'center'; g.fillStyle = '#8B93A7'; g.font = '11px system-ui';
  g.fillText('Mô phỏng bố cục hiển thị – không phải giao diện thật của nền tảng', Wv / 2, Hv - 8);
}

// =============================================================================
// MODULE 2 — Đóng logo (vị trí, độ đậm, kích thước, preset, hàng loạt)
// =============================================================================
async function useLogo(blobOrUrl) {
  const blob = typeof blobOrUrl === 'string' ? await (await fetch(blobOrUrl)).blob() : blobOrUrl;
  S.logo = await createImageBitmap(blob);
  S.logoUrl = await new Promise(r => { const fr = new FileReader(); fr.onload = () => r(fr.result); fr.readAsDataURL(blob); });
  const c = $('edLogoThumb'), g = c.getContext('2d'); g.clearRect(0, 0, c.width, c.height);
  const k = Math.min(c.width / S.logo.width, c.height / S.logo.height);
  g.drawImage(S.logo, (c.width - S.logo.width * k) / 2, (c.height - S.logo.height * k) / 2, S.logo.width * k, S.logo.height * k);
  schedule();
}

function buildPositions() {
  const arrows = { tl: '↖', tc: '↑', tr: '↗', ml: '←', c: '●', mr: '→', bl: '↙', bc: '↓', br: '↘' };
  for (const key of ['tl', 'tc', 'tr', 'ml', 'c', 'mr', 'bl', 'bc', 'br']) {
    const b = document.createElement('button'); b.className = 'eb'; b.textContent = arrows[key]; b.dataset.key = key; b.title = O.POSITIONS[key][0];
    b.onclick = () => { S.wm.position = key; markPos(); schedule(); };
    $('edPos').appendChild(b);
  }
  markPos();
}
function markPos() { document.querySelectorAll('#edPos .eb').forEach(b => b.classList.toggle('on', b.dataset.key === S.wm.position)); }
function syncWmControls() { setSlider('edWmOpacity', S.wm.opacity, v => v + '%'); setSlider('edWmScale', S.wm.scale, v => v + '%'); setSlider('edWmMargin', S.wm.margin, v => v + '%'); markPos(); }

async function applyWatermark() {
  if (!cur() || !S.logo) return;
  await commitAdj('Điều chỉnh', true);
  const out = O.composeWatermark(cur(), S.logo, S.wm);
  $('edWmShow').checked = false;            // tránh vẽ logo hai lần
  pushState(out, `Đóng logo (${O.POSITIONS[S.wm.position][0].toLowerCase()})`);
  status('Đã đóng logo vào ảnh (đã tắt lớp xem trước để không chồng hai lần).');
}

const PKEY = 'airemove_wm_presets';
const presets = () => { try { return JSON.parse(localStorage.getItem(PKEY) || '{}'); } catch (_) { return {}; } };
function reloadPresets() {
  const sel = $('edPreset'); sel.innerHTML = '<option value="">— Chọn mẫu đã lưu —</option>';
  for (const name of Object.keys(presets()).sort()) { const o = document.createElement('option'); o.value = name; o.textContent = name; sel.appendChild(o); }
}

async function startBulk() {
  if (!S.logo) { status('Hãy chọn logo trước.'); return; }
  if (!S.bulk.length) { status('Hãy chọn thư mục ảnh (hoặc nhiều ảnh) để đóng logo.'); return; }
  let dir = null;
  if (window.showDirectoryPicker) { try { dir = await window.showDirectoryPicker({ mode: 'readwrite' }); } catch (err) { if (err.name === 'AbortError') return; } }
  S.bulkCancel = false; S.bulkRunning = true;
  $('edBulkStart').disabled = true; $('edBulkStop').disabled = false;
  const zip = dir ? null : new window.JSZip(), total = S.bulk.length, wm = { ...S.wm };
  let ok = 0, fail = 0;
  for (let i = 0; i < total; i++) {
    if (S.bulkCancel) break;
    const f = S.bulk[i];
    $('edBulkBar').style.width = `${Math.round(i / total * 100)}%`; $('edBulkText').textContent = `${i}/${total} – ${f.name}`;
    await tick();                             // nhường luồng cho giao diện (chạy nền, không đơ trang)
    try {
      const bmp = await createImageBitmap(f), c = new OffscreenCanvas(bmp.width, bmp.height), g = c.getContext('2d', { willReadFrequently: true });
      g.drawImage(bmp, 0, 0);
      const id = g.getImageData(0, 0, bmp.width, bmp.height);
      const out = O.composeWatermark({ width: id.width, height: id.height, data: id.data }, S.logo, wm);
      let fmt = $('edBulkFmt').value, e = ext(f.name);
      if (fmt === 'keep') fmt = { jpeg: 'jpg', jfif: 'jpg' }[e] || (['jpg', 'png', 'webp'].includes(e) ? e : 'png');
      const blob = await O.encode(out, fmt, 92);
      const rel = (f.webkitRelativePath || f.name).replace(/\.[^.]+$/, '') + '.' + fmt;
      if (dir) {
        const parts = rel.split('/'); let d = dir;
        for (const p of parts.slice(0, -1)) d = await d.getDirectoryHandle(p, { create: true });
        const fh = await d.getFileHandle(parts[parts.length - 1], { create: true }), w = await fh.createWritable(); await w.write(blob); await w.close();
      } else zip.file(rel, blob);
      ok++;
    } catch (err) { console.error(err); fail++; }
  }
  const stopped = S.bulkCancel;
  if (zip && ok) {
    $('edBulkText').textContent = 'Đang nén .zip…';
    const blob = await zip.generateAsync({ type: 'blob' });
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'AI-Remove-dong-logo.zip'; a.click();
  }
  $('edBulkBar').style.width = stopped ? `${Math.round((ok + fail) / total * 100)}%` : '100%';
  $('edBulkText').textContent = `${stopped ? 'Đã ngưng' : 'Xong'}: ${ok} ảnh, ${fail} lỗi`;
  status(`Đóng logo hàng loạt ${stopped ? 'đã ngưng' : 'xong'}: ${ok} ảnh, ${fail} lỗi.`);
  S.bulkRunning = false; $('edBulkStart').disabled = false; $('edBulkStop').disabled = true;
}

// =============================================================================
// Gắn sự kiện
// =============================================================================
function init() {
  buildFilters(); buildPresets(); buildPositions(); reloadPresets();
  // mở/xuất
  $('edOpen').onchange = e => { if (e.target.files[0]) openFile(e.target.files[0]); e.target.value = ''; };
  $('edExport').onclick = $('edExportTop').onclick = exportImage;
  $('edSmart').onchange = () => { $('edQuality').disabled = $('edSmart').checked; };
  bindSlider('edQuality', () => {});
  // lịch sử
  $('edUndo').onclick = $('edUndo2').onclick = undo;
  $('edRedo').onclick = $('edRedo2').onclick = redo;
  // so sánh trước/sau: nhấn giữ = ảnh gốc, thả ra = ảnh đang sửa
  const cmp = $('edCompare');
  const showOrig = () => {
    if (!S.original) return;
    cmp.classList.add('held');
    const [p] = O.resizeLongSide(S.original, PREVIEW_MAX);
    V.tag = 'ẢNH GỐC'; lastPreview = O.toCanvas(p); V.set(lastPreview, S.original.width, S.original.height); drawSocial();
  };
  const hideOrig = () => { if (cmp.classList.contains('held')) { cmp.classList.remove('held'); refreshPreview(); } };
  cmp.addEventListener('pointerdown', showOrig);
  cmp.addEventListener('pointerup', hideOrig);
  cmp.addEventListener('pointerleave', hideOrig);
  $('edFit').onclick = () => V.fit(); $('edZoom100').onclick = () => V.actual();
  // module 3
  for (const [id, key] of Object.entries(SLIDERS)) bindSlider(id, v => { S.adj[key] = v; schedule(); }, id === 'edFAmount' ? v => v + '%' : v => v);
  $('edAdjApply').onclick = () => commitAdj('Bộ lọc & điều chỉnh');
  $('edAdjReset').onclick = () => resetAdj();
  // module 4
  $('edAuto').onclick = autoEnhance;
  bindSlider('edAutoStrength', () => {}, v => v + '%');
  $('edSkinApply').onclick = () => commitAdj('Làm mịn da');
  $('edBrush').onclick = () => setBrush(V.mode !== 'brush');
  bindSlider('edBrushSize', v => { V.brush = v; V.draw(); }, v => v + ' px');
  $('edRemove').onclick = removeObjects;
  $('edClearMask').onclick = () => { if (S.mask) S.mask.fill(0); V.maskCanvas = null; V.draw(); updateButtons(); };
  // module 1
  $('edCropApply').onclick = applyCrop; $('edCropCancel').onclick = cancelCrop;
  for (const id of ['edSocial', 'edPuzzle', 'edSquare']) $(id).onchange = drawSocial;
  document.querySelectorAll('input[name=edSocialMode]').forEach(r => (r.onchange = drawSocial));
  // module 2
  $('edLogo').onchange = async e => { if (e.target.files[0]) await useLogo(e.target.files[0]); e.target.value = ''; };
  $('edWmShow').onchange = schedule;
  bindSlider('edWmOpacity', v => { S.wm.opacity = v; schedule(); }, v => v + '%');
  bindSlider('edWmScale', v => { S.wm.scale = v; schedule(); }, v => v + '%');
  bindSlider('edWmMargin', v => { S.wm.margin = v; schedule(); }, v => v + '%');
  $('edWmApply').onclick = applyWatermark;
  $('edPresetSave').onclick = () => {
    if (!S.logoUrl) { status('Hãy chọn logo trước khi lưu mẫu.'); return; }
    const name = (prompt('Tên mẫu logo:') || '').trim(); if (!name) return;
    const p = presets(); p[name] = { logo: S.logoUrl, ...S.wm };
    try { localStorage.setItem(PKEY, JSON.stringify(p)); } catch (_) { status('Không lưu được (logo quá lớn cho bộ nhớ trình duyệt).'); return; }
    reloadPresets(); $('edPreset').value = name; status(`Đã lưu mẫu logo "${name}".`);
  };
  $('edPreset').onchange = async () => {
    const d = presets()[$('edPreset').value]; if (!d) return;
    await useLogo(d.logo);
    S.wm = { position: d.position, opacity: d.opacity, scale: d.scale, margin: d.margin };
    syncWmControls(); schedule();
  };
  $('edPresetDel').onclick = () => { const p = presets(); delete p[$('edPreset').value]; localStorage.setItem(PKEY, JSON.stringify(p)); reloadPresets(); };
  const pickBulk = e => {
    S.bulk = [...e.target.files].filter(f => ['jpg', 'jpeg', 'jfif', 'png', 'webp', 'bmp'].includes(ext(f.name)));
    $('edBulkInfo').textContent = S.bulk.length ? `Đã chọn ${S.bulk.length} ảnh` : 'Không có ảnh phù hợp';
    e.target.value = '';
  };
  $('edBulkDir').onchange = pickBulk; $('edBulkFiles').onchange = pickBulk;
  $('edBulkStart').onclick = startBulk;
  $('edBulkStop').onclick = () => { S.bulkCancel = true; $('edBulkStop').disabled = true; };
  // accordion: mở một mục thì đóng các mục khác; rời mục 4 thì tắt cọ
  document.querySelectorAll('.ed-sec').forEach(d => d.addEventListener('toggle', () => {
    if (d.open) document.querySelectorAll('.ed-sec').forEach(o => { if (o !== d) o.open = false; });
    if (d.dataset.sec === 'retouch' && !d.open && V.mode === 'brush') setBrush(false);
  }));
  // phím tắt
  window.addEventListener('keydown', e => {
    if ($('editor').hidden || ['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName) && document.activeElement.type !== 'range') return;
    const k = e.key.toLowerCase();
    if ((e.ctrlKey || e.metaKey) && k === 'z' && !e.shiftKey) { e.preventDefault(); undo(); }
    else if ((e.ctrlKey || e.metaKey) && (k === 'y' || (k === 'z' && e.shiftKey))) { e.preventDefault(); redo(); }
  });
  // kéo thả ảnh khi đang ở tab chỉnh sửa
  window.addEventListener('editor-drop', e => { const f = e.detail.find(x => IMG_EXT.includes(ext(x.name))); if (f) openFile(f); });
  window.addEventListener('editor-show', () => { V.fitted ? V.fit() : V.draw(); drawSocial(); });
  updateButtons();
  window.__editor = { S, O, V };              // phục vụ kiểm thử
}

init();
