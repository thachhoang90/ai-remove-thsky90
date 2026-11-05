// =============================================================================
// Lõi xử lý ảnh cho tab "Chỉnh sửa ảnh" (bản web) — tương đương airemove/editor/ops.py
//
// Quy ước: ảnh là đối tượng {width, height, data: Uint8ClampedArray RGBA}
// (giống ImageData). Kênh alpha được giữ nguyên qua mọi phép chỉnh màu.
// Mọi hàm đều trả về ẢNH MỚI, không sửa ảnh đầu vào -> Undo/Redo chỉ cần lưu ảnh.
// =============================================================================

export const LUMA = [0.299, 0.587, 0.114];   // trọng số độ sáng Rec.601

// ---------------------------------------------------------------- tiện ích
export function newImage(w, h) { return { width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }; }
export function cloneImage(img) { return { width: img.width, height: img.height, data: new Uint8ClampedArray(img.data) }; }

/** Ảnh -> 3 mặt phẳng float [0..1] (R, G, B) để tính toán chính xác. */
function toPlanes(img) {
  const n = img.width * img.height, d = img.data;
  const r = new Float32Array(n), g = new Float32Array(n), b = new Float32Array(n);
  for (let i = 0, p = 0; i < n; i++, p += 4) { r[i] = d[p] / 255; g[i] = d[p + 1] / 255; b[i] = d[p + 2] / 255; }
  return [r, g, b];
}

/** 3 mặt phẳng float -> ảnh mới (alpha lấy từ ảnh mẫu). */
function fromPlanes([r, g, b], like) {
  const out = newImage(like.width, like.height), d = out.data, s = like.data, n = r.length;
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    d[p] = r[i] * 255 + 0.5; d[p + 1] = g[i] * 255 + 0.5; d[p + 2] = b[i] * 255 + 0.5; d[p + 3] = s[p + 3];
  }
  return out;  // Uint8ClampedArray tự cắt về 0..255
}

/** Đổi kích thước bằng canvas (trình duyệt dùng nội suy chất lượng cao). */
export function resizeImage(img, w, h) {
  w = Math.max(1, Math.round(w)); h = Math.max(1, Math.round(h));
  if (w === img.width && h === img.height) return cloneImage(img);
  let src = toCanvas(img);
  // Thu nhỏ nhiều lần, mỗi lần tối đa 1/2 -> tránh răng cưa khi thu nhỏ mạnh.
  let cw = img.width, ch = img.height;
  while (cw / 2 >= w && ch / 2 >= h) {
    const c = new OffscreenCanvas(Math.round(cw / 2), Math.round(ch / 2));
    const g = c.getContext('2d'); g.imageSmoothingQuality = 'high'; g.drawImage(src, 0, 0, c.width, c.height);
    src = c; cw = c.width; ch = c.height;
  }
  const c = new OffscreenCanvas(w, h);
  const g = c.getContext('2d', { willReadFrequently: true }); g.imageSmoothingQuality = 'high';
  g.drawImage(src, 0, 0, w, h);
  const id = g.getImageData(0, 0, w, h);
  return { width: w, height: h, data: id.data };
}

export function resizeLongSide(img, maxSide) {
  const s = Math.max(img.width, img.height);
  if (s <= maxSide) return [img, 1];
  const k = maxSide / s;
  return [resizeImage(img, img.width * k, img.height * k), k];
}

export function toCanvas(img) {
  const c = new OffscreenCanvas(img.width, img.height);
  c.getContext('2d').putImageData(new ImageData(img.data, img.width, img.height), 0, 0);
  return c;
}

/** Làm mờ Gauss tách lớp (ngang rồi dọc) trên một mặt phẳng float. */
export function gaussBlur(src, w, h, sigma) {
  if (sigma <= 0) return Float32Array.from(src);
  const r = Math.max(1, Math.ceil(sigma * 3)), ker = new Float32Array(2 * r + 1);
  let sum = 0;
  for (let i = -r; i <= r; i++) { ker[i + r] = Math.exp(-(i * i) / (2 * sigma * sigma)); sum += ker[i + r]; }
  for (let i = 0; i < ker.length; i++) ker[i] /= sum;
  const tmp = new Float32Array(w * h), out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let a = 0;
      for (let i = -r; i <= r; i++) { let xx = x + i; xx = xx < 0 ? -xx : xx >= w ? 2 * w - 2 - xx : xx; a += src[row + Math.max(0, Math.min(w - 1, xx))] * ker[i + r]; }
      tmp[row + x] = a;
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let a = 0;
      for (let i = -r; i <= r; i++) { let yy = y + i; yy = yy < 0 ? -yy : yy >= h ? 2 * h - 2 - yy : yy; a += tmp[Math.max(0, Math.min(h - 1, yy)) * w + x] * ker[i + r]; }
      out[y * w + x] = a;
    }
  }
  return out;
}

/** Lọc trung bình hộp bán kính r bằng ảnh tích phân — O(1) mỗi điểm ảnh (dùng cho Guided Filter). */
function boxMean(src, w, h, r) {
  const W1 = w + 1, I = new Float64Array(W1 * (h + 1)), out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    let rs = 0;
    for (let x = 0; x < w; x++) { rs += src[y * w + x]; I[(y + 1) * W1 + x + 1] = I[y * W1 + x + 1] + rs; }
  }
  for (let y = 0; y < h; y++) {
    const y0 = Math.max(0, y - r), y1 = Math.min(h - 1, y + r);
    for (let x = 0; x < w; x++) {
      const x0 = Math.max(0, x - r), x1 = Math.min(w - 1, x + r);
      const s = I[(y1 + 1) * W1 + x1 + 1] - I[y0 * W1 + x1 + 1] - I[(y1 + 1) * W1 + x0] + I[y0 * W1 + x0];
      out[y * w + x] = s / ((y1 - y0 + 1) * (x1 - x0 + 1));
    }
  }
  return out;
}

/** Bảng tra đường cong tông màu từ các điểm điều khiển (0..255), nội suy tuyến tính. */
function curveLUT(points) {
  const lut = new Float32Array(256);
  for (let v = 0; v < 256; v++) {
    let k = 0;
    while (k < points.length - 2 && v > points[k + 1][0]) k++;
    const [x0, y0] = points[k], [x1, y1] = points[k + 1];
    const t = x1 === x0 ? 0 : Math.min(1, Math.max(0, (v - x0) / (x1 - x0)));
    lut[v] = Math.min(255, Math.max(0, y0 + (y1 - y0) * t)) / 255;
  }
  return lut;
}

function applyCurves(P, { r, g, b, all } = {}) {
  const look = (arr, lut) => { for (let i = 0; i < arr.length; i++) arr[i] = lut[Math.max(0, Math.min(255, Math.round(arr[i] * 255)))]; };
  if (all) { const L = curveLUT(all); P.forEach(a => look(a, L)); }
  if (r) look(P[0], curveLUT(r));
  if (g) look(P[1], curveLUT(g));
  if (b) look(P[2], curveLUT(b));
  return P;
}

// =============================================================================
// MODULE 3 — ĐIỀU CHỈNH BẰNG MA TRẬN MÀU
// Sáng, tương phản, bão hòa đều là phép biến đổi tuyến tính trên vector [R,G,B].
// Gộp cả ba thành MỘT ma trận 3×4 rồi áp một lần cho mỗi điểm ảnh -> mượt, nhanh.
// =============================================================================

/** Ma trận bão hòa 3×3: M = (1 - s)·L + s·I (giữ nguyên độ sáng). */
export function saturationMatrix(s) {
  const m = [];
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) m.push((1 - s) * LUMA[j] + (i === j ? s : 0));
  return m;  // 9 phần tử, theo hàng
}

/** Gộp 3 thanh trượt (-100..100) thành ma trận 3×4 [m00..m22, o0, o1, o2].
 *  x3 = k·(S·x + b - 0.5) + 0.5  =>  M = k·S,  offset = k·b + 0.5·(1 - k). */
export function adjustmentMatrix(brightness = 0, contrast = 0, saturation = 0) {
  const S = saturationMatrix(1 + saturation / 100);
  const b = brightness / 100 * 0.35;
  const k = contrast < 0 ? 1 + contrast / 100 : 1 + contrast / 100 * 1.5;
  const off = k * b + 0.5 * (1 - k);
  return { m: S.map(v => v * k), o: [off, off, off] };
}

function applyMatrix(P, { m, o }) {
  const [r, g, b] = P, n = r.length;
  for (let i = 0; i < n; i++) {
    const R = r[i], G = g[i], B = b[i];
    r[i] = m[0] * R + m[1] * G + m[2] * B + o[0];
    g[i] = m[3] * R + m[4] * G + m[5] * B + o[1];
    b[i] = m[6] * R + m[7] * G + m[8] * B + o[2];
  }
  return P;
}

const matrix3 = m => ({ m, o: [0, 0, 0] });

/** Làm nét (Unsharp Mask) trên kênh độ sáng; bỏ qua dao động rất nhỏ để không nổi nhiễu. */
function sharpenPlanes(P, w, h, amount, scale = 1) {
  if (amount <= 0) return P;
  const a = amount / 100 * 1.6, n = w * h, y = new Float32Array(n);
  for (let i = 0; i < n; i++) y[i] = LUMA[0] * P[0][i] + LUMA[1] * P[1][i] + LUMA[2] * P[2][i];
  const bl = gaussBlur(y, w, h, Math.max(0.5, scale));
  for (let i = 0; i < n; i++) {
    let d = y[i] - bl[i];
    d = Math.sign(d) * Math.max(Math.abs(d) - 0.004, 0) * a;
    P[0][i] += d; P[1][i] += d; P[2][i] += d;
  }
  return P;
}

// ---------------------------------------------------------------- bộ lọc màu
function vignette(P, w, h, strength) {
  const cx = (w - 1) / 2, cy = (h - 1) / 2;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const r2 = ((x - cx) / (w / 2)) ** 2 + ((y - cy) / (h / 2)) ** 2;
    const v = 1 - strength * Math.min(1, r2 / 2), i = y * w + x;
    P[0][i] *= v; P[1][i] *= v; P[2][i] *= v;
  }
}
function clamp01(P) { for (const a of P) for (let i = 0; i < a.length; i++) a[i] = a[i] < 0 ? 0 : a[i] > 1 ? 1 : a[i]; return P; }

const FILTER_FNS = {
  // Vintage: sepia nhẹ, nâng điểm đen, nén vùng sáng, giảm bão hòa, tối góc.
  vintage(P, w, h) {
    const sep = [0.393, 0.769, 0.189, 0.349, 0.686, 0.168, 0.272, 0.534, 0.131];
    const s = saturationMatrix(0.8);
    applyMatrix(P, matrix3(s.map((v, i) => 0.75 * v + 0.25 * sep[i])));
    clamp01(P);
    applyCurves(P, { r: [[0, 28], [128, 140], [255, 238]], g: [[0, 20], [128, 128], [255, 228]], b: [[0, 38], [128, 112], [255, 205]] });
    vignette(P, w, h, 0.35);
  },
  // Cinematic: vùng tối ngả xanh mòng két, vùng sáng ngả cam, tương phản chữ S.
  cinematic(P) {
    const [r, g, b] = P;
    for (let i = 0; i < r.length; i++) {
      const y = LUMA[0] * r[i] + LUMA[1] * g[i] + LUMA[2] * b[i], sh = (1 - y) ** 2, hi = y * y;
      r[i] += -0.06 * sh + 0.07 * hi; g[i] += 0.015 * sh + 0.02 * hi; b[i] += 0.07 * sh - 0.07 * hi;
    }
    applyMatrix(P, matrix3(saturationMatrix(0.9))); clamp01(P);
    applyCurves(P, { all: [[0, 8], [64, 52], [128, 128], [192, 206], [255, 248]] });
  },
  // Sáng / tối giản: tăng phơi sáng, giảm tương phản, nâng vùng tối, màu nhạt.
  bright(P) {
    for (const a of P) for (let i = 0; i < a.length; i++) a[i] = Math.pow(Math.min(1, Math.max(0, a[i])), 0.82);
    applyMatrix(P, adjustmentMatrix(4, -18, -15)); clamp01(P);
    applyCurves(P, { all: [[0, 22], [128, 140], [255, 255]], b: [[0, 4], [255, 255]] });
  },
  // Đen trắng: trộn kênh + tương phản chữ S.
  bw(P) {
    const [r, g, b] = P;
    for (let i = 0; i < r.length; i++) { const y = 0.36 * r[i] + 0.53 * g[i] + 0.11 * b[i]; r[i] = g[i] = b[i] = y; }
    clamp01(P); applyCurves(P, { all: [[0, 0], [60, 46], [128, 128], [200, 214], [255, 255]] });
  },
  // Tông ấm: tăng đỏ/vàng, giảm xanh dương.
  warm(P) {
    for (let i = 0; i < P[0].length; i++) { P[0][i] *= 1.08; P[1][i] *= 1.02; P[2][i] *= 0.88; }
    applyMatrix(P, matrix3(saturationMatrix(1.08))); clamp01(P);
    applyCurves(P, { r: [[0, 6], [255, 255]], b: [[0, 0], [255, 236]] });
  },
  // Tông lạnh.
  cool(P) {
    for (let i = 0; i < P[0].length; i++) { P[0][i] *= 0.92; P[2][i] *= 1.08; }
    clamp01(P); applyCurves(P, { b: [[0, 10], [255, 255]] });
  },
  // Phim bạc màu (fade).
  film(P) {
    applyMatrix(P, adjustmentMatrix(0, -12, -10)); clamp01(P);
    applyCurves(P, { r: [[0, 26], [255, 244]], g: [[0, 32], [255, 246]], b: [[0, 26], [255, 232]] });
  },
  // Rực rỡ.
  vivid(P) { applyMatrix(P, adjustmentMatrix(0, 12, 35)); },
};

export const FILTERS = [
  ['none', 'Gốc'], ['vintage', 'Vintage'], ['cinematic', 'Cinematic'], ['bright', 'Sáng / Tối giản'],
  ['bw', 'Đen trắng'], ['warm', 'Tông ấm'], ['cool', 'Tông lạnh'], ['film', 'Phim bạc màu'], ['vivid', 'Rực rỡ'],
];

export function applyFilter(img, name, amount = 100) {
  const fn = FILTER_FNS[name];
  if (!fn || amount <= 0) return img;
  const src = toPlanes(img), P = src.map(a => Float32Array.from(a));
  fn(P, img.width, img.height);
  if (amount < 100) { const t = amount / 100; for (let c = 0; c < 3; c++) for (let i = 0; i < P[c].length; i++) P[c][i] = t * P[c][i] + (1 - t) * src[c][i]; }
  return fromPlanes(P, img);
}

export const defaultAdjustments = () => ({ filter: 'none', filterAmount: 100, brightness: 0, contrast: 0, saturation: 0, sharpness: 0, skin: 0 });
export const isIdentity = a => (a.filter === 'none' || a.filterAmount <= 0) && !a.brightness && !a.contrast && !a.saturation && !a.sharpness && !a.skin;

/** Thứ tự cố định: bộ lọc -> ma trận (sáng/tương phản/bão hòa) -> làm nét -> mịn da. */
export function renderAdjustments(img, a, scale = 1) {
  let out = applyFilter(img, a.filter, a.filterAmount);
  if (a.brightness || a.contrast || a.saturation || a.sharpness) {
    const P = toPlanes(out);
    if (a.brightness || a.contrast || a.saturation) applyMatrix(P, adjustmentMatrix(a.brightness, a.contrast, a.saturation));
    if (a.sharpness) sharpenPlanes(P, img.width, img.height, a.sharpness, scale);
    out = fromPlanes(P, img);
  }
  if (a.skin > 0) out = smoothSkin(out, a.skin);
  return out;
}

// =============================================================================
// MODULE 4 — LÀM ĐẸP VÀ XỬ LÝ TỰ ĐỘNG
// =============================================================================

/** Tối ưu một chạm dựa trên histogram: cân bằng trắng (gray-world trên vùng trung tính),
 *  giãn mức đen/trắng theo phân vị 0.5%/99.5%, hiệu chỉnh gamma về xám giữa, tăng
 *  tương phản cục bộ (unsharp bán kính lớn — thay cho CLAHE của bản Windows). */
export function autoEnhance(img) {
  const w = img.width, h = img.height, n = w * h, P = toPlanes(img), notes = [];
  const [r, g, b] = P;
  // 1. cân bằng trắng
  let sr = 0, sg = 0, sb = 0, cnt = 0;
  for (let i = 0; i < n; i++) { const y = LUMA[0] * r[i] + LUMA[1] * g[i] + LUMA[2] * b[i]; if (y > 0.12 && y < 0.92) { sr += r[i]; sg += g[i]; sb += b[i]; cnt++; } }
  if (cnt > 500) {
    const m = [sr / cnt, sg / cnt, sb / cnt], gray = (m[0] + m[1] + m[2]) / 3;
    const gains = m.map(v => 1 + 0.7 * (Math.min(1.25, Math.max(0.8, gray / Math.max(v, 1e-4))) - 1));
    for (let i = 0; i < n; i++) { r[i] *= gains[0]; g[i] *= gains[1]; b[i] *= gains[2]; }
    notes.push(`cân bằng trắng R×${gains[0].toFixed(2)} G×${gains[1].toFixed(2)} B×${gains[2].toFixed(2)}`);
  }
  // 2. giãn mức theo histogram độ sáng
  const hist = new Uint32Array(1024);
  for (let i = 0; i < n; i++) { const y = Math.min(1, Math.max(0, LUMA[0] * r[i] + LUMA[1] * g[i] + LUMA[2] * b[i])); hist[Math.min(1023, (y * 1023) | 0)]++; }
  const pct = q => { let acc = 0; const t = q * n; for (let k = 0; k < 1024; k++) { acc += hist[k]; if (acc >= t) return k / 1023; } return 1; };
  const lo = Math.min(pct(0.005), 0.10), hi = Math.max(pct(0.995), 0.80);
  if (hi - lo > 0.05) {
    for (const a of P) for (let i = 0; i < n; i++) a[i] = (a[i] - lo) / (hi - lo);
    notes.push(`giãn mức ${Math.round(lo * 255)}→0, ${Math.round(hi * 255)}→255`);
  }
  clamp01(P);
  // 3. gamma
  let mean = 0;
  for (let i = 0; i < n; i++) mean += LUMA[0] * r[i] + LUMA[1] * g[i] + LUMA[2] * b[i];
  mean /= n;
  if (mean > 0.05 && mean < 0.95) {
    let gm = Math.log(0.48) / Math.log(mean);
    gm = 1 + 0.6 * (Math.min(1.35, Math.max(0.75, gm)) - 1);
    if (Math.abs(gm - 1) > 0.02) { for (const a of P) for (let i = 0; i < n; i++) a[i] = Math.pow(a[i], gm); notes.push(`gamma ${gm.toFixed(2)}`); }
  }
  // 4. tương phản cục bộ + bão hòa nhẹ
  const y = new Float32Array(n);
  for (let i = 0; i < n; i++) y[i] = LUMA[0] * r[i] + LUMA[1] * g[i] + LUMA[2] * b[i];
  const [ys, k] = smallPlane(y, w, h, 600);
  const blur = upPlane(gaussBlur(ys.p, ys.w, ys.h, Math.max(2, Math.min(ys.w, ys.h) / 60)), ys.w, ys.h, w, h);
  for (let i = 0; i < n; i++) { const d = (y[i] - blur[i]) * 0.25; r[i] += d; g[i] += d; b[i] += d; }
  notes.push('tương phản cục bộ');
  applyMatrix(P, adjustmentMatrix(0, 0, 8));
  return [fromPlanes(clamp01(P), img), notes.join(', ')];
}

// thu nhỏ / phóng to một mặt phẳng float (nội suy song tuyến)
function smallPlane(p, w, h, maxSide) {
  const k = Math.min(1, maxSide / Math.max(w, h)), nw = Math.max(1, Math.round(w * k)), nh = Math.max(1, Math.round(h * k));
  if (k === 1) return [{ p, w, h }, 1];
  const o = new Float32Array(nw * nh), fx = w / nw, fy = h / nh;
  for (let y = 0; y < nh; y++) for (let x = 0; x < nw; x++) {
    let s = 0, c = 0;
    for (let yy = Math.floor(y * fy); yy < Math.min(h, Math.floor((y + 1) * fy)); yy++)
      for (let xx = Math.floor(x * fx); xx < Math.min(w, Math.floor((x + 1) * fx)); xx++) { s += p[yy * w + xx]; c++; }
    o[y * nw + x] = c ? s / c : 0;
  }
  return [{ p: o, w: nw, h: nh }, k];
}
function upPlane(p, w, h, W, H) {
  if (w === W && h === H) return p;
  const o = new Float32Array(W * H);
  for (let y = 0; y < H; y++) {
    const sy = Math.min(h - 1, Math.max(0, (y + 0.5) * h / H - 0.5)), y0 = Math.floor(sy), y1 = Math.min(h - 1, y0 + 1), fy = sy - y0;
    for (let x = 0; x < W; x++) {
      const sx = Math.min(w - 1, Math.max(0, (x + 0.5) * w / W - 0.5)), x0 = Math.floor(sx), x1 = Math.min(w - 1, x0 + 1), fx = sx - x0;
      o[y * W + x] = (p[y0 * w + x0] * (1 - fx) + p[y0 * w + x1] * fx) * (1 - fy) + (p[y1 * w + x0] * (1 - fx) + p[y1 * w + x1] * fx) * fy;
    }
  }
  return o;
}

/** Mặt nạ vùng da (0..1) theo ngưỡng màu da trong không gian YCrCb, có làm mềm mép. */
export function skinMask(img) {
  const w = img.width, h = img.height, n = w * h, d = img.data, m = new Float32Array(n);
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    const R = d[p], G = d[p + 1], B = d[p + 2];
    const Y = 0.299 * R + 0.587 * G + 0.114 * B, Cr = (R - Y) * 0.713 + 128, Cb = (B - Y) * 0.564 + 128;
    m[i] = (Cr >= 133 && Cr <= 178 && Cb >= 80 && Cb <= 132 && Y > 35) ? 1 : 0;
  }
  // "mở" hình thái học xấp xỉ: lọc trung bình rồi ngưỡng -> bỏ chấm lẻ, lấp lỗ nhỏ
  const r = Math.max(1, Math.round(Math.min(w, h) / 250));
  const a = boxMean(m, w, h, r);
  for (let i = 0; i < n; i++) m[i] = a[i] > 0.55 ? 1 : 0;
  return gaussBlur(m, w, h, Math.max(2, Math.min(w, h) / 120));
}

/** Guided Filter (He và cộng sự) — làm mờ GIỮ CẠNH với chi phí O(N):
 *  q = a·I + b, trong đó a, b được ước lượng trong từng cửa sổ: vùng phẳng (phương sai
 *  nhỏ hơn eps) bị làm mịn mạnh, vùng có cạnh (phương sai lớn) được giữ nguyên. */
function guidedFilter(I, w, h, r, eps) {
  const mI = boxMean(I, w, h, r), II = new Float32Array(I.length);
  for (let i = 0; i < I.length; i++) II[i] = I[i] * I[i];
  const mII = boxMean(II, w, h, r), A = new Float32Array(I.length), B = new Float32Array(I.length);
  for (let i = 0; i < I.length; i++) { const v = mII[i] - mI[i] * mI[i]; A[i] = v / (v + eps); B[i] = mI[i] - A[i] * mI[i]; }
  const mA = boxMean(A, w, h, r), mB = boxMean(B, w, h, r), q = new Float32Array(I.length);
  for (let i = 0; i < I.length; i++) q[i] = mA[i] * I[i] + mB[i];
  return q;
}

/** Làm mịn da: tách tần số — nền mịn (Guided Filter) + 30% chi tiết nhỏ (lỗ chân lông),
 *  chỉ áp lên vùng da, trộn theo cường độ 0..100 -> mịn mà không bệt. */
export function smoothSkin(img, strength) {
  if (strength <= 0) return img;
  const w = img.width, h = img.height, n = w * h, s = strength / 100;
  const mask = skinMask(img), P = toPlanes(img);
  const r = Math.max(3, Math.round(Math.min(w, h) / 110)), eps = (0.03 + 0.07 * s) ** 2;
  for (let c = 0; c < 3; c++) {
    const I = P[c], base = guidedFilter(I, w, h, r, eps), fine = gaussBlur(I, w, h, 1.2);
    for (let i = 0; i < n; i++) {
      const smoothed = base[i] + 0.3 * (I[i] - fine[i]);
      I[i] = I[i] + (smoothed - I[i]) * mask[i] * Math.min(1, s * 1.1);
    }
  }
  return fromPlanes(P, img);
}

/** XÓA VẬT THỂ — thuật toán Telea (Fast Marching Method), giống cv2.INPAINT_TELEA.
 *  Vùng trống được lấp dần từ viền vào trong theo "khoảng cách tới viền" T: mỗi điểm
 *  lấy trung bình có trọng số của các điểm đã biết trong bán kính 'radius', trọng số
 *  ưu tiên điểm gần (dst), cùng mức khoảng cách (lev) và nằm theo hướng lan (dir). */
export function inpaintTelea(img, mask, radius) {
  const w = img.width, h = img.height, n = w * h;
  const KNOWN = 0, BAND = 1, INSIDE = 2, INF = 1e6;
  const f = new Uint8Array(n), T = new Float32Array(n);
  const out = cloneImage(img), d = out.data;
  // mặt nạ nới thêm 1 px
  const m = new Uint8Array(n);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let v = 0;
    for (let j = -1; j <= 1 && !v; j++) for (let i = -1; i <= 1; i++) { const xx = x + i, yy = y + j; if (xx >= 0 && yy >= 0 && xx < w && yy < h && mask[yy * w + xx]) { v = 1; break; } }
    m[y * w + x] = v;
  }
  // hàng đợi ưu tiên (min-heap) theo T
  const heap = [], hp = (a, b) => T[heap[a]] < T[heap[b]];
  const push = i => { heap.push(i); let k = heap.length - 1; while (k > 0) { const p = (k - 1) >> 1; if (hp(k, p)) { [heap[k], heap[p]] = [heap[p], heap[k]]; k = p; } else break; } };
  const pop = () => { const top = heap[0], last = heap.pop(); if (heap.length) { heap[0] = last; let k = 0; for (;;) { const l = 2 * k + 1, r = l + 1; let s = k; if (l < heap.length && hp(l, s)) s = l; if (r < heap.length && hp(r, s)) s = r; if (s === k) break; [heap[k], heap[s]] = [heap[s], heap[k]]; k = s; } } return top; };
  for (let i = 0; i < n; i++) { f[i] = m[i] ? INSIDE : KNOWN; T[i] = m[i] ? INF : 0; }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    if (f[i] !== INSIDE) continue;
    if ((x > 0 && f[i - 1] === KNOWN) || (x < w - 1 && f[i + 1] === KNOWN) || (y > 0 && f[i - w] === KNOWN) || (y < h - 1 && f[i + w] === KNOWN)) {
      f[i] = BAND; T[i] = 0; push(i);
    }
  }
  const solve = (i1, i2) => {           // nghiệm phương trình eikonal |∇T| = 1
    let sol = INF;
    if (i1 >= 0 && f[i1] === KNOWN) {
      if (i2 >= 0 && f[i2] === KNOWN) {
        const r = Math.sqrt(Math.max(0, 2 - (T[i1] - T[i2]) ** 2));
        let s = (T[i1] + T[i2] - r) / 2;
        if (s >= T[i1] && s >= T[i2]) sol = s; else { s += r; if (s >= T[i1] && s >= T[i2]) sol = s; }
      } else sol = 1 + T[i1];
    } else if (i2 >= 0 && f[i2] === KNOWN) sol = 1 + T[i2];
    return sol;
  };
  const R = Math.max(1, Math.round(radius));
  const fill = (x, y) => {
    const i = y * w + x;
    // hướng lan truyền = gradient của T
    const tx = (x > 0 && x < w - 1 && f[i - 1] !== INSIDE && f[i + 1] !== INSIDE) ? (T[i + 1] - T[i - 1]) / 2 : 0;
    const ty = (y > 0 && y < h - 1 && f[i - w] !== INSIDE && f[i + w] !== INSIDE) ? (T[i + w] - T[i - w]) / 2 : 0;
    let sw = 0, s0 = 0, s1 = 0, s2 = 0;
    for (let j = Math.max(0, y - R); j <= Math.min(h - 1, y + R); j++) for (let k = Math.max(0, x - R); k <= Math.min(w - 1, x + R); k++) {
      const q = j * w + k;
      if (f[q] === INSIDE || q === i) continue;
      const rx = x - k, ry = y - j, d2 = rx * rx + ry * ry;
      if (d2 > R * R) continue;
      const dist = Math.sqrt(d2);
      let dir = Math.abs(rx * tx + ry * ty) / dist; if (dir < 1e-6) dir = 1e-6;
      const dst = 1 / d2, lev = 1 / (1 + Math.abs(T[q] - T[i]));
      const wq = dir * dst * lev;
      const p = q * 4; s0 += wq * d[p]; s1 += wq * d[p + 1]; s2 += wq * d[p + 2]; sw += wq;
    }
    if (sw > 0) { const p = i * 4; d[p] = s0 / sw; d[p + 1] = s1 / sw; d[p + 2] = s2 / sw; }
  };
  while (heap.length) {
    const i = pop();
    if (f[i] === KNOWN) continue;
    f[i] = KNOWN;
    const x = i % w, y = (i / w) | 0;
    for (const [dx, dy] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
      const xx = x + dx, yy = y + dy;
      if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
      const q = yy * w + xx;
      if (f[q] === KNOWN) continue;
      const L = xx > 0 ? q - 1 : -1, Rr = xx < w - 1 ? q + 1 : -1, U = yy > 0 ? q - w : -1, D = yy < h - 1 ? q + w : -1;
      T[q] = Math.min(solve(U, L), solve(D, L), solve(U, Rr), solve(D, Rr));
      if (f[q] === INSIDE) { f[q] = BAND; fill(xx, yy); }
      push(q);
    }
  }
  return out;
}

/** Xóa vật thể: Telea + thêm nhiễu hạt khớp vùng xung quanh để mảng vá không mịn bất thường. */
export function removeObjects(img, mask) {
  const w = img.width, h = img.height, n = w * h;
  const grow = Math.max(2, Math.round(Math.min(w, h) / 500)), m = new Uint8Array(n);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    if (!mask[y * w + x]) continue;
    for (let j = -grow; j <= grow; j++) for (let i = -grow; i <= grow; i++) {
      const xx = x + i, yy = y + j;
      if (xx >= 0 && yy >= 0 && xx < w && yy < h && i * i + j * j <= grow * grow) m[yy * w + xx] = 1;
    }
  }
  const out = inpaintTelea(img, m, Math.max(3, Math.round(Math.min(w, h) / 300)));
  // độ nhiễu xung quanh
  const g = new Float32Array(n);
  for (let i = 0, p = 0; i < n; i++, p += 4) g[i] = 0.299 * img.data[p] + 0.587 * img.data[p + 1] + 0.114 * img.data[p + 2];
  const bl = gaussBlur(g, w, h, 1.0);
  let s = 0, s2 = 0, c = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    if (m[i]) continue;
    let near = false;
    for (let j = -7; j <= 7 && !near; j += 7) for (let k = -7; k <= 7; k += 7) { const xx = x + k, yy = y + j; if (xx >= 0 && yy >= 0 && xx < w && yy < h && m[yy * w + xx]) { near = true; break; } }
    if (near) { const v = g[i] - bl[i]; s += v; s2 += v * v; c++; }
  }
  const sigma = c > 50 ? Math.min(8, Math.sqrt(Math.max(0, s2 / c - (s / c) ** 2))) : 0;
  if (sigma > 0.6) {
    let seed = 12345;
    const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
    const d = out.data;
    for (let i = 0; i < n; i++) {
      if (!m[i]) continue;
      const z = Math.sqrt(-2 * Math.log(rnd() + 1e-9)) * Math.cos(2 * Math.PI * rnd()) * sigma * 0.8;
      d[4 * i] += z; d[4 * i + 1] += z; d[4 * i + 2] += z;
    }
  }
  return out;
}

// =============================================================================
// MODULE 2 — ĐÓNG LOGO
// =============================================================================
export const POSITIONS = {
  tl: ['Góc trên trái', 0, 0], tc: ['Trên giữa', 0.5, 0], tr: ['Góc trên phải', 1, 0],
  ml: ['Giữa trái', 0, 0.5], c: ['Chính giữa', 0.5, 0.5], mr: ['Giữa phải', 1, 0.5],
  bl: ['Góc dưới trái', 0, 1], bc: ['Dưới giữa', 0.5, 1], br: ['Góc dưới phải', 1, 1],
};
export const defaultWatermark = () => ({ position: 'br', opacity: 80, scale: 18, margin: 3 });

/** Vị trí và kích thước logo trên ảnh: rộng = scale% chiều rộng ảnh; lề = margin% cạnh ngắn. */
export function watermarkRect(W, H, lw0, lh0, st) {
  let lw = Math.max(1, Math.round(W * st.scale / 100)), lh = Math.max(1, Math.round(lw * lh0 / lw0));
  if (lh > H) { lh = H; lw = Math.max(1, Math.round(lh * lw0 / lh0)); }
  const margin = Math.round(Math.min(W, H) * st.margin / 100), [, fx, fy] = POSITIONS[st.position] || POSITIONS.br;
  const x = Math.round(margin + fx * (W - lw - 2 * margin)), y = Math.round(margin + fy * (H - lh - 2 * margin));
  return [Math.max(0, Math.min(x, W - lw)), Math.max(0, Math.min(y, H - lh)), lw, lh];
}

/** Chồng logo (ImageBitmap/canvas có nền trong suốt) lên ảnh với độ đậm 'opacity':
 *  kết quả = α·logo + (1-α)·ảnh — canvas tự trộn alpha khi đặt globalAlpha. */
export function composeWatermark(img, logo, st) {
  if (!logo || st.opacity <= 0) return img;
  const c = toCanvas(img), g = c.getContext('2d', { willReadFrequently: true });
  const [x, y, lw, lh] = watermarkRect(img.width, img.height, logo.width, logo.height, st);
  g.imageSmoothingQuality = 'high'; g.globalAlpha = st.opacity / 100;
  g.drawImage(logo, x, y, lw, lh);
  const id = g.getImageData(0, 0, img.width, img.height);
  // giữ alpha gốc của ảnh (canvas có thể làm thay đổi vùng trong suốt)
  for (let p = 3; p < id.data.length; p += 4) id.data[p] = Math.max(id.data[p], img.data[p]);
  return { width: img.width, height: img.height, data: id.data };
}

// =============================================================================
// MODULE 1 — CẮT ẢNH THEO TỈ LỆ MẠNG XÃ HỘI & NÉN THÔNG MINH
// =============================================================================
export const CROP_PRESETS = [
  ['free', 'Tự do', null, null],
  ['fb_cover', 'FB Cover 851×315', 851 / 315, [1702, 630]],
  ['fb_post', 'FB Post 1.91:1', 1.91, [1200, 628]],
  ['ig_post', 'IG Post 1:1', 1, [1080, 1080]],
  ['ig_portrait', 'IG dọc 4:5', 4 / 5, [1080, 1350]],
  ['story', 'Story/TikTok 9:16', 9 / 16, [1080, 1920]],
  ['yt_thumb', 'YouTube 16:9', 16 / 9, [1280, 720]],
];

export function maxRectForRatio(W, H, ratio) {
  if (!ratio) return [0, 0, W, H];
  let w, h;
  if (W / H > ratio) { h = H; w = h * ratio; } else { w = W; h = w / ratio; }
  return [(W - w) / 2, (H - h) / 2, w, h];
}

/** Cắt theo khung và (tùy chọn) thu về kích thước chuẩn; thu nhỏ mạnh thì làm nét nhẹ. */
export function cropImage(img, [x, y, w, h], target = null) {
  const x0 = Math.max(0, Math.round(x)), y0 = Math.max(0, Math.round(y));
  const x1 = Math.min(img.width, Math.round(x + w)), y1 = Math.min(img.height, Math.round(y + h));
  const cw = x1 - x0, ch = y1 - y0, out = newImage(cw, ch);
  for (let j = 0; j < ch; j++) out.data.set(img.data.subarray(((y0 + j) * img.width + x0) * 4, ((y0 + j) * img.width + x1) * 4), j * cw * 4);
  if (target && cw > target[0]) {
    const k = target[0] / cw;
    let r = resizeImage(out, target[0], target[1]);
    if (k < 0.8) { const P = toPlanes(r); sharpenPlanes(P, r.width, r.height, 25, 0.6); r = fromPlanes(P, r); }
    return r;
  }
  return out;
}

export const PLATFORMS = {
  facebook: { label: 'Facebook', maxLong: 2048, ssim: 0.985 },
  instagram: { label: 'Instagram', maxW: 1080, ssim: 0.985 },
  youtube: { label: 'YouTube Thumbnail', maxLong: 1280, ssim: 0.98, maxBytes: 2000000 },
  general: { label: 'Giữ kích thước', ssim: 0.99 },
};

/** SSIM trên kênh độ sáng (1 = giống hệt) — đo khác biệt theo cảm nhận của mắt. */
export function ssim(a, b) {
  const w = a.width, h = a.height, n = w * h, x = new Float32Array(n), y = new Float32Array(n);
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    x[i] = 0.299 * a.data[p] + 0.587 * a.data[p + 1] + 0.114 * a.data[p + 2];
    y[i] = 0.299 * b.data[p] + 0.587 * b.data[p + 1] + 0.114 * b.data[p + 2];
  }
  const xx = new Float32Array(n), yy = new Float32Array(n), xy = new Float32Array(n);
  for (let i = 0; i < n; i++) { xx[i] = x[i] * x[i]; yy[i] = y[i] * y[i]; xy[i] = x[i] * y[i]; }
  const bl = v => gaussBlur(v, w, h, 1.5);
  const mx = bl(x), my = bl(y), sxx = bl(xx), syy = bl(yy), sxy = bl(xy);
  const C1 = (0.01 * 255) ** 2, C2 = (0.03 * 255) ** 2;
  let s = 0;
  for (let i = 0; i < n; i++) {
    const vx = sxx[i] - mx[i] * mx[i], vy = syy[i] - my[i] * my[i], cv = sxy[i] - mx[i] * my[i];
    s += ((2 * mx[i] * my[i] + C1) * (2 * cv + C2)) / ((mx[i] * mx[i] + my[i] * my[i] + C1) * (vx + vy + C2));
  }
  return s / n;
}

const MIME = { jpg: 'image/jpeg', png: 'image/png', webp: 'image/webp' };

export async function encode(img, fmt, quality = 92) {
  let src = img;
  if (fmt === 'jpg') {                         // JPEG không có nền trong suốt -> lót nền trắng
    const c = new OffscreenCanvas(img.width, img.height), g = c.getContext('2d');
    g.fillStyle = '#fff'; g.fillRect(0, 0, c.width, c.height); g.drawImage(toCanvas(img), 0, 0);
    return c.convertToBlob({ type: MIME.jpg, quality: quality / 100 });
  }
  return toCanvas(src).convertToBlob({ type: MIME[fmt] || MIME.png, quality: quality / 100 });
}

async function decodeBlob(blob) {
  const bmp = await createImageBitmap(blob), c = new OffscreenCanvas(bmp.width, bmp.height);
  const g = c.getContext('2d', { willReadFrequently: true }); g.drawImage(bmp, 0, 0);
  const id = g.getImageData(0, 0, bmp.width, bmp.height);
  return { width: id.width, height: id.height, data: id.data };
}

/** Đưa ảnh về kích thước hiển thị của nền tảng + làm nét đầu ra nhẹ. */
export function prepareForPlatform(img, platform) {
  const rule = PLATFORMS[platform] || PLATFORMS.general;
  let k = 1;
  if (rule.maxLong && Math.max(img.width, img.height) > rule.maxLong) k = rule.maxLong / Math.max(img.width, img.height);
  if (rule.maxW && img.width > rule.maxW) k = Math.min(k, rule.maxW / img.width);
  if (k >= 1) return [img, 1];
  let r = resizeImage(img, img.width * k, img.height * k);
  const P = toPlanes(r); sharpenPlanes(P, r.width, r.height, 22, 0.6); r = fromPlanes(P, r);
  return [r, k];
}

/** NÉN THÔNG MINH: thu về đúng kích thước nền tảng (Facebook ≤ 2048 px cạnh dài, Instagram
 *  rộng 1080 px) rồi tìm nhị phân chất lượng THẤP NHẤT mà SSIM vẫn ≥ ngưỡng (~0.985). File
 *  nhỏ, đúng kích thước => nền tảng gần như không nén lại, ảnh giữ được độ sắc nét. */
export async function smartCompress(img, fmt, platform, onStep) {
  const rule = PLATFORMS[platform] || PLATFORMS.general;
  const [ref, k] = prepareForPlatform(img, platform);
  const info = { width: ref.width, height: ref.height, scale: k, format: fmt };
  if (fmt === 'png') { const b = await encode(ref, 'png'); return [b, { ...info, size: b.size, quality: null, ssim: 1 }]; }
  let lo = 55, hi = 96, best = null;
  while (lo <= hi) {
    const q = (lo + hi) >> 1;
    const blob = await encode(ref, fmt, q);
    const s = ssim(ref, await decodeBlob(blob));
    onStep && onStep(q, s);
    if (s >= rule.ssim) { best = [q, blob, s]; hi = q - 1; } else lo = q + 1;
  }
  if (!best) { const b = await encode(ref, fmt, 96); best = [96, b, ssim(ref, await decodeBlob(b))]; }
  let [q, blob, s] = best;
  while (rule.maxBytes && blob.size > rule.maxBytes && q > 40) { q -= 5; blob = await encode(ref, fmt, q); }
  return [blob, { ...info, size: blob.size, quality: q, ssim: s }];
}
