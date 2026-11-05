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
  // Chân dung trong trẻo: sáng hơn, da ấm hồng nhẹ, tương phản thấp, mềm ảo 12%.
  portrait(P, w, h) {
    for (const a of P) for (let i = 0; i < a.length; i++) a[i] = Math.pow(Math.min(1, Math.max(0, a[i])), 0.9);
    for (let i = 0; i < P[0].length; i++) { P[0][i] *= 1.03; P[2][i] *= 0.97; }
    applyMatrix(P, adjustmentMatrix(3, -12, -4)); clamp01(P);
    softFocus(P, w, h, 0.12);
    applyCurves(P, { all: [[0, 12], [128, 136], [255, 252]] });
  },
  // Giờ vàng: vùng sáng ngả cam vàng, nâng vùng tối, tối góc nhẹ.
  golden(P, w, h) {
    const [r, g, b] = P;
    for (let i = 0; i < r.length; i++) {
      const y = LUMA[0] * r[i] + LUMA[1] * g[i] + LUMA[2] * b[i], hi = Math.pow(Math.max(0, y), 1.5), sh = (1 - y) ** 2;
      r[i] += 0.08 * hi + 0.02 * sh; g[i] += 0.035 * hi; b[i] += -0.06 * hi - 0.02 * sh;
    }
    clamp01(P); applyMatrix(P, matrix3(saturationMatrix(1.1))); clamp01(P);
    applyCurves(P, { all: [[0, 16], [128, 134], [255, 250]] });
    vignette(P, w, h, 0.22);
  },
  // Pastel Hàn Quốc: tương phản thấp, nâng điểm đen, màu dịu, vùng sáng phớt hồng.
  pastel(P) {
    applyMatrix(P, adjustmentMatrix(6, -28, -22)); clamp01(P);
    const [r, g, b] = P;
    for (let i = 0; i < r.length; i++) { const y = LUMA[0] * r[i] + LUMA[1] * g[i] + LUMA[2] * b[i]; r[i] += 0.035 * y * y; b[i] += 0.03 * y * y; }
    clamp01(P); applyCurves(P, { all: [[0, 34], [128, 140], [255, 250]] });
  },
  // Moody tối: giảm sáng, giảm màu, vùng tối ngả xanh lạnh.
  moody(P) {
    applyMatrix(P, adjustmentMatrix(-8, 8, -28)); clamp01(P);
    const [r, g, b] = P;
    for (let i = 0; i < r.length; i++) { const y = LUMA[0] * r[i] + LUMA[1] * g[i] + LUMA[2] * b[i], sh = (1 - y) ** 2; r[i] -= 0.03 * sh; g[i] += 0.01 * sh; b[i] += 0.045 * sh; }
    clamp01(P); applyCurves(P, { all: [[0, 8], [90, 70], [128, 112], [255, 232]] });
  },
  // Phim Kodak: ấm, đỏ/vàng đậm đà, điểm đen sâu.
  kodak(P) {
    for (let i = 0; i < P[0].length; i++) { P[0][i] *= 1.06; P[1][i] *= 1.02; P[2][i] *= 0.92; }
    clamp01(P); applyMatrix(P, matrix3(saturationMatrix(1.15))); clamp01(P);
    applyCurves(P, { all: [[0, 6], [64, 56], [128, 132], [192, 204], [255, 250]], b: [[0, 8], [255, 240]] });
  },
  // Fuji cổ điển: màu trầm, xanh lá ngả lam, vùng tối hơi lạnh.
  fuji(P) {
    applyMatrix(P, matrix3([0.98, 0.04, -0.02, 0.0, 0.94, 0.06, -0.02, 0.06, 0.96])); clamp01(P);
    applyMatrix(P, matrix3(saturationMatrix(0.85))); clamp01(P);
    applyCurves(P, { all: [[0, 10], [64, 58], [128, 130], [192, 200], [255, 248]], b: [[0, 14], [255, 250]] });
  },
  // Sepia cổ điển.
  sepia(P, w, h) {
    applyMatrix(P, matrix3([0.393, 0.769, 0.189, 0.349, 0.686, 0.168, 0.272, 0.534, 0.131])); clamp01(P);
    applyCurves(P, { all: [[0, 18], [255, 240]] });
    vignette(P, w, h, 0.3);
  },
  // Matte lạnh (Bắc Âu).
  matte(P) {
    for (let i = 0; i < P[0].length; i++) { P[0][i] *= 0.95; P[2][i] *= 1.05; }
    clamp01(P); applyMatrix(P, adjustmentMatrix(0, -15, -12)); clamp01(P);
    applyCurves(P, { all: [[0, 38], [128, 132], [255, 240]] });
  },
};

/** Làm mềm ảo: trộn với bản mờ Gauss (dùng cho bộ lọc chân dung). */
function softFocus(P, w, h, amount) {
  const s = Math.max(1, Math.min(w, h) / 250);
  for (let c = 0; c < 3; c++) { const b = gaussBlur(P[c], w, h, s); for (let i = 0; i < b.length; i++) P[c][i] = P[c][i] * (1 - amount) + b[i] * amount; }
}

export const FILTERS = [
  ['none', 'Gốc'], ['vintage', 'Vintage'], ['cinematic', 'Cinematic'], ['bright', 'Sáng / Tối giản'],
  ['bw', 'Đen trắng'], ['warm', 'Tông ấm'], ['cool', 'Tông lạnh'], ['film', 'Phim bạc màu'], ['vivid', 'Rực rỡ'],
  ['portrait', 'Chân dung trong trẻo'], ['golden', 'Giờ vàng'], ['pastel', 'Pastel Hàn Quốc'], ['moody', 'Moody tối'],
  ['kodak', 'Phim Kodak'], ['fuji', 'Fuji cổ điển'], ['sepia', 'Sepia cổ điển'], ['matte', 'Matte lạnh'],
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

// ---- chuyển đổi sRGB <-> CIE Lab (D65) — Lab tách độ sáng (L) khỏi màu (a, b),
//      giúp chỉnh sáng/màu da độc lập và tự nhiên hơn so với chỉnh trực tiếp R, G, B.
const SRGB2LIN = (() => { const t = new Float32Array(256); for (let i = 0; i < 256; i++) { const c = i / 255; t[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); } return t; })();
const LIN2SRGB = (() => { const t = new Float32Array(4097); for (let i = 0; i <= 4096; i++) { const c = i / 4096; t[i] = c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055; } return t; })();
const labF = t => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
const labFi = t => (t > 0.206893 ? t * t * t : (t - 16 / 116) / 7.787);

function toLab(img) {
  const n = img.width * img.height, d = img.data, L = new Float32Array(n), A = new Float32Array(n), B = new Float32Array(n);
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    const r = SRGB2LIN[d[p]], g = SRGB2LIN[d[p + 1]], b = SRGB2LIN[d[p + 2]];
    const fx = labF((0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047);
    const fy = labF(0.2126729 * r + 0.7151522 * g + 0.0721750 * b);
    const fz = labF((0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883);
    L[i] = 116 * fy - 16; A[i] = 500 * (fx - fy); B[i] = 200 * (fy - fz);
  }
  return [L, A, B];
}

function fromLab([L, A, B], like) {
  const out = newImage(like.width, like.height), d = out.data, s = like.data, n = L.length;
  const enc = v => LIN2SRGB[Math.max(0, Math.min(4096, Math.round(v * 4096)))] * 255 + 0.5;
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    const fy = (L[i] + 16) / 116, fx = fy + A[i] / 500, fz = fy - B[i] / 200;
    const X = labFi(fx) * 0.95047, Y = labFi(fy), Z = labFi(fz) * 1.08883;
    d[p] = enc(3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z);
    d[p + 1] = enc(-0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z);
    d[p + 2] = enc(0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z);
    d[p + 3] = s[p + 3];
  }
  return out;
}

/** Tìm các mảng da đủ lớn (thay cho nhận diện khuôn mặt bằng AI của bản Windows):
 *  trả về mặt nạ "vùng mặt/cổ" và bề rộng ước lượng của khuôn mặt (px). */
function skinRegions(skin, w, h) {
  const k = Math.min(1, 360 / Math.max(w, h)), sw = Math.max(1, Math.round(w * k)), sh = Math.max(1, Math.round(h * k));
  const small = new Uint8Array(sw * sh);
  for (let y = 0; y < sh; y++) for (let x = 0; x < sw; x++) small[y * sw + x] = skin[Math.min(h - 1, Math.floor((y + 0.5) / k)) * w + Math.min(w - 1, Math.floor((x + 0.5) / k))] > 0.5 ? 1 : 0;
  const lab = new Int32Array(sw * sh), comps = [];
  let id = 0;
  for (let i = 0; i < sw * sh; i++) {
    if (!small[i] || lab[i]) continue;
    id++; let area = 0, minx = sw, maxx = 0, miny = sh, maxy = 0; const st = [i]; lab[i] = id;
    while (st.length) {
      const q = st.pop(), x = q % sw, y = (q / sw) | 0; area++;
      if (x < minx) minx = x; if (x > maxx) maxx = x; if (y < miny) miny = y; if (y > maxy) maxy = y;
      for (const nq of [q - 1, q + 1, q - sw, q + sw]) if (nq >= 0 && nq < sw * sh && Math.abs((nq % sw) - x) <= 1 && small[nq] && !lab[nq]) { lab[nq] = id; st.push(nq); }
    }
    comps.push({ id, area, bw: maxx - minx + 1, bh: maxy - miny + 1 });
  }
  const tot = sw * sh;
  const good = comps.filter(c => c.area > 0.003 * tot && c.area < 0.3 * tot && c.bw < sw * 0.8).sort((a, b) => b.area - a.area).slice(0, 3);
  if (!good.length) return [null, 0];
  const ids = new Set(good.map(c => c.id)), sel = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const q = Math.min(sh - 1, Math.floor(y * k)) * sw + Math.min(sw - 1, Math.floor(x * k));
    if (ids.has(lab[q]) && skin[y * w + x] > 0.5) sel[y * w + x] = 1;
  }
  return [sel, Math.round(Math.min(good[0].bw, good[0].bh) / k)];
}

function median(arr, idx) {
  const v = new Float32Array(idx.length); for (let i = 0; i < idx.length; i++) v[i] = arr[idx[i]];
  v.sort(); return v.length ? v[v.length >> 1] : 0;
}

/** Xóa mụn / đốm thâm nhỏ: điểm TỐI hoặc ĐỎ hơn hẳn vùng da xung quanh, kích thước nhỏ
 *  (loại mắt, lông mày, lỗ mũi) -> lấp bằng Telea rồi trộn mềm. */
function removeBlemishes(img, skin, faceW) {
  const w = img.width, h = img.height, n = w * h, k = Math.max(5, Math.min(31, Math.round(faceW / 18))) | 1, r = k >> 1;
  const [L, A] = toLab(img), mL = boxMean(L, w, h, r), mA = boxMean(A, w, h, r);
  const cand = new Uint8Array(n);
  for (let i = 0; i < n; i++) if (skin[i] > 0.6 && (mL[i] - L[i] > 3.5 || (A[i] - mA[i] > 6 && mL[i] - L[i] > 1))) cand[i] = 1;
  const seen = new Uint8Array(n), keep = new Uint8Array(n), maxArea = (k * 0.9) ** 2;
  let count = 0;
  for (let i = 0; i < n; i++) {
    if (!cand[i] || seen[i]) continue;
    const st = [i], pix = []; seen[i] = 1; let minx = w, maxx = 0, miny = h, maxy = 0;
    while (st.length) {
      const q = st.pop(), x = q % w, y = (q / w) | 0; pix.push(q);
      if (x < minx) minx = x; if (x > maxx) maxx = x; if (y < miny) miny = y; if (y > maxy) maxy = y;
      if (pix.length > maxArea * 2) continue;
      for (const nq of [q - 1, q + 1, q - w, q + w]) if (nq >= 0 && nq < n && Math.abs((nq % w) - x) <= 1 && cand[nq] && !seen[nq]) { seen[nq] = 1; st.push(nq); }
    }
    const bw = maxx - minx + 1, bh = maxy - miny + 1;
    if (pix.length >= 3 && pix.length <= maxArea && Math.max(bw, bh) <= k * 1.3 && pix.length >= 0.3 * bw * bh) { for (const q of pix) keep[q] = 1; count++; }
  }
  if (!count || count > 80) return [img, 0];     // quá nhiều "đốm" => kết cấu/hoa văn, không phải mụn
  const grown = new Uint8Array(n);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (keep[y * w + x]) for (let j = -2; j <= 2; j++) for (let i = -2; i <= 2; i++) { const xx = x + i, yy = y + j; if (xx >= 0 && yy >= 0 && xx < w && yy < h) grown[yy * w + xx] = 1; }
  const fixed = inpaintTelea(img, grown, 3), wgt = gaussBlur(Float32Array.from(grown), w, h, 1.5), out = cloneImage(img);
  for (let i = 0, p = 0; i < n; i++, p += 4) { const t = wgt[i]; if (t > 0) for (let c = 0; c < 3; c++) out.data[p + c] = img.data[p + c] * (1 - t) + fixed.data[p + c] * t; }
  return [out, count];
}

/** Ảnh đồ họa / hoạt hình? — > 30% điểm ảnh bão hòa cao và nhiều cạnh sắc. Ảnh loại này
 *  không chạy bước làm đẹp da (dễ xóa nhầm chi tiết vẽ). */
export function isGraphic(img) {
  const [sm] = resizeLongSide(img, 600), w = sm.width, h = sm.height, n = w * h, d = sm.data, g = new Float32Array(n);
  let sat = 0;
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    const r = d[p], gg = d[p + 1], b = d[p + 2], mx = Math.max(r, gg, b), mn = Math.min(r, gg, b);
    if (mx > 0 && (mx - mn) / mx * 255 > 150) sat++;
    g[i] = 0.299 * r + 0.587 * gg + 0.114 * b;
  }
  const g2 = g.map(v => v * v), m = boxMean(g, w, h, 2), m2 = boxMean(g2, w, h, 2);
  let edges = 0;
  for (let i = 0; i < n; i++) if (Math.sqrt(Math.max(0, m2[i] - m[i] * m[i])) > 12) edges++;
  return sat / n > 0.30 && edges / n > 0.22;
}

/** TỐI ƯU MỘT CHẠM — làm đẹp tự nhiên như ảnh máy ảnh chuyên nghiệp (tương đương bản Windows).
 *  1. Giảm nhiễu màu. 2. Cân bằng trắng theo vùng trung tính + màu da. 3. Chỉnh sáng theo vùng
 *  da/mặt + "đánh đèn" vùng mặt. 4. Giữ chi tiết vùng sáng/tối bằng Guided Filter (không quầng).
 *  5. Vibrance bảo vệ màu da. 6. Chân dung: xóa mụn, mờ nếp nhăn, mịn nhẹ, sáng da. 7. Làm nét
 *  chi tiết (trừ vùng da). Bản web không có AI nhận diện khuôn mặt nên dùng các mảng màu da lớn. */
export function autoEnhance(img, strength = 70) {
  const s = Math.max(0, Math.min(1, strength / 100));
  if (s <= 0) return [img, 'không thay đổi'];
  const w = img.width, h = img.height, n = w * h, notes = [];
  // 1. giảm nhiễu màu: làm mịn kênh màu Cb/Cr, giữ nguyên độ sáng
  let work = cloneImage(img);
  {
    const d = work.data, Y = new Float32Array(n), Cb = new Float32Array(n), Cr = new Float32Array(n);
    for (let i = 0, p = 0; i < n; i++, p += 4) { const r = d[p], g = d[p + 1], b = d[p + 2]; Y[i] = 0.299 * r + 0.587 * g + 0.114 * b; Cb[i] = -0.168736 * r - 0.331264 * g + 0.5 * b; Cr[i] = 0.5 * r - 0.418688 * g - 0.081312 * b; }
    const sg = Math.max(1, Math.min(w, h) / 700), bCb = gaussBlur(Cb, w, h, sg), bCr = gaussBlur(Cr, w, h, sg), t = 0.2 + 0.5 * s;
    for (let i = 0, p = 0; i < n; i++, p += 4) {
      const cb = Cb[i] * (1 - t) + bCb[i] * t, cr = Cr[i] * (1 - t) + bCr[i] * t;
      d[p] = Y[i] + 1.402 * cr; d[p + 1] = Y[i] - 0.344136 * cb - 0.714136 * cr; d[p + 2] = Y[i] + 1.772 * cb;
    }
  }
  const graphic = isGraphic(img);
  const skin = skinMask(work), [faceSel, faceW] = graphic ? [null, 0] : skinRegions(skin, w, h);
  const portrait = !!faceSel;
  notes.push(graphic ? 'ảnh đồ họa/hoạt hình – chỉ tối ưu nhẹ ánh sáng và màu' : portrait ? 'phát hiện vùng da/khuôn mặt' : 'ảnh không có vùng da lớn – tối ưu ánh sáng/màu');
  let [L, A, B] = toLab(work);
  // 2. cân bằng trắng
  const neutral = [];
  for (let i = 0; i < n; i += 3) if (Math.hypot(A[i], B[i]) < 14 && L[i] > 20 && L[i] < 92) neutral.push(i);
  let da = 0, db = 0;
  if (neutral.length > 0.02 * n / 3 && !graphic) { da = -median(A, neutral); db = -median(B, neutral); }
  let faceIdx = null;
  if (portrait) {
    faceIdx = []; for (let i = 0; i < n; i += 2) if (faceSel[i]) faceIdx.push(i);
    if (faceIdx.length > 50) {
      const fa = median(A, faceIdx), fb = median(B, faceIdx), hue = Math.atan2(fb, fa) * 180 / Math.PI, target = Math.min(Math.max(hue, 45), 62);
      if (Math.abs(target - hue) > 1) {
        const c = Math.hypot(fa, fb), ta = c * Math.cos(target * Math.PI / 180), tb = c * Math.sin(target * Math.PI / 180);
        da = 0.5 * da + 0.5 * (ta - fa); db = 0.5 * db + 0.5 * (tb - fb);
      }
    } else faceIdx = null;
  }
  da = Math.max(-7, Math.min(7, da)) * (0.5 + 0.4 * s); db = Math.max(-7, Math.min(7, db)) * (0.5 + 0.4 * s);
  if (Math.abs(da) > 0.3 || Math.abs(db) > 0.3) { for (let i = 0; i < n; i++) { A[i] += da; B[i] += db; } notes.push(`cân bằng trắng (a ${da >= 0 ? '+' : ''}${da.toFixed(1)}, b ${db >= 0 ? '+' : ''}${db.toFixed(1)})`); }
  // 3. độ sáng: chân dung lấy vùng da làm chuẩn, chỉ nâng toàn ảnh một phần + đánh sáng riêng vùng mặt
  let ref, target;
  if (faceIdx) { ref = median(L, faceIdx); target = 64; }
  else { const all = []; for (let i = 0; i < n; i += 7) all.push(i); ref = median(L, all); target = graphic ? ref : ref < 50 ? 50 : ref; }
  ref = Math.min(Math.max(ref, 5), 95);
  let g = Math.log(target / 100) / Math.log(ref / 100); g = Math.min(1.12, Math.max(0.78, g));
  const gAll = 1 + (g - 1) * ((faceIdx ? 0.45 : 0.6) + 0.4 * s);
  if (Math.abs(gAll - 1) > 0.015) { for (let i = 0; i < n; i++) L[i] = 100 * Math.pow(Math.min(1, Math.max(0, L[i] / 100)), gAll); notes.push(`chỉnh sáng (gamma ${gAll.toFixed(2)})`); }
  if (faceIdx) {
    const gap = Math.min(Math.max(target - median(L, faceIdx), 0), 12) * (0.5 + 0.4 * s);
    if (gap > 0.5) {
      const sel = new Float32Array(n); for (let i = 0; i < n; i++) sel[i] = skin[i] > 0.35 ? 1 : 0;
      const light = gaussBlur(sel, w, h, Math.max(3, faceW / 6)); let mx = 0; for (const v of light) mx = Math.max(mx, v);
      for (let i = 0; i < n; i++) L[i] += gap * (light[i] / Math.max(mx, 1e-6)) * (1 - L[i] / 115);
      notes.push(`đánh sáng khuôn mặt +${gap.toFixed(1)}`);
    }
  }
  // 4. giữ chi tiết vùng sáng/tối (Guided Filter tách lớp nền)
  const Ln = new Float32Array(n); for (let i = 0; i < n; i++) Ln[i] = L[i] / 100;
  const base = guidedFilter(Ln, w, h, Math.max(8, Math.round(Math.min(w, h) / 25)), 0.02);
  for (let i = 0; i < n; i++) {
    let b = base[i]; const det = Ln[i] - b, hi = Math.min(1, Math.max(0, (b - 0.72) / 0.28)), lo = Math.min(1, Math.max(0, (0.28 - b) / 0.28));
    b = b - 0.10 * s * hi * (b - 0.72) * 2.2 + 0.08 * s * lo * (0.28 - b) * 2.5;
    let v = Math.min(1, Math.max(0, b + det * (1 + 0.06 * s)));
    v = v + 0.06 * s * Math.sin(2 * Math.PI * (v - 0.5)) / (2 * Math.PI) * 4 * v * (1 - v);
    L[i] = Math.min(1, Math.max(0, v)) * 100;
  }
  notes.push('giữ chi tiết vùng sáng/tối');
  // 5. vibrance, bảo vệ màu da
  for (let i = 0; i < n; i++) {
    const c = Math.hypot(A[i], B[i]); let bo = 1 + 0.22 * s * Math.min(1, Math.max(0, 1 - c / 55));
    if (portrait) bo = 1 + (bo - 1) * (1 - 0.75 * skin[i]);
    A[i] *= bo; B[i] *= bo;
  }
  let out = fromLab([L, A, B], img);
  // 6. làm đẹp chân dung
  if (portrait) {
    if (faceW >= 110) { const [o, nb] = removeBlemishes(out, skin, faceW); out = o; if (nb) notes.push(`xóa ${nb} đốm mụn/thâm`); }
    out = smoothSkin(out, 22 + 30 * s, skin);
    [L, A, B] = toLab(out);
    for (let i = 0; i < n; i++) { const m = skin[i]; if (m <= 0) continue; L[i] = Math.min(100, L[i] + (3 + 3 * s) * m * (1 - L[i] / 100) * 1.4); A[i] *= 1 - 0.10 * s * m; }
    out = fromLab([L, A, B], img);
    notes.push('làm mịn nhẹ, mờ nếp nhăn, sáng da');
  }
  // 7. làm nét chi tiết, trừ vùng da
  const P = toPlanes(out), y = new Float32Array(n);
  for (let i = 0; i < n; i++) y[i] = LUMA[0] * P[0][i] + LUMA[1] * P[1][i] + LUMA[2] * P[2][i];
  const bl = gaussBlur(y, w, h, Math.max(0.8, Math.min(w, h) / 1600));
  for (let i = 0; i < n; i++) {
    let d = y[i] - bl[i]; d = Math.sign(d) * Math.max(Math.abs(d) - 0.006, 0) * 0.45 * s * (portrait ? 1 - 0.9 * skin[i] : 1);
    P[0][i] += d; P[1][i] += d; P[2][i] += d;
  }
  notes.push('làm nét chi tiết');
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
export function smoothSkin(img, strength, maskIn = null) {
  if (strength <= 0) return img;
  const w = img.width, h = img.height, n = w * h, s = strength / 100;
  const mask = maskIn || skinMask(img), P = toPlanes(img);
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
