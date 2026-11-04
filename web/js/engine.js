// AI Remove - thsky90 — browser port of airemove/engine.py
// Reverse alpha blending of the Gemini sparkle with sub-pixel geometry fit.
// All arrays are Float32Array in row-major order.

export const SHARP_SIGMA = 0.5;
export const K_MIN = 0.3;
export const K_MAX = 1.45;
const K_RANGE = { v1: [0.15, 1.9], n48: [0.15, 1.9], v2: [0.3, 1.2] };

// ---------------------------------------------------------------------------
// Templates
// ---------------------------------------------------------------------------
export const T = { ready: false };

async function loadAlpha(url) {
  const img = await createImageBitmap(await (await fetch(url)).blob());
  const c = new OffscreenCanvas(img.width, img.height);
  const g = c.getContext('2d');
  g.drawImage(img, 0, 0);
  const d = g.getImageData(0, 0, img.width, img.height).data;
  const n = img.width;
  const a = new Float32Array(n * n);
  for (let i = 0; i < n * n; i++) a[i] = Math.max(d[4 * i], d[4 * i + 1], d[4 * i + 2]) / 255;
  return { n, a };
}

export async function loadTemplates(base = 'assets/') {
  if (T.ready) return T;
  T.v1 = await loadAlpha(base + 'gemini_alpha_96.png');
  T.v2 = await loadAlpha(base + 'gemini_alpha_96_v2.png');
  T.n48 = await loadAlpha(base + 'gemini_alpha_48.png');
  T.unit = {};
  for (const k of ['v1', 'v2']) {
    const t = T[k];
    let m = 0;
    for (const v of t.a) m = Math.max(m, v);
    T.unit[k] = { n: t.n, a: t.a.map(v => v / m) };
  }
  T.ready = true;
  return T;
}

function tpl(variant) { return T[variant] || T.v1; }

// ---------------------------------------------------------------------------
// Basic image helpers
// ---------------------------------------------------------------------------
export function renderAlpha(t, x0, y0, s, bx, by, bw, bh) {
  const n = t.n, A = t.a, ss = 4;
  const out = new Float32Array(bw * bh);
  const k = n / s;
  const samp = (i, j) => (i >= 0 && j >= 0 && i < n && j < n) ? A[j * n + i] : 0;
  for (let oy = 0; oy < bh; oy++) {
    for (let ox = 0; ox < bw; ox++) {
      let acc = 0;
      for (let sy = 0; sy < ss; sy++) {
        const v = (by + oy + (sy + 0.5) / ss - y0) * k - 0.5;
        const j0 = Math.floor(v), fy = v - j0;
        for (let sx = 0; sx < ss; sx++) {
          const u = (bx + ox + (sx + 0.5) / ss - x0) * k - 0.5;
          const i0 = Math.floor(u), fx = u - i0;
          if (i0 < -1 || j0 < -1 || i0 >= n || j0 >= n) continue;
          acc += (samp(i0, j0) * (1 - fx) + samp(i0 + 1, j0) * fx) * (1 - fy)
               + (samp(i0, j0 + 1) * (1 - fx) + samp(i0 + 1, j0 + 1) * fx) * fy;
        }
      }
      out[oy * bw + ox] = acc / (ss * ss);
    }
  }
  return out;
}

export function gaussBlur(src, w, h, sigma) {
  if (sigma <= 0) return Float32Array.from(src);
  const r = Math.max(1, Math.ceil(sigma * 3));
  const ker = new Float32Array(2 * r + 1);
  let sum = 0;
  for (let i = -r; i <= r; i++) { ker[i + r] = Math.exp(-(i * i) / (2 * sigma * sigma)); sum += ker[i + r]; }
  for (let i = 0; i < ker.length; i++) ker[i] /= sum;
  const refl = (i, n) => { if (n === 1) return 0; while (i < 0 || i >= n) { i = i < 0 ? -i : 2 * n - 2 - i; } return i; };
  const tmp = new Float32Array(w * h), out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let a = 0;
    for (let i = -r; i <= r; i++) a += src[y * w + refl(x + i, w)] * ker[i + r];
    tmp[y * w + x] = a;
  }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let a = 0;
    for (let i = -r; i <= r; i++) a += tmp[refl(y + i, h) * w + x] * ker[i + r];
    out[y * w + x] = a;
  }
  return out;
}

function dilate(mask, w, h, r) {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let v = 0;
    for (let j = Math.max(0, y - r); j <= Math.min(h - 1, y + r) && !v; j++)
      for (let i = Math.max(0, x - r); i <= Math.min(w - 1, x + r); i++) if (mask[j * w + i]) { v = 1; break; }
    out[y * w + x] = v;
  }
  return out;
}

function erode(mask, w, h, r) {
  const inv = mask.map(v => (v ? 0 : 1));
  return dilate(inv, w, h, r).map(v => (v ? 0 : 1));
}

function gradEnergy(g, wm, w, h) {
  let e = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (x + 1 < w && wm[i + 1]) { const d = g[i + 1] - g[i]; e += d * d * wm[i + 1]; }
      if (y + 1 < h && wm[i + w]) { const d = g[i + w] - g[i]; e += d * d * wm[i + w]; }
    }
  }
  return e;
}

/** Smoothest (Laplace) continuation of P inside mask from its border. */
export function harmonicFill(P, mask, w, h, iters = 300) {
  const B = Float32Array.from(P);
  const known = new Uint8Array(w * h);
  let any = false;
  for (let i = 0; i < w * h; i++) { known[i] = mask[i] ? 0 : 1; if (mask[i]) any = true; }
  if (!any) return B;
  // onion-peel initialisation
  let changed = true, guard = 0;
  while (changed && guard++ < w + h) {
    changed = false;
    const add = [];
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (known[i]) continue;
      let s = 0, c = 0;
      if (x > 0 && known[i - 1]) { s += B[i - 1]; c++; }
      if (x < w - 1 && known[i + 1]) { s += B[i + 1]; c++; }
      if (y > 0 && known[i - w]) { s += B[i - w]; c++; }
      if (y < h - 1 && known[i + w]) { s += B[i + w]; c++; }
      if (c) add.push([i, s / c]);
    }
    for (const [i, v] of add) { B[i] = v; known[i] = 1; changed = true; }
  }
  const idx = [];
  for (let i = 0; i < w * h; i++) if (mask[i]) idx.push(i);
  for (let it = 0; it < iters; it++) {
    for (const i of idx) {
      const x = i % w, y = (i / w) | 0;
      const l = x > 0 ? B[i - 1] : B[i], r = x < w - 1 ? B[i + 1] : B[i];
      const u = y > 0 ? B[i - w] : B[i], d = y < h - 1 ? B[i + w] : B[i];
      B[i] = 0.25 * (l + r + u + d);
    }
  }
  return B;
}

function median(arr) {
  if (!arr.length) return 0;
  const a = Float32Array.from(arr).sort();
  const m = a.length >> 1;
  return a.length % 2 ? a[m] : 0.5 * (a[m - 1] + a[m]);
}

function golden(f, lo, hi, iters = 18) {
  const gr = 0.6180339887;
  let a = lo, b = hi, c = b - gr * (b - a), d = a + gr * (b - a);
  let fc = f(c), fd = f(d);
  for (let i = 0; i < iters; i++) {
    if (fc < fd) { b = d; d = c; fd = fc; c = b - gr * (b - a); fc = f(c); }
    else { a = c; c = d; fc = fd; d = a + gr * (b - a); fd = f(d); }
  }
  return fc < fd ? [c, fc] : [d, fd];
}

export function boxWithMargin(x, y, s, W, H, margin) {
  let bx = Math.floor(x - margin), by = Math.floor(y - margin);
  let ex = Math.ceil(x + s + margin), ey = Math.ceil(y + s + margin);
  bx = Math.max(0, bx); by = Math.max(0, by); ex = Math.min(W, ex); ey = Math.min(H, ey);
  return [bx, by, Math.max(1, ex - bx), Math.max(1, ey - by)];
}

function crop(plane, W, bx, by, bw, bh) {
  const out = new Float32Array(bw * bh);
  for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) out[y * bw + x] = plane[(by + y) * W + bx + x];
  return out;
}

export function grayFromRGBA(data, W, H) {
  const g = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) g[i] = 0.299 * data[4 * i] + 0.587 * data[4 * i + 1] + 0.114 * data[4 * i + 2];
  return g;
}

// ---------------------------------------------------------------------------
// Coarse detection: normalised cross correlation
// ---------------------------------------------------------------------------
function resizeArea(src, w, h, nw, nh) {
  const out = new Float32Array(nw * nh);
  const fx = w / nw, fy = h / nh;
  for (let y = 0; y < nh; y++) {
    const y0 = y * fy, y1 = (y + 1) * fy;
    for (let x = 0; x < nw; x++) {
      const x0 = x * fx, x1 = (x + 1) * fx;
      let acc = 0, wsum = 0;
      for (let j = Math.floor(y0); j < Math.ceil(y1) && j < h; j++) {
        const wy = Math.min(y1, j + 1) - Math.max(y0, j);
        for (let i = Math.floor(x0); i < Math.ceil(x1) && i < w; i++) {
          const wx = Math.min(x1, i + 1) - Math.max(x0, i);
          acc += src[j * w + i] * wx * wy; wsum += wx * wy;
        }
      }
      out[y * nw + x] = wsum ? acc / wsum : 0;
    }
  }
  return out;
}

function nccBest(img, W, H, rx, ry, rw, rh, t, tw) {
  if (tw >= rw || tw >= rh) return [-1, 0, 0];
  const N = tw * tw;
  let tm = 0;
  for (let i = 0; i < N; i++) tm += t[i];
  tm /= N;
  const tz = new Float32Array(N);
  let tss = 0;
  for (let i = 0; i < N; i++) { tz[i] = t[i] - tm; tss += tz[i] * tz[i]; }
  if (tss < 1e-9) return [-1, 0, 0];
  // integral images of the ROI
  const iw = rw + 1;
  const S = new Float64Array((rw + 1) * (rh + 1)), S2 = new Float64Array((rw + 1) * (rh + 1));
  for (let y = 0; y < rh; y++) {
    let rs = 0, rs2 = 0;
    for (let x = 0; x < rw; x++) {
      const v = img[(ry + y) * W + rx + x];
      rs += v; rs2 += v * v;
      S[(y + 1) * iw + x + 1] = S[y * iw + x + 1] + rs;
      S2[(y + 1) * iw + x + 1] = S2[y * iw + x + 1] + rs2;
    }
  }
  let best = -1, bx = 0, by = 0;
  for (let y = 0; y <= rh - tw; y++) {
    for (let x = 0; x <= rw - tw; x++) {
      const s1 = S[(y + tw) * iw + x + tw] - S[y * iw + x + tw] - S[(y + tw) * iw + x] + S[y * iw + x];
      const s2 = S2[(y + tw) * iw + x + tw] - S2[y * iw + x + tw] - S2[(y + tw) * iw + x] + S2[y * iw + x];
      const v = s2 - s1 * s1 / N;
      if (v < 1e-6) continue;
      let num = 0;
      for (let j = 0; j < tw; j++) {
        const row = (ry + y + j) * W + rx + x, tr = j * tw;
        for (let i = 0; i < tw; i++) num += tz[tr + i] * img[row + i];
      }
      const sc = num / Math.sqrt(v * tss);
      if (sc > best) { best = sc; bx = x; by = y; }
    }
  }
  return [best, rx + bx, ry + by];
}

function tplResized(variant, size) {
  const u = T.unit[variant] || T.unit.v1;
  return resizeArea(u.a, u.n, u.n, size, size);
}

export function cornerRoi(W, H, frac = 0.36, minPx = 260) {
  const rw = Math.min(W, Math.max(minPx, Math.floor(W * frac)));
  const rh = Math.min(H, Math.max(minPx, Math.floor(H * frac)));
  return [W - rw, H - rh, rw, rh];
}

/** Multi-scale search. Returns [score, size, x, y]. prior=[x,y,s] limits the search. */
export function coarseDetect(gray, W, H, prior = null) {
  let rx, ry, rw, rh, smin, smax;
  if (prior) {
    const [px, py, ps] = prior;
    const m = Math.max(6, Math.floor(ps * 0.35));
    rx = Math.max(0, Math.floor(px - m)); ry = Math.max(0, Math.floor(py - m));
    rw = Math.min(W, Math.floor(px + ps + m)) - rx; rh = Math.min(H, Math.floor(py + ps + m)) - ry;
    smin = Math.max(8, Math.floor(ps * 0.7)); smax = Math.floor(ps * 1.3) + 1;
  } else {
    [rx, ry, rw, rh] = cornerRoi(W, H);
    smin = 16;
    smax = Math.min(Math.floor(Math.min(rw, rh) * 0.8), Math.max(140, Math.floor(Math.min(W, H) * 0.12)));
  }
  // stage 1: downscaled search
  const f = Math.max(1, Math.floor(Math.min(smin, 48) / 8));
  let cand;
  if (f > 1 && !prior) {
    const dw = Math.floor(rw / f), dh = Math.floor(rh / f);
    const roi = resizeArea(crop(gray, W, rx, ry, rw, rh), rw, rh, dw, dh);
    let best = [-1, 0, 0, 0];
    for (let s = Math.max(4, Math.floor(smin / f)); s <= Math.floor(smax / f); s++) {
      const r = nccBest(roi, dw, dh, 0, 0, dw, dh, tplResized('v1', s), s);
      if (r[0] > best[0]) best = [r[0], s * f, rx + r[1] * f, ry + r[2] * f];
    }
    cand = best;
  }
  // stage 2: full resolution around the candidate (or the whole prior window)
  let best = [-1, 0, 0, 0];
  if (cand && cand[1]) {
    const [, cs, cx, cy] = cand;
    const m = 2 * f + 2;
    const x0 = Math.max(0, cx - m), y0 = Math.max(0, cy - m);
    for (let s = Math.max(smin, cs - f - 1); s <= Math.min(smax, cs + f + 1); s++) {
      const w2 = Math.min(W - x0, s + 2 * m), h2 = Math.min(H - y0, s + 2 * m);
      const r = nccBest(gray, W, H, x0, y0, w2, h2, tplResized('v1', s), s);
      if (r[0] > best[0]) best = [r[0], s, r[1], r[2]];
    }
  } else {
    for (let s = smin; s <= smax; s++) {
      const r = nccBest(gray, W, H, rx, ry, rw, rh, tplResized('v1', s), s);
      if (r[0] > best[0]) best = [r[0], s, r[1], r[2]];
    }
  }
  return best;
}

// ---------------------------------------------------------------------------
// Fitting
// ---------------------------------------------------------------------------
class Problem {
  constructor(grays, box, white, geom) {
    this.grays = grays; this.box = box; this.white = white;
    const [bx, by, bw, bh] = box;
    const [x, y, s] = geom;
    const sup = renderAlpha(T.v1, x, y, s, bx, by, bw, bh).map(v => (v > 0.01 ? 1 : 0));
    const k = Math.max(3, Math.round(s * 0.08));
    this.w = Float32Array.from(dilate(sup, bw, bh, k));
    this.cache = new Map();
  }
  alpha(v, x, y, s) {
    const key = `${v}|${x.toFixed(3)}|${y.toFixed(3)}|${s.toFixed(3)}`;
    let a = this.cache.get(key);
    if (!a) {
      const [bx, by, bw, bh] = this.box;
      a = renderAlpha(tpl(v), x, y, s, bx, by, bw, bh);
      if (this.cache.size > 3000) this.cache.clear();
      this.cache.set(key, a);
    }
    return a;
  }
  energy(v, x, y, s, k, beta = 0) {
    const [, , bw, bh] = this.box;
    let A = this.alpha(v, x, y, s);
    if (beta) {
      const b = gaussBlur(A, bw, bh, SHARP_SIGMA);
      A = A.map((val, i) => val + beta * (val - b[i]));
    }
    let tot = 0;
    const r = new Float32Array(bw * bh);
    for (const g of this.grays) {
      for (let i = 0; i < r.length; i++) {
        const a = Math.min(0.92, Math.max(0, A[i] * k));
        r[i] = (g[i] - a * this.white) / (1 - a);
      }
      tot += gradEnergy(r, this.w, bw, bh);
    }
    return tot;
  }
  base() {
    const [, , bw, bh] = this.box;
    let t = 0;
    for (const g of this.grays) t += gradEnergy(g, this.w, bw, bh);
    return t;
  }
}

function refine(prob, x, y, s, variants, rounds = 3) {
  const res = [];
  for (const v of variants) {
    if (v === 'n48' && Math.abs(s - 48) > 3) continue;
    const p = { x, y, s, k: 1, b: 0 };
    const E = (o = {}) => { const q = { ...p, ...o }; return prob.energy(v, q.x, q.y, q.s, q.k, q.b); };
    const [klo, khi] = K_RANGE[v] || [0.15, 1.9];
    let e;
    [p.k, e] = golden(kk => E({ k: kk }), klo, khi, 16);
    let best = [e, 0];
    for (const bb of [-0.3, 0.3, 0.6, 1.0]) {
      const [kk, ee] = golden(q => E({ k: q, b: bb }), Math.max(klo, p.k * 0.7), Math.min(khi, p.k * 1.3), 10);
      if (ee < best[0]) { best = [ee, bb]; p.k = kk; }
    }
    p.b = best[1];
    for (let r = 0; r < rounds; r++) {
      const f = 0.5 ** r, span = 1.2 * f;
      [p.x] = golden(q => E({ x: q }), p.x - span, p.x + span, 10);
      [p.y] = golden(q => E({ y: q }), p.y - span, p.y + span, 10);
      const ss = Math.max(1, p.s * 0.03) * f;
      [p.s] = golden(q => E({ s: q }), p.s - ss, p.s + ss, 10);
      [p.k] = golden(q => E({ k: q }), Math.max(klo, p.k * (1 - 0.2 * f)), Math.min(khi, p.k * (1 + 0.2 * f)), 10);
      [p.b] = golden(q => E({ b: q }), Math.max(-0.5, p.b - 0.5 * f), Math.min(1.6, p.b + 0.5 * f), 10);
    }
    res.push([E(), v, p.x, p.y, p.s, p.k, p.b]);
  }
  res.sort((a, b) => a[0] - b[0]);
  return res[0];
}

function ringRoughness(g, sup, w, h) {
  const inner = dilate(sup, w, h, 3), outer = dilate(sup, w, h, 9);
  const bl = gaussBlur(g, w, h, 1.5);
  let s = 0, c = 0;
  for (let i = 0; i < w * h; i++) if (outer[i] && !inner[i]) { s += Math.abs(g[i] - bl[i]); c++; }
  return c < 10 ? 99 : s / c;
}

export function estimateStrengthDC(g, A, w, h, white) {
  let am = 0;
  for (const v of A) am = Math.max(am, v);
  const au = A.map(v => v / Math.max(am, 1e-6));
  const sup0 = au.map(v => (v > 0.01 ? 1 : 0));
  const sup = dilate(sup0, w, h, 2);
  const rough = ringRoughness(g, sup0, w, h);
  const B = harmonicFill(g, sup, w, h, 300);
  const inner = erode(au.map(v => (v > 0.6 ? 1 : 0)), w, h, 1);
  const idx = [];
  for (let i = 0; i < w * h; i++) if (inner[i]) idx.push(i);
  if (idx.length < 12) return [1, 0];
  const contrast = median(idx.map(i => white - B[i]));
  if (contrast < 25) return [1, 0];
  const f = k => median(idx.map(i => { const a = Math.min(0.92, A[i] * k); return (g[i] - a * white) / (1 - a) - B[i]; }));
  let lo = 0.05, hi = 1.9;
  if (f(lo) <= 0) return [lo, 0];
  if (f(hi) >= 0) return [hi, 0];
  for (let i = 0; i < 32; i++) { const m = 0.5 * (lo + hi); if (f(m) > 0) lo = m; else hi = m; }
  let trust = 1 / (1 + (rough / 3) ** 2);
  trust *= Math.min(1, contrast / 80);
  return [0.5 * (lo + hi), trust];
}

/** frames: array of gray planes (full size W x H). */
export function fitGeometry(frames, coarse, white, W, H, score, variants = ['v1', 'v2', 'n48'], minGain = 0.12) {
  const [x, y, s] = coarse;
  const margin = Math.max(4, Math.floor(s * 0.25));
  const box = boxWithMargin(x, y, s, W, H, margin);
  const [bx, by, bw, bh] = box;
  const grays = frames.map(f => crop(f, W, bx, by, bw, bh));
  const prob = new Problem(grays, box, white, [x, y, s]);
  const e0 = prob.base();
  const [e, v, cx, cy, cs, kg, beta] = refine(prob, x, y, s, variants);
  const gain = 1 - e / Math.max(e0, 1e-6);
  const need = score >= 0.5 ? minGain : Math.max(minGain, 0.3);
  const geo = { x: cx, y: cy, size: cs, strength: kg, variant: v, sharp: beta, chromaBlur: 0, score, gain,
    found: gain >= need && kg >= K_MIN && kg <= K_MAX, note: '' };
  if (!geo.found) { geo.note = 'weak'; return geo; }
  const A = renderAlpha(tpl(v), cx, cy, cs, bx, by, bw, bh);
  const ks = [], ws = [];
  for (const g of grays.slice(0, 24)) {
    const [kd, tr] = estimateStrengthDC(g, A, bw, bh, white);
    if (tr > 0) { ks.push(kd); ws.push(tr); }
  }
  if (ks.length) {
    const wsum = ws.reduce((a, b) => a + b, 0);
    const kd = ks.reduce((a, k, i) => a + k * ws[i], 0) / wsum;
    const tr = Math.min(1, wsum / ws.length);
    if (tr >= 0.3 && kd < K_MIN * 0.8) { geo.found = false; geo.note = 'no-lift'; return geo; }
    if (kd >= 0.75 * kg && kd <= 1.3 * kg) geo.strength = tr * kd + (1 - tr) * kg;
  }
  const [lo, hi] = K_RANGE[v] || [0.15, 1.9];
  geo.strength = Math.min(hi, Math.max(lo, geo.strength));
  geo.note = 'ok';
  return geo;
}

export function alphaFor(geo, bx, by, bw, bh) {
  let A = renderAlpha(tpl(geo.variant), geo.x, geo.y, geo.size, bx, by, bw, bh);
  if (geo.sharp) {
    const b = gaussBlur(A, bw, bh, SHARP_SIGMA);
    A = A.map((v, i) => v + geo.sharp * (v - b[i]));
  }
  return A.map(v => Math.min(0.95, Math.max(0, v * geo.strength)));
}

// ---------------------------------------------------------------------------
// Removal on RGBA ImageData
// ---------------------------------------------------------------------------
function seamWeights(a, w, h) {
  let m = 0;
  for (const v of a) m = Math.max(m, v);
  const u = a.map(v => v / Math.max(m, 1e-6));
  const g = new Float32Array(w * h);
  let gm = 0;
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const i = y * w + x;
    const gx = (u[i - w + 1] + 2 * u[i + 1] + u[i + w + 1]) - (u[i - w - 1] + 2 * u[i - 1] + u[i + w - 1]);
    const gy = (u[i + w - 1] + 2 * u[i + w] + u[i + w + 1]) - (u[i - w - 1] + 2 * u[i - w] + u[i - w + 1]);
    g[i] = Math.hypot(gx, gy); gm = Math.max(gm, g[i]);
  }
  const ms = g.map(v => Math.min(1, Math.max(0, (v / Math.max(gm, 1e-6) - 0.12) / 0.4)));
  return gaussBlur(ms, w, h, 0.8);
}

function fitChromaBlur(Cb, Cr, a, w, h) {
  const wm = Float32Array.from(dilate(a.map(v => (v > 0.005 ? 1 : 0)), w, h, 2));
  let best = [Infinity, 0];
  for (const sg of [0, 0.5, 0.8, 1.1, 1.5]) {
    const ac = sg ? gaussBlur(a, w, h, sg) : a;
    let e = 0;
    for (const C of [Cb, Cr]) {
      const r = C.map((v, i) => { const q = Math.min(0.95, ac[i]); return (v - q * 128) / (1 - q); });
      e += gradEnergy(r, wm, w, h);
    }
    if (e < best[0]) best = [e, sg];
  }
  return best[1];
}

function extractYCC(data, W, bx, by, bw, bh) {
  const n = bw * bh, Y = new Float32Array(n), Cb = new Float32Array(n), Cr = new Float32Array(n);
  for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) {
    const p = 4 * ((by + y) * W + bx + x), i = y * bw + x;
    const r = data[p], g = data[p + 1], b = data[p + 2];
    Y[i] = 0.299 * r + 0.587 * g + 0.114 * b;
    Cb[i] = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b;
    Cr[i] = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b;
  }
  return [Y, Cb, Cr];
}

function writeYCC(data, W, bx, by, bw, bh, Y, Cb, Cr) {
  const cl = v => Math.max(0, Math.min(255, Math.round(v)));
  for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) {
    const p = 4 * ((by + y) * W + bx + x), i = y * bw + x;
    data[p] = cl(Y[i] + 1.402 * (Cr[i] - 128));
    data[p + 1] = cl(Y[i] - 0.344136 * (Cb[i] - 128) - 0.714136 * (Cr[i] - 128));
    data[p + 2] = cl(Y[i] + 1.772 * (Cb[i] - 128));
  }
}

function seamClean(P, a, w, h) {
  const sw = seamWeights(a, w, h);
  const hard = sw.map(v => (v > 0.25 ? 1 : 0));
  const fill = harmonicFill(P, hard, w, h, 60);
  return P.map((v, i) => v * (1 - sw[i]) + fill[i] * sw[i]);
}

/** In-place removal on an ImageData-like {data,width,height}. */
export function removeWithGeometry(img, geo, { seam = true, method = 'alpha' } = {}) {
  const W = img.width, H = img.height, d = img.data;
  const margin = Math.max(4, Math.floor(geo.size * 0.25));
  const [bx, by, bw, bh] = boxWithMargin(geo.x, geo.y, geo.size, W, H, margin);
  if (method === 'inpaint' || !geo.found) {
    const a = renderAlpha(T.v1, geo.x, geo.y, geo.size, bx, by, bw, bh);
    const k = Math.max(1, Math.round(geo.size * 0.06));
    const mask = dilate(a.map(v => (v > 0.02 ? 1 : 0)), bw, bh, k);
    for (let c = 0; c < 3; c++) {
      const P = new Float32Array(bw * bh);
      for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) P[y * bw + x] = d[4 * ((by + y) * W + bx + x) + c];
      const F = harmonicFill(P, mask, bw, bh, 400);
      for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) {
        d[4 * ((by + y) * W + bx + x) + c] = Math.max(0, Math.min(255, Math.round(F[y * bw + x])));
      }
    }
    return;
  }
  const a = alphaFor(geo, bx, by, bw, bh);
  let [Y, Cb, Cr] = extractYCC(d, W, bx, by, bw, bh);
  if (geo.chromaBlur === undefined || geo.chromaBlur === null) geo.chromaBlur = fitChromaBlur(Cb, Cr, a, bw, bh);
  const ac = geo.chromaBlur ? gaussBlur(a, bw, bh, geo.chromaBlur) : a;
  Y = Y.map((v, i) => (v - a[i] * 255) / (1 - a[i]));
  Cb = Cb.map((v, i) => { const q = Math.min(0.95, ac[i]); return (v - q * 128) / (1 - q); });
  Cr = Cr.map((v, i) => { const q = Math.min(0.95, ac[i]); return (v - q * 128) / (1 - q); });
  if (seam) { Y = seamClean(Y, a, bw, bh); Cb = seamClean(Cb, a, bw, bh); Cr = seamClean(Cr, a, bw, bh); }
  writeYCC(d, W, bx, by, bw, bh, Y, Cb, Cr);
}

export function residualScore(img, geo) {
  const W = img.width, H = img.height;
  const margin = Math.max(4, Math.floor(geo.size * 0.25));
  const [bx, by, bw, bh] = boxWithMargin(geo.x, geo.y, geo.size, W, H, margin);
  const A = renderAlpha(T.v1, geo.x, geo.y, geo.size, bx, by, bw, bh);
  const sup0 = A.map(v => (v > 0.01 ? 1 : 0));
  const sup = dilate(sup0, bw, bh, 2);
  const inner = erode(A.map(v => (v > 0.3 ? 1 : 0)), bw, bh, 1);
  const idx = [];
  for (let i = 0; i < bw * bh; i++) if (inner[i]) idx.push(i);
  if (idx.length < 10) return [0, 0];
  let diff = 0;
  const gray = new Float32Array(bw * bh);
  for (let c = 0; c < 3; c++) {
    const P = new Float32Array(bw * bh);
    for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) P[y * bw + x] = img.data[4 * ((by + y) * W + bx + x) + c];
    for (let i = 0; i < P.length; i++) gray[i] += P[i] * [0.299, 0.587, 0.114][c];
    const B = harmonicFill(P, sup, bw, bh, 250);
    diff += Math.abs(median(idx.map(i => P[i] - B[i])));
  }
  const rough = ringRoughness(gray, sup0, bw, bh);
  return [diff / 3, 1 / (1 + (rough / 3) ** 2)];
}

function cloneImg(img) { return { width: img.width, height: img.height, data: new Uint8ClampedArray(img.data) }; }

/** Full pipeline on ImageData. opts: {mode:'auto'|'manual', box:[x,y,s], method}. Returns {img, geo}. */
export function processImage(src, opts = {}) {
  const W = src.width, H = src.height;
  const gray = grayFromRGBA(src.data, W, H);
  const prior = opts.mode === 'manual' && opts.box ? opts.box : null;
  if (prior && opts.method === 'inpaint') {
    const geo = { x: prior[0], y: prior[1], size: prior[2], found: false, note: 'manual-inpaint' };
    const out = cloneImg(src); removeWithGeometry(out, geo, { method: 'inpaint' });
    return { img: out, geo };
  }
  const [score, s, x, y] = coarseDetect(gray, W, H, prior);
  let geo;
  if (!s || (!prior && score < 0.3)) {
    geo = { x, y, size: s, score, found: false, note: 'not-found' };
  } else {
    geo = fitGeometry([gray], [x, y, s], 255, W, H, score);
  }
  if (!geo.found) {
    if (prior) {
      geo = { x: prior[0], y: prior[1], size: prior[2], found: false, note: 'manual-inpaint' };
      const out = cloneImg(src); removeWithGeometry(out, geo, { method: 'inpaint' });
      return { img: out, geo };
    }
    return { img: cloneImg(src), geo };
  }
  geo.chromaBlur = null;
  let out = cloneImg(src);
  removeWithGeometry(out, geo, { seam: true });
  // quality guard: visible residue on a smooth background -> inpaint instead
  const [res, trust] = residualScore(out, geo);
  if ((trust >= 0.45 && res >= 5) || res >= 12) {
    const alt = cloneImg(src);
    removeWithGeometry(alt, { ...geo, found: false }, { method: 'inpaint' });
    const [res2] = residualScore(alt, geo);
    if (res2 < res * (trust >= 0.45 ? 0.6 : 0.35)) { out = alt; geo.note = 'inpaint-fallback'; }
  }
  return { img: out, geo };
}

export function applyGeometry(src, geo, method = 'alpha') {
  const out = cloneImg(src);
  removeWithGeometry(out, geo, { method: geo.found && method !== 'inpaint' ? 'alpha' : 'inpaint' });
  return out;
}
