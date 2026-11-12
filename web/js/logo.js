// AI Remove - thsky90 — still-image logo detection + removal (v2.3)
// Browser port of airemove/detect.py and airemove/restore.py.
//
// Detection: a semi-transparent white overlay MUST lift every pixel just
// inside its outline relative to the neighbour just outside, whatever lies
// underneath.  Each pixel pair straddling the outline gives its own estimate
// of the logo strength k; on a real logo most pairs agree ("inlier ratio"),
// elsewhere they scatter.  Text, faces or busy artwork under the logo only
// spoil the pairs they cross, which the weighted median ignores.
//
// Removal: exact reverse blend, colour planes with the codec's chroma blur,
// alignment of the interior with its surroundings when a faint "diamond" is
// left, and a guarded clean-up of the 1-2 px outline seam.

import { T, renderAlpha, gaussBlur, boxWithMargin, harmonicFill, cornerRoi,
         grayFromRGBA, golden, SHARP_SIGMA } from './engine.js';

const PRIOR_RATIOS = [2 / 3, 2];
const K_MIN = 0.28, K_MAX = 1.5;
const DIRS = [[0, 1], [1, 0], [1, 1], [1, -1]];            // [dy, dx]
const T_ON_PRIOR = 0.28, T_OFF_PRIOR = 0.60, T_MANUAL = 0.22;
const K_AUTO_MIN = 0.40, K_AUTO_MAX = 1.18;
const PLATEAU_MIN = 0.20, PLATEAU_MAX = 0.60, PLATEAU_MANUAL = [0.12, 0.78];
const PROFILE_MAX = 0.45;
const A_MAX = 0.92;
const R_MAX = 45;

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
const tplOf = v => T[v] || T.v1;
const pyRound = v => { const r = Math.round(v); return (Math.abs(v % 1) === 0.5 && r % 2) ? r - 1 : r; };
export const strideFor = s => Math.max(2, pyRound(s / 16));

function percentile(arr, q) {
  if (!arr.length) return 0;
  const a = Float32Array.from(arr).sort();
  const pos = (q / 100) * (a.length - 1), lo = Math.floor(pos);
  return lo + 1 < a.length ? a[lo] + (pos - lo) * (a[lo + 1] - a[lo]) : a[lo];
}

const _plateau = {};
export function plateauOf(variant) {
  if (!(variant in _plateau)) _plateau[variant] = percentile(tplOf(variant).a, 90);
  return _plateau[variant];
}

/** Luma and min(R,G,B): both obey the blending model exactly. */
export function planesOf(img) {
  const n = img.width * img.height, d = img.data;
  const y = new Float32Array(n), mn = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const r = d[4 * i], g = d[4 * i + 1], b = d[4 * i + 2];
    y[i] = 0.299 * r + 0.587 * g + 0.114 * b;
    mn[i] = Math.min(r, g, b);
  }
  return [y, mn];
}

function wmedian(vals, wts, n) {
  const idx = new Uint32Array(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  idx.sort((a, b) => vals[a] - vals[b]);
  let tot = 0;
  for (let i = 0; i < n; i++) tot += wts[i];
  let acc = 0;
  for (let i = 0; i < n; i++) { acc += wts[idx[i]]; if (acc >= tot * 0.5) return vals[idx[i]]; }
  return vals[idx[n - 1]];
}

// ---------------------------------------------------------------------------
// Outline voting
// ---------------------------------------------------------------------------
const voteOk = v => v.k >= K_MIN && v.k <= K_MAX;
const rank = v => (voteOk(v) ? v.inlier : v.inlier * 0.25);

/** Pixel pairs d apart that straddle the outline of alpha patch A (aw x ah).
 *  Depends on the alpha only, so one set serves every position it is tried at. */
function pairSet(A, aw, ah, d, cx = null, cy = null, frac = 0.5) {
  let am = 0;
  for (let i = 0; i < A.length; i++) if (A[i] > am) am = A[i];
  const thr = frac * am;
  let cap = 2048, n = 0;
  let py = new Int16Array(cap), px = new Int16Array(cap), sy = new Int16Array(cap), sx = new Int16Array(cap);
  let ap = new Float32Array(cap), aq = new Float32Array(cap), qd = new Uint8Array(cap);
  const grow = () => {
    cap *= 2;
    const g = (old, C) => { const t = new C(cap); t.set(old); return t; };
    py = g(py, Int16Array); px = g(px, Int16Array); sy = g(sy, Int16Array); sx = g(sx, Int16Array);
    ap = g(ap, Float32Array); aq = g(aq, Float32Array); qd = g(qd, Uint8Array);
  };
  if (am > 1e-6) {
    for (const [uy, ux] of DIRS) {
      const dy = uy * d, dx = ux * d;
      const xa = dx >= 0 ? 0 : -dx, xb = dx >= 0 ? aw - dx : aw;
      for (let y = 0; y < ah - dy; y++) for (let x = xa; x < xb; x++) {
        const vp = A[y * aw + x], vq = A[(y + dy) * aw + x + dx];
        if (Math.abs(vp - vq) < thr) continue;
        if (n === cap) grow();
        py[n] = y; px[n] = x; sy[n] = dy; sx[n] = dx; ap[n] = vp; aq[n] = vq;
        qd[n] = cx === null ? 0 : ((y >= cy ? 2 : 0) + (x >= cx ? 1 : 0));
        n++;
      }
    }
  }
  return { n, py, px, sy, sx, ap, aq, qd, am, quads: cx !== null };
}

const HB = 8192, HLO = -4, HSCALE = HB / 8;               // histogram of k over [-4, 4)
const _hist = new Float64Array(HB);
let _ks = new Float32Array(1 << 15), _ws = new Float32Array(1 << 15), _qs = new Uint8Array(1 << 15);

/** Strength vote of a pair set placed with its patch origin at (ax, ay) of the image planes. */
function voteSet(planes, W, ps, ax, ay) {
  const none = { k: 0, inlier: 0, n: 0, cover: 0, quad: 0 };
  if (!ps.n) return none;
  const need = ps.n * planes.length;
  if (need > _ks.length) { const c = 1 << Math.ceil(Math.log2(need)); _ks = new Float32Array(c); _ws = new Float32Array(c); _qs = new Uint8Array(c); }
  const denMin = 0.5 * ps.am * 10;
  let n = 0, wAll = 0;
  _hist.fill(0);
  for (let i = 0; i < ps.n; i++) {
    const gp = (ay + ps.py[i]) * W + ax + ps.px[i], gq = gp + ps.sy[i] * W + ps.sx[i];
    const a1 = ps.ap[i], a2 = ps.aq[i];
    for (let c = 0; c < planes.length; c++) {
      const vp = planes[c][gp], vq = planes[c][gq];
      const den = a1 * (255 - vq) - a2 * (255 - vp), w = Math.abs(den);
      if (w <= denMin) continue;
      const k = (vp - vq) / den;
      _ks[n] = k; _ws[n] = w; _qs[n] = ps.qd[i]; n++;
      wAll += w;
      _hist[clamp(((k - HLO) * HSCALE) | 0, 0, HB - 1)] += w;
    }
  }
  if (!n) return none;
  if (n < 24) return { ...none, n };
  // weighted median from the histogram (bin width 0.001, interpolated)
  let acc = 0, km = 0;
  const half = wAll * 0.5;
  for (let b = 0; b < HB; b++) {
    if (acc + _hist[b] >= half) { km = HLO + (b + (_hist[b] > 0 ? (half - acc) / _hist[b] : 0.5)) / HSCALE; break; }
    acc += _hist[b];
  }
  const tol = 0.2 * Math.max(Math.abs(km), 0.3);
  let wIn = 0;
  const qa = [0, 0, 0, 0], qi = [0, 0, 0, 0];
  for (let i = 0; i < n; i++) {
    const w = _ws[i], q = _qs[i];
    qa[q] += w;
    if (Math.abs(_ks[i] - km) < tol) { wIn += w; qi[q] += w; }
  }
  let quad = 0;
  if (ps.quads) quad = Math.min(...[0, 1, 2, 3].map(i => (qa[i] > 0 ? qi[i] / qa[i] : 0)));
  return { k: km, inlier: wIn / wAll, n, cover: n / Math.max(need, 1), quad };
}

/** A: alpha patch (aw x ah) whose top-left pixel sits at (ax, ay) of the image planes. */
function vote(planes, W, A, aw, ah, ax, ay, d, cx = null, cy = null) {
  return voteSet(planes, W, pairSet(A, aw, ah, d, cx, cy), ax, ay);
}

/** Is the brightness step confined to the outline?  Returns [r_in, r_out, n]. */
function profileTest(planes, W, A, aw, ah, ax, ay, k, d) {
  let am = 0;
  for (let i = 0; i < A.length; i++) if (A[i] > am) am = A[i];
  const rin = [], rout = [], wts = [];
  for (const [uy, ux] of DIRS) {
    for (const sgn of [1, -1]) {
      const dy = uy * d * sgn, dx = ux * d * sgn;
      const y0 = Math.max(0, dy, -dy, -2 * dy), y1 = ah - Math.max(0, dy, 2 * dy, -dy);
      const x0 = Math.max(0, dx, -dx, -2 * dx), x1 = aw - Math.max(0, dx, 2 * dx, -dx);
      if (y1 - y0 < 2 || x1 - x0 < 2) continue;
      for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
        const Ap = A[y * aw + x];
        if (Ap < 0.9 * am || A[(y - dy) * aw + x - dx] < 0.9 * am) continue;
        if (A[(y + dy) * aw + x + dx] > 0.04 * am || A[(y + 2 * dy) * aw + x + 2 * dx] > 0.04 * am) continue;
        const g = (ay + y) * W + ax + x, st = dy * W + dx;
        for (const P of planes) {
          const ip2 = P[g - st], ip = P[g], iq = P[g + st], iq2 = P[g + 2 * st];
          const ref = Ap * k * (255 - iq);
          if (ref > 4) { rin.push((ip2 - ip) / ref); rout.push((iq - iq2) / ref); wts.push(ref); }
        }
      }
    }
  }
  const n = rin.length;
  if (n < 16) return [0, 0, n];
  return [wmedian(rin, wts, n), wmedian(rout, wts, n), n];
}

const _g5 = (() => { const k = [-2, -1, 0, 1, 2].map(i => Math.exp(-(i * i) / (2 * SHARP_SIGMA * SHARP_SIGMA))); const t = k.reduce((x, y) => x + y); return k.map(v => v / t); })();
/** Gaussian blur with sigma = SHARP_SIGMA (5 taps, clamped borders). */
function blur5(A, w, h) {
  const tmp = new Float32Array(w * h), out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const r = y * w;
    for (let x = 0; x < w; x++) {
      const x0 = x > 1 ? x - 2 : 0, x1 = x > 0 ? x - 1 : 0, x3 = x < w - 1 ? x + 1 : w - 1, x4 = x < w - 2 ? x + 2 : w - 1;
      tmp[r + x] = _g5[0] * A[r + x0] + _g5[1] * A[r + x1] + _g5[2] * A[r + x] + _g5[3] * A[r + x3] + _g5[4] * A[r + x4];
    }
  }
  for (let y = 0; y < h; y++) {
    const y0 = (y > 1 ? y - 2 : 0) * w, y1 = (y > 0 ? y - 1 : 0) * w, y3 = (y < h - 1 ? y + 1 : h - 1) * w, y4 = (y < h - 2 ? y + 2 : h - 1) * w, r = y * w;
    for (let x = 0; x < w; x++) out[r + x] = _g5[0] * tmp[y0 + x] + _g5[1] * tmp[y1 + x] + _g5[2] * tmp[r + x] + _g5[3] * tmp[y3 + x] + _g5[4] * tmp[y4 + x];
  }
  return out;
}

function sharpen(A, w, h, beta) {
  if (!beta) return A;
  const b = blur5(A, w, h);
  const out = new Float32Array(A.length);
  for (let i = 0; i < A.length; i++) out[i] = A[i] + beta * (A[i] - b[i]);
  return out;
}

function voteAt(planes, W, H, x, y, s, variant = 'v1', d = null, beta = 0, ss = 4) {
  d = d || strideFor(s);
  const [bx, by, bw, bh] = boxWithMargin(x, y, s, W, H, d + 2);
  const A = sharpen(renderAlpha(tplOf(variant), x, y, s, bx, by, bw, bh, ss), bw, bh, beta);
  return vote(planes, W, A, bw, bh, bx, by, d, x + s / 2 - bx, y + s / 2 - by);
}

function profileAt(planes, W, H, x, y, s, k) {
  const d = strideFor(s);
  const [bx, by, bw, bh] = boxWithMargin(x, y, s, W, H, 3 * d + 2);
  const A = renderAlpha(T.v1, x, y, s, bx, by, bw, bh);
  return profileTest(planes, W, A, bw, bh, bx, by, Math.max(k, 0.05), d);
}

export function isOnPrior(x, y, s, W, H, tol = 0.12) {
  const mr = W - x - s, mb = H - y - s;
  if (Math.abs(mr - mb) > Math.max(3, 0.12 * s)) return false;
  const m = 0.5 * (mr + mb);
  return PRIOR_RATIOS.some(r => Math.abs(m - r * s) <= Math.max(3, tol * (r + 1) * s));
}

function priorRatio(x, y, s, W, H) {
  if (!isOnPrior(x, y, s, W, H)) return null;
  const m = 0.5 * ((W - x - s) + (H - y - s));
  return PRIOR_RATIOS.reduce((a, r) => (Math.abs(m - r * s) < Math.abs(m - a * s) ? r : a));
}

/** Maximise the inlier ratio in a small neighbourhood of (x, y, s). */
function localVoteSearch(planes, W, H, x, y, s, ratio = null) {
  let best = null;
  const tryAt = (vx, vy, vs, pen = 0) => {
    if (vs < 6 || vx < 0 || vy < 0 || vx + vs > W || vy + vs > H) return;
    const v = voteAt(planes, W, H, vx, vy, vs);
    const r = rank(v) - pen;
    if (!best || r > best.r) best = { r, v, x: vx, y: vy, s: vs };
  };
  if (ratio !== null) {
    const s0 = 0.5 * ((W - x) + (H - y)) / (ratio + 1);
    for (const ds of [0, -0.02, 0.02, -0.04, 0.04, -0.07, 0.07]) {
      const ss = s0 * (1 + ds);
      const gx = W - (ratio + 1) * ss, gy = H - (ratio + 1) * ss;
      for (const dy of [0, -1, 1]) for (const dx of [0, -1, 1]) {
        tryAt(gx + dx, gy + dy, ss, 0.01 * (Math.abs(dx) + Math.abs(dy)));
      }
    }
    return best;
  }
  const span = Math.max(1.5, s * 0.04);
  for (const ds of [0, -0.06, 0.06]) {
    const ss = s * (1 + ds);
    const cx = x + (s - ss) / 2, cy = y + (s - ss) / 2;
    for (const dy of [0, -span, span]) for (const dx of [0, -span, span]) tryAt(cx + dx, cy + dy, ss);
  }
  return best;
}

// ---------------------------------------------------------------------------
// Candidates
// ---------------------------------------------------------------------------
/** Known Gemini layouts: margin = r x size from the bottom-right corner, any scale. */
function priorScan(planes, W, H, roi, smin, smax) {
  const out = [];
  for (const r of PRIOR_RATIOS) {
    const list = [];
    for (let s = smin; s <= smax; s += Math.max(0.5, s * 0.015)) {
      const x = Math.round(W - (r + 1) * s), y = Math.round(H - (r + 1) * s);
      if (x < roi[0] || y < roi[1] || x + s > W || y + s > H) continue;
      list.push({ score: rank(voteAt(planes, W, H, x, y, s, 'v1', null, 0, 2)), x, y, s });
    }
    list.sort((a, b) => b.score - a.score);
    const kept = [];
    for (const c of list) {
      if (kept.every(o => Math.abs(Math.log(c.s / o.s)) > 0.08)) kept.push(c);
      if (kept.length >= 2) break;
    }
    out.push(...kept);
  }
  return out;
}

/** Anywhere in the corner (cropped pictures, unusual layouts): correlate the
 *  sign of the brightness steps with the outline pattern, on a pyramid so the
 *  template is always 12-23 px.  Returns up to ``top`` candidates {x, y, s}. */
function denseScan(planes, W, H, roi, smin, smax, top = 3) {
  const [rx, ry, rw, rh] = roi, d = 2, cands = [];
  for (let f = 2; f <= 16; f *= 2) {
    if (12 * f > smax) break;
    if (24 * f <= smin) continue;
    const lw = Math.floor(rw / f), lh = Math.floor(rh / f);
    if (lw < 16 || lh < 16) break;
    // step maps: clip((P[p] - P[p + d*e]) / 4, -1, 1), mean over planes, on the area-averaged level
    const D = DIRS.map(() => new Float32Array(lw * lh));
    for (const P of planes) {
      const L = new Float32Array(lw * lh), inv = 1 / (f * f);
      for (let y = 0; y < lh; y++) for (let x = 0; x < lw; x++) {
        let acc = 0;
        for (let j = 0; j < f; j++) { const r = (ry + y * f + j) * W + rx + x * f; for (let i = 0; i < f; i++) acc += P[r + i]; }
        L[y * lw + x] = acc * inv;
      }
      DIRS.forEach(([uy, ux], k) => {
        const dy = uy * d, dx = ux * d, xa = dx >= 0 ? 0 : -dx, xb = dx >= 0 ? lw - dx : lw, Dk = D[k];
        for (let y = 0; y < lh - dy; y++) for (let x = xa; x < xb; x++) {
          Dk[y * lw + x] += clamp((L[y * lw + x] - L[(y + dy) * lw + x + dx]) * 0.25, -1, 1) / planes.length;
        }
      });
    }
    for (let t = Math.max(12, Math.ceil(smin / f)); t <= Math.min(23, Math.floor(smax / f)) && t < Math.min(lw, lh) - 1; t++) {
      const A = renderAlpha(T.v1, 0, 0, t, 0, 0, t, t);
      let am = 0;
      for (const v of A) if (v > am) am = v;
      const pos = DIRS.map(() => []), neg = DIRS.map(() => []);
      let norm = 0;
      DIRS.forEach(([uy, ux], k) => {
        const dy = uy * d, dx = ux * d, xa = dx >= 0 ? 0 : -dx, xb = dx >= 0 ? t - dx : t;
        for (let y = 0; y < t - dy; y++) for (let x = xa; x < xb; x++) {
          const df = A[y * t + x] - A[(y + dy) * t + x + dx];
          if (Math.abs(df) < 0.5 * am) continue;
          (df > 0 ? pos : neg)[k].push(y * lw + x); norm++;
        }
      });
      const nw = lw - t + 1, nh = lh - t + 1, S = new Float32Array(nw * nh);
      for (let k = 0; k < DIRS.length; k++) {
        const Dk = D[k], pk = Int32Array.from(pos[k]), nk = Int32Array.from(neg[k]);
        for (let y = 0; y < nh; y++) for (let x = 0; x < nw; x++) {
          const b0 = y * lw + x;
          let acc = 0;
          for (let i = 0; i < pk.length; i++) acc += Dk[b0 + pk[i]];
          for (let i = 0; i < nk.length; i++) acc -= Dk[b0 + nk[i]];
          S[y * nw + x] += acc;
        }
      }
      for (let rep = 0; rep < 2; rep++) {
        let bi = 0;
        for (let i = 1; i < S.length; i++) if (S[i] > S[bi]) bi = i;
        const sc = S[bi] / Math.max(norm, 1);
        if (sc <= 0.05) break;
        const bx = bi % nw, by = (bi / nw) | 0, hh = t >> 1;
        cands.push({ score: sc, x: rx + bx * f, y: ry + by * f, s: t * f });
        for (let y = Math.max(0, by - hh); y <= Math.min(nh - 1, by + hh); y++)
          for (let x = Math.max(0, bx - hh); x <= Math.min(nw - 1, bx + hh); x++) S[y * nw + x] = -1e9;
      }
    }
  }
  cands.sort((p, q) => q.score - p.score);
  const kept = [];
  for (const c of cands) {
    if (kept.every(o => Math.abs(c.x - o.x) > 0.35 * c.s || Math.abs(c.y - o.y) > 0.35 * c.s ||
                        Math.abs(Math.log(c.s / o.s)) > 0.2)) kept.push(c);
    if (kept.length >= top) break;
  }
  return kept;
}

/** Sizes x positions around a box drawn by the user (integer shifts reuse one rendered alpha). */
function manualScan(planes, W, H, prior) {
  const [px, py, ps] = prior;
  const cxp = px + ps / 2, cyp = py + ps / 2;
  const reach = Math.max(8, ps * 0.45);
  const scanned = [];
  for (let f = 0.6; f <= 1.5001; f *= 1.05) {
    const s = ps * f, d = strideFor(s), m = d + 2;
    const aw = Math.ceil(s) + 2 * m, ah = aw;
    // logo at (m, m) of its own patch; the same pairs are tried at every position
    const ps0 = pairSet(renderAlpha(T.v1, m, m, s, 0, 0, aw, ah, 2), aw, ah, d, m + s / 2, m + s / 2);
    const step = Math.max(2, Math.round(s / 14));
    const x0 = Math.round(cxp - s / 2), y0 = Math.round(cyp - s / 2);
    const nst = Math.floor(reach / step);
    for (let j = -nst; j <= nst; j++) for (let i = -nst; i <= nst; i++) {
      const x = x0 + i * step, y = y0 + j * step;
      const ax = x - m, ay = y - m;
      if (ax < 0 || ay < 0 || ax + aw > W || ay + ah > H) continue;
      scanned.push({ score: rank(voteSet(planes, W, ps0, ax, ay)), x, y, s });
    }
  }
  scanned.sort((a, b) => b.score - a.score);
  const kept = [];
  for (const c of scanned) {
    if (kept.every(o => Math.abs(c.x - o.x) > 0.35 * c.s || Math.abs(c.y - o.y) > 0.35 * c.s ||
                        Math.abs(Math.log(c.s / o.s)) > 0.2)) kept.push(c);
    if (kept.length >= 3) break;
  }
  // coarse-to-fine: halve the grid step around each peak
  const out = [];
  for (const c0 of kept) {
    let c = { ...c0 };
    for (let step = Math.max(1, Math.round(c.s / 28)), fs = 0.025; ; step = Math.floor(step / 2), fs /= 2) {
      let best = c;
      for (const f of [1, 1 - fs, 1 + fs]) {
        const s = c.s * f, d = strideFor(s), m = d + 2;
        const aw = Math.ceil(s) + 2 * m;
        const ps1 = pairSet(renderAlpha(T.v1, m, m, s, 0, 0, aw, aw, 2), aw, aw, d, m + s / 2, m + s / 2);
        const x0 = Math.round(c.x + (c.s - s) / 2), y0 = Math.round(c.y + (c.s - s) / 2);
        for (const j of [-1, 0, 1]) for (const i of [-1, 0, 1]) {
          const x = x0 + i * step, y = y0 + j * step;
          if (x - m < 0 || y - m < 0 || x - m + aw > W || y - m + aw > H) continue;
          const sc = rank(voteSet(planes, W, ps1, x - m, y - m));
          if (sc > best.score) best = { score: sc, x, y, s };
        }
      }
      c = best;
      if (step <= 1) break;
    }
    out.push(c);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Sub-pixel refinement (robust L1 outline cost on the luma plane)
// ---------------------------------------------------------------------------
function sobel(u, w, h) {
  const gx = new Float32Array(w * h), gy = new Float32Array(w * h);
  const at = (x, y) => u[clamp(y, 0, h - 1) * w + clamp(x, 0, w - 1)];
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    gx[y * w + x] = (at(x + 1, y - 1) + 2 * at(x + 1, y) + at(x + 1, y + 1)) - (at(x - 1, y - 1) + 2 * at(x - 1, y) + at(x - 1, y + 1));
    gy[y * w + x] = (at(x - 1, y + 1) + 2 * at(x, y + 1) + at(x + 1, y + 1)) - (at(x - 1, y - 1) + 2 * at(x, y - 1) + at(x + 1, y - 1));
  }
  return [gx, gy];
}

function dilateDisc(mask, w, h, r) {
  const out = new Uint8Array(w * h), r2 = r * r + r;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    if (!mask[y * w + x]) continue;
    for (let j = -r; j <= r; j++) for (let i = -r; i <= r; i++) {
      if (i * i + j * j > r2) continue;
      const yy = y + j, xx = x + i;
      if (yy >= 0 && yy < h && xx >= 0 && xx < w) out[yy * w + xx] = 1;
    }
  }
  return out;
}

function dilateBox(mask, w, h, r) {
  const tmp = new Uint8Array(w * h), out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let v = 0;
    for (let i = Math.max(0, x - r); i <= Math.min(w - 1, x + r); i++) if (mask[y * w + i]) { v = 1; break; }
    tmp[y * w + x] = v;
  }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let v = 0;
    for (let j = Math.max(0, y - r); j <= Math.min(h - 1, y + r); j++) if (tmp[j * w + x]) { v = 1; break; }
    out[y * w + x] = v;
  }
  return out;
}

/** Band around the outline of template alpha Tm (any scale). */
function outlineBand(Tm, w, h, s) {
  let m = 0;
  for (let i = 0; i < Tm.length; i++) if (Tm[i] > m) m = Tm[i];
  const u = Tm.map(v => v / Math.max(m, 1e-6));
  const [gx, gy] = sobel(u, w, h);
  const edge = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) edge[i] = Math.hypot(gx[i], gy[i]) > 0.25 ? 1 : 0;
  return dilateDisc(edge, w, h, Math.max(2, pyRound(s * 0.05)));
}

/** L1 norm of "neutral" neighbour differences inside the band (see detect._Outline). */
function neutralCost(P, a, white, band, w, h) {
  const n = w * h, B = new Float32Array(n), sq = new Float32Array(n);
  for (let i = 0; i < n; i++) { const om = 1 - a[i]; sq[i] = Math.sqrt(om); B[i] = (P[i] - a[i] * white) / om; }
  let tot = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    if (x + 1 < w && (band[i] || band[i + 1])) tot += Math.abs((B[i + 1] - B[i]) * sq[i + 1] * sq[i]);
    if (y + 1 < h && (band[i] || band[i + w])) tot += Math.abs((B[i + w] - B[i]) * sq[i + w] * sq[i]);
  }
  return tot;
}

class Outline {
  constructor(luma, W, box, geom) {
    const [bx, by, bw, bh] = box;
    this.box = box; this.w = bw; this.h = bh;
    this.P = new Float32Array(bw * bh);
    for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) this.P[y * bw + x] = luma[(by + y) * W + bx + x];
    this.band = outlineBand(renderAlpha(T.v1, geom[0], geom[1], geom[2], bx, by, bw, bh), bw, bh, geom[2]);
    this.cache = new Map();
  }
  alpha(variant, x, y, s, beta) {
    const key = `${variant}|${x.toFixed(3)}|${y.toFixed(3)}|${s.toFixed(3)}|${beta.toFixed(3)}`;
    let a = this.cache.get(key);
    if (!a) {
      const [bx, by, bw, bh] = this.box;
      a = sharpen(renderAlpha(tplOf(variant), x, y, s, bx, by, bw, bh), bw, bh, beta);
      if (this.cache.size > 3000) this.cache.clear();
      this.cache.set(key, a);
    }
    return a;
  }
  cost(variant, x, y, s, k, beta = 0) {
    const A = this.alpha(variant, x, y, s, beta), a = new Float32Array(A.length);
    for (let i = 0; i < A.length; i++) a[i] = clamp(A[i] * k, 0, A_MAX);
    return neutralCost(this.P, a, 255, this.band, this.w, this.h);
  }
}

function refine(planes, W, H, x, y, s, k, variants = ['v1', 'n48', 'v2'], rounds = 3) {
  const margin = Math.max(4, Math.floor(s * 0.25));
  const box = boxWithMargin(x, y, s, W, H, margin);
  const prob = new Outline(planes[0], W, box, [x, y, s]);
  const base = prob.cost('v1', x, y, s, 0);
  const ratio = priorRatio(x, y, s, W, H);
  const results = [];
  for (const v of variants) {
    if (v === 'n48' && Math.abs(s - 48) > 3) continue;
    const p = { x, y, s, k, b: 0, ex: 0, ey: 0 };
    if (ratio !== null) {
      p.ex = clamp(x - (W - (ratio + 1) * s), -0.5, 0.5);
      p.ey = clamp(y - (H - (ratio + 1) * s), -0.5, 0.5);
    }
    if (v !== 'v1') p.k = k * plateauOf('v1') / Math.max(plateauOf(v), 1e-6);
    const geom = q => (ratio === null ? [q.x, q.y, q.s]
      : [W - (ratio + 1) * q.s + q.ex, H - (ratio + 1) * q.s + q.ey, q.s]);
    const C = (kw = {}) => { const q = { ...p, ...kw }; const [gx, gy, gs] = geom(q); return prob.cost(v, gx, gy, gs, q.k, q.b); };
    let best = [C(), 0];
    for (const bb of [-0.3, 0.3, 0.6, 1.0]) { const c = C({ b: bb }); if (c < best[0]) best = [c, bb]; }
    p.b = best[1];
    for (let r = 0; r < rounds; r++) {
      const f = 0.5 ** r, ss = Math.max(1, p.s * 0.03) * f;
      p.s = golden(q => C({ s: q }), p.s - ss, p.s + ss, 9)[0];
      if (ratio === null) {
        const span = 1.2 * f;
        p.x = golden(q => C({ x: q }), p.x - span, p.x + span, 9)[0];
        p.y = golden(q => C({ y: q }), p.y - span, p.y + span, 9)[0];
      } else {
        p.ex = golden(q => C({ ex: q }), -0.5, 0.5, 8)[0];
        p.ey = golden(q => C({ ey: q }), -0.5, 0.5, 8)[0];
      }
      p.b = golden(q => C({ b: q }), Math.max(-0.5, p.b - 0.5 * f), Math.min(1.6, p.b + 0.5 * f), 9)[0];
      p.k = golden(q => C({ k: q }), Math.max(0.1, p.k * (1 - 0.15 * f)), Math.min(1.9, p.k * (1 + 0.15 * f)), 9)[0];
    }
    let [gx, gy, gs] = geom(p), cost = C();
    // files straight from Gemini carry the logo at exactly 48 or 96 px on whole pixels
    if (ratio !== null) {
      for (const s0 of [48, 96]) {
        if (Math.abs(gs - s0) > 0.03 * s0) continue;
        const nx = pyRound(W - (ratio + 1) * s0), ny = pyRound(H - (ratio + 1) * s0);
        if (Math.abs(nx - gx) >= 1 || Math.abs(ny - gy) >= 1) continue;
        for (const bb of [0, p.b]) {
          const cn = prob.cost(v, nx, ny, s0, p.k, bb);
          if (cn <= cost * 1.02) { gx = nx; gy = ny; gs = s0; cost = cn; p.b = bb; break; }
        }
      }
    }
    results.push({ ratio: cost / Math.max(base, 1e-6), variant: v, x: gx, y: gy, s: gs, k: p.k, beta: p.b });
  }
  results.sort((a, b) => a.ratio * (a.variant === 'v1' ? 1 : 1.08) - b.ratio * (b.variant === 'v1' ? 1 : 1.08));
  return results[0];
}

// ---------------------------------------------------------------------------
// Public: detect
// ---------------------------------------------------------------------------
// strength (v1 template) Gemini uses on each layout: 48 px @ 96 px margin -> ~0.58, 48@32 and 96@64 -> ~1.0
const nominalK = r => (r === null ? null : (r === PRIOR_RATIOS[0] ? 1.0 : 0.58));

/** v3.0 - grainy photographic backgrounds (sand, grass, wood): single pixel pairs
 *  disagree, but the strength measured in 8 sectors all round the outline is the
 *  same.  Returns [sectors compatible with k0, usable sectors, z-score]. */
function evidence(planes, W, H, x, y, s, k0) {
  const d = strideFor(s), NS = 8;
  const [bx, by, bw, bh] = boxWithMargin(x, y, s, W, H, d + 2);
  const A = renderAlpha(T.v1, x, y, s, bx, by, bw, bh);
  const cx = x + s / 2 - bx, cy = y + s / 2 - by;
  let am = 0;
  for (let i = 0; i < A.length; i++) if (A[i] > am) am = A[i];
  if (am <= 1e-6) return [0, 0, 0];
  const ks = Array.from({ length: NS }, () => []), ws = Array.from({ length: NS }, () => []);
  const denMin = 0.5 * am * 10;
  for (const [uy, ux] of DIRS) {
    const dy = uy * d, dx = ux * d, xa = dx >= 0 ? 0 : -dx, xb = dx >= 0 ? bw - dx : bw;
    for (let yy = 0; yy < bh - dy; yy++) for (let xx = xa; xx < xb; xx++) {
      const ap = A[yy * bw + xx], aq = A[(yy + dy) * bw + xx + dx];
      if (Math.abs(ap - aq) < 0.5 * am) continue;
      const sec = Math.floor((Math.atan2(yy - cy, xx - cx) + Math.PI) / (2 * Math.PI) * NS) % NS;
      const gp = (by + yy) * W + bx + xx, gq = gp + dy * W + dx;
      for (const P of planes) {
        const den = ap * (255 - P[gq]) - aq * (255 - P[gp]);
        if (Math.abs(den) > denMin) { ks[sec].push((P[gp] - P[gq]) / den); ws[sec].push(Math.abs(den)); }
      }
    }
  }
  const val = [];
  for (let i = 0; i < NS; i++) if (ks[i].length >= 12) val.push(wmedian(ks[i], ws[i], ks[i].length));
  if (val.length < 5) return [0, val.length, 0];
  const med = percentile(val, 50), spread = 1.4826 * percentile(val.map(v => Math.abs(v - med)), 50);
  const z = med / (spread / Math.sqrt(val.length) + 0.02);
  return [val.filter(v => v >= 0.4 * k0 && v <= 1.9 * k0).length, val.length, z];
}

function evidenceOk(planes, W, H, v, x, y, s) {
  const k0 = nominalK(priorRatio(x, y, s, W, H));
  if (k0 === null || v.inlier < 0.10 || v.k < 0.7 * k0 || v.k > 1.3 * k0) return false;
  const [ok, n, z] = evidence(planes, W, H, x, y, s, k0);
  if (n < 7 || ok < 7 || z < 7) return false;
  const [rin, rout, np] = profileAt(planes, W, H, x, y, s, v.k);
  return !(np >= 16 && (rin > 0.25 || Math.abs(rout) > 0.25));
}

function accept(planes, W, H, v, x, y, s, on, manual) {
  if (!voteOk(v) || v.n < 60) return false;
  if (manual) {
    if (v.inlier >= T_MANUAL) return true;
    // grainy background inside the user's box: same sector test, against the two strengths Gemini uses
    if (v.inlier < 0.12) return false;
    const k0 = Math.abs(v.k - 0.58) <= Math.abs(v.k - 1.0) ? 0.58 : 1.0;
    if (v.k < 0.7 * k0 || v.k > 1.3 * k0) return false;
    const [ok, n, z] = evidence(planes, W, H, x, y, s, k0);
    if (n < 7 || ok < 7 || z < 8) return false;
    const [rin, rout, np] = profileAt(planes, W, H, x, y, s, v.k);
    return !(np >= 16 && (rin > 0.25 || Math.abs(rout) > 0.25));
  }
  if (v.k < K_AUTO_MIN || v.k > K_AUTO_MAX) return false;
  if (on) { if (v.inlier < T_ON_PRIOR) return evidenceOk(planes, W, H, v, x, y, s); }
  else if (!(s >= 24 && v.inlier >= T_OFF_PRIOR && v.quad >= 0.35)) return false;
  const [rin, rout, n] = profileAt(planes, W, H, x, y, s, v.k);
  return !(n >= 16 && (rin > PROFILE_MAX || Math.abs(rout) > PROFILE_MAX));
}

/** Find the sparkle on ImageData-like {data,width,height}. prior = [x, y, size] (manual box). */
export function detect(img, prior = null) {
  const W = img.width, H = img.height;
  const planes = planesOf(img);
  const manual = !!prior;
  let cands = [];
  if (manual) {
    cands = manualScan(planes, W, H, prior);
    cands.push({ x: prior[0], y: prior[1], s: prior[2] });
  } else {
    const roi = cornerRoi(W, H);
    cands = priorScan(planes, W, H, roi, 14, Math.max(60, 0.14 * Math.min(W, H)));
    // files straight from Gemini: always try the exact native placements
    for (const [sz, mg] of [[48, 96], [48, 32], [96, 64]]) {
      if (W - mg - sz >= roi[0] && H - mg - sz >= roi[1]) cands.push({ x: W - mg - sz, y: H - mg - sz, s: sz });
    }
    // anywhere else in the corner (only logos of 24 px and more are accepted off the known layouts)
    cands.push(...denseScan(planes, W, H, roi, 24, Math.max(60, 0.14 * Math.min(W, H))));
  }
  let best = null;
  for (const c of cands) {
    if (c.s < 8 || c.x < 0 || c.y < 0 || c.x + c.s > W || c.y + c.s > H) continue;
    const r = localVoteSearch(planes, W, H, c.x, c.y, c.s, priorRatio(c.x, c.y, c.s, W, H));
    if (!r) continue;
    const on = isOnPrior(r.x, r.y, r.s, W, H);
    const good = accept(planes, W, H, r.v, r.x, r.y, r.s, on, manual);
    const rk = (good ? 2 : 0) + (on && voteOk(r.v) ? 1 : 0) + rank(r.v);
    if (!best || rk > best.rk) best = { rk, v: r.v, x: r.x, y: r.y, s: r.s, on, good };
  }
  const miss = note => (manual
    ? { x: prior[0], y: prior[1], size: prior[2], strength: 0, found: false, note: 'manual-inpaint' }
    : { x: best ? best.x : 0, y: best ? best.y : 0, size: best ? best.s : 0, strength: best ? Math.max(best.v.k, 0) : 0,
        score: best ? best.v.inlier : 0, found: false, note });
  if (!best) return miss('not-found');
  if (!best.good) return miss(voteOk(best.v) ? 'weak' : 'no-step');
  const f = refine(planes, W, H, best.x, best.y, best.s, best.v.k);
  const v2 = voteAt(planes, W, H, f.x, f.y, f.s, f.variant, null, f.beta);
  let k = f.k;
  if (v2.n >= 60 && v2.inlier >= 0.7 * best.v.inlier) k = v2.k;
  // on a known layout the strength is known too: keep the estimate close to it
  const k0 = best.on ? nominalK(priorRatio(f.x, f.y, f.s, W, H)) : null;
  if (k0 !== null && f.variant === 'v1' && f.s >= 30) {
    const tol = Math.min(Math.abs(f.s - 48), Math.abs(f.s - 96)) <= 1.5 ? 0.08 : 0.15;
    k = clamp(k, k0 * (1 - tol), k0 * (1 + tol));
  }
  const plateau = k * plateauOf(f.variant);
  const [lo, hi] = manual ? PLATEAU_MANUAL : [PLATEAU_MIN, PLATEAU_MAX];
  if (!(plateau >= lo && plateau <= hi)) return miss('implausible-strength');
  return { x: f.x, y: f.y, size: f.s, strength: k, variant: f.variant, sharp: f.beta, chromaBlur: null,
           score: Math.max(best.v.inlier, v2.inlier), gain: 1 - f.ratio, found: true, note: best.on ? 'ok' : 'ok-offgrid' };
}

// ---------------------------------------------------------------------------
// Removal
// ---------------------------------------------------------------------------
const ycc = (r, g, b) => [0.299 * r + 0.587 * g + 0.114 * b,
                          128 - 0.168736 * r - 0.331264 * g + 0.5 * b,
                          128 + 0.5 * r - 0.418688 * g - 0.081312 * b];

function fitChromaSigma(R, G, B, a, band, w, h) {
  const n = w * h, cb = new Float32Array(n), cr = new Float32Array(n);
  for (let i = 0; i < n; i++) { const c = ycc(R[i], G[i], B[i]); cb[i] = c[1]; cr[i] = c[2]; }
  let best = [null, 0];
  for (const sg of [0, 0.5, 0.8, 1.1, 1.5, 2.0]) {
    const ac = (sg ? gaussBlur(a, w, h, sg) : a).map(v => clamp(v, 0, 0.95));
    const c = neutralCost(cb, ac, 128, band, w, h) + neutralCost(cr, ac, 128, band, w, h);
    if (best[0] === null || c < best[0] * 0.985) best = [c, sg];
  }
  return best[1];
}

/** Pairs (p just inside the flat part of the logo, q just outside) in calm surroundings. */
function boundarySamples(res, obs, Tm, pl, w, h, steps = [3, 5], cap = 700) {
  const C = [], Rr = [], P = [];
  for (const d of steps) for (const [uy, ux] of DIRS) for (const sg of [1, -1]) {
    const dy = uy * d * sg, dx = ux * d * sg;
    const y0 = Math.max(0, dy, -dy, -2 * dy), y1 = h - Math.max(0, dy, 2 * dy, -dy);
    const x0 = Math.max(0, dx, -dx, -2 * dx), x1 = w - Math.max(0, dx, 2 * dx, -dx);
    if (y1 - y0 < 2 || x1 - x0 < 2) continue;
    const st = dy * w + dx;
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
      const i = y * w + x;
      if (Tm[i - st] < 0.85 * pl || Tm[i] < 0.85 * pl || Tm[i + st] > 0.01 || Tm[i + 2 * st] > 0.01) continue;
      let ok = true;
      const col = [0, 0, 0], r = [0, 0, 0];
      for (let c = 0; c < 3 && ok; c++) {
        const p2 = res[c][i - st], pp = res[c][i], q = obs[c][i + st], q2 = obs[c][i + 2 * st];
        col[c] = pp; r[c] = pp - q;
        if (Math.abs(r[c]) > R_MAX || Math.abs(q - q2) > 12 || Math.abs(pp - p2) > 20) ok = false;
      }
      if (ok) { C.push(col); Rr.push(r); P.push([x, y]); }
    }
  }
  if (!C.length) return null;
  if (C.length > cap) {
    const pick = [];
    for (let j = 0; j < cap; j++) pick.push(Math.floor(j * (C.length - 1) / (cap - 1)));
    return { C: pick.map(j => C[j]), R: pick.map(j => Rr[j]), P: pick.map(j => P[j]) };
  }
  return { C, R: Rr, P };
}

/** Colour difference of the restored patch to its surroundings (per channel arrays), or null. */
function colourOffsets(res, Tm, pl, size, smp, w, h, sigmaC = 30, dead = 1.0) {
  if (!smp || smp.C.length < 24) return null;
  const M = smp.C.length, n = w * h;
  const off = [new Float32Array(n), new Float32Array(n), new Float32Array(n)];
  const sigmaS = Math.max(5, 0.2 * size);
  const ic = 0.5 / (sigmaC * sigmaC), is = 0.5 / (sigmaS * sigmaS);
  const cR = Float32Array.from(smp.C, c => c[0]), cG = Float32Array.from(smp.C, c => c[1]), cB = Float32Array.from(smp.C, c => c[2]);
  const pX = Float32Array.from(smp.P, p => p[0]), pY = Float32Array.from(smp.P, p => p[1]);
  const rr = [0, 1, 2].map(c => Float32Array.from(smp.R, r => r[c]));
  const wt = new Float32Array(M);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    if (Tm[i] <= 0.02 * pl) continue;
    const r0 = res[0][i], g0 = res[1][i], b0 = res[2][i];
    let wsum = 0, w2 = 0;
    for (let j = 0; j < M; j++) {
      const dr = r0 - cR[j], dg = g0 - cG[j], db = b0 - cB[j], dx = x - pX[j], dy = y - pY[j];
      const v = Math.exp(-(dr * dr + dg * dg + db * db) * ic - (dx * dx + dy * dy) * is);
      wt[j] = v; wsum += v; w2 += v * v;
    }
    const conf = clamp(wsum / 3, 0, 1);
    if (conf <= 0) continue;
    wsum = Math.max(wsum, 1e-6);
    const nEff = wsum * wsum / Math.max(w2, 1e-9);
    const al = clamp(Tm[i] / pl, 0, 1);
    for (let c = 0; c < 3; c++) {
      const r = rr[c];
      let dl = 0;
      for (let j = 0; j < M; j++) dl += wt[j] * r[j];
      dl /= wsum;
      for (let it = 0; it < 8; it++) {                          // weighted mean -> weighted median
        let num = 0, den = 0;
        for (let j = 0; j < M; j++) { const q = wt[j] / Math.max(Math.abs(r[j] - dl), 1); num += q * r[j]; den += q; }
        dl = num / Math.max(den, 1e-6);
      }
      let spread = 0;
      for (let j = 0; j < M; j++) spread += wt[j] * Math.abs(r[j] - dl);
      const se = 1.25 * (spread / wsum) / Math.sqrt(Math.max(nEff, 1)) + 0.25;
      const sure = clamp((Math.abs(dl) / se - 2) / 2, 0, 1);    // only what is statistically certain
      off[c][i] = Math.sign(dl) * clamp(Math.abs(dl) - dead, 0, R_MAX) * sure * conf * al;
    }
  }
  return off;
}

/** Chamfer distance (3x3, like cv2.DIST_L2 mask 3) to the nearest pixel where mask != 0. */
function distanceTo(mask, w, h) {
  const a = 0.955, b = 1.3693, INF = 1e6, d = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) d[i] = mask[i] ? 0 : INF;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x; let v = d[i];
    if (x > 0) v = Math.min(v, d[i - 1] + a);
    if (y > 0) { v = Math.min(v, d[i - w] + a); if (x > 0) v = Math.min(v, d[i - w - 1] + b); if (x < w - 1) v = Math.min(v, d[i - w + 1] + b); }
    d[i] = v;
  }
  for (let y = h - 1; y >= 0; y--) for (let x = w - 1; x >= 0; x--) {
    const i = y * w + x; let v = d[i];
    if (x < w - 1) v = Math.min(v, d[i + 1] + a);
    if (y < h - 1) { v = Math.min(v, d[i + w] + a); if (x < w - 1) v = Math.min(v, d[i + w + 1] + b); if (x > 0) v = Math.min(v, d[i + w - 1] + b); }
    d[i] = v;
  }
  return d;
}

function bilinear(P, w, h, x, y) {
  x = clamp(x, 0, w - 1); y = clamp(y, 0, h - 1);
  const x0 = Math.floor(x), y0 = Math.floor(y), x1 = Math.min(w - 1, x0 + 1), y1 = Math.min(h - 1, y0 + 1);
  const fx = x - x0, fy = y - y0;
  return (P[y0 * w + x0] * (1 - fx) + P[y0 * w + x1] * fx) * (1 - fy) + (P[y1 * w + x0] * (1 - fx) + P[y1 * w + x1] * fx) * fy;
}

/** Guarded clean-up of the outline seam (port of restore.clean_outline). Modifies res in place. */
function cleanOutline(res, Tm, pl, a, w, h, strength = 1) {
  const n = w * h;
  const u = Tm.map(v => clamp(v / pl, 0, 1.5));
  let [gx, gy] = sobel(u, w, h);
  let seam = new Uint8Array(n);
  for (let i = 0; i < n; i++) seam[i] = (Math.hypot(gx[i], gy[i]) / 8 > 0.05 || (u[i] > 0.03 && u[i] < 0.9)) ? 1 : 0;
  seam = dilateBox(seam, w, h, 1);
  const outside = new Uint8Array(n), inside = new Uint8Array(n);
  let nSeam = 0, anyIn = false;
  for (let i = 0; i < n; i++) {
    if (seam[i]) { nSeam++; continue; }
    if (u[i] < 0.03) outside[i] = 1; else { inside[i] = 1; anyIn = true; }
  }
  if (nSeam < 8 || strength <= 0) return 0;
  [gx, gy] = sobel(gaussBlur(u, w, h, 1.5), w, h);
  const dOut = distanceTo(outside, w, h), dIn = anyIn ? distanceTo(inside, w, h) : null;
  const sample = (i, span) => {                    // -> [vo(3), vi(3), wo]
    const x = i % w, y = (i / w) | 0;
    const nr = Math.max(Math.hypot(gx[i], gy[i]), 1e-6), nx = -gx[i] / nr, ny = -gy[i] / nr;
    const to = span === null ? clamp(dOut[i] + 0.5, 1, 5) : span;
    const ti = span === null ? clamp((dIn ? dIn[i] : 4.5) + 0.5, 1, 5) : span;
    const vo = [0, 1, 2].map(c => bilinear(res[c], w, h, x + nx * to, y + ny * to));
    const vi = [0, 1, 2].map(c => bilinear(res[c], w, h, x - nx * ti, y - ny * ti));
    return [vo, vi, ti / (to + ti)];
  };
  const near = dilateBox(seam, w, h, 5), far = dilateBox(seam, w, h, 2);
  const dSeam = [];
  for (let i = 0; i < n; i++) if (seam[i]) dSeam.push(dOut[i]);
  const span = clamp(percentile(dSeam, 50) + 0.5, 1.5, 4);
  // texture level of the artwork next to the seam
  const ctrl = [], ctrlSides = [];
  for (let i = 0; i < n; i++) {
    if (!near[i] || far[i]) continue;
    const [co, ci] = sample(i, span);
    let dv = 0, sd = 0;
    for (let c = 0; c < 3; c++) { dv = Math.max(dv, Math.abs(res[c][i] - 0.5 * (co[c] + ci[c]))); sd = Math.max(sd, Math.abs(co[c] - ci[c])); }
    ctrl.push(dv); ctrlSides.push(sd);
  }
  const tex = ctrl.length > 20 ? percentile(ctrl, 50) : 2;
  // seam pixels
  const idx = [], pred = [], dev = [], calm = [];
  for (let i = 0; i < n; i++) {
    if (!seam[i]) continue;
    const [vo, vi, wo] = sample(i, null);
    const p = [0, 1, 2].map(c => vo[c] * wo + vi[c] * (1 - wo));
    let dv = 0, sd = 0;
    for (let c = 0; c < 3; c++) { dv = Math.max(dv, Math.abs(res[c][i] - p[c])); sd = Math.max(sd, Math.abs(vo[c] - vi[c])); }
    idx.push(i); pred.push(p); dev.push(dv); calm.push(clamp((20 + 2 * tex - sd) / 10, 0, 1));
  }
  const sm = dev.filter((_, j) => calm[j] > 0.5);
  if (sm.length < 12) return 0;
  const ctrlCalm = ctrl.filter((_, j) => ctrlSides[j] <= 15 + 2 * tex);
  const excess = ctrlCalm.length > 20 ? (percentile(sm, 90) + 1) / (percentile(ctrlCalm, 90) + 1) : 2;
  const gate = clamp((excess - 1.25) / 0.5, 0, 1);
  if (gate <= 0) return 0;
  const str = clamp(strength, 0, 1) * gate, tau = 1.5 * tex + 1.5;
  let changed = 0;
  const upd = [];
  for (let j = 0; j < idx.length; j++) {
    const i = idx[j], x = i % w, y = (i / w) | 0, p = pred[j];
    const outOfLine = clamp((dev[j] - tau) / Math.max(tau, 1), 0, 1);
    if (outOfLine <= 0 || calm[j] <= 0) continue;
    // a thin stroke of the artwork also sticks out, but its colour continues in the clean picture nearby
    let nearest = 1e6;
    for (let oy = -4; oy <= 4; oy++) for (let ox = -4; ox <= 4; ox++) {
      if (oy * oy + ox * ox > 18 || (!oy && !ox)) continue;
      const yy = y + oy, xx = x + ox;
      if (yy < 0 || yy >= h || xx < 0 || xx >= w || seam[yy * w + xx]) continue;
      const k2 = yy * w + xx;
      const dc = Math.max(Math.abs(res[0][i] - res[0][k2]), Math.abs(res[1][i] - res[1][k2]), Math.abs(res[2][i] - res[2][k2]));
      if (dc < nearest) nearest = dc;
    }
    const orphan = clamp((nearest - 0.4 * dev[j]) / (0.2 * dev[j] + 1), 0, 1);
    // a remnant is an alpha error of at most ~0.15 of the headroom to white
    let over = 0;
    for (let c = 0; c < 3; c++) {
      const bound = 0.15 * (255 - p[c]) / (1 - Math.min(a[i], 0.6)) + 6;
      over = Math.max(over, Math.abs(res[c][i] - p[c]) / bound);
    }
    const wgt = calm[j] * outOfLine * orphan * clamp((1.5 - over) / 0.5, 0, 1) * str;
    if (wgt > 0.05) changed++;
    if (wgt > 0) upd.push([i, wgt, p]);
  }
  for (const [i, wgt, p] of upd) for (let c = 0; c < 3; c++) res[c][i] = res[c][i] * (1 - wgt) + p[c] * wgt;
  return changed / nSeam;
}

const cloneImg = img => ({ width: img.width, height: img.height, data: new Uint8ClampedArray(img.data) });

/** Returns a copy of img with the sparkle of ``geo`` removed (exact reverse blend + refinements). */
export function remove(img, geo, { seam = 1, match = true } = {}) {
  const W = img.width, H = img.height, out = cloneImg(img), d = out.data;
  const s = geo.size, margin = Math.max(10, Math.floor(s * 0.3));
  const [bx, by, bw, bh] = boxWithMargin(geo.x, geo.y, s, W, H, margin);
  const n = bw * bh;
  const obs = [new Float32Array(n), new Float32Array(n), new Float32Array(n)];
  for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) {
    const p = 4 * ((by + y) * W + bx + x), i = y * bw + x;
    obs[0][i] = d[p]; obs[1][i] = d[p + 1]; obs[2][i] = d[p + 2];
  }
  const Tm = renderAlpha(tplOf(geo.variant), geo.x, geo.y, s, bx, by, bw, bh);
  const pl = plateauOf(geo.variant);
  const a = sharpen(Tm, bw, bh, geo.sharp || 0).map(v => clamp(Math.max(v, 0) * geo.strength, 0, A_MAX));
  let res = obs.map(P => P.map((v, i) => (v - a[i] * 255) / (1 - a[i])));
  // colour planes with the codec's chroma blur
  const sg = fitChromaSigma(obs[0], obs[1], obs[2], a, outlineBand(Tm, bw, bh, s), bw, bh);
  geo.chromaBlur = sg;
  if (sg > 0) {
    const ab = gaussBlur(a, bw, bh, sg).map(v => clamp(v, 0, A_MAX));
    for (let i = 0; i < n; i++) {
      const y1 = ycc(res[0][i], res[1][i], res[2][i])[0];
      const q = 1 - ab[i], w255 = ab[i] * 255;
      const c2 = ycc((obs[0][i] - w255) / q, (obs[1][i] - w255) / q, (obs[2][i] - w255) / q);
      res[0][i] = y1 + 1.402 * (c2[2] - 128);
      res[1][i] = y1 - 0.344136 * (c2[1] - 128) - 0.714136 * (c2[2] - 128);
      res[2][i] = y1 + 1.772 * (c2[1] - 128);
    }
  }
  // align the interior with the surroundings
  if (match) {
    const off = colourOffsets(res, Tm, pl, s, boundarySamples(res, obs, Tm, pl, bw, bh), bw, bh);
    if (off) for (let c = 0; c < 3; c++) for (let i = 0; i < n; i++) res[c][i] -= off[c][i];
  }
  // outline seam
  if (seam > 0) cleanOutline(res, Tm, pl, a, bw, bh, seam);
  for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) {
    const p = 4 * ((by + y) * W + bx + x), i = y * bw + x;
    d[p] = Math.round(clamp(res[0][i], 0, 255)); d[p + 1] = Math.round(clamp(res[1][i], 0, 255)); d[p + 2] = Math.round(clamp(res[2][i], 0, 255));
  }
  return out;
}

/** Fill a region from its surroundings (invents content; for marks that are not a Gemini sparkle). */
function fillMask(img, maskFn, x0, y0, x1, y1) {
  const W = img.width, H = img.height, out = cloneImg(img), d = out.data;
  const pad = Math.max(16, Math.floor(Math.max(x1 - x0, y1 - y0) / 2));
  const bx = Math.max(0, Math.floor(x0 - pad)), by = Math.max(0, Math.floor(y0 - pad));
  const bw = Math.min(W, Math.ceil(x1 + pad)) - bx, bh = Math.min(H, Math.ceil(y1 + pad)) - by;
  const mask = new Uint8Array(bw * bh);
  for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) mask[y * bw + x] = maskFn(bx + x, by + y) ? 1 : 0;
  for (let c = 0; c < 3; c++) {
    const P = new Float32Array(bw * bh);
    for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) P[y * bw + x] = d[4 * ((by + y) * W + bx + x) + c];
    const F = harmonicFill(P, mask, bw, bh, 400);
    for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) {
      if (mask[y * bw + x]) d[4 * ((by + y) * W + bx + x) + c] = Math.round(clamp(F[y * bw + x], 0, 255));
    }
  }
  return out;
}

export function inpaintBox(img, box) {
  const [x, y, s] = box, r = s * 0.18;
  const inside = (px, py) => {
    if (px < x || py < y || px >= x + s || py >= y + s) return false;
    const cx = clamp(px, x + r, x + s - r), cy = clamp(py, y + r, y + s - r);
    return (px - cx) ** 2 + (py - cy) ** 2 <= r * r;
  };
  return fillMask(img, inside, x, y, x + s, y + s);
}

export function inpaintStar(img, geo) {
  const W = img.width, H = img.height;
  const [bx, by, bw, bh] = boxWithMargin(geo.x, geo.y, geo.size, W, H, Math.max(4, Math.floor(geo.size * 0.25)));
  const A = renderAlpha(T.v1, geo.x, geo.y, geo.size, bx, by, bw, bh);
  const m = dilateBox(A.map(v => (v > 0.02 ? 1 : 0)), bw, bh, Math.max(1, Math.round(geo.size * 0.06)));
  const inside = (px, py) => px >= bx && py >= by && px < bx + bw && py < by + bh && m[(py - by) * bw + px - bx];
  return fillMask(img, inside, geo.x, geo.y, geo.x + geo.size, geo.y + geo.size);
}

/** Remove the logo with a known geometry. opts: {mode, box:[x,y,s], method}. */
export function applyGeometry(src, geo, opts = {}) {
  const manual = opts.mode === 'manual' && opts.box;
  if (geo && geo.found && opts.method !== 'inpaint') return remove(src, geo);
  if (manual) return inpaintBox(src, opts.box);
  if (geo && geo.found) return inpaintStar(src, geo);
  return cloneImg(src);
}

/** Full pipeline on ImageData. opts: {mode:'auto'|'manual', box:[x,y,s], method}. Returns {img, geo}. */
export function processImage(src, opts = {}) {
  const manual = opts.mode === 'manual' && opts.box;
  if (manual && opts.method === 'inpaint') {
    const geo = { x: opts.box[0], y: opts.box[1], size: opts.box[2], found: false, note: 'manual-inpaint' };
    return { img: applyGeometry(src, geo, opts), geo };
  }
  const geo = detect(src, manual ? opts.box : null);
  return { img: applyGeometry(src, geo, opts), geo };
}

export { grayFromRGBA };
