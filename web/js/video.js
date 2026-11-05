// Video pipeline for the browser (ffmpeg.wasm, single thread).
// Only the small box around the logo is decoded/processed in JavaScript;
// ffmpeg overlays the cleaned box back and re-encodes to MP4 (H.264).
import { FFmpeg } from '../vendor/ffmpeg/index.js';
import * as E from './engine.js';

let ff = null;
let logBuf = [];

export async function getFFmpeg(onStatus) {
  if (ff) return ff;
  onStatus && onStatus('Đang tải bộ xử lý video (~31 MB, chỉ lần đầu)…');
  const inst = new FFmpeg();
  inst.on('log', ({ message }) => { logBuf.push(message); if (logBuf.length > 4000) logBuf.splice(0, 2000); });
  const base = new URL('../vendor/core/', import.meta.url).href;
  await inst.load({ coreURL: base + 'ffmpeg-core.js', wasmURL: base + 'ffmpeg-core.wasm' });
  ff = inst;
  return ff;
}

/** Stop whatever ffmpeg is doing right now (the worker is terminated). */
export function abort() {
  if (ff) { try { ff.terminate(); } catch (_) { /* ignore */ } }
  ff = null;
}

async function run(args) {
  logBuf = [];
  const rc = await ff.exec(args);
  return { rc, log: logBuf.join('\n') };
}

export async function probe(name) {
  const { log } = await run(['-hide_banner', '-i', name]);
  const info = { width: 0, height: 0, fps: 25, fpsStr: '25', duration: 0, range: 'tv', hasAudio: false, acodec: '' };
  let m = log.match(/Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)/);
  if (m) info.duration = +m[1] * 3600 + +m[2] * 60 + +m[3];
  const vline = log.split('\n').find(l => l.includes('Video:'));
  if (!vline) throw new Error('Tệp không có luồng video');
  m = vline.match(/,\s*(\d{2,5})x(\d{2,5})[\s,\[]/);
  if (m) { info.width = +m[1]; info.height = +m[2]; }
  m = vline.match(/([\d.]+)\s*fps/) || vline.match(/([\d.]+)\s*tbr/);
  if (m) { info.fps = +m[1]; info.fpsStr = String(+(+m[1]).toFixed(3)); }
  if (Math.abs(info.fps - 23.976) < 0.01) info.fpsStr = '24000/1001';
  if (Math.abs(info.fps - 29.97) < 0.01) info.fpsStr = '30000/1001';
  if (Math.abs(info.fps - 59.94) < 0.01) info.fpsStr = '60000/1001';
  if (/\bpc\b|yuvj/.test(vline)) info.range = 'pc';
  const aline = log.split('\n').find(l => l.includes('Audio:'));
  if (aline) { info.hasAudio = true; const am = aline.match(/Audio:\s*([\w-]+)/); info.acodec = am ? am[1] : ''; }
  m = log.match(/rotate\s*:\s*(-?\d+)/) || log.match(/rotation of (-?[\d.]+) degrees/);
  if (m && Math.abs(+m[1]) % 180 === 90) { const t = info.width; info.width = info.height; info.height = t; }
  if (!info.width) throw new Error('Không đọc được kích thước video');
  info.frames = Math.max(1, Math.round(info.duration * info.fps));
  return info;
}

const even = v => v - (v % 2);

/** Decode a rectangle (even-aligned) of evenly spaced frames as Y/U/V planes. */
async function grabRegion(name, info, rect, { every = 1, max = 0 } = {}) {
  const [x, y, w, h] = rect;
  const vf = [];
  if (every > 1) vf.push(`select='not(mod(n\\,${every}))'`);
  vf.push(`crop=${w}:${h}:${x}:${y}`);
  const args = ['-v', 'error', '-i', name, '-an', '-vf', vf.join(','), '-fps_mode', 'passthrough'];
  if (max) args.push('-frames:v', String(max));
  args.push('-f', 'rawvideo', '-pix_fmt', 'yuv420p', 'region.yuv');
  await run(args);
  const data = await ff.readFile('region.yuv');
  await ff.deleteFile('region.yuv');
  const ys = w * h, cs = (w / 2) * (h / 2), fb = ys + 2 * cs;
  const frames = [];
  for (let o = 0; o + fb <= data.length; o += fb) {
    frames.push({ y: data.subarray(o, o + ys), u: data.subarray(o + ys, o + ys + cs), v: data.subarray(o + ys + cs, o + fb) });
  }
  return frames;
}

export async function grabFrameRGBA(name, info, t) {
  const args = ['-v', 'error', '-ss', t.toFixed(3), '-i', name, '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'rgba', 'frame.rgba'];
  await run(args);
  const data = await ff.readFile('frame.rgba');
  await ff.deleteFile('frame.rgba');
  return new ImageData(new Uint8ClampedArray(data.buffer, data.byteOffset, info.width * info.height * 4), info.width, info.height);
}

function toF32(u8) { const f = new Float32Array(u8.length); for (let i = 0; i < u8.length; i++) f[i] = u8[i]; return f; }

function wmedianPerPixel(stack, wts, n) {
  const out = new Float32Array(n);
  const k = stack.length;
  const vals = new Array(k);
  for (let p = 0; p < n; p++) {
    for (let i = 0; i < k; i++) vals[i] = [stack[i][p], wts[i][p]];
    vals.sort((a, b) => a[0] - b[0]);
    let tot = 0;
    for (const v of vals) tot += v[1];
    let acc = 0, m = vals[k - 1][0];
    for (const v of vals) { acc += v[1]; if (acc >= tot / 2) { m = v[0]; break; } }
    out[p] = m;
  }
  return out;
}

/** Detect the logo over sampled frames; returns geometry (+ calibrated alpha). */
export async function detect(name, info, { manualBox = null, onStatus } = {}) {
  const W = info.width, H = info.height, wy = info.range === 'pc' ? 255 : 235;
  let roi;
  if (manualBox) {
    const [px, py, ps] = manualBox, m = Math.max(8, Math.floor(ps * 0.4));
    const x0 = even(Math.max(0, Math.floor(px - m))), y0 = even(Math.max(0, Math.floor(py - m)));
    roi = [x0, y0, even(Math.min(W - x0, Math.ceil(ps + 2 * m) + 2)), even(Math.min(H - y0, Math.ceil(ps + 2 * m) + 2))];
  } else {
    const [rx, ry, rw, rh] = E.cornerRoi(W, H);
    const x0 = even(rx), y0 = even(ry);
    roi = [x0, y0, even(W - x0), even(H - y0)];
  }
  onStatus && onStatus('Đang lấy mẫu khung hình…');
  const every = Math.max(1, Math.floor(info.frames / 40));
  const fr = await grabRegion(name, info, roi, { every, max: 40 });
  if (!fr.length) throw new Error('Không đọc được khung hình');
  const [rx, ry, rw, rh] = roi;
  const Ys = fr.map(f => toF32(f.y));
  // temporal median: the static logo survives, moving content is suppressed
  const med = new Float32Array(rw * rh);
  const tmp = new Float32Array(Ys.length);
  for (let p = 0; p < rw * rh; p++) { for (let i = 0; i < Ys.length; i++) tmp[i] = Ys[i][p]; tmp.sort(); med[p] = tmp[tmp.length >> 1]; }
  onStatus && onStatus('Đang dò vị trí logo…');
  const prior = manualBox ? [manualBox[0] - rx, manualBox[1] - ry, manualBox[2]] : null;
  const [score, s, x, y] = E.coarseDetect(med, rw, rh, prior);
  if (!s || (!prior && score < 0.25)) return { found: false, note: 'not-found', x: 0, y: 0, size: 0 };
  // rank frames: dark and flat around the logo are the most informative
  const rank = Ys.map((Y, i) => {
    let sum = 0, c = 0;
    for (let j = Math.floor(y); j < Math.floor(y + s); j++) for (let k = Math.floor(x); k < Math.floor(x + s); k++) { sum += Y[j * rw + k]; c++; }
    return [(wy - sum / Math.max(1, c)), i];
  }).sort((a, b) => b[0] - a[0]);
  const pick = rank.slice(0, 8).map(r => Ys[r[1]]);
  const geo = E.fitGeometry(pick, [x, y, s], wy, rw, rh, score);
  if (!geo.found) {
    if (manualBox) return { x: manualBox[0], y: manualBox[1], size: manualBox[2], found: false, note: 'manual-inpaint' };
    return geo;
  }
  // per-video calibration of the alpha map
  onStatus && onStatus('Đang hiệu chuẩn độ mờ logo theo video…');
  const box = boxFor(geo, rw, rh);
  const [bx, by, bw, bh] = box;
  const model = E.alphaFor(geo, bx, by, bw, bh);
  const sup = model.map(v => (v > 0.004 ? 1 : 0));
  const mask = dil(sup, bw, bh, 2);
  const stack = [], wts = [], trs = [];
  for (const Y of Ys) {
    const P = new Float32Array(bw * bh);
    for (let j = 0; j < bh; j++) for (let k = 0; k < bw; k++) P[j * bw + k] = Y[(by + j) * rw + bx + k];
    const rough = roughness(P, sup, bw, bh);
    const tr = 1 / (1 + (rough / 2.5) ** 2);
    if (tr < 0.15) continue;
    const B = E.harmonicFill(P, mask, bw, bh, 200);
    const a = new Float32Array(bw * bh), w = new Float32Array(bw * bh);
    let cs = [];
    for (let i = 0; i < a.length; i++) {
      const con = Math.max(wy - B[i], 1);
      a[i] = (P[i] - B[i]) / con;
      w[i] = tr * Math.min(1, con / 110) ** 2 + 1e-6;
      if (sup[i]) cs.push(con);
    }
    cs.sort((p, q) => p - q);
    stack.push(a); wts.push(w); trs.push(tr * Math.min(1, (cs[cs.length >> 1] || 0) / 110));
  }
  if (stack.length >= 3 && trs.reduce((p, q) => p + q, 0) >= 1.5) {
    const cal = wmedianPerPixel(stack, wts, bw * bh);
    let mm = 0; for (const v of model) mm = Math.max(mm, v);
    const ratios = [];
    for (let i = 0; i < cal.length; i++) { cal[i] = mask[i] ? Math.min(0.9, Math.max(0, cal[i])) : 0; if (model[i] > mm * 0.6) ratios.push(cal[i] / Math.max(model[i], 1e-4)); }
    ratios.sort((p, q) => p - q);
    const ratio = ratios[ratios.length >> 1] || 0;
    if (ratio > 0.5 && ratio < 2) {
      const wsum = new Float32Array(bw * bh);
      for (const w of wts) for (let i = 0; i < w.length; i++) wsum[i] += w[i];
      const out = new Float32Array(bw * bh);
      for (let i = 0; i < out.length; i++) {
        const sw = Math.min(1, wsum[i] / 2);
        out[i] = mask[i] ? sw * cal[i] + (1 - sw) * model[i] * ratio : 0;
      }
      geo.calib = out; geo.strength *= ratio; geo.note = 'ok+calib';
    }
  }
  // back to frame coordinates
  geo.x += rx; geo.y += ry; geo.roi = roi;
  return geo;
}

function dil(m, w, h, r) {
  const o = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let v = 0;
    for (let j = Math.max(0, y - r); j <= Math.min(h - 1, y + r) && !v; j++)
      for (let i = Math.max(0, x - r); i <= Math.min(w - 1, x + r); i++) if (m[j * w + i]) { v = 1; break; }
    o[y * w + x] = v;
  }
  return o;
}

function roughness(g, sup, w, h) {
  const inner = dil(sup, w, h, 3), outer = dil(sup, w, h, 9);
  const bl = E.gaussBlur(g, w, h, 1.5);
  let s = 0, c = 0;
  for (let i = 0; i < w * h; i++) if (outer[i] && !inner[i]) { s += Math.abs(g[i] - bl[i]); c++; }
  return c < 10 ? 99 : s / c;
}

function boxFor(geo, W, H) {
  const margin = Math.max(4, Math.floor(geo.size * 0.25));
  let [bx, by, bw, bh] = E.boxWithMargin(geo.x, geo.y, geo.size, W, H, margin);
  let ex = bx + bw, ey = by + bh;
  bx = even(bx); by = even(by);
  ex = Math.min(W, ex + (ex % 2)); ey = Math.min(H, ey + (ey % 2));
  return [bx, by, ex - bx, ey - by];
}

function chromaAlpha(a, w, h) {
  const cw = w / 2, ch = h / 2, o = new Float32Array(cw * ch);
  for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) {
    const i = 2 * y * w + 2 * x;
    o[y * cw + x] = Math.min(0.95, 0.25 * (a[i] + a[i + 1] + a[i + w] + a[i + w + 1]));
  }
  return o;
}

/** Builds a per-frame remover working on Y/U/V planes of the logo box. */
function makeRemover(geo, info, method) {
  const W = info.width, H = info.height, wy = info.range === 'pc' ? 255 : 235;
  const box = boxFor(geo, W, H);
  const [bx, by, bw, bh] = box;
  if (method === 'inpaint' || !geo.found) {
    const a = E.renderAlpha(E.T.v1, geo.x, geo.y, geo.size, bx, by, bw, bh);
    const k = Math.max(1, Math.round(geo.size * 0.06));
    const mask = dil(a.map(v => (v > 0.02 ? 1 : 0)), bw, bh, k);
    const maskc = new Uint8Array((bw / 2) * (bh / 2));
    for (let y = 0; y < bh / 2; y++) for (let x = 0; x < bw / 2; x++) maskc[y * (bw / 2) + x] = mask[2 * y * bw + 2 * x] | mask[2 * y * bw + 2 * x + 1] | mask[(2 * y + 1) * bw + 2 * x];
    return {
      box, fn(f) {
        const fill = (plane, m, w, h) => { const P = toF32(plane); const F = E.harmonicFill(P, m, w, h, 200); for (let i = 0; i < P.length; i++) plane[i] = Math.max(0, Math.min(255, Math.round(F[i]))); };
        fill(f.y, mask, bw, bh); fill(f.u, maskc, bw / 2, bh / 2); fill(f.v, maskc, bw / 2, bh / 2);
      },
    };
  }
  let a;
  if (geo.calib && geo.roi) {
    // calibrated map was measured in ROI coordinates on the same even grid
    a = Float32Array.from(geo.calib, v => Math.min(0.92, v));
    if (a.length !== bw * bh) a = E.alphaFor(geo, bx, by, bw, bh);
  } else {
    a = E.alphaFor(geo, bx, by, bw, bh);
  }
  const ac = chromaAlpha(a, bw, bh);
  // seam weights (outline where the codec blurred the edge)
  let am = 0; for (const v of a) am = Math.max(am, v);
  const sw = new Float32Array(bw * bh);
  for (let y = 1; y < bh - 1; y++) for (let x = 1; x < bw - 1; x++) {
    const i = y * bw + x;
    const gx = a[i + 1] - a[i - 1], gy = a[i + bw] - a[i - bw];
    sw[i] = Math.min(1, Math.max(0, (Math.hypot(gx, gy) / Math.max(am, 1e-6) - 0.12) / 0.5));
  }
  const hard = sw.map(v => (v > 0.25 ? 1 : 0));
  return {
    box, fn(f) {
      const R = new Float32Array(bw * bh);
      for (let i = 0; i < R.length; i++) R[i] = (f.y[i] - a[i] * wy) / (1 - a[i]);
      const F = E.harmonicFill(R, hard, bw, bh, 40);
      for (let i = 0; i < R.length; i++) f.y[i] = Math.max(0, Math.min(255, Math.round(R[i] * (1 - sw[i]) + F[i] * sw[i])));
      for (const c of [f.u, f.v]) for (let i = 0; i < c.length; i++) c[i] = Math.max(0, Math.min(255, Math.round((c[i] - ac[i] * 128) / (1 - ac[i]))));
    },
  };
}

export async function previewAfter(name, info, geo, t, method) {
  const img = await grabFrameRGBA(name, info, t);
  if (!geo || !geo.size) return img;
  const rem = makeRemover(geo, info, method);
  const [bx, by, bw, bh] = rem.box;
  // work on a YUV copy of the box built from RGBA (preview only)
  const Y = new Uint8Array(bw * bh), U = new Uint8Array(bw * bh / 4), V = new Uint8Array(bw * bh / 4);
  const d = img.data, W = info.width, full = info.range === 'pc';
  for (let j = 0; j < bh; j++) for (let i = 0; i < bw; i++) {
    const p = 4 * ((by + j) * W + bx + i), r = d[p], g = d[p + 1], b = d[p + 2];
    let yy = 0.299 * r + 0.587 * g + 0.114 * b;
    Y[j * bw + i] = full ? yy : 16 + yy * 219 / 255;
  }
  for (let j = 0; j < bh / 2; j++) for (let i = 0; i < bw / 2; i++) {
    let su = 0, sv = 0;
    for (const [dx, dy] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
      const p = 4 * ((by + 2 * j + dy) * W + bx + 2 * i + dx), r = d[p], g = d[p + 1], b = d[p + 2];
      su += -0.168736 * r - 0.331264 * g + 0.5 * b; sv += 0.5 * r - 0.418688 * g - 0.081312 * b;
    }
    const k = full ? 1 : 224 / 255;
    U[j * bw / 2 + i] = 128 + su / 4 * k; V[j * bw / 2 + i] = 128 + sv / 4 * k;
  }
  const f = { y: Y, u: U, v: V };
  rem.fn(f);
  for (let j = 0; j < bh; j++) for (let i = 0; i < bw; i++) {
    const p = 4 * ((by + j) * W + bx + i);
    let yy = f.y[j * bw + i], uu = f.u[(j >> 1) * bw / 2 + (i >> 1)] - 128, vv = f.v[(j >> 1) * bw / 2 + (i >> 1)] - 128;
    if (!full) { yy = (yy - 16) * 255 / 219; uu *= 255 / 224; vv *= 255 / 224; }
    d[p] = yy + 1.402 * vv; d[p + 1] = yy - 0.344136 * uu - 0.714136 * vv; d[p + 2] = yy + 1.772 * uu;
  }
  return img;
}

/** Full processing. Returns a Uint8Array of the output MP4. */
export async function processVideo(name, info, geo, { method = 'alpha', strip = true, onProgress } = {}) {
  const rem = makeRemover(geo, info, method);
  const [bx, by, bw, bh] = rem.box;
  onProgress && onProgress(0.05, 'Đang tách vùng logo…');
  const frames = await grabRegion(name, info, rem.box, {});
  const fb = bw * bh * 1.5;
  const out = new Uint8Array(frames.length * fb);
  for (let i = 0; i < frames.length; i++) {
    const f = { y: new Uint8Array(frames[i].y), u: new Uint8Array(frames[i].u), v: new Uint8Array(frames[i].v) };
    rem.fn(f);
    out.set(f.y, i * fb); out.set(f.u, i * fb + bw * bh); out.set(f.v, i * fb + bw * bh * 1.25);
    if (i % 20 === 0) { onProgress && onProgress(0.05 + 0.15 * i / frames.length, `Xử lý khung ${i}/${frames.length}`); await new Promise(r => setTimeout(r)); }
  }
  await ff.writeFile('box.yuv', out);
  const meta = strip
    ? ['-map_metadata', '-1', '-map_chapters', '-1', '-fflags', '+bitexact', '-flags:v', '+bitexact', '-flags:a', '+bitexact']
    : ['-map_metadata', '0'];
  const audio = info.hasAudio ? (/^(aac|mp3)$/.test(info.acodec) ? ['-c:a', 'copy'] : ['-c:a', 'aac', '-b:a', '192k']) : [];
  const total = frames.length;
  const handler = ({ progress }) => onProgress && onProgress(0.2 + 0.8 * Math.max(0, Math.min(1, progress)), 'Đang mã hóa MP4…');
  ff.on('progress', handler);
  const { rc, log } = await run(['-v', 'error', '-i', name,
    '-f', 'rawvideo', '-pix_fmt', 'yuv420p', '-s', `${bw}x${bh}`, '-r', info.fpsStr, '-i', 'box.yuv',
    '-filter_complex', `[0:v]setpts=PTS-STARTPTS[a];[1:v]setpts=PTS-STARTPTS[b];[a][b]overlay=${bx}:${by}:eof_action=pass,format=yuv420p[v]`,
    '-map', '[v]', '-map', '0:a?', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', ...audio,
    '-color_range', info.range === 'pc' ? 'pc' : 'tv', ...meta, '-movflags', '+faststart', 'out.mp4']);
  ff.off('progress', handler);
  await ff.deleteFile('box.yuv');
  if (rc !== 0) throw new Error('ffmpeg lỗi: ' + log.slice(-400));
  const data = await ff.readFile('out.mp4');
  await ff.deleteFile('out.mp4');
  onProgress && onProgress(1, `Xong (${total} khung)`);
  return data;
}
