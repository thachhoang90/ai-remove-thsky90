// Container-level metadata helpers (JPEG segments / PNG chunks).

const td = new TextDecoder('latin1');

export function scan(bytes) {
  const s = td.decode(bytes.subarray(0, Math.min(bytes.length, 8 * 1024 * 1024)));
  const f = [];
  if (/c2pa|caBX|jumb|jumd/.test(s)) f.push('C2PA (Content Credentials)');
  if (/<x:xmpmeta|ns\.adobe\.com\/xap|XML:com\.adobe\.xmp/.test(s)) f.push('XMP');
  if (/trainedAlgorithmicMedia|compositeSynthetic/.test(s)) f.push('IPTC: gắn nhãn AI');
  if (/Google|Gemini/.test(s)) f.push("chuỗi 'Google/Gemini'");
  if (/Exif\0\0|eXIf/.test(s)) f.push('EXIF');
  return f;
}

function jpegSegments(b) {
  const segs = [];
  if (b[0] !== 0xff || b[1] !== 0xd8) return segs;
  let i = 2;
  while (i + 4 <= b.length) {
    if (b[i] !== 0xff) break;
    const m = b[i + 1];
    if (m === 0xda || m === 0xd9) break;
    const len = (b[i + 2] << 8) | b[i + 3];
    segs.push({ marker: m, bytes: b.subarray(i, i + 2 + len) });
    i += 2 + len;
  }
  return segs;
}

function isIcc(seg) { return seg.marker === 0xe2 && td.decode(seg.bytes.subarray(4, 15)) === 'ICC_PROFILE'; }

export function rebuildJpeg(newBytes, origBytes, keepAll) {
  const keep = jpegSegments(origBytes).filter(s =>
    isIcc(s) || (keepAll && (s.marker === 0xe1 || s.marker === 0xed || s.marker === 0xfe)));
  if (!keep.length) return newBytes;
  const ns = jpegSegments(newBytes);
  // insert after SOI + APP0 (JFIF) of the new file, drop its own APP2 ICC if any
  let pos = 2;
  for (const s of ns) { if (s.marker === 0xe0) pos += s.bytes.length; else break; }
  const tail = newBytes.subarray(pos);
  const total = pos + keep.reduce((a, s) => a + s.bytes.length, 0) + tail.length;
  const out = new Uint8Array(total);
  out.set(newBytes.subarray(0, pos), 0);
  let o = pos;
  for (const s of keep) { out.set(s.bytes, o); o += s.bytes.length; }
  out.set(tail, o);
  return out;
}

function pngChunks(b) {
  const out = [];
  let i = 8;
  while (i + 12 <= b.length) {
    const len = ((b[i] << 24) | (b[i + 1] << 16) | (b[i + 2] << 8) | b[i + 3]) >>> 0;
    const type = td.decode(b.subarray(i + 4, i + 8));
    out.push({ type, bytes: b.subarray(i, i + 12 + len) });
    i += 12 + len;
    if (type === 'IEND') break;
  }
  return out;
}

export function rebuildPng(newBytes, origBytes, keepAll) {
  const want = keepAll ? ['iCCP', 'tEXt', 'iTXt', 'zTXt', 'eXIf', 'pHYs'] : ['iCCP'];
  const keep = pngChunks(origBytes).filter(c => want.includes(c.type));
  if (!keep.length) return newBytes;
  const nc = pngChunks(newBytes);
  const parts = [newBytes.subarray(0, 8)];
  for (const c of nc) {
    if (keep.some(k => k.type === 'iCCP') && (c.type === 'iCCP' || c.type === 'sRGB' || c.type === 'gAMA')) continue;
    parts.push(c.bytes);
    if (c.type === 'IHDR') for (const k of keep) parts.push(k.bytes);
  }
  const total = parts.reduce((a, p) => a + p.length, 0);
  const out = new Uint8Array(total);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}
