"""Video pipeline: detect the static sparkle over many frames, then remove it
frame by frame directly on the decoded Y/U/V planes (the rest of the picture is
passed through untouched apart from the final re-encode)."""
from __future__ import annotations

import os
import tempfile
import threading

import cv2
import numpy as np

from . import engine as E
from .ffmpeg_tools import ffmpeg_path, popen, probe, run, encoders, VideoInfo

VIDEO_EXTS = (".mp4", ".mov", ".m4v", ".avi", ".webm", ".wmv", ".mkv")


class Cancelled(Exception):
    pass


# --------------------------------------------------------------------------
# Frame access
# --------------------------------------------------------------------------

def _dims(info: VideoInfo):
    w, h = info.width, info.height
    if abs(info.rotation) % 180 == 90:
        w, h = h, w
    return w, h


def _split_yuv(buf: bytes, w: int, h: int):
    cw, ch = (w + 1) // 2, (h + 1) // 2
    a = np.frombuffer(buf, np.uint8)
    y = a[:w * h].reshape(h, w)
    u = a[w * h:w * h + cw * ch].reshape(ch, cw)
    v = a[w * h + cw * ch:w * h + 2 * cw * ch].reshape(ch, cw)
    return y, u, v


def _frame_bytes(w, h):
    return w * h + 2 * ((w + 1) // 2) * ((h + 1) // 2)


def grab_yuv(path: str, t: float, info: VideoInfo | None = None):
    info = info or probe(path)
    w, h = _dims(info)
    p = run([ffmpeg_path(), "-v", "error", "-ss", "%.3f" % max(0.0, t), "-i", path,
             "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
            capture_output=True)
    n = _frame_bytes(w, h)
    if len(p.stdout) < n:
        p = run([ffmpeg_path(), "-v", "error", "-i", path, "-frames:v", "1",
                 "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"], capture_output=True)
        if len(p.stdout) < n:
            raise RuntimeError("Không đọc được khung hình video")
    y, u, v = _split_yuv(p.stdout[:n], w, h)
    return y.copy(), u.copy(), v.copy()


def yuv_to_rgb(y, u, v, full_range=False):
    h, w = y.shape
    uu = cv2.resize(u, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) - 128
    vv = cv2.resize(v, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) - 128
    yy = y.astype(np.float32)
    if not full_range:
        yy = (yy - 16) * (255 / 219)
        uu *= 255 / 224
        vv *= 255 / 224
    r = yy + 1.402 * vv
    g = yy - 0.344136 * uu - 0.714136 * vv
    b = yy + 1.772 * uu
    return np.clip(np.stack([r, g, b], -1), 0, 255).astype(np.uint8)


def grab_rgb(path: str, t: float, info: VideoInfo | None = None):
    info = info or probe(path)
    y, u, v = grab_yuv(path, t, info)
    return yuv_to_rgb(y, u, v, info.color_range == "pc")


def sample_frames(path: str, info: VideoInfo, n: int = 16, cancel=None):
    """Return ``n`` frames (Y, U, V) spread over the clip."""
    w, h = _dims(info)
    dur = max(info.duration, 0.04)
    out = []
    if dur <= 75:
        # one sequential decode, keep evenly spaced frames
        total = max(1, info.frames)
        step = max(1, total // n)
        fb = _frame_bytes(w, h)
        proc = popen([ffmpeg_path(), "-v", "error", "-i", path, "-an",
                      "-vf", "select='not(mod(n\\,%d))'" % step, "-fps_mode", "passthrough",
                      "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
                     stdout=__import__("subprocess").PIPE,
                     stderr=__import__("subprocess").DEVNULL)
        try:
            while len(out) < n * 2:
                if cancel and cancel():
                    raise Cancelled()
                buf = proc.stdout.read(fb)
                if not buf or len(buf) < fb:
                    break
                y, u, v = _split_yuv(buf, w, h)
                out.append((y.copy(), u.copy(), v.copy()))
        finally:
            proc.stdout.close()
            proc.kill()
            proc.wait()
        if len(out) > n:
            idx = np.linspace(0, len(out) - 1, n).round().astype(int)
            out = [out[i] for i in idx]
    if not out:
        for t in np.linspace(dur * 0.04, dur * 0.96, n):
            if cancel and cancel():
                raise Cancelled()
            try:
                out.append(grab_yuv(path, float(t), info))
            except Exception:
                pass
    if not out:
        raise RuntimeError("Không đọc được khung hình video")
    return out


# --------------------------------------------------------------------------
# Detection over many frames
# --------------------------------------------------------------------------

def _white_y(info: VideoInfo) -> float:
    return 255.0 if info.color_range == "pc" else 235.0


def detect_video(frames, info: VideoInfo, opts: E.Options | None = None) -> E.Geometry:
    opts = opts or E.Options()
    Ys = [f[0].astype(np.float32) for f in frames]
    H, W = Ys[0].shape
    prior = None
    if opts.mode == "manual" and opts.manual_box:
        bx, by, bw, bh = opts.manual_box
        s = float(min(bw, bh))
        prior = (bx + (bw - s) / 2, by + (bh - s) / 2, s)
    # accumulate NCC maps: a static logo adds up, moving content averages out
    if prior is not None:
        px, py, ps = prior
        m = max(6, int(ps * 0.35))
        roi = (max(0, int(px - m)), max(0, int(py - m)),
               min(W, int(px + ps + m)) - max(0, int(px - m)),
               min(H, int(py + ps + m)) - max(0, int(py - m)))
        sizes = range(max(8, int(ps * 0.7)), int(ps * 1.3) + 2)
    else:
        roi = E.corner_roi(W, H)
        smax = min(int(min(roi[2], roi[3]) * 0.8), max(140, int(min(W, H) * 0.12)))
        sizes = range(16, smax + 1, 2)
    Yd = Ys if len(Ys) <= 16 else [Ys[i] for i in np.linspace(0, len(Ys) - 1, 16).round().astype(int)]
    acc = {}
    for y in Yd:
        for s, r in E.ncc_maps(y, sizes, roi).items():
            r = np.nan_to_num(r, nan=0.0)
            acc[s] = r if s not in acc else acc[s] + r
    best = (-1.0, 0, 0, 0)
    for s, r in acc.items():
        _, mx, _, loc = cv2.minMaxLoc(r)
        mx /= len(Yd)
        if mx > best[0]:
            best = (float(mx), s, roi[0] + loc[0], roi[1] + loc[1])
    if prior is None and best[1]:
        s0 = best[1]
        acc2 = {}
        for y in Yd:
            for s, r in E.ncc_maps(y, range(max(16, s0 - 2), s0 + 3), roi).items():
                r = np.nan_to_num(r, nan=0.0)
                acc2[s] = r if s not in acc2 else acc2[s] + r
        for s, r in acc2.items():
            _, mx, _, loc = cv2.minMaxLoc(r)
            mx /= len(Yd)
            if mx > best[0]:
                best = (float(mx), s, roi[0] + loc[0], roi[1] + loc[1])
    score, s, x, y = best
    if s == 0:
        if prior:
            return E.Geometry(*prior, found=False, note="manual-no-match")
        return E.Geometry(0, 0, 0, note="not-found")
    if prior is None and score < 0.25:
        return E.Geometry(x, y, s, score=score, found=False, note="low-score")
    # refine on the frames where the logo is most visible: dark and flat
    # surroundings (texture behind the logo only adds noise to the fit)
    wy = _white_y(info)
    m = max(6, int(s * 0.25))
    bx0, by0 = max(0, int(x) - m), max(0, int(y) - m)
    A0 = E.render_alpha(E.Templates.get().template("v1"), x, y, s, bx0, by0,
                        min(W, int(x + s) + m) - bx0, min(H, int(y + s) + m) - by0)
    rank = []
    for i, yy in enumerate(Ys):
        sub = yy[by0:by0 + A0.shape[0], bx0:bx0 + A0.shape[1]]
        rough = E.ring_roughness(sub, A0 > 0.01)
        contrast = wy - float(np.median(sub))
        rank.append((contrast / (1.0 + rough) ** 2, i))
    rank.sort(reverse=True)
    pick = [Ys[i] for q, i in rank[:8] if q > 0] or [Ys[rank[0][1]]]
    geo = E.fit_geometry(pick, (x, y, s), wy, W, H, score)
    geo.extra_frames = len(pick)
    if geo.found and opts.extra.get("calibrate", True):
        cal = calibrate_alpha(Ys, geo, wy, W, H)
        if cal is not None:
            geo.calib = cal
            geo.note = "ok+calib"
            geo.chroma_blur = 0.0  # the measured map already holds the codec blur
    if prior is not None and not geo.found:
        geo.x, geo.y, geo.size = prior
        geo.note = "manual-inpaint"
    if geo.found and getattr(geo, "calib", None) is None:
        geo.chroma_blur = _fit_chroma(frames, geo)
    return geo


def _boxes(geo: E.Geometry, W: int, H: int):
    margin = max(4, int(geo.size * 0.25))
    bx, by, bw, bh = E._box_with_margin(geo.x, geo.y, geo.size, W, H, margin)
    # align to even coordinates so chroma maps 2:1
    ex, ey = bx + bw, by + bh
    bx -= bx % 2
    by -= by % 2
    ex = min(W, ex + (ex % 2))
    ey = min(H, ey + (ey % 2))
    return bx, by, ex - bx, ey - by


def _chroma_alpha(a, blur):
    h, w = a.shape
    ac = cv2.resize(a, ((w + 1) // 2, (h + 1) // 2), interpolation=cv2.INTER_AREA)
    if blur > 0:
        ac = cv2.GaussianBlur(ac, (0, 0), blur)
    return ac


def _fit_chroma(frames, geo):
    H, W = frames[0][0].shape
    bx, by, bw, bh = _boxes(geo, W, H)
    a = E.alpha_for(geo, bx, by, bw, bh)
    best = None
    for blur in (0.0, 0.4, 0.7, 1.0):
        ac = np.clip(_chroma_alpha(a, blur), 0, 0.95)
        wmask = (cv2.dilate((ac > 0.005).astype(np.uint8), np.ones((3, 3), np.uint8))).astype(np.float32)
        e = 0.0
        for _, u, v in frames[:10]:
            for c in (u, v):
                p = c[by // 2:by // 2 + ac.shape[0], bx // 2:bx // 2 + ac.shape[1]].astype(np.float32)
                if p.shape != ac.shape:
                    continue
                e += E._grad_energy((p - ac * 128.0) / (1 - ac), wmask)
        if best is None or e < best[0]:
            best = (e, blur)
    return float(best[1])


# --------------------------------------------------------------------------
# Per-frame removal
# --------------------------------------------------------------------------

def _wmedian(stack: np.ndarray, wts: np.ndarray) -> np.ndarray:
    idx = np.argsort(stack, axis=0)
    sv = np.take_along_axis(stack, idx, 0)
    sw = np.take_along_axis(wts, idx, 0)
    cw = np.cumsum(sw, 0)
    half = cw[-1] / 2.0
    pos = (cw >= half[None]).argmax(0)
    return np.take_along_axis(sv, pos[None], 0)[0]


def calibrate_alpha(Ys, geo: E.Geometry, white: float, W: int, H: int):
    """Measure the real alpha map of the logo from the clip itself.

    For every frame the background under the logo is predicted by a smooth
    (harmonic) continuation of its surroundings; ``(I - B) / (white - B)``
    is then a per-pixel alpha sample.  A weighted median over many frames
    (weights favour flat, dark surroundings) gives a map that already
    contains the codec's blur and the exact opacity of this video."""
    box = _boxes(geo, W, H)
    bx, by, bw, bh = box
    model = E.alpha_for(geo, bx, by, bw, bh)
    sup = model > 0.004
    if sup.sum() < 20:
        return None
    mask = cv2.dilate(sup.astype(np.uint8), np.ones((5, 5), np.uint8))
    inner = cv2.erode((model > model.max() * 0.6).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    prep = []
    for Y in Ys:
        P = Y[by:by + bh, bx:bx + bw].astype(np.float32)
        if P.shape != model.shape:
            continue
        rough = E.ring_roughness(P, sup)
        tr = 1.0 / (1.0 + (rough / 2.5) ** 2)
        if tr < 0.15:
            continue
        B = E.harmonic_fill(P, mask, 300)
        prep.append((P, B, tr, rough))
    level = white
    stack, wts, trs = [], [], []
    for P, B, tr, _ in prep:
        con = np.maximum(level - B, 1.0)
        stack.append((P - B) / con)
        wts.append(tr * np.clip(con / 110.0, 0, 1) ** 2 + 1e-6)
        trs.append(tr * float(np.clip(np.median(con[sup]) / 110.0, 0, 1)))
    if len(stack) < 3 or sum(trs) < 1.5:
        return None
    a = _wmedian(np.array(stack), np.array(wts))
    a = np.clip(a, 0, 0.9)
    a[mask == 0] = 0
    inner = model > model.max() * 0.6
    if inner.sum() < 10:
        return None
    ratio = float(np.median(a[inner] / np.maximum(model[inner], 1e-4)))
    if not (0.5 < ratio < 2.0):
        return None
    scaled = model * ratio
    corr = np.corrcoef(a[mask > 0], scaled[mask > 0])[0, 1]
    if not np.isfinite(corr) or corr < 0.85:
        return None
    # trust the measurement where it is well supported, the model elsewhere
    support_w = np.clip(np.sum(np.array(wts), 0) / 2.0, 0, 1)
    out = support_w * a + (1 - support_w) * scaled
    out[mask == 0] = 0
    geo.strength *= ratio  # keep the reported strength meaningful
    return (box, out.astype(np.float32))


class FrameRemover:
    def __init__(self, geo: E.Geometry, W: int, H: int, white_y: float,
                 method: str = "alpha", seam: float = 1.0):
        self.geo = geo
        self.box = _boxes(geo, W, H)
        bx, by, bw, bh = self.box
        self.white_y = white_y
        self.method = "inpaint" if (method == "inpaint" or not geo.found) else "alpha"
        self.seam = seam
        if self.method == "alpha":
            cal = getattr(geo, "calib", None)
            if cal is not None and tuple(cal[0]) == tuple(self.box):
                self.a = np.clip(cal[1], 0, 0.92).astype(np.float32)
            else:
                self.a = E.alpha_for(geo, bx, by, bw, bh).astype(np.float32)
            self.ac = np.clip(_chroma_alpha(self.a, geo.chroma_blur), 0, 0.95).astype(np.float32)
            au = self.a / max(self.a.max(), 1e-6)
            self.seam_w = np.clip(E.seam_mask(au) * seam, 0, 1).astype(np.float32)
            self.seam_hard = (self.seam_w > 0.25).astype(np.uint8)
        else:
            T = E.Templates.get().variants["v1"]
            a = E.render_alpha(T, geo.x, geo.y, geo.size, bx, by, bw, bh)
            k = max(1, int(round(geo.size * 0.06)))
            m = cv2.dilate((a > 0.02).astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8))
            self.mask = m
            self.mask_c = (cv2.resize(m, ((bw + 1) // 2, (bh + 1) // 2),
                                      interpolation=cv2.INTER_AREA) > 0).astype(np.uint8)
            self.rad = max(3, int(geo.size * 0.12))

    def __call__(self, y: np.ndarray, u: np.ndarray, v: np.ndarray, scale: float = 1.0):
        bx, by, bw, bh = self.box
        cx, cy = bx // 2, by // 2
        ys = y[by:by + bh, bx:bx + bw]
        if self.method == "inpaint":
            y[by:by + bh, bx:bx + bw] = cv2.inpaint(np.ascontiguousarray(ys), self.mask,
                                                    self.rad, cv2.INPAINT_TELEA)
            for c in (u, v):
                cs = c[cy:cy + self.mask_c.shape[0], cx:cx + self.mask_c.shape[1]]
                m = self.mask_c[:cs.shape[0], :cs.shape[1]]
                c[cy:cy + cs.shape[0], cx:cx + cs.shape[1]] = cv2.inpaint(
                    np.ascontiguousarray(cs), m, max(2, self.rad // 2), cv2.INPAINT_TELEA)
            return
        a = self.a if scale == 1.0 else np.clip(self.a * scale, 0, 0.92)
        r = (ys.astype(np.float32) - a * self.white_y) / (1 - a)
        if self.seam > 0 and self.seam_hard.any():
            src = np.clip(r, 0, 255).astype(np.uint8)
            inp = cv2.inpaint(src, self.seam_hard, 2, cv2.INPAINT_TELEA).astype(np.float32)
            r = r * (1 - self.seam_w) + inp * self.seam_w
        y[by:by + bh, bx:bx + bw] = np.clip(np.rint(r), 0, 255).astype(np.uint8)
        ac = self.ac if scale == 1.0 else np.clip(self.ac * scale, 0, 0.95)
        for c in (u, v):
            cs = c[cy:cy + ac.shape[0], cx:cx + ac.shape[1]]
            aa = ac[:cs.shape[0], :cs.shape[1]]
            rc = (cs.astype(np.float32) - aa * 128.0) / (1 - aa)
            c[cy:cy + cs.shape[0], cx:cx + cs.shape[1]] = np.clip(np.rint(rc), 0, 255).astype(np.uint8)


def scene_scales(src: str, info: VideoInfo, remover: "FrameRemover", cancel=None,
                 progress=None, window_s: float = 0.5) -> np.ndarray | None:
    """Per-frame correction of the logo strength, smoothed over time.

    The opacity measured on flat, dark surroundings can differ a little from
    scene to scene (encoder / compositing).  For every frame the strength that
    makes the logo interior match a smooth continuation of its surroundings
    is measured; a trust-weighted running median over ~1 s keeps it stable
    and frames without reliable evidence fall back to the global value."""
    if remover.method != "alpha":
        return None
    W, H = _dims(info)
    bx, by, bw, bh = remover.box
    a0 = remover.a
    au = a0 / max(float(a0.max()), 1e-6)
    sup = cv2.dilate((au > 0.01).astype(np.uint8), np.ones((5, 5), np.uint8))
    inner = cv2.erode((au > 0.6).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    if inner.sum() < 10:
        return None
    L = remover.white_y
    total = info.frames
    step = max(1, total // 900)          # long clips: measure a subset
    proc = popen([ffmpeg_path(), "-v", "error", "-i", src, "-an", "-sn", "-dn",
                  "-fps_mode", "cfr", "-r", info.fps_str,
                  "-vf", f"crop={bw}:{bh}:{bx}:{by}", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
                 stdout=__import__("subprocess").PIPE, stderr=__import__("subprocess").DEVNULL)
    fbb = _frame_bytes(bw, bh)
    cs, ts = [], []
    n = 0
    try:
        while True:
            if cancel and cancel():
                raise Cancelled()
            buf = proc.stdout.read(fbb)
            if not buf or len(buf) < fbb:
                break
            if n % step:
                cs.append(np.nan)
                ts.append(0.0)
                n += 1
                continue
            P = np.frombuffer(buf[:bw * bh], np.uint8).reshape(bh, bw).astype(np.float32)
            rough = E.ring_roughness(P, au > 0.01)
            tr = 1.0 / (1.0 + (rough / 2.0) ** 2)
            c_est, ok = 1.0, False
            if tr > 0.2:
                B = E.harmonic_fill(P, sup, 150)
                con = float(np.median((L - B)[inner]))
                tr *= min(1.0, con / 60.0)
                Pi, Bi, Ai = P[inner], B[inner], a0[inner]

                def f(c):
                    aa = np.clip(Ai * c, 0, 0.92)
                    return float(np.median((Pi - aa * L) / (1 - aa) - Bi))
                lo, hi = 0.6, 1.5
                if con > 20 and f(lo) > 0 > f(hi):
                    for _ in range(22):
                        m = 0.5 * (lo + hi)
                        if f(m) > 0:
                            lo = m
                        else:
                            hi = m
                    c_est, ok = 0.5 * (lo + hi), True
            cs.append(c_est if ok else np.nan)
            ts.append(tr if ok else 0.0)
            n += 1
            if progress and n % 25 == 0:
                progress(min(0.15, 0.15 * n / max(total, 1)), "Đang đo độ mờ theo cảnh…")
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()
    if not cs:
        return None
    cs, ts = np.array(cs), np.array(ts)
    N = len(cs)
    half = max(3, int(round(info.fps * window_s)))
    out = np.ones(N, np.float32)
    for i in range(N):
        lo, hi = max(0, i - half), min(N, i + half + 1)
        w, c = ts[lo:hi], cs[lo:hi]
        m = w > 0
        if w[m].sum() < 0.8:
            continue
        o = np.argsort(c[m])
        cw = np.cumsum(w[m][o])
        out[i] = c[m][o][np.searchsorted(cw, cw[-1] / 2)]
    return np.clip(out, 0.8, 1.3)


def preview_pair(path: str, t: float, geo: E.Geometry, info: VideoInfo | None = None,
                 method="alpha"):
    """(before_rgb, after_rgb) of the frame at time t, for the GUI preview."""
    info = info or probe(path)
    y, u, v = grab_yuv(path, t, info)
    full = info.color_range == "pc"
    before = yuv_to_rgb(y, u, v, full)
    if geo is not None and geo.size > 0:
        H, W = y.shape
        FrameRemover(geo, W, H, _white_y(info), method)(y, u, v)
    after = yuv_to_rgb(y, u, v, full)
    return before, after


# --------------------------------------------------------------------------
# Full pipeline
# --------------------------------------------------------------------------

_ENC_CACHE = None


def _available_encoders():
    global _ENC_CACHE
    if _ENC_CACHE is None:
        try:
            _ENC_CACHE = encoders()
        except Exception:
            _ENC_CACHE = set()
    return _ENC_CACHE


def codec_args(ext: str, info: VideoInfo, quality: str = "high"):
    enc = _available_encoders()
    ext = ext.lower()
    crf = {"high": 16, "normal": 20}.get(quality, 16)
    v, a = [], []
    ac = (info.acodec or "").lower()
    if ext in (".webm",):
        if "libvpx-vp9" in enc:
            v = ["-c:v", "libvpx-vp9", "-crf", str(crf + 12), "-b:v", "0", "-row-mt", "1",
                 "-deadline", "good", "-cpu-used", "2"]
        else:
            v = ["-c:v", "libvpx", "-crf", "10", "-b:v", "8M"]
        a = ["-c:a", "copy"] if ac in ("opus", "vorbis") else ["-c:a", "libopus" if "libopus" in enc else "libvorbis", "-b:a", "160k"]
    elif ext == ".avi":
        v = ["-c:v", "mpeg4", "-q:v", "2", "-vtag", "xvid"]
        a = ["-c:a", "copy"] if ac in ("mp3", "pcm_s16le", "ac3", "mp2") else ["-c:a", "libmp3lame" if "libmp3lame" in enc else "mp2", "-b:a", "192k"]
    elif ext == ".wmv":
        br = max(4000, int(info.bitrate_kbps * 1.6) if info.bitrate_kbps else 6000)
        v = ["-c:v", "wmv2", "-b:v", "%dk" % br]
        a = ["-c:a", "copy"] if ac in ("wmav2", "wmav1") else ["-c:a", "wmav2", "-b:a", "192k"]
    else:  # mp4 / mov / m4v / mkv
        if "libx264" in enc:
            v = ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-profile:v", "high"]
        elif "libopenh264" in enc:
            v = ["-c:v", "libopenh264", "-b:v", "%dk" % max(6000, info.bitrate_kbps * 2)]
        else:
            v = ["-c:v", "mpeg4", "-q:v", "2"]
        okcopy = ("aac", "mp3", "ac3", "eac3", "alac") if ext != ".mkv" else ("aac", "mp3", "ac3", "eac3", "opus", "vorbis", "flac", "alac")
        a = ["-c:a", "copy"] if ac in okcopy else ["-c:a", "aac", "-b:a", "192k"]
        if ext in (".mp4", ".m4v", ".mov"):
            v += ["-movflags", "+faststart"]
    v += ["-pix_fmt", "yuv420p"]
    return v, a


def _color_args(info: VideoInfo):
    out = ["-color_range", "pc" if info.color_range == "pc" else "tv"]
    valid_cs = {"bt709", "smpte170m", "bt470bg", "bt2020nc", "smpte240m", "fcc"}
    valid_pr = {"bt709", "smpte170m", "bt470bg", "bt2020", "smpte240m", "bt470m", "film"}
    valid_tr = {"bt709", "smpte170m", "iec61966-2-1", "gamma22", "gamma28", "bt2020-10",
                "linear", "smpte240m"}
    if info.colorspace in valid_cs:
        out += ["-colorspace", info.colorspace]
    if info.color_primaries in valid_pr:
        out += ["-color_primaries", info.color_primaries]
    if info.color_trc in valid_tr:
        out += ["-color_trc", info.color_trc]
    return out


def metadata_args(strip: bool):
    if strip:
        return ["-map_metadata", "-1", "-map_metadata:s:v", "-1", "-map_metadata:s:a", "-1",
                "-map_chapters", "-1", "-fflags", "+bitexact", "-flags:v", "+bitexact",
                "-flags:a", "+bitexact", "-metadata", "creation_time=",
                "-metadata:s:v", "handler_name=VideoHandler",
                "-metadata:s:a", "handler_name=SoundHandler"]
    return ["-map_metadata", "1", "-map_metadata:s:v", "1:s:v:0", "-map_metadata:s:a", "1:s:a:0"]


def process_video(src: str, dst: str, opts: E.Options | None = None, strip_metadata=True,
                  progress=None, cancel=None, log=None, quality="high", geo=None):
    """Remove the watermark from ``src`` and write ``dst``.

    progress(fraction 0..1, text) is called regularly; cancel() -> True aborts.
    Returns the Geometry used."""
    opts = opts or E.Options()
    log = log or (lambda *_: None)
    progress = progress or (lambda *_: None)
    info = probe(src)
    W, H = _dims(info)
    progress(0.0, "Đang phân tích video…")
    if geo is None:
        frames = sample_frames(src, info, 40, cancel)
        if opts.mode == "manual" and opts.manual_box and opts.method == "inpaint":
            bx, by, bw, bh = opts.manual_box
            s = float(min(bw, bh))
            geo = E.Geometry(bx + (bw - s) / 2, by + (bh - s) / 2, s, found=False,
                             note="manual-inpaint")
        else:
            geo = detect_video(frames, info, opts)
    if not geo.found and not (opts.mode == "manual" and opts.manual_box):
        log("Không phát hiện logo Gemini – video được giữ nguyên chất lượng (chỉ xử lý metadata).")
    method = "alpha" if geo.found and opts.method != "inpaint" else "inpaint"
    remover = None
    if geo.found or (opts.mode == "manual" and opts.manual_box):
        remover = FrameRemover(geo, W, H, _white_y(info), method, opts.seam)
    os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
    ext = os.path.splitext(dst)[1].lower()

    if remover is None:
        # nothing to remove: remux (copy streams) so quality is untouched
        args = [ffmpeg_path(), "-y", "-v", "error", "-i", src, "-map", "0:v:0", "-map", "0:a?",
                "-c", "copy"]
        args += ["-map_metadata", "-1", "-map_chapters", "-1", "-fflags", "+bitexact"] if strip_metadata else []
        args += [dst]
        p = run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if p.returncode != 0:
            # container change needs a re-encode
            v, a = codec_args(ext, info, quality)
            args = [ffmpeg_path(), "-y", "-v", "error", "-i", src, "-map", "0:v:0", "-map", "0:a?"] + v + a
            if strip_metadata:
                args += ["-map_metadata", "-1", "-map_chapters", "-1", "-fflags", "+bitexact"]
            p = run(args + [dst], capture_output=True, text=True, encoding="utf-8", errors="replace")
            if p.returncode != 0:
                raise RuntimeError(p.stderr.strip()[-600:])
        progress(1.0, "Xong")
        return geo

    scales = None
    if opts.extra.get("scene_adapt", True):
        try:
            scales = scene_scales(src, info, remover, cancel, progress)
        except Cancelled:
            raise
        except Exception as ex:  # noqa: BLE001
            log(f"Bỏ qua bước đo theo cảnh: {ex}")
            scales = None
    fb = _frame_bytes(W, H)
    dec = popen([ffmpeg_path(), "-v", "error", "-i", src, "-an", "-sn", "-dn",
                 "-fps_mode", "cfr", "-r", info.fps_str,
                 "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
                stdout=__import__("subprocess").PIPE, stderr=__import__("subprocess").DEVNULL,
                bufsize=fb * 2)
    v, a = codec_args(ext, info, quality)
    errf = tempfile.TemporaryFile()
    enc_args = [ffmpeg_path(), "-y", "-v", "error",
                "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{W}x{H}", "-r", info.fps_str,
                ] + _color_args(info) + ["-i", "-", "-i", src,
                "-map", "0:v:0", "-map", "1:a?"] + v + a + _color_args(info) \
        + metadata_args(strip_metadata) + [dst]
    enc = popen(enc_args, stdin=__import__("subprocess").PIPE, stderr=errf, bufsize=fb * 2)
    total = info.frames
    n = 0
    ok = False
    try:
        while True:
            if cancel and cancel():
                raise Cancelled()
            buf = dec.stdout.read(fb)
            if not buf or len(buf) < fb:
                break
            arr = bytearray(buf)
            y, u, vv = _split_yuv(arr, W, H)
            y = y.copy(); u = u.copy(); vv = vv.copy()
            sc = float(scales[min(n, len(scales) - 1)]) if scales is not None else 1.0
            remover(y, u, vv, sc)
            enc.stdin.write(y.tobytes())
            enc.stdin.write(u.tobytes())
            enc.stdin.write(vv.tobytes())
            n += 1
            if n % 5 == 0:
                base = 0.15 if scales is not None else 0.0
                progress(min(0.99, base + (1 - base) * n / max(total, 1)), f"Khung {n}/{total}")
        ok = True
    finally:
        try:
            dec.stdout.close()
        except Exception:
            pass
        dec.kill()
        dec.wait()
        try:
            enc.stdin.close()
        except Exception:
            pass
        if not ok:
            enc.kill()
        rc = enc.wait()
        errf.seek(0)
        err = errf.read().decode("utf-8", "replace")
        errf.close()
        if not ok:
            try:
                os.remove(dst)
            except OSError:
                pass
    if rc != 0:
        try:
            os.remove(dst)
        except OSError:
            pass
        raise RuntimeError("ffmpeg lỗi: " + err.strip()[-800:])
    if n == 0:
        raise RuntimeError("Không giải mã được khung hình nào")
    progress(1.0, "Xong")
    return geo
