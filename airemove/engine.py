"""Core watermark engine for AI Remove - thsky90.

Model
-----
Gemini stamps a white, semi-transparent sparkle on top of the picture with
ordinary alpha compositing:

    observed = (1 - a) * original + a * white

where ``a = k * A(x, y)``.  ``A`` is the calibrated alpha map of the sparkle
(captured on a pure black background) and ``k`` is a strength factor that
absorbs opacity changes, resizing and compression.  Once the exact position,
size and strength are known the original pixels are recovered analytically:

    original = (observed - a * white) / (1 - a)

The engine therefore spends most of its effort on *finding* the sparkle with
sub-pixel accuracy:

1. coarse multi-scale normalised cross-correlation in the bottom-right corner
2. coordinate-descent refinement of (x, y, size, strength, template variant)
   that minimises the gradient energy left behind after removal
3. validation: the removal must clearly reduce the watermark signature,
   otherwise nothing is touched (no false positives on clean images)
4. a thin "seam" pass that smooths the 1-2 px outline where lossy
   compression (JPEG/WebP/H.264) has blurred the sparkle edge.

The same functions work on RGB images and on the Y/U/V planes of video.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, asdict

import cv2
import numpy as np

cv2.setNumThreads(max(1, (os.cpu_count() or 2) - 1))

SHARP_SIGMA = 0.5
# plausible strength range (k = 1 is the classic 50% mark, ~0.58 the 2026 one)
K_MIN, K_MAX = 0.3, 1.45
# search range of k per template (v2 is already a low-opacity capture)
K_RANGE = {"v1": (0.15, 1.9), "n48": (0.15, 1.9), "v2": (0.3, 1.2)}

# --------------------------------------------------------------------------
# Templates
# --------------------------------------------------------------------------

def _asset_dir() -> str:
    base = getattr(sys, "_MEIPASS", None)
    if base:
        p = os.path.join(base, "airemove", "assets")
        if os.path.isdir(p):
            return p
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def _load_alpha(name: str) -> np.ndarray:
    path = os.path.join(_asset_dir(), name)
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.float32).max(axis=2) / 255.0


class Templates:
    """Calibrated alpha maps (96 px reference + native 48 px capture)."""

    _inst = None

    def __init__(self):
        self.variants = {
            "v1": _load_alpha("gemini_alpha_96.png"),
            "v2": _load_alpha("gemini_alpha_96_v2.png"),
        }
        self.native48 = _load_alpha("gemini_alpha_48.png")
        # unit-peak copies used for correlation-based search
        self.unit = {k: v / max(v.max(), 1e-6) for k, v in self.variants.items()}

    def template(self, variant: str) -> np.ndarray:
        if variant == "n48":
            return self.native48
        return self.variants.get(variant, self.variants["v1"])

    @classmethod
    def get(cls) -> "Templates":
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst


def render_alpha(T: np.ndarray, x0: float, y0: float, s: float,
                 bx: int, by: int, bw: int, bh: int) -> np.ndarray:
    """Render template ``T`` placed at (x0, y0) with size ``s`` (float, image
    pixels) onto the integer grid of box (bx, by, bw, bh).  4x supersampling
    followed by area averaging gives correct anti-aliasing at any scale."""
    n = T.shape[0]
    ss = 4
    a = n / (ss * s)
    bxo = (bx + 0.5 / ss - x0) * n / s - 0.5
    byo = (by + 0.5 / ss - y0) * n / s - 0.5
    M = np.array([[a, 0, bxo], [0, a, byo]], dtype=np.float64)
    fine = cv2.warpAffine(T, M, (bw * ss, bh * ss),
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return cv2.resize(fine, (bw, bh), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------

@dataclass
class Geometry:
    x: float            # top-left x of the sparkle box (image px, float)
    y: float            # top-left y
    size: float         # side length of the sparkle box
    strength: float = 1.0
    variant: str = "v1"
    sharp: float = 0.0   # edge overshoot (resampling / encoder sharpening)
    chroma_blur: float = 0.0  # 4:2:0 chroma smoothing of lossy files (JPEG/WebP)
    score: float = 0.0  # detection confidence (NCC 0..1)
    gain: float = 0.0   # relative energy reduction achieved by removal
    found: bool = False
    note: str = ""

    def box(self):
        return (self.x, self.y, self.size, self.size)

    def to_dict(self):
        return asdict(self)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.float32)
    f = img.astype(np.float32)
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def _grad_energy(g: np.ndarray, w: np.ndarray) -> float:
    gx = g[:, 1:] - g[:, :-1]
    gy = g[1:, :] - g[:-1, :]
    return float((gx * gx * w[:, 1:]).sum() + (gy * gy * w[1:, :]).sum())


def _box_with_margin(x, y, s, W, H, margin):
    bx = int(np.floor(x - margin))
    by = int(np.floor(y - margin))
    ex = int(np.ceil(x + s + margin))
    ey = int(np.ceil(y + s + margin))
    bx, by = max(0, bx), max(0, by)
    ex, ey = min(W, ex), min(H, ey)
    return bx, by, max(1, ex - bx), max(1, ey - by)


def harmonic_fill(P: np.ndarray, mask: np.ndarray, iters: int = 400) -> np.ndarray:
    """Smoothest (Laplace) interpolation of P inside ``mask`` from its border.

    Two-level Jacobi solve: coarse pass on a half-size grid, then refine."""
    P = P.astype(np.float32)
    m = mask.astype(bool)
    if not m.any():
        return P.copy()
    init = cv2.inpaint(np.clip(P, 0, 255).astype(np.uint8), m.astype(np.uint8), 3,
                       cv2.INPAINT_TELEA).astype(np.float32)
    h, w = P.shape
    if h >= 24 and w >= 24:
        ps = cv2.resize(init, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        ms = cv2.resize(m.astype(np.uint8), (w // 2, h // 2), interpolation=cv2.INTER_NEAREST) > 0
        for _ in range(iters):
            avg = 0.25 * (np.roll(ps, 1, 0) + np.roll(ps, -1, 0) + np.roll(ps, 1, 1) + np.roll(ps, -1, 1))
            ps[ms] = avg[ms]
        up = cv2.resize(ps, (w, h), interpolation=cv2.INTER_LINEAR)
        init[m] = up[m]
    B = init
    for _ in range(iters // 2):
        avg = 0.25 * (np.roll(B, 1, 0) + np.roll(B, -1, 0) + np.roll(B, 1, 1) + np.roll(B, -1, 1))
        B[m] = avg[m]
    return B


def reverse_blend(patch: np.ndarray, alpha: np.ndarray, white) -> np.ndarray:
    """Analytic inverse of alpha compositing with a white logo."""
    a = np.clip(alpha, 0.0, 0.92)
    if patch.ndim == 3:
        a3 = a[..., None]
        w = np.asarray(white, dtype=np.float32).reshape(1, 1, -1)
    else:
        a3 = a
        w = float(white if np.isscalar(white) else white[0])
    return (patch - a3 * w) / (1.0 - a3)


# --------------------------------------------------------------------------
# Coarse detection
# --------------------------------------------------------------------------

def corner_roi(W: int, H: int, frac: float = 0.36, min_px: int = 260):
    rw = min(W, max(min_px, int(W * frac)))
    rh = min(H, max(min_px, int(H * frac)))
    return W - rw, H - rh, rw, rh


def ncc_maps(gray: np.ndarray, sizes, roi, variant="v1"):
    """Return {size: NCC map} for the template at each integer size."""
    T = Templates.get().unit[variant]
    rx, ry, rw, rh = roi
    sub = gray[ry:ry + rh, rx:rx + rw].astype(np.float32)
    out = {}
    for s in sizes:
        if s >= rw or s >= rh:
            continue
        t = cv2.resize(T, (s, s), interpolation=cv2.INTER_AREA)
        if t.std() < 1e-6:
            continue
        out[s] = cv2.matchTemplate(sub, t, cv2.TM_CCOEFF_NORMED)
    return out


def coarse_detect(gray: np.ndarray, roi=None, smin=16, smax=None, variant="v1",
                  prior=None):
    """Multi-scale search; returns (score, size, x, y) of the best match.

    ``prior`` (x, y, s) restricts the search to a neighbourhood (manual mode).
    """
    H, W = gray.shape
    if prior is not None:
        px, py, ps = prior
        m = max(6, int(ps * 0.35))
        roi = (max(0, int(px - m)), max(0, int(py - m)),
               min(W, int(px + ps + m)) - max(0, int(px - m)),
               min(H, int(py + ps + m)) - max(0, int(py - m)))
        smin = max(8, int(ps * 0.7))
        smax = int(ps * 1.3) + 1
    if roi is None:
        roi = corner_roi(W, H)
    rx, ry, rw, rh = roi
    if smax is None:
        smax = int(min(rw, rh) * 0.8)
        smax = min(smax, max(140, int(min(W, H) * 0.12)))
    # coarse sizes then local refinement of size
    step = 2 if smax - smin > 40 else 1
    best = (-1.0, 0, 0, 0)
    for s, r in ncc_maps(gray, range(smin, smax + 1, step), roi, variant).items():
        _, mx, _, loc = cv2.minMaxLoc(r)
        if mx > best[0]:
            best = (float(mx), s, rx + loc[0], ry + loc[1])
    if step > 1 and best[1]:
        s0 = best[1]
        for s, r in ncc_maps(gray, range(max(smin, s0 - 2), s0 + 3), roi, variant).items():
            _, mx, _, loc = cv2.minMaxLoc(r)
            if mx > best[0]:
                best = (float(mx), s, rx + loc[0], ry + loc[1])
    return best


# --------------------------------------------------------------------------
# Refinement
# --------------------------------------------------------------------------

class _Problem:
    """Gradient-energy objective on a fixed working box.

    ``frames`` is a list of float32 patches (H, W[, C]) that share one
    watermark geometry (1 for an image, several for video)."""

    def __init__(self, frames, box, white, support_geom):
        self.frames = frames
        self.box = box  # bx, by, bw, bh
        self.white = white
        bx, by, bw, bh = box
        x, y, s = support_geom
        T = Templates.get().variants["v1"]
        sup = render_alpha(T, x, y, s, bx, by, bw, bh) > 0.01
        k = max(3, int(round(s * 0.08)))
        sup = cv2.dilate(sup.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8))
        self.w = sup.astype(np.float32)
        self.grays = [to_gray(f) if f.ndim == 3 else f for f in frames]
        self._cache = {}

    def alpha(self, variant, x, y, s):
        bx, by, bw, bh = self.box
        key = (variant, round(x, 3), round(y, 3), round(s, 3))
        a = self._cache.get(key)
        if a is None:
            tm = Templates.get()
            if variant == "n48":
                a = render_alpha(tm.native48, x, y, s, bx, by, bw, bh)
            else:
                a = render_alpha(tm.variants[variant], x, y, s, bx, by, bw, bh)
            if len(self._cache) > 4000:
                self._cache.clear()
            self._cache[key] = a
        return a

    def energy(self, variant, x, y, s, k, beta=0.0):
        A = self.alpha(variant, x, y, s)
        if beta:
            A = A + beta * (A - cv2.GaussianBlur(A, (0, 0), SHARP_SIGMA))
        a = np.clip(A * k, 0, 0.92)
        wl = self.white if np.isscalar(self.white) else float(np.mean(self.white))
        tot = 0.0
        for g in self.grays:
            r = (g - a * wl) / (1.0 - a)
            tot += _grad_energy(r, self.w)
        return tot

    def base_energy(self):
        tot = 0.0
        for g in self.grays:
            tot += _grad_energy(g, self.w)
        return tot


def _golden(f, lo, hi, iters=18):
    gr = 0.6180339887
    a, b = lo, hi
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
    return (c, fc) if fc < fd else (d, fd)


def refine(prob: _Problem, x, y, s, variants=("v1", "v2"), rounds=3,
           fix_geometry=False, fit_beta=True):
    """Coordinate descent (golden-section per coordinate) over position, size,
    strength and edge sharpness, for each template variant.

    Returns (energy, variant, x, y, s, k, beta) of the best variant."""
    results = []
    for var in variants:
        if var == "n48" and abs(s - 48) > 3:
            continue
        p = {"x": float(x), "y": float(y), "s": float(s), "k": 1.0, "b": 0.0}

        def E(**kw):
            q = dict(p)
            q.update(kw)
            return prob.energy(var, q["x"], q["y"], q["s"], q["k"], q["b"])

        klo, khi = K_RANGE.get(var, (0.15, 1.9))
        p["k"], e = _golden(lambda v: E(k=v), klo, khi, 16)
        if fit_beta:
            best = (e, 0.0)
            for bb in (-0.3, 0.3, 0.6, 1.0):
                kk, ee = _golden(lambda v: E(k=v, b=bb), max(klo, p["k"] * 0.7),
                                 min(khi, p["k"] * 1.3), 10)
                if ee < best[0]:
                    best = (ee, bb)
                    p["k"] = kk
            p["b"] = best[1]
        for r in range(rounds if not fix_geometry else 1):
            f = 0.5 ** r
            if not fix_geometry:
                span = 1.2 * f
                p["x"], _ = _golden(lambda v: E(x=v), p["x"] - span, p["x"] + span, 10)
                p["y"], _ = _golden(lambda v: E(y=v), p["y"] - span, p["y"] + span, 10)
                ss = max(1.0, p["s"] * 0.03) * f
                p["s"], _ = _golden(lambda v: E(s=v), p["s"] - ss, p["s"] + ss, 10)
            p["k"], _ = _golden(lambda v: E(k=v), max(klo, p["k"] * (1 - 0.2 * f)),
                                min(khi, p["k"] * (1 + 0.2 * f)), 10)
            if fit_beta:
                p["b"], _ = _golden(lambda v: E(b=v), max(-0.5, p["b"] - 0.5 * f),
                                    min(1.6, p["b"] + 0.5 * f), 10)
        e = E()
        results.append((e, var, p["x"], p["y"], p["s"], p["k"], p["b"]))
    results.sort(key=lambda t: t[0])
    return results[0]


# --------------------------------------------------------------------------
# Removal on a patch
# --------------------------------------------------------------------------

def seam_mask(alpha_unit: np.ndarray, width: float = 1.0) -> np.ndarray:
    """Weight map (0..1) on the sparkle outline where compression blurs edges."""
    gx = cv2.Sobel(alpha_unit, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(alpha_unit, cv2.CV_32F, 0, 1, ksize=3)
    g = np.sqrt(gx * gx + gy * gy)
    if g.max() <= 0:
        return np.zeros_like(alpha_unit)
    g = g / g.max()
    m = np.clip((g - 0.12) / 0.4, 0, 1)
    if width > 1:
        k = int(round(width))
        m = cv2.dilate(m, np.ones((2 * k - 1, 2 * k - 1), np.uint8))
    return cv2.GaussianBlur(m, (3, 3), 0)


def clean_seam(restored: np.ndarray, alpha: np.ndarray, strength: float = 1.0):
    """Blend the outline toward an edge-preserving local estimate.

    After the analytic inverse the inside and outside of the sparkle are both
    correct; only a 1-2 px seam may carry residue from lossy compression.
    The seam is re-estimated from its neighbours with a masked inpaint and
    blended in proportionally to how strong the template edge is there."""
    if strength <= 0:
        return restored
    au = alpha / max(alpha.max(), 1e-6)
    m = seam_mask(au) * strength
    if m.max() <= 0:
        return restored
    hard = (m > 0.25).astype(np.uint8)
    src = np.clip(restored, 0, 255).astype(np.uint8)
    if src.ndim == 3 and src.shape[2] == 1:
        src = src[..., 0]
    inp = cv2.inpaint(src, hard, 2, cv2.INPAINT_TELEA).astype(np.float32)
    if restored.ndim == 3 and inp.ndim == 2:
        inp = inp[..., None]
    mm = np.clip(m, 0, 1)
    if restored.ndim == 3:
        mm = mm[..., None]
    return restored * (1 - mm) + inp * mm


def inpaint_region(img: np.ndarray, mask: np.ndarray, radius: int = 5):
    """Fallback for non-standard logos: classic inpainting (uint8 image)."""
    src = np.clip(img, 0, 255).astype(np.uint8)
    out = cv2.inpaint(src, mask.astype(np.uint8), radius, cv2.INPAINT_TELEA)
    return out


# --------------------------------------------------------------------------
# Public API: still images (RGB uint8)
# --------------------------------------------------------------------------

@dataclass
class Options:
    mode: str = "auto"            # "auto" | "manual"
    manual_box: tuple | None = None   # (x, y, w, h) in image pixels
    method: str = "alpha"         # "alpha" (reverse blend) | "inpaint"
    seam: float = 1.0             # 0 = off, 1 = normal
    min_score: float = 0.30       # minimum NCC for auto detection
    min_gain: float = 0.12        # minimum energy reduction to accept
    extra: dict = field(default_factory=dict)


def detect_image(rgb: np.ndarray, opts: Options | None = None) -> Geometry:
    """Find the sparkle on a still image.

    Since v2.3 this is the outline-voting detector of ``airemove.detect``: it
    looks at the brightness step a semi-transparent white overlay must cause
    all around its outline, so it also works when the logo sits on text, a
    face or busy artwork.  (Videos keep using coarse_detect + fit_geometry on
    time-averaged frames, see video.py.)"""
    from . import detect as D
    opts = opts or Options()
    prior = None
    if opts.mode == "manual" and opts.manual_box:
        bx, by, bw, bh = opts.manual_box
        s = float(min(bw, bh))
        prior = (bx + (bw - s) / 2, by + (bh - s) / 2, s)
    geo = D.detect(rgb, prior)
    if prior is not None and not geo.found:
        # no sparkle inside the user's box -> the box itself will be filled in
        geo.x, geo.y, geo.size = prior
        geo.note = "manual-inpaint"
    return geo


def ring_roughness(g: np.ndarray, support: np.ndarray) -> float:
    """Mean high-frequency energy in a ring around the logo (texture level)."""
    sup = support.astype(np.uint8)
    k = 3
    inner = cv2.dilate(sup, np.ones((2 * k + 1, 2 * k + 1), np.uint8)) > 0
    outer = cv2.dilate(sup, np.ones((6 * k + 1, 6 * k + 1), np.uint8)) > 0
    ring = outer & ~inner
    if ring.sum() < 10:
        return 99.0
    hp = np.abs(g - cv2.GaussianBlur(g, (0, 0), 1.5))
    return float(hp[ring].mean())


def estimate_strength_dc(g: np.ndarray, A: np.ndarray, white: float, base=None):
    """Strength that makes the restored interior match the smooth extension of
    its surroundings (what the eye compares).  Returns (k, trust 0..1)."""
    au = A / max(A.max(), 1e-6)
    sup = cv2.dilate((au > 0.01).astype(np.uint8), np.ones((5, 5), np.uint8))
    rough = ring_roughness(g, au > 0.01)
    B = harmonic_fill(g, sup) if base is None else base
    inner = cv2.erode((au > 0.6).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    if inner.sum() < 12:
        return 1.0, 0.0
    gi, Ai, Bi = g[inner], A[inner], B[inner]
    # contrast available: on bright backgrounds a white logo is nearly invisible
    contrast = float(np.median(white - Bi))
    if contrast < 25:
        return 1.0, 0.0

    def f(k):
        a = np.clip(Ai * k, 0, 0.92)
        return float(np.median((gi - a * white) / (1 - a) - Bi))

    lo, hi = 0.05, 1.9
    if f(lo) <= 0:
        return lo, 0.0
    if f(hi) >= 0:
        return hi, 0.0
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    k = 0.5 * (lo + hi)
    trust = 1.0 / (1.0 + (rough / 3.0) ** 2)
    trust *= min(1.0, contrast / 80.0)
    return k, float(trust)


def fit_geometry(frames, coarse, white, W, H, score, variants=("v1", "v2", "n48"),
                 min_gain=0.12, fit_sharp=True):
    x, y, s = coarse
    margin = max(4, int(s * 0.25))
    box = _box_with_margin(x, y, s, W, H, margin)
    bx, by, bw, bh = box
    patches = [f[by:by + bh, bx:bx + bw] for f in frames]
    prob = _Problem(patches, box, white, (x, y, s))
    e0 = prob.base_energy()
    e, var, cx, cy, cs, kg, beta = refine(prob, x, y, s, variants=variants, fit_beta=fit_sharp)
    gain = 1.0 - e / max(e0, 1e-6)
    need = min_gain if score >= 0.5 else max(min_gain, 0.3)
    found = gain >= need and K_MIN <= kg <= K_MAX
    geo = Geometry(cx, cy, cs, strength=kg, variant=var, sharp=float(beta), score=float(score),
                   gain=float(gain), found=bool(found), note="ok" if found else "weak")
    if not found:
        return geo
    # --- strength: the level-based estimate is authoritative on smooth
    # backgrounds (it is what the eye compares), the edge-based one elsewhere
    tm = Templates.get()
    A = render_alpha(tm.template(var), cx, cy, cs, bx, by, bw, bh)
    wl = white if np.isscalar(white) else float(np.mean(white))
    ks, ws = [], []
    for g in prob.grays[:24]:
        kd, tr = estimate_strength_dc(g, A, wl)
        if tr > 0:
            ks.append(kd)
            ws.append(tr)
    if ks:
        kd = float(np.average(ks, weights=ws))
        tr = float(min(1.0, np.mean(ws)))
        if tr >= 0.3 and kd < K_MIN * 0.8:
            # smooth background and no brightness lift: there is no watermark
            geo.found = False
            geo.note = "no-lift"
            return geo
        if 0.75 * kg <= kd <= 1.3 * kg:
            geo.strength = tr * kd + (1 - tr) * kg
        geo.strength = float(np.clip(geo.strength, *K_RANGE.get(var, (0.15, 1.9))))
    # --- chroma: lossy codecs store colour at half resolution ---------------
    if frames[0].ndim == 3 and frames[0].shape[2] == 3:
        geo.chroma_blur = fit_chroma_blur(patches[0], geo, box)
    return geo


def rgb_to_ycc(p):
    r, g, b = p[..., 0], p[..., 1], p[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return y, cb, cr


def ycc_to_rgb(y, cb, cr):
    r = y + 1.402 * (cr - 128.0)
    g = y - 0.344136 * (cb - 128.0) - 0.714136 * (cr - 128.0)
    b = y + 1.772 * (cb - 128.0)
    return np.stack([r, g, b], axis=-1)


def _chroma_alpha(a, sigma):
    if sigma <= 0:
        return a
    return cv2.GaussianBlur(a, (0, 0), sigma)


def fit_chroma_blur(patch, geo, box) -> float:
    bx, by, bw, bh = box
    a = alpha_for(geo, bx, by, bw, bh)
    _, cb, cr = rgb_to_ycc(patch.astype(np.float32))
    w = cv2.dilate((a > 0.005).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
    best = None
    for sg in (0.0, 0.5, 0.8, 1.1, 1.5):
        ac = np.clip(_chroma_alpha(a, sg), 0, 0.95)
        e = sum(_grad_energy((c - ac * 128.0) / (1 - ac), w) for c in (cb, cr))
        if best is None or e < best[0]:
            best = (e, sg)
    return float(best[1])


def alpha_for(geo: Geometry, bx, by, bw, bh) -> np.ndarray:
    tm = Templates.get()
    A = render_alpha(tm.template(geo.variant), geo.x, geo.y, geo.size, bx, by, bw, bh)
    if geo.sharp:
        A = A + geo.sharp * (A - cv2.GaussianBlur(A, (0, 0), SHARP_SIGMA))
    return np.clip(A * geo.strength, 0, 0.95)


def remove_with_geometry(img: np.ndarray, geo: Geometry, white=255.0,
                         seam: float = 1.0, method: str = "alpha") -> np.ndarray:
    """Return a copy of ``img`` (uint8, HxW or HxWxC) with the sparkle removed."""
    H, W = img.shape[:2]
    margin = max(4, int(geo.size * 0.25))
    bx, by, bw, bh = _box_with_margin(geo.x, geo.y, geo.size, W, H, margin)
    out = img.copy()
    patch = img[by:by + bh, bx:bx + bw].astype(np.float32)
    if method == "inpaint" or not geo.found:
        tm = Templates.get()
        a = render_alpha(tm.variants["v1"], geo.x, geo.y, geo.size, bx, by, bw, bh)
        mask = a > 0.02
        k = max(1, int(round(geo.size * 0.06)))
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8))
        rad = max(3, int(geo.size * 0.12))
        res = inpaint_region(patch, mask, rad)
        if res.ndim == 2 and out.ndim == 3:
            res = res[..., None]
        out[by:by + bh, bx:bx + bw] = res.reshape(out[by:by + bh, bx:bx + bw].shape)
        return out
    a = alpha_for(geo, bx, by, bw, bh)
    if patch.ndim == 3 and patch.shape[2] == 3 and np.isscalar(white):
        # luminance at full resolution, chroma with the codec-smoothed alpha
        y, cb, cr = rgb_to_ycc(patch)
        ac = np.clip(_chroma_alpha(a, geo.chroma_blur), 0, 0.95)
        y = (y - a * white) / (1 - a)
        cb = (cb - ac * 128.0) / (1 - ac)
        cr = (cr - ac * 128.0) / (1 - ac)
        res = ycc_to_rgb(y, cb, cr)
    else:
        wv = white
        if patch.ndim == 3 and np.isscalar(white):
            wv = [white] * patch.shape[2]
        res = reverse_blend(patch, a, wv)
    if seam > 0:
        res = clean_seam(res, a, seam)
    out[by:by + bh, bx:bx + bw] = np.clip(np.rint(res), 0, 255).astype(img.dtype)
    return out


def residual_score(img: np.ndarray, geo: Geometry):
    """How visible is whatever is left inside the logo area?  Compares the
    interior with a smooth continuation of the surroundings.  Returns
    (mean colour difference in 0..255 levels, background smoothness trust)."""
    H, W = img.shape[:2]
    margin = max(4, int(geo.size * 0.25))
    bx, by, bw, bh = _box_with_margin(geo.x, geo.y, geo.size, W, H, margin)
    A = render_alpha(Templates.get().variants["v1"], geo.x, geo.y, geo.size, bx, by, bw, bh)
    sup = cv2.dilate((A > 0.01).astype(np.uint8), np.ones((5, 5), np.uint8))
    inner = cv2.erode((A > 0.3).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    if inner.sum() < 10:
        return 0.0, 0.0
    P = img[by:by + bh, bx:bx + bw].astype(np.float32)
    chans = [P[..., c] for c in range(P.shape[2])] if P.ndim == 3 else [P]
    diffs = []
    for c in chans:
        B = harmonic_fill(c, sup, 250)
        diffs.append(np.abs(np.median((c - B)[inner])))
    rough = ring_roughness(to_gray(P) if P.ndim == 3 else P, A > 0.01)
    trust = 1.0 / (1.0 + (rough / 3.0) ** 2)
    return float(np.mean(diffs)), float(trust)


def quality_guard(src: np.ndarray, out: np.ndarray, geo: Geometry, limit: float = 5.0):
    """If the analytic result still shows a visible star on a smooth background
    (palette-quantised GIF, very heavy JPEG...), use inpainting instead."""
    res, trust = residual_score(out, geo)
    if not ((trust >= 0.45 and res >= limit) or res >= 12.0):
        return out
    alt = remove_with_geometry(src, Geometry(geo.x, geo.y, geo.size, found=False), method="inpaint")
    res2, _ = residual_score(alt, geo)
    if res2 < res * (0.6 if trust >= 0.45 else 0.35):
        geo.note = "inpaint-fallback"
        return alt
    return out


def apply_image(rgb: np.ndarray, geo: Geometry | None, opts: Options | None = None) -> np.ndarray:
    """Remove the logo of a still image whose geometry is already known.

    * sparkle found          -> exact reverse blend (airemove.restore.remove):
                                the artwork under the logo is recovered, not painted
    * manual box, no sparkle -> the box is filled in from its surroundings
    * method "inpaint"       -> always fill in (box in manual mode, star shape otherwise)"""
    from . import restore as R
    opts = opts or Options()
    manual = opts.mode == "manual" and bool(opts.manual_box)
    if geo is not None and geo.found and opts.method != "inpaint":
        return R.remove(rgb, geo, opts.seam)
    if manual:
        return R.inpaint(rgb, R.box_mask(rgb.shape, opts.manual_box))
    if geo is not None and geo.found:
        return R.inpaint(rgb, R.star_mask(rgb.shape, geo))
    return rgb.copy()


def process_image(rgb: np.ndarray, opts: Options | None = None):
    """Detect and remove. Returns (result_rgb, geometry)."""
    opts = opts or Options()
    if opts.mode == "manual" and opts.manual_box and opts.method == "inpaint":
        bx, by, bw, bh = opts.manual_box
        s = float(min(bw, bh))
        geo = Geometry(bx + (bw - s) / 2, by + (bh - s) / 2, s, found=False, note="manual-inpaint")
        return apply_image(rgb, geo, opts), geo
    geo = detect_image(rgb, opts)
    # v3.1 "Áp dụng vị trí xóa logo thủ công": pictures of the same size made
    # by the same tool carry the logo at exactly the same place.  When the
    # sparkle cannot be confirmed inside the user's box, reuse the geometry
    # that was fitted on the template picture instead of smearing the box.
    fb = opts.extra.get("fallback_geo")
    if fb and tuple(fb[0]) == (rgb.shape[1], rgb.shape[0]):
        t = Geometry(**fb[1])
        agrees = (geo.found and abs(geo.x - t.x) <= 2.5 and abs(geo.y - t.y) <= 2.5
                  and abs(geo.size - t.size) <= 0.05 * t.size)
        if not agrees:                       # nothing (or something else) found in the box
            geo = t
            geo.found, geo.note = True, "template"
    return apply_image(rgb, geo, opts), geo
