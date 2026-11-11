"""Removing the sparkle from a still image once its geometry is known (v2.3).

The logo is semi-transparent, so the artwork underneath is still in the file:

    observed = (1 - a) * original + a * white     =>     original = (observed - a * white) / (1 - a)

Nothing is painted or guessed - text, faces and textures under the logo come
back exactly as far as the file still holds them.  Three refinements deal
with what happens to real files (resizing, lossy re-compression, colour
conversion, pictures that went through Gemini more than once):

1. colour planes    - lossy formats store colour at half resolution: colour is
                      restored with an alpha blurred by the amount that fits.
2. surroundings     - when a faint dull "diamond" is left, the interior is
                      aligned with what lies just outside the outline.  Only
                      differences that are statistically certain are applied,
                      so files that follow the formula stay exact.
3. outline          - a 1-2 px seam can remain where the logo edge was
                      resampled.  Seam pixels that stick out are replaced by
                      an interpolation ACROSS the seam, but only where the
                      artwork is calm there; strokes of the artwork that
                      cross or touch the outline are kept untouched.

``inpaint`` is the last resort for marks that are not a Gemini sparkle.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from . import detect as D
from . import engine as E

A_MAX = 0.92           # alpha is never trusted beyond this (division by 1 - a)


# --------------------------------------------------------------------------
# 1. colour planes
# --------------------------------------------------------------------------

def _band(T: np.ndarray, s: float) -> np.ndarray:
    u = T / max(float(T.max()), 1e-6)
    gx = cv2.Sobel(u, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(u, cv2.CV_32F, 0, 1, ksize=3)
    edge = (np.hypot(gx, gy) > 0.25).astype(np.uint8)
    r = max(2, int(round(s * 0.05)))
    return cv2.dilate(edge, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))


def _neutral_cost(P: np.ndarray, a: np.ndarray, white: float, band: np.ndarray) -> float:
    """Robust (L1) outline cost that neither rewards nor punishes strength on
    pairs with equal alpha - see detect._Outline."""
    om = 1.0 - a
    sq = np.sqrt(om)
    B = (P - a * white) / om
    dx = np.abs((B[:, 1:] - B[:, :-1]) * sq[:, 1:] * sq[:, :-1]) * (band[:, 1:] | band[:, :-1])
    dy = np.abs((B[1:, :] - B[:-1, :]) * sq[1:, :] * sq[:-1, :]) * (band[1:, :] | band[:-1, :])
    return float(dx.sum() + dy.sum())


def fit_chroma_sigma(patch: np.ndarray, a: np.ndarray, band: np.ndarray) -> float:
    """Blur of the colour planes (0 for lossless files, ~1 for 4:2:0 codecs)."""
    _, cb, cr = E.rgb_to_ycc(patch)
    best = (None, 0.0)
    for sg in (0.0, 0.5, 0.8, 1.1, 1.5, 2.0):
        ac = np.clip(cv2.GaussianBlur(a, (0, 0), sg) if sg else a, 0, 0.95)
        c = _neutral_cost(cb, ac, 128.0, band) + _neutral_cost(cr, ac, 128.0, band)
        if best[0] is None or c < best[0] * 0.985:           # prefer the sharper one on ties
            best = (c, sg)
    return float(best[1])


# --------------------------------------------------------------------------
# 2. matching the surroundings
# --------------------------------------------------------------------------
#
# Some files do not follow the blending formula exactly: they were converted
# to another colour space or a palette after Gemini drew the logo, or the
# picture went through Gemini before and already carries a painted remnant of
# an older logo underneath the new one.  The restored interior then differs
# slightly from its surroundings and a faint "diamond" stays visible.
#
# The picture itself tells how large that difference is: just across the
# outline the artwork continues, so a restored pixel just inside should look
# like its neighbour just outside.  The differences measured there are spread
# over the interior (nearest parts of the outline and similar colours count
# most) and subtracted.  Details under the logo keep their shape - only their
# colour level is aligned.

R_MAX = 45.0          # larger differences are edges of the artwork, not a colour error


def boundary_samples(res: np.ndarray, obs: np.ndarray, T: np.ndarray, pl: float,
                     steps=(3, 5), cap: int = 700):
    """Pairs (p just inside the flat part of the logo, q just outside).

    Only pairs in calm surroundings are kept - the artwork is flat one step
    further in and one step further out and the two sides differ moderately -
    so that an edge of the artwork running between p and q is not mistaken
    for a colour error.

    Returns (colour of the restored pixel p, restored p - observed q, position of p)."""
    h, w = T.shape
    cols, dif, pos = [], [], []
    for d in steps:
        for uy, ux in D.DIRS:
            for sg in (1, -1):
                dy, dx = uy * d * sg, ux * d * sg
                y0, y1 = max(0, dy, -dy, -2 * dy), h - max(0, dy, 2 * dy, -dy)
                x0, x1 = max(0, dx, -dx, -2 * dx), w - max(0, dx, 2 * dx, -dx)
                if y1 - y0 < 2 or x1 - x0 < 2:
                    continue

                def sl(n):
                    return (slice(y0 + n * dy, y1 + n * dy), slice(x0 + n * dx, x1 + n * dx))

                sel = (T[sl(-1)] >= 0.85 * pl) & (T[sl(0)] >= 0.85 * pl) & (T[sl(1)] <= 0.01) & (T[sl(2)] <= 0.01)
                if not sel.any():
                    continue
                p2, pp, q, q2 = res[sl(-1)][sel], res[sl(0)][sel], obs[sl(1)][sel], obs[sl(2)][sel]
                r = pp - q
                ok = (np.abs(r).max(1) <= R_MAX) & (np.abs(q - q2).max(1) <= 12.0) & (np.abs(pp - p2).max(1) <= 20.0)
                if not ok.any():
                    continue
                yy, xx = np.mgrid[y0:y1, x0:x1]
                cols.append(pp[ok])
                dif.append(r[ok])
                pos.append(np.stack([xx[sel][ok], yy[sel][ok]], 1))
    if not cols:
        return None
    C, R, P = np.concatenate(cols), np.concatenate(dif), np.concatenate(pos)
    if len(C) > cap:                               # deterministic thinning
        idx = np.linspace(0, len(C) - 1, cap).astype(np.int64)
        C, R, P = C[idx], R[idx], P[idx]
    return C.astype(np.float32), R.astype(np.float32), P.astype(np.float32)


def colour_offsets(res: np.ndarray, T: np.ndarray, pl: float, size: float, samples,
                   sigma_c: float = 30.0, dead: float = 1.0) -> np.ndarray:
    """Per-pixel colour difference of the restored patch to its surroundings
    (to be subtracted), HxWx3.  Zero wherever the evidence is not conclusive."""
    out = np.zeros_like(res)
    if samples is None or len(samples[0]) < 24:
        return out
    C, R, P = samples
    ys, xs = np.where(T > 0.02 * pl)
    if len(ys) == 0:
        return out
    X = res[ys, xs]                                                    # N x 3
    sigma_s = max(5.0, 0.2 * size)
    w = np.zeros((len(ys), len(C)), np.float32)
    for c in range(3):
        w += (X[:, c:c + 1] - C[None, :, c]) ** 2
    w *= -0.5 / (sigma_c * sigma_c)
    w -= ((xs[:, None].astype(np.float32) - P[None, :, 0]) ** 2 +
          (ys[:, None].astype(np.float32) - P[None, :, 1]) ** 2) * (0.5 / (sigma_s * sigma_s))
    np.exp(w, out=w)
    wsum = w.sum(1)
    conf = np.clip(wsum / 3.0, 0.0, 1.0)             # enough samples nearby / of a similar colour?
    wsum = np.maximum(wsum, 1e-6)
    n_eff = wsum * wsum / np.maximum((w * w).sum(1), 1e-9)
    delta = np.zeros((len(ys), 3), np.float32)
    for c in range(3):
        r = R[:, c]
        dlt = (w @ r) / wsum                                           # weighted mean ...
        for _ in range(8):                                             # ... refined to a weighted median
            ww = w / np.maximum(np.abs(r[None, :] - dlt[:, None]), 1.0)
            dlt = (ww @ r) / np.maximum(ww.sum(1), 1e-6)
        # is the difference certain?  compare it with its own standard error
        spread = (w * np.abs(r[None, :] - dlt[:, None])).sum(1) / wsum
        se = 1.25 * spread / np.sqrt(np.maximum(n_eff, 1.0)) + 0.25
        sure = np.clip((np.abs(dlt) / se - 2.0) / 2.0, 0.0, 1.0)
        delta[:, c] = np.sign(dlt) * np.clip(np.abs(dlt) - dead, 0.0, R_MAX) * sure
    delta *= conf[:, None]
    # the difference belongs to the logo layer: fade it out with the logo's alpha
    a = np.clip(T[ys, xs] / pl, 0.0, 1.0)
    out[ys, xs] = delta * a[:, None]
    return out


# --------------------------------------------------------------------------
# 3. outline seam
# --------------------------------------------------------------------------

def seam_regions(T: np.ndarray, pl: float):
    """Split a patch into (level, seam, outside, inside) from the template alpha.

    The seam is where alpha changes quickly (the resampled logo edge); the
    flat interior and the untouched background are the two clean sides."""
    u = np.clip(T / pl, 0.0, 1.5)
    gx = cv2.Sobel(u, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(u, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    seam = ((np.hypot(gx, gy) > 0.05) | ((u > 0.03) & (u < 0.9))).astype(np.uint8)
    seam = cv2.dilate(seam, np.ones((3, 3), np.uint8))
    outside = ((u < 0.03) & (seam == 0)).astype(np.uint8)
    inside = ((u >= 0.03) & (seam == 0)).astype(np.uint8)
    return u, seam, outside, inside


def _bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return cv2.remap(img, x.astype(np.float32), y.astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


def cross_seam(res: np.ndarray, u: np.ndarray, outside: np.ndarray, inside: np.ndarray, span=None):
    """Sample the clean picture on both sides of the seam, along the outline
    normal.  Returns (value outside, value inside, weight of the outside one).

    With ``span`` the two samples are taken at that fixed distance instead of
    at the nearest clean pixels (used to measure how textured the artwork is)."""
    ub = cv2.GaussianBlur(u, (0, 0), 1.5)
    gx = cv2.Sobel(ub, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(ub, cv2.CV_32F, 0, 1, ksize=3)
    nrm = np.maximum(np.hypot(gx, gy), 1e-6)
    nx, ny = -gx / nrm, -gy / nrm                                    # outward normal
    h, w = u.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if span is None:
        d_out = cv2.distanceTransform(1 - outside, cv2.DIST_L2, 3)
        d_in = cv2.distanceTransform(1 - inside, cv2.DIST_L2, 3) if inside.any() else np.full_like(d_out, 5.0)
        to = np.clip(d_out + 0.5, 1.0, 5.0)
        ti = np.clip(d_in + 0.5, 1.0, 5.0)
    else:
        to = ti = np.full(u.shape, float(span), np.float32)
    vo = _bilinear(res, xx + nx * to, yy + ny * to)
    vi = _bilinear(res, xx - nx * ti, yy - ny * ti)
    return vo, vi, (ti / (to + ti))[..., None]


def clean_outline(res: np.ndarray, T: np.ndarray, pl: float, a: np.ndarray, strength: float = 1.0):
    """Remove what is left of the logo edge without smearing the artwork.

    A seam pixel is replaced by the straight interpolation between the clean
    picture just outside and just inside the logo edge - but only where

      * those two sides agree (the artwork is calm across the seam, so no
        stroke, letter or facial line can be lost), and
      * the pixel sticks out more than the artwork's own texture does
        (measured on control rings next to the seam).

    Where the artwork has an edge of its own at the seam, the restored pixel
    is kept as it is - the edge hides whatever is left.

    Returns (patch, share of seam pixels changed)."""
    u, seam, outside, inside = seam_regions(T, pl)
    n_seam = int(seam.sum())
    if n_seam < 8 or strength <= 0:
        return res, 0.0
    vo, vi, wo = cross_seam(res, u, outside, inside)
    pred = vo * wo + vi * (1 - wo)
    dev = np.abs(res - pred).max(axis=2)
    sides = np.abs(vo - vi).max(axis=2)
    # texture level of the artwork next to the seam (same kind of interpolation)
    d_o = cv2.distanceTransform(1 - outside, cv2.DIST_L2, 3)[seam > 0]
    co, ci, _ = cross_seam(res, u, outside, inside, span=float(np.clip(np.median(d_o) + 0.5, 1.5, 4.0)))
    ctrl_dev = np.abs(res - 0.5 * (co + ci)).max(axis=2)
    near = cv2.dilate(seam, np.ones((11, 11), np.uint8)) > 0
    far = cv2.dilate(seam, np.ones((5, 5), np.uint8)) == 0
    ctrl = ctrl_dev[near & far]
    tex = float(np.median(ctrl)) if ctrl.size > 20 else 2.0
    calm = np.clip((20.0 + 2.0 * tex - sides) / 10.0, 0.0, 1.0)       # 1: sides agree, 0: edge of the artwork
    # is there a seam at all?  on files that follow the formula the seam pixels
    # deviate no more than the artwork next to them - then nothing is touched
    sm = dev[(seam > 0) & (calm > 0.5)]
    if sm.size < 12:
        return res, 0.0
    ctrl_calm = ctrl_dev[near & far & (np.abs(co - ci).max(axis=2) <= 15.0 + 2.0 * tex)]
    excess = (float(np.percentile(sm, 90)) + 1.0) / (float(np.percentile(ctrl_calm, 90)) + 1.0) \
        if ctrl_calm.size > 20 else 2.0
    gate = float(np.clip((excess - 1.25) / 0.5, 0.0, 1.0))
    if gate <= 0:
        return res, 0.0
    strength = strength * gate
    tau = 1.5 * tex + 1.5
    out_of_line = np.clip((dev - tau) / max(tau, 1.0), 0.0, 1.0)
    # a thin stroke of the artwork crossing the seam also "sticks out" - but its
    # colour continues in the clean picture right next to it, a remnant's does not
    clean = (seam == 0)
    nearest = np.full(dev.shape, 1e6, np.float32)
    h, w = dev.shape
    for oy in range(-4, 5):
        for ox in range(-4, 5):
            if oy * oy + ox * ox > 18 or (oy == 0 and ox == 0):
                continue
            ya, yb = max(0, oy), h + min(0, oy)
            xa, xb = max(0, ox), w + min(0, ox)
            dst = (slice(ya - oy, yb - oy), slice(xa - ox, xb - ox))
            src = (slice(ya, yb), slice(xa, xb))
            dcol = np.abs(res[dst] - res[src]).max(axis=2)
            dcol[~clean[src]] = 1e6
            np.minimum(nearest[dst], dcol, out=nearest[dst])
    orphan = np.clip((nearest - 0.4 * dev) / (0.2 * dev + 1.0), 0.0, 1.0)
    # a remnant of the logo edge is an alpha error of at most ~0.15: it cannot
    # move a pixel further than that share of its headroom to white.  Anything
    # beyond is artwork (a letter, a highlight) and stays.
    bound = 0.15 * (255.0 - pred) / (1.0 - np.minimum(a, 0.6))[..., None] + 6.0
    over = (np.abs(res - pred) / bound).max(axis=2)
    plausible = np.clip((1.5 - over) / 0.5, 0.0, 1.0)
    wgt = (calm * out_of_line * orphan * plausible * (seam > 0) *
           float(np.clip(strength, 0, 1))).astype(np.float32)
    changed = float((wgt > 0.05).sum() / n_seam)
    if changed <= 0:
        return res, 0.0
    wgt = wgt[..., None]
    return res * (1 - wgt) + pred * wgt, changed


# --------------------------------------------------------------------------
# Public: remove / inpaint
# --------------------------------------------------------------------------

def _model_alpha(geo: E.Geometry, box) -> np.ndarray:
    """Alpha model for strength 1 (template + edge sharpness), not clipped."""
    A = E.render_alpha(E.Templates.get().template(geo.variant), geo.x, geo.y, geo.size, *box)
    if geo.sharp:
        A = A + geo.sharp * (A - cv2.GaussianBlur(A, (0, 0), E.SHARP_SIGMA))
    return np.maximum(A, 0.0)


def remove(rgb: np.ndarray, geo: E.Geometry, seam: float = 1.0, match: bool = True,
           info: dict | None = None) -> np.ndarray:
    """Return a copy of ``rgb`` (uint8 HxWx3) with the sparkle removed."""
    H, W = rgb.shape[:2]
    s = geo.size
    margin = max(10, int(s * 0.3))
    box = E._box_with_margin(geo.x, geo.y, s, W, H, margin)
    bx, by, bw, bh = box
    patch = rgb[by:by + bh, bx:bx + bw].astype(np.float32)
    T = E.render_alpha(E.Templates.get().template(geo.variant), geo.x, geo.y, s, *box)
    pl = D.plateau_of(geo.variant)
    k = float(geo.strength)
    a = np.clip(_model_alpha(geo, box) * k, 0.0, A_MAX)

    # ---- reverse blend; colour planes with the codec's chroma blur
    res = (patch - a[..., None] * 255.0) / (1.0 - a[..., None])
    sg = fit_chroma_sigma(patch, a, _band(T, s))
    if sg > 0:
        ab = np.clip(cv2.GaussianBlur(a, (0, 0), sg), 0, A_MAX)[..., None]
        res2 = (patch - ab * 255.0) / (1.0 - ab)
        y1, _, _ = E.rgb_to_ycc(res)
        _, cb2, cr2 = E.rgb_to_ycc(res2)
        res = E.ycc_to_rgb(y1, cb2, cr2)
    geo.chroma_blur = sg

    # ---- align the interior with the surroundings
    shift = 0.0
    if match:
        off = colour_offsets(res, T, pl, s, boundary_samples(res, patch, T, pl))
        shift = float(np.abs(off).max())
        res = res - off

    # ---- outline seam
    touched = 0.0
    if seam > 0:
        res, touched = clean_outline(res, T, pl, a, seam)
    if info is not None:
        info.update(seam_touched=touched, chroma_sigma=sg, colour_shift=shift)

    out = rgb.copy()
    out[by:by + bh, bx:bx + bw] = np.clip(np.rint(res), 0, 255).astype(np.uint8)
    return out


def residual_vote(out: np.ndarray, geo: E.Geometry) -> D.Vote:
    """Outline vote on a processed image: is any of the logo left?"""
    H, W = out.shape[:2]
    return D.vote_at(D.planes_of(out), geo.x, geo.y, geo.size, W, H, variant=geo.variant)


def box_mask(shape, box, round_frac: float = 0.18) -> np.ndarray:
    """Rounded-rectangle mask (uint8 0/255) of a user box (x, y, w, h)."""
    H, W = shape[:2]
    x, y, w, h = box
    m = np.zeros((H, W), np.uint8)
    x0, y0 = int(max(0, math.floor(x))), int(max(0, math.floor(y)))
    x1, y1 = int(min(W, math.ceil(x + w))), int(min(H, math.ceil(y + h)))
    if x1 <= x0 or y1 <= y0:
        return m
    r = int(min(x1 - x0, y1 - y0) * round_frac)
    cv2.rectangle(m, (x0 + r, y0), (x1 - r - 1, y1 - 1), 255, -1)
    cv2.rectangle(m, (x0, y0 + r), (x1 - 1, y1 - r - 1), 255, -1)
    for cx, cy in ((x0 + r, y0 + r), (x1 - r - 1, y0 + r), (x0 + r, y1 - r - 1), (x1 - r - 1, y1 - r - 1)):
        cv2.circle(m, (cx, cy), r, 255, -1)
    return m


def star_mask(shape, geo: E.Geometry, grow: float = 0.06) -> np.ndarray:
    """Mask (uint8 0/255) of the sparkle shape, slightly grown."""
    H, W = shape[:2]
    box = E._box_with_margin(geo.x, geo.y, geo.size, W, H, max(4, int(geo.size * 0.25)))
    bx, by, bw, bh = box
    a = E.render_alpha(E.Templates.get().variants["v1"], geo.x, geo.y, geo.size, *box)
    k = max(1, int(round(geo.size * grow)))
    m = cv2.dilate((a > 0.02).astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)) * 255
    full = np.zeros((H, W), np.uint8)
    full[by:by + bh, bx:bx + bw] = m
    return full


def inpaint(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill ``mask`` from its surroundings (for marks that are not a Gemini sparkle).

    Unlike ``remove`` this INVENTS the content: colours and lines flow in from
    the border of the mask (cv2.inpaint), so fine detail inside is lost.  The
    grain of the neighbourhood is matched afterwards so that the patch does
    not look unnaturally smooth on photos (flat artwork gets none)."""
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return rgb.copy()
    H, W = rgb.shape[:2]
    ext = int(max(ys.max() - ys.min(), xs.max() - xs.min())) + 1
    pad = max(16, ext // 2)
    y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad + 1)
    crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
    m = cv2.dilate((mask[y0:y1, x0:x1] > 0).astype(np.uint8), np.ones((3, 3), np.uint8))
    filled = cv2.inpaint(crop, m, max(3, int(ext * 0.08)), cv2.INPAINT_TELEA).astype(np.float32)
    # grain of the surroundings (robust: edges of the artwork do not count)
    f = crop.astype(np.float32)
    hp = (f - cv2.GaussianBlur(f, (0, 0), 1.0)).mean(axis=2)
    ring = (cv2.dilate(m, np.ones((17, 17), np.uint8)) > 0) & (cv2.dilate(m, np.ones((5, 5), np.uint8)) == 0)
    sigma = 1.4826 * float(np.median(np.abs(hp[ring]))) if ring.sum() > 50 else 0.0
    if sigma > 0.6:
        rng = np.random.default_rng(12345)
        noise = cv2.GaussianBlur(rng.normal(0.0, 1.0, m.shape).astype(np.float32), (0, 0), 0.6)
        noise *= min(sigma, 6.0) / max(float(noise.std()), 1e-6)
        filled += noise[..., None]
    soft = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.0)[..., None]
    soft = np.maximum(soft, m[..., None].astype(np.float32) * 0.0)
    res = f * (1.0 - soft) + filled * soft
    out = rgb.copy()
    out[y0:y1, x0:x1] = np.clip(np.rint(res), 0, 255).astype(np.uint8)
    return out
