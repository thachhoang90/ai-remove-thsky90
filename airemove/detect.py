"""Robust detection of the Gemini sparkle on still images (v2.3).

Why a second detector?  The first one (engine.coarse_detect + fit_geometry)
compares raw brightness with the logo shape and minimises the gradient energy
left after removal.  That works on calm backgrounds but breaks when the logo
sits on text, faces or other busy artwork: the background's own edges dominate
both scores, so the logo is missed or its strength is mis-estimated.

This module only looks at what a semi-transparent white overlay MUST do,
whatever is underneath: every pixel just inside the logo outline is lifted
towards white relative to its neighbour just outside,

    I_in = (1 - a) * B + a * 255      and      I_out = B        (B: background)

Each pair of pixels straddling the outline therefore gives its own estimate of
the logo strength k (a = k * A, A = calibrated alpha map):

    k_i = (I_p - I_q) / (A_p * (255 - I_q) - A_q * (255 - I_p))

On a real logo most pairs agree on one k inside the plausible range; anywhere
else the estimates scatter around zero.  The weighted median of the k_i is the
strength, the share of pairs agreeing with it ("inlier ratio") is the detection
confidence.  Background edges only spoil the pairs they cross, which the median
ignores - so text or a face under the logo no longer matters.

Pipeline
    1. dense search  - "step-sign" maps (does brightness rise across the outline
                       in the right direction?) correlated with the outline
                       pattern at many sizes, on an image pyramid
    2. candidates    - best peaks + the layouts Gemini is known to use
                       (margin = 2/3 x size or 2 x size from the corner)
    3. verification  - outline voting at full resolution, small local search
    4. refinement    - sub-pixel position / size / edge sharpness with a robust
                       (L1) outline cost; template variant selection
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import engine as E

# margin / size of the layouts Gemini uses (48@32, 96@64 -> 2/3; 48@96 -> 2)
PRIOR_RATIOS = (2.0 / 3.0, 2.0)
K_MIN, K_MAX = 0.28, 1.5
DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))          # (dy, dx) unit steps


# --------------------------------------------------------------------------
# Planes
# --------------------------------------------------------------------------

def planes_of(rgb: np.ndarray):
    """Luma and min(R,G,B) as float32.

    Both obey the blending model exactly (the map c -> (1-a)c + a*255 is
    monotone, so it commutes with min).  min(R,G,B) is the plane on which a
    white overlay is most visible: on saturated colours (yellow, red, blue)
    one channel is near 0 and jumps the most."""
    f = rgb.astype(np.float32)
    y = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    mn = f.min(axis=2)
    return [y, mn]


# --------------------------------------------------------------------------
# Outline voting
# --------------------------------------------------------------------------

@dataclass
class Vote:
    k: float = 0.0          # robust strength (weighted median)
    inlier: float = 0.0     # share of evidence agreeing with k
    n: int = 0              # number of usable pixel pairs
    cover: float = 0.0      # share of outline pairs with usable contrast
    quad: float = 0.0       # weakest of the four quadrants' inlier ratios

    @property
    def ok(self) -> bool:
        return K_MIN <= self.k <= K_MAX


def _shift_slices(h, w, dy, dx):
    if dx >= 0:
        return (slice(0, h - dy), slice(0, w - dx)), (slice(dy, h), slice(dx, w))
    return (slice(0, h - dy), slice(-dx, w)), (slice(dy, h), slice(0, w + dx))


def _wmedian(v: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    return float(v[o][min(len(v) - 1, int(np.searchsorted(cw, cw[-1] * 0.5)))])


def vote(planes, A: np.ndarray, d: int, white: float = 255.0, frac: float = 0.5,
         center=None) -> Vote:
    """Strength estimate from all pixel pairs ``d`` apart that straddle the
    outline of alpha map ``A`` (template alpha for k = 1, same grid as planes).

    ``center`` (cx, cy in patch coordinates) additionally reports how the
    weakest quadrant of the star agrees - a real logo shows on all four sides,
    a bright blob or a stroke of the artwork usually on one or two."""
    am = float(A.max())
    if am <= 1e-6:
        return Vote()
    h, w = A.shape
    ks, ws, qs = [], [], []
    total = 0
    den_min = 0.5 * am * 10.0            # needs >= ~10 grey levels of headroom to white
    if center is not None:
        yy, xx = np.mgrid[0:h, 0:w]
        quad_map = ((yy >= center[1]).astype(np.int8) * 2 + (xx >= center[0]).astype(np.int8))
    for uy, ux in DIRS:
        p, q = _shift_slices(h, w, uy * d, ux * d)
        Ap, Aq = A[p], A[q]
        sel = np.abs(Ap - Aq) >= frac * am
        if not sel.any():
            continue
        ap, aq = Ap[sel], Aq[sel]
        qd = quad_map[p][sel] if center is not None else None
        for I in planes:
            ip, iq = I[p][sel], I[q][sel]
            den = ap * (white - iq) - aq * (white - ip)
            okm = np.abs(den) > den_min
            total += len(den)
            if okm.any():
                ks.append((ip[okm] - iq[okm]) / den[okm])
                ws.append(np.abs(den[okm]))
                if qd is not None:
                    qs.append(qd[okm])
    if not ks:
        return Vote()
    k = np.concatenate(ks)
    wt = np.concatenate(ws)
    if len(k) < 24:
        return Vote(n=len(k))
    km = _wmedian(k, wt)
    tol = 0.2 * max(abs(km), 0.3)
    inl_mask = np.abs(k - km) < tol
    inl = float(wt[inl_mask].sum() / wt.sum())
    quad = 0.0
    if center is not None:
        qq = np.concatenate(qs)
        vals = []
        for i in range(4):
            m = qq == i
            vals.append(float(wt[m & inl_mask].sum() / wt[m].sum()) if m.any() and wt[m].sum() > 0 else 0.0)
        quad = min(vals)
    return Vote(km, inl, len(k), len(k) / max(total, 1), quad)


def profile_test(planes, A: np.ndarray, k: float, d: int, white: float = 255.0):
    """Is the brightness step confined to the outline?

    A uniform semi-transparent overlay lifts the picture by a fixed share
    everywhere inside: one step further in, and one step further out, the
    picture is as flat as the artwork itself.  Glints, sparks and soft bright
    blobs drawn by the artist keep getting brighter towards their centre (or
    carry a halo outside).  For every outline-crossing pair (p inside, q
    outside) the samples one more stride inside (p2) and outside (q2) are
    compared with the step expected from strength ``k``.

    Returns (r_in, r_out): median extra step inside / outside, as a fraction
    of the outline step.  Both are close to 0 for a real logo."""
    am = float(A.max())
    h, w = A.shape
    rin, rout, wts = [], [], []
    for uy, ux in DIRS:
        for sgn in (1, -1):
            dy, dx = uy * d * sgn, ux * d * sgn
            # index ranges so that p2 = p - e, p, q = p + e, q2 = p + 2e are all inside the patch
            y0, y1 = max(0, dy, -dy, -2 * dy), h - max(0, dy, 2 * dy, -dy)
            x0, x1 = max(0, dx, -dx, -2 * dx), w - max(0, dx, 2 * dx, -dx)
            if y1 - y0 < 2 or x1 - x0 < 2:
                continue

            def sl(k_):
                return (slice(y0 + k_ * dy, y1 + k_ * dy), slice(x0 + k_ * dx, x1 + k_ * dx))

            Ap2, Ap, Aq, Aq2 = A[sl(-1)], A[sl(0)], A[sl(1)], A[sl(2)]
            sel = (Ap >= 0.9 * am) & (Ap2 >= 0.9 * am) & (Aq <= 0.04 * am) & (Aq2 <= 0.04 * am)
            if not sel.any():
                continue
            a = Ap[sel] * k
            for I in planes:
                ip2, ip, iq, iq2 = I[sl(-1)][sel], I[sl(0)][sel], I[sl(1)][sel], I[sl(2)][sel]
                ref = a * (white - iq)
                okm = ref > 4.0
                if okm.any():
                    rin.append((ip2[okm] - ip[okm]) / ref[okm])
                    rout.append((iq[okm] - iq2[okm]) / ref[okm])
                    wts.append(ref[okm])
    if not rin:
        return 0.0, 0.0, 0
    rin, rout, wts = np.concatenate(rin), np.concatenate(rout), np.concatenate(wts)
    if len(rin) < 16:
        return 0.0, 0.0, len(rin)
    return _wmedian(rin, wts), _wmedian(rout, wts), len(rin)


def plateau_of(variant: str) -> float:
    """Alpha of the flat interior of a template (k = 1)."""
    return float(np.percentile(E.Templates.get().template(variant), 90))


def stride_for(s: float) -> int:
    return max(2, int(round(s / 16.0)))


# --------------------------------------------------------------------------
# Dense step-sign search
# --------------------------------------------------------------------------

def _step_maps(planes, d: int, c: float = 4.0):
    """For each direction: clip((P[p] - P[p + d*e]) / c, -1, 1), averaged over planes."""
    h, w = planes[0].shape
    out = []
    for uy, ux in DIRS:
        p, q = _shift_slices(h, w, uy * d, ux * d)
        D = np.zeros((h, w), np.float32)
        acc = np.zeros(planes[0][p].shape, np.float32)
        for P in planes:
            acc += np.clip((P[p] - P[q]) * (1.0 / c), -1.0, 1.0)
        D[p] = acc / len(planes)
        out.append(D)
    return out


_KERNEL_CACHE: dict = {}


def _kernels(t: int, d: int):
    """Outline sign pattern of the template at size t (list of 4 float32 kernels, norm)."""
    key = (t, d)
    if key not in _KERNEL_CACHE:
        T = E.Templates.get().variants["v1"]
        A = E.render_alpha(T, 0.0, 0.0, float(t), 0, 0, t, t)
        am = float(A.max())
        ks, norm = [], 0.0
        for uy, ux in DIRS:
            p, q = _shift_slices(t, t, uy * d, ux * d)
            K = np.zeros((t, t), np.float32)
            diff = A[p] - A[q]
            K[p] = np.where(np.abs(diff) >= 0.5 * am, np.sign(diff), 0.0)
            ks.append(K)
            norm += float(np.abs(K).sum())
        _KERNEL_CACHE[key] = (ks, max(norm, 1.0))
    return _KERNEL_CACHE[key]


def _score_map(D, t: int, d: int):
    ks, norm = _kernels(t, d)
    acc = None
    for De, K in zip(D, ks):
        r = cv2.matchTemplate(De, K, cv2.TM_CCORR)
        acc = r if acc is None else acc + r
    return acc / norm


def _resize_planes(planes, f: float):
    if f == 1.0:
        return planes
    h, w = planes[0].shape
    size = (max(8, int(round(w * f))), max(8, int(round(h * f))))
    return [cv2.resize(P, size, interpolation=cv2.INTER_AREA) for P in planes]


def dense_candidates(planes, roi, smin: float, smax: float, W: int, H: int, top: int = 10):
    """Search the ROI for outline-like step patterns at all sizes.

    Returns a list of (score, x, y, size, on_prior) in full-resolution pixels."""
    rx, ry, rw, rh = roi
    sub = [P[ry:ry + rh, rx:rx + rw] for P in planes]
    cands = []
    T_LO, T_HI = 20, 40                       # template size range searched per level
    level = 0
    while True:
        f = 0.5 ** level
        lo_full, hi_full = T_LO / f, T_HI / f
        if level == 0:
            lo_full = min(lo_full, smin)
        if lo_full > smax:
            break
        if hi_full >= smin:
            lv = _resize_planes(sub, f)
            lh, lw = lv[0].shape
            d = 2
            D = _step_maps(lv, d)
            t0 = max(12, int(np.floor(max(lo_full, smin) * f)))
            t1 = min(int(np.ceil(min(hi_full, smax) * f)), min(lw, lh) - 2)
            for t in range(t0, t1 + 1):
                S = _score_map(D, t, d)
                s_full = t / f
                # best peaks of this size
                Sc = S.copy()
                for _ in range(2):
                    _, mx, _, loc = cv2.minMaxLoc(Sc)
                    if mx <= 0.05:
                        break
                    cands.append((float(mx), rx + loc[0] / f, ry + loc[1] / f, s_full, False))
                    x0, y0 = max(0, loc[0] - t // 2), max(0, loc[1] - t // 2)
                    Sc[y0:loc[1] + t // 2 + 1, x0:loc[0] + t // 2 + 1] = -1
                # known layouts: margin = r * size from the bottom-right corner
                for r in PRIOR_RATIOS:
                    px = (W - (r + 1.0) * s_full - rx) * f
                    py = (H - (r + 1.0) * s_full - ry) * f
                    ix, iy = int(round(px)), int(round(py))
                    if 1 <= ix < S.shape[1] - 1 and 1 <= iy < S.shape[0] - 1:
                        win = S[iy - 1:iy + 2, ix - 1:ix + 2]
                        j = int(np.argmax(win))
                        cands.append((float(win.max()), rx + (ix - 1 + j % 3) / f,
                                      ry + (iy - 1 + j // 3) / f, s_full, True))
        level += 1
        if level > 5:
            break
    # keep the strongest of each kind, drop near-duplicates.  Layout candidates
    # get their own slots so that busy artwork cannot crowd the real logo out.
    def nms(items, limit):
        kept = []
        for c in sorted(items, key=lambda c: -c[0]):
            if all(abs(c[1] - o[1]) > 0.35 * c[3] or abs(c[2] - o[2]) > 0.35 * c[3] or
                   abs(np.log(c[3] / o[3])) > 0.2 for o in kept):
                kept.append(c)
            if len(kept) >= limit:
                break
        return kept

    return nms([c for c in cands if c[4]], top) + nms([c for c in cands if not c[4]], max(3, top // 2))


# --------------------------------------------------------------------------
# Verification + refinement
# --------------------------------------------------------------------------

def _patch(planes, x, y, s, W, H, margin):
    bx, by, bw, bh = E._box_with_margin(x, y, s, W, H, margin)
    return (bx, by, bw, bh), [P[by:by + bh, bx:bx + bw] for P in planes]


def vote_at(planes, x, y, s, W, H, variant="v1", d=None, beta=0.0) -> Vote:
    d = d or stride_for(s)
    box, sub = _patch(planes, x, y, s, W, H, d + 2)
    A = E.render_alpha(E.Templates.get().template(variant), x, y, s, *box)
    if beta:
        A = A + beta * (A - cv2.GaussianBlur(A, (0, 0), E.SHARP_SIGMA))
    return vote(sub, A, d, center=(x + s / 2.0 - box[0], y + s / 2.0 - box[1]))


def is_on_prior(x, y, s, W, H, tol: float = 0.12) -> bool:
    """Does (x, y, s) match a layout Gemini uses (equal right/bottom margins)?"""
    mr, mb = W - x - s, H - y - s
    if abs(mr - mb) > max(3.0, 0.12 * s):
        return False
    m = 0.5 * (mr + mb)
    return any(abs(m - r * s) <= max(3.0, tol * (r + 1) * s) for r in PRIOR_RATIOS)


def prior_ratio(x, y, s, W, H):
    """Margin/size ratio of the nearest known layout, or None if off-grid."""
    if not is_on_prior(x, y, s, W, H):
        return None
    m = 0.5 * ((W - x - s) + (H - y - s))
    return min(PRIOR_RATIOS, key=lambda r: abs(m - r * s))


def _rank(v: Vote) -> float:
    return v.inlier if v.ok else v.inlier * 0.25


def local_vote_search(planes, x, y, s, W, H, ratio=None):
    """Maximise the inlier ratio in a small neighbourhood of (x, y, s).

    On a known layout the position is tied to the size (margins are a fixed
    multiple of it), so the search runs along that line: the size follows from
    the far more precise position."""
    best = None
    if ratio is not None:
        s0 = 0.5 * ((W - x) + (H - y)) / (ratio + 1.0)
        for ds in (0.0, -0.02, 0.02, -0.04, 0.04, -0.07, 0.07):
            ss = s0 * (1.0 + ds)
            gx, gy = W - (ratio + 1.0) * ss, H - (ratio + 1.0) * ss
            for dy in (0.0, -1.0, 1.0):
                for dx in (0.0, -1.0, 1.0):
                    v = vote_at(planes, gx + dx, gy + dy, ss, W, H)
                    r = _rank(v) - 0.01 * (abs(dx) + abs(dy))
                    if best is None or r > best[0]:
                        best = (r, v, gx + dx, gy + dy, ss)
        return best
    span = max(1.5, s * 0.04)
    for ds in (0.0, -0.06, 0.06):
        ss = s * (1.0 + ds)
        cx, cy = x + (s - ss) / 2.0, y + (s - ss) / 2.0     # keep the centre
        for dy in (0.0, -span, span):
            for dx in (0.0, -span, span):
                v = vote_at(planes, cx + dx, cy + dy, ss, W, H)
                r = _rank(v)
                if best is None or r > best[0]:
                    best = (r, v, cx + dx, cy + dy, ss)
    return best


class _Outline:
    """Robust sub-pixel cost: L1 norm of "neutral" neighbour differences in a
    band around the outline.

    For neighbours p, q the restored difference (B_p - B_q) is scaled by
    sqrt((1 - a_p)(1 - a_q)).  Pairs with equal alpha (both inside or both
    outside) then do not depend on the strength at all, so background texture
    neither rewards nor punishes any k - only pairs across the outline count.
    L1 makes the minimum a (weighted) median: edges of the artwork crossing the
    outline are outliers and get ignored."""

    def __init__(self, planes, box, geom, white=255.0):
        self.planes = planes
        self.box = box
        self.white = white
        bx, by, bw, bh = box
        x, y, s = geom
        T = E.Templates.get().variants["v1"]
        A = E.render_alpha(T, x, y, s, bx, by, bw, bh)
        u = A / max(float(A.max()), 1e-6)
        gx = cv2.Sobel(u, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(u, cv2.CV_32F, 0, 1, ksize=3)
        edge = (np.hypot(gx, gy) > 0.25).astype(np.uint8)
        r = max(2, int(round(s * 0.05)))
        band = cv2.dilate(edge, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        self.wx = (band[:, 1:] | band[:, :-1]).astype(np.float32)
        self.wy = (band[1:, :] | band[:-1, :]).astype(np.float32)
        self._cache = {}

    def alpha(self, variant, x, y, s, beta):
        key = (variant, round(x, 3), round(y, 3), round(s, 3), round(beta, 3))
        a = self._cache.get(key)
        if a is None:
            a = E.render_alpha(E.Templates.get().template(variant), x, y, s, *self.box)
            if beta:
                a = a + beta * (a - cv2.GaussianBlur(a, (0, 0), E.SHARP_SIGMA))
            if len(self._cache) > 3000:
                self._cache.clear()
            self._cache[key] = a
        return a

    def cost(self, variant, x, y, s, k, beta=0.0):
        a = np.clip(self.alpha(variant, x, y, s, beta) * k, 0.0, 0.92)
        om = 1.0 - a
        sq = np.sqrt(om)
        tot = 0.0
        for P in self.planes:
            B = (P - a * self.white) / om
            dx = (B[:, 1:] - B[:, :-1]) * sq[:, 1:] * sq[:, :-1]
            dy = (B[1:, :] - B[:-1, :]) * sq[1:, :] * sq[:-1, :]
            tot += float((np.abs(dx) * self.wx).sum() + (np.abs(dy) * self.wy).sum())
        return tot


def refine(planes, x, y, s, k, W, H, variants=("v1", "n48", "v2"), rounds=3):
    """Sub-pixel refinement of position, size and edge sharpness.

    Works on the luma plane only: lossy formats store colour at half
    resolution, so colour planes blur the outline and mislead the fit.

    When the logo sits on a known Gemini layout its geometry has ONE real
    degree of freedom - the scale the picture was resized by: the margins are a
    fixed multiple of the size.  The fit then moves along that line and may
    leave it by at most half a pixel, which keeps it stable on busy artwork.

    Returns (cost_ratio, variant, x, y, s, k, beta)."""
    luma = planes[:1]
    margin = max(4, int(s * 0.25))
    box, sub = _patch(luma, x, y, s, W, H, margin)
    prob = _Outline(sub, box, (x, y, s))
    base = prob.cost("v1", x, y, s, 0.0)
    ratio = prior_ratio(x, y, s, W, H)
    tm = E.Templates.get()
    results = []
    for var in variants:
        if var == "n48" and abs(s - 48) > 3:
            continue
        p = {"x": float(x), "y": float(y), "s": float(s), "k": float(k), "b": 0.0, "ex": 0.0, "ey": 0.0}
        if ratio is not None:
            p["ex"] = float(np.clip(x - (W - (ratio + 1) * s), -0.5, 0.5))
            p["ey"] = float(np.clip(y - (H - (ratio + 1) * s), -0.5, 0.5))
        # template variants have different plateau levels: keep the plateau alpha
        # that the vote measured with the v1 template
        if var != "v1":
            p["k"] = float(k) * plateau_of("v1") / max(plateau_of(var), 1e-6)

        def geom(q):
            if ratio is None:
                return q["x"], q["y"], q["s"]
            return (W - (ratio + 1) * q["s"] + q["ex"], H - (ratio + 1) * q["s"] + q["ey"], q["s"])

        def C(**kw):
            q = dict(p)
            q.update(kw)
            gx, gy, gs = geom(q)
            return prob.cost(var, gx, gy, gs, q["k"], q["b"])

        best = (C(), 0.0)
        for bb in (-0.3, 0.3, 0.6, 1.0):
            c = C(b=bb)
            if c < best[0]:
                best = (c, bb)
        p["b"] = best[1]
        for r in range(rounds):
            f = 0.5 ** r
            ss = max(1.0, p["s"] * 0.03) * f
            p["s"], _ = E._golden(lambda v: C(s=v), p["s"] - ss, p["s"] + ss, 9)
            if ratio is None:
                span = 1.2 * f
                p["x"], _ = E._golden(lambda v: C(x=v), p["x"] - span, p["x"] + span, 9)
                p["y"], _ = E._golden(lambda v: C(y=v), p["y"] - span, p["y"] + span, 9)
            else:
                p["ex"], _ = E._golden(lambda v: C(ex=v), -0.5, 0.5, 8)
                p["ey"], _ = E._golden(lambda v: C(ey=v), -0.5, 0.5, 8)
            p["b"], _ = E._golden(lambda v: C(b=v), max(-0.5, p["b"] - 0.5 * f),
                                  min(1.6, p["b"] + 0.5 * f), 9)
            p["k"], _ = E._golden(lambda v: C(k=v), max(0.1, p["k"] * (1 - 0.15 * f)),
                                  min(1.9, p["k"] * (1 + 0.15 * f)), 9)
        gx, gy, gs = geom(p)
        cost = C()
        # Files straight from Gemini carry the logo at exactly 48 or 96 px on
        # whole pixels: prefer that exact geometry when it explains the picture
        # as well as the fitted one (recovery is then lossless).
        if ratio is not None:
            for s0 in (48.0, 96.0):
                if abs(gs - s0) <= 0.03 * s0:
                    nx, ny = round(W - (ratio + 1) * s0), round(H - (ratio + 1) * s0)
                    if abs(nx - gx) < 1.0 and abs(ny - gy) < 1.0:
                        for bb in (0.0, p["b"]):
                            cn = prob.cost(var, nx, ny, s0, p["k"], bb)
                            if cn <= cost * 1.02:
                                gx, gy, gs, cost = float(nx), float(ny), s0, cn
                                p["b"] = bb
                                break
        results.append((cost / max(base, 1e-6), var, gx, gy, gs, p["k"], p["b"]))
    # "v1" is the shape Gemini draws today; another template has to explain
    # the picture clearly better to be preferred (palette / heavily compressed
    # files otherwise flip between look-alikes)
    results.sort(key=lambda t: t[0] * (1.0 if t[1] == "v1" else 1.08))
    return results[0]


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

# Thresholds calibrated on ~200 logo / ~200 logo-free corners of real pictures
# (PNG, resized WebP, resized JPEG): 0 false alarms, ~96 % of the logos accepted.
T_ON_PRIOR = 0.28      # minimum inlier ratio at a known Gemini layout
T_OFF_PRIOR = 0.60     # ... anywhere else in the corner (needs all four sides too)
T_MANUAL = 0.22        # ... inside a box the user drew
K_AUTO_MIN, K_AUTO_MAX = 0.40, 1.18       # strength (v1 template) of the logos Gemini draws: ~0.58 and ~1.0
PLATEAU_MIN, PLATEAU_MAX = 0.20, 0.60     # the same as interior alpha: ~0.29 (2026) and ~0.50 (classic)
PLATEAU_MANUAL = (0.12, 0.78)
PROFILE_MAX = 0.45     # how much brighter the picture may still get beyond the outline (share of the step)


def profile_at(planes, x, y, s, W, H, k):
    d = stride_for(s)
    box, sub = _patch(planes, x, y, s, W, H, 3 * d + 2)
    A = E.render_alpha(E.Templates.get().variants["v1"], x, y, s, *box)
    return profile_test(sub, A, max(k, 0.05), d)


def _accept(planes, v: Vote, x, y, s, W, H, on: bool, manual: bool) -> bool:
    if not v.ok or v.n < 60:
        return False
    if manual:
        return v.inlier >= T_MANUAL
    if not (K_AUTO_MIN <= v.k <= K_AUTO_MAX):
        return False
    if on:
        if v.inlier < T_ON_PRIOR:
            return False
    elif not (s >= 24.0 and v.inlier >= T_OFF_PRIOR and v.quad >= 0.35):
        return False
    # glints, sparks and soft highlights drawn by the artist: the step is not
    # confined to the outline
    r_in, r_out, n = profile_at(planes, x, y, s, W, H, v.k)
    if n >= 16 and (r_in > PROFILE_MAX or abs(r_out) > PROFILE_MAX):
        return False
    return True


def detect(rgb: np.ndarray, prior=None) -> E.Geometry:
    """Find the sparkle. ``prior`` = (x, y, size) restricts the search to the
    neighbourhood of a box drawn by the user."""
    H, W = rgb.shape[:2]
    planes = planes_of(rgb)
    manual = prior is not None
    if manual:
        px, py, ps = prior
        m = max(8.0, ps * 0.45)
        rx, ry = int(max(0, px - m)), int(max(0, py - m))
        roi = (rx, ry, int(min(W, px + ps + m)) - rx, int(min(H, py + ps + m)) - ry)
        smin, smax = max(10.0, ps * 0.6), ps * 1.5
    else:
        roi = E.corner_roi(W, H)
        smin = 14.0
        smax = max(60.0, 0.14 * min(W, H))
    cands = dense_candidates(planes, roi, smin, smax, W, H, top=8 if not manual else 4)
    if manual:
        cands.append((1.0, float(prior[0]), float(prior[1]), float(prior[2]), False))
    best = None
    for sc, x, y, s, _ in cands:
        if s < 8 or x < 0 or y < 0 or x + s > W or y + s > H:
            continue
        ratio = prior_ratio(x, y, s, W, H)
        r = local_vote_search(planes, x, y, s, W, H, ratio)
        if r is None:
            continue
        _, v, vx, vy, vs = r
        on = is_on_prior(vx, vy, vs, W, H)
        good = _accept(planes, v, vx, vy, vs, W, H, on, manual)
        rank = (2.0 if good else 0.0) + (1.0 if on and v.ok else 0.0) + _rank(v)
        if best is None or rank > best[0]:
            best = (rank, v, vx, vy, vs, on, good)
    if best is None:
        if manual:
            return E.Geometry(prior[0], prior[1], prior[2], found=False, note="manual-no-match")
        return E.Geometry(0, 0, 0, note="not-found")
    _, v, x, y, s, on, good = best
    if not good:
        g = E.Geometry(x, y, s, strength=max(v.k, 0.0), score=float(v.inlier), found=False,
                       note="weak" if v.ok else "no-step")
        if manual:
            g.x, g.y, g.size = prior
            g.note = "manual-no-match"
        return g
    ratio_c, var, x, y, s, k, beta = refine(planes, x, y, s, v.k, W, H)
    # Final strength: vote once more at the refined geometry with the alpha
    # model that will be used for removal (template + edge sharpness), so that
    # the level step across the outline is explained exactly.
    v2 = vote_at(planes, x, y, s, W, H, variant=var, beta=beta)
    if v2.n >= 60 and v2.inlier >= 0.7 * v.inlier:
        k = v2.k
    plateau = k * plateau_of(var)
    lo, hi = PLATEAU_MANUAL if manual else (PLATEAU_MIN, PLATEAU_MAX)
    if not (lo <= plateau <= hi):
        g = E.Geometry(x, y, s, strength=float(k), variant=var, score=float(v.inlier), found=False,
                       note="implausible-strength")
        if manual:
            g.x, g.y, g.size = prior
            g.note = "manual-no-match"
        return g
    return E.Geometry(x, y, s, strength=float(k), variant=var, sharp=float(beta),
                      score=float(max(v.inlier, v2.inlier)), gain=float(1.0 - ratio_c), found=True,
                      note="ok" if on else "ok-offgrid")
