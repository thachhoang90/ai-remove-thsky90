"""Render the app icon (PNG + multi-size ICO)."""
import os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

def render(size=1024):
    S = size
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    # rounded-square background with vertical gradient
    grad = np.zeros((S, S, 4), np.uint8)
    t = np.linspace(0, 1, S)[:, None]
    c1 = np.array([67, 56, 202]); c2 = np.array([30, 27, 75])
    col = (c1 * (1 - t) + c2 * t)
    grad[..., :3] = col[:, None, :].repeat(S, 1)[:, :, :3] if col.ndim == 2 else col
    grad[..., 3] = 255
    bg = Image.fromarray(grad, "RGBA")
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, S - 1, S - 1], radius=int(S * 0.22), fill=255)
    img.paste(bg, (0, 0), mask)
    # sparkle (4-point star with concave sides)
    def star(cx, cy, r, n=200):
        pts = []
        import math
        for i in range(4):
            a0 = i * math.pi / 2
            tip = (cx + r * math.cos(a0), cy + r * math.sin(a0))
            nxt = (cx + r * math.cos(a0 + math.pi / 2), cy + r * math.sin(a0 + math.pi / 2))
            for k in range(n):
                u = k / n
                # quadratic bezier with control at centre
                x = (1 - u) ** 2 * tip[0] + 2 * (1 - u) * u * cx + u * u * nxt[0]
                y = (1 - u) ** 2 * tip[1] + 2 * (1 - u) * u * cy + u * u * nxt[1]
                pts.append((x, y))
        return pts
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.polygon(star(S * 0.47, S * 0.47, S * 0.33), fill=(255, 255, 255, 255))
    glow = layer.filter(ImageFilter.GaussianBlur(S * 0.03))
    img = Image.alpha_composite(img, Image.fromarray((np.asarray(glow) * [1, 1, 1, 0.6]).astype(np.uint8)))
    img = Image.alpha_composite(img, layer)
    # eraser slash
    d2 = ImageDraw.Draw(img)
    w = int(S * 0.075)
    d2.line([(S * 0.20, S * 0.80), (S * 0.80, S * 0.20)], fill=(244, 63, 94, 255), width=w)
    d2.ellipse([S * 0.66, S * 0.66, S * 0.86, S * 0.86], fill=(34, 197, 94, 255))
    # check mark
    d2.line([(S * 0.705, S * 0.765), (S * 0.75, S * 0.81), (S * 0.825, S * 0.715)], fill=(255, 255, 255, 255), width=int(S * 0.028), joint="curve")
    return img

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    big = render(1024)
    big.resize((256, 256), Image.LANCZOS).save(os.path.join(out, "app_icon.png"))
    big.save(os.path.join(out, "app_icon.ico"), sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print("ok")
