"""Image reading/writing with optional metadata stripping."""
from __future__ import annotations

import io
import os

import numpy as np
from PIL import Image, ImageOps, PngImagePlugin

try:
    import pillow_heif  # type: ignore

    pillow_heif.register_heif_opener()
    HAS_HEIF = True
except Exception:  # pragma: no cover
    HAS_HEIF = False

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".jfif", ".heic", ".heif", ".gif", ".webp",
              ".bmp", ".tif", ".tiff")

Image.MAX_IMAGE_PIXELS = 400_000_000


class LoadedImage:
    def __init__(self):
        self.frames: list[np.ndarray] = []   # RGB uint8 frames
        self.alphas: list[np.ndarray | None] = []
        self.durations: list[int] = []
        self.loop = 0
        self.format = ""
        self.exif: bytes | None = None
        self.icc: bytes | None = None
        self.xmp = None
        self.png_text: dict = {}
        self.orientation = 1
        self.dpi = None

    @property
    def rgb(self):
        return self.frames[0]


def load(path: str) -> LoadedImage:
    li = LoadedImage()
    with Image.open(path) as im:
        li.format = (im.format or "").upper()
        li.icc = im.info.get("icc_profile")
        li.dpi = im.info.get("dpi")
        try:
            ex = im.getexif()
            li.orientation = int(ex.get(0x0112, 1) or 1)
            li.exif = im.info.get("exif") or (ex.tobytes() if len(ex) else None)
        except Exception:
            li.exif = im.info.get("exif")
        li.xmp = im.info.get("xmp") or im.info.get("XML:com.adobe.xmp")
        if li.format == "PNG":
            li.png_text = {k: v for k, v in getattr(im, "text", {}).items()}
        n = getattr(im, "n_frames", 1)
        li.loop = im.info.get("loop", 0)
        for i in range(n):
            if n > 1:
                im.seek(i)
            fr = im
            if fr.mode in ("RGBA", "LA", "PA") or (fr.mode == "P" and "transparency" in fr.info):
                fr = fr.convert("RGBA")
                arr = np.asarray(fr)
                li.frames.append(np.ascontiguousarray(arr[..., :3]))
                a = arr[..., 3]
                li.alphas.append(None if a.min() == 255 else np.ascontiguousarray(a))
            else:
                if fr.mode in ("I;16", "I;16B", "I", "F"):
                    a = np.asarray(fr).astype(np.float32)
                    a = (a / max(a.max(), 1) * 255).clip(0, 255).astype(np.uint8)
                    fr = Image.fromarray(a)
                li.frames.append(np.ascontiguousarray(np.asarray(fr.convert("RGB"))))
                li.alphas.append(None)
            li.durations.append(int(im.info.get("duration", 100) or 100))
    return li


def _strip_orientation(exif_bytes):
    try:
        ex = Image.Exif()
        ex.load(exif_bytes)
        if 0x0112 in ex:
            ex[0x0112] = 1
        return ex.tobytes()
    except Exception:
        return None


def save(li: LoadedImage, frames: list[np.ndarray], path: str, strip_metadata: bool,
         jpeg_quality: int = 95):
    ext = os.path.splitext(path)[1].lower()
    ims = []
    for i, fr in enumerate(frames):
        a = li.alphas[i] if i < len(li.alphas) else None
        if a is not None and ext not in (".jpg", ".jpeg", ".jfif", ".bmp"):
            im = Image.fromarray(np.dstack([fr, a]), "RGBA")
        else:
            im = Image.fromarray(fr, "RGB")
        ims.append(im)
    # Without metadata the EXIF orientation flag is gone: bake it into pixels.
    if strip_metadata and li.orientation not in (0, 1):
        ims = [ImageOps.exif_transpose(_with_orientation(im, li.orientation)) for im in ims]

    kw = {}
    if li.icc:
        kw["icc_profile"] = li.icc  # colour profile is not "hidden" data; keep colours right
    if not strip_metadata:
        if li.exif:
            kw["exif"] = li.exif
        if li.dpi:
            kw["dpi"] = li.dpi
    tmp = path + ".part"
    try:
        if ext in (".jpg", ".jpeg", ".jfif"):
            if not strip_metadata and li.xmp:
                kw["xmp"] = li.xmp if isinstance(li.xmp, bytes) else str(li.xmp).encode("utf-8")
            ims[0].convert("RGB").save(tmp, "JPEG", quality=jpeg_quality, subsampling=0,
                                       optimize=True, **kw)
        elif ext == ".png":
            if not strip_metadata and li.png_text:
                pi = PngImagePlugin.PngInfo()
                for k, v in li.png_text.items():
                    try:
                        pi.add_itxt(k, str(v))
                    except Exception:
                        pass
                kw["pnginfo"] = pi
            ims[0].save(tmp, "PNG", compress_level=6, **kw)
        elif ext == ".webp":
            if not strip_metadata and li.xmp:
                kw["xmp"] = li.xmp if isinstance(li.xmp, (bytes, str)) else str(li.xmp)
            if len(ims) > 1:
                ims[0].save(tmp, "WEBP", save_all=True, append_images=ims[1:], quality=95,
                            method=4, duration=li.durations, loop=li.loop, **kw)
            else:
                ims[0].save(tmp, "WEBP", quality=95, method=4, **kw)
        elif ext in (".heic", ".heif"):
            if not HAS_HEIF:
                raise RuntimeError("Thiếu bộ mã HEIF")
            if not strip_metadata and li.xmp:
                kw["xmp"] = li.xmp if isinstance(li.xmp, bytes) else str(li.xmp).encode("utf-8")
            ims[0].save(tmp, "HEIF", quality=95, **kw)
        elif ext == ".gif":
            pal = [im.convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT,
                                              dither=Image.Dither.NONE) for im in ims]
            dur = li.durations if len(pal) > 1 else (li.durations[0] if li.durations else 100)
            gkw = {"loop": li.loop or 0, "duration": dur, "disposal": 2}
            pal[0].save(tmp, "GIF", save_all=len(pal) > 1, append_images=pal[1:], optimize=False, **gkw)
        elif ext in (".tif", ".tiff"):
            ims[0].save(tmp, "TIFF", compression="tiff_lzw", **kw)
        elif ext == ".bmp":
            ims[0].convert("RGB").save(tmp, "BMP")
        else:
            ims[0].save(tmp, "PNG", **kw)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _with_orientation(im: Image.Image, orientation: int) -> Image.Image:
    ex = Image.Exif()
    ex[0x0112] = orientation
    im.info["exif"] = ex.tobytes()
    im.getexif()[0x0112] = orientation
    return im


# --------------------------------------------------------------------------
# Metadata scan (what hidden data does a file carry?)
# --------------------------------------------------------------------------

_MARKERS = [
    ("C2PA (Content Credentials)", [b"c2pa", b"caBX", b"jumb", b"jumd"]),
    ("XMP", [b"<x:xmpmeta", b"http://ns.adobe.com/xap/", b"XML:com.adobe.xmp"]),
    ("IPTC: trainedAlgorithmicMedia (gắn nhãn AI)", [b"trainedAlgorithmicMedia", b"compositeSynthetic"]),
    ("Chuỗi 'Google' / 'Gemini'", [b"Google", b"Gemini", b"Made with Google AI"]),
    ("EXIF", [b"Exif\x00\x00", b"eXIf"]),
]


def scan_metadata(path: str, limit: int = 8 * 1024 * 1024) -> list[str]:
    """Rough scan for hidden metadata blocks. Returns human readable findings."""
    found = []
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            head = f.read(min(size, limit))
            tail = b""
            if size > limit:
                f.seek(max(limit, size - 2 * 1024 * 1024))
                tail = f.read()
        data = head + tail
    except OSError:
        return found
    for name, keys in _MARKERS:
        if any(k in data for k in keys):
            found.append(name)
    return found
