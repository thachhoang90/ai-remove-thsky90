"""File-level operations shared by the GUI and the command line."""
from __future__ import annotations

import os

import numpy as np

from . import engine as E
from . import imagefile as IF
from .video import VIDEO_EXTS, process_video


def list_files(paths, exts, recursive=True):
    """Expand files/folders into (file, base_folder_or_None) pairs."""
    out = []
    seen = set()
    for p in paths:
        p = os.path.abspath(p)
        if os.path.isdir(p):
            if recursive:
                walker = os.walk(p)
            else:
                walker = [(p, [], [f for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))])]
            for root, dirs, files in walker:
                dirs.sort()
                for f in sorted(files):
                    if f.lower().endswith(exts) and not f.endswith(".part"):
                        fp = os.path.join(root, f)
                        if fp not in seen:
                            seen.add(fp)
                            out.append((fp, p))
        elif os.path.isfile(p) and p.lower().endswith(exts):
            if p not in seen:
                seen.add(p)
                out.append((p, None))
    return out


def output_path(src: str, base: str | None, out_dir: str, new_ext: str | None = None,
                unique: bool = True) -> str:
    if base:
        rel = os.path.relpath(src, base)
        rel = os.path.join(os.path.basename(base.rstrip("\\/")), rel)
    else:
        rel = os.path.basename(src)
    stem, ext = os.path.splitext(rel)
    if new_ext:
        ext = new_ext
    dst = os.path.join(out_dir, stem + ext)
    if os.path.abspath(dst) == os.path.abspath(src):
        dst = os.path.join(out_dir, stem + "_clean" + ext)
    if unique and os.path.exists(dst):
        i = 1
        while os.path.exists(os.path.join(out_dir, f"{stem} ({i}){ext}")):
            i += 1
        dst = os.path.join(out_dir, f"{stem} ({i}){ext}")
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    return dst


def process_image_file(src: str, dst: str, opts: E.Options | None = None,
                       strip_metadata: bool = True, geo: E.Geometry | None = None):
    """Returns the Geometry that was used (geo.found tells if a logo was removed)."""
    opts = opts or E.Options()
    li = IF.load(src)
    frames = li.frames
    if geo is None:
        if len(frames) == 1:
            out, geo = E.process_image(frames[0], opts)
            outs = [out]
        else:
            geo = None
            for fr in frames[:min(len(frames), 6)]:
                g = E.detect_image(fr, opts)
                if g.found:
                    geo = g
                    break
            geo = geo or g
            outs = [apply_geometry(fr, geo, opts) for fr in frames]
    else:
        outs = [apply_geometry(fr, geo, opts) for fr in frames]
    IF.save(li, outs, dst, strip_metadata)
    return geo


def apply_geometry(rgb: np.ndarray, geo: E.Geometry, opts: E.Options):
    return E.apply_image(rgb, geo, opts)


def process_video_file(src, dst, opts=None, strip_metadata=True, progress=None,
                       cancel=None, log=None, geo=None):
    return process_video(src, dst, opts, strip_metadata, progress, cancel, log, geo=geo)


def is_video(path: str) -> bool:
    return path.lower().endswith(VIDEO_EXTS)


def is_image(path: str) -> bool:
    return path.lower().endswith(IF.IMAGE_EXTS)
