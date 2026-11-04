"""Locate the ffmpeg binary and read basic stream information."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass

_FFMPEG = None

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def _candidates():
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    roots = []
    if getattr(sys, "frozen", False):
        roots.append(os.path.dirname(sys.executable))
    base = getattr(sys, "_MEIPASS", None)
    if base:
        roots.append(base)
    here = os.path.dirname(os.path.abspath(__file__))
    app_dir = os.path.dirname(here)
    # Windows bundle layout: <root>/app/airemove, <root>/runtime/python.exe, <root>/ffmpeg/
    roots += [here, app_dir, os.path.dirname(app_dir),
              os.path.dirname(os.path.dirname(os.path.abspath(sys.executable)))]
    for r in roots:
        for sub in ("", "ffmpeg", os.path.join("ffmpeg", "bin"), "bin"):
            yield os.path.join(r, sub, exe)


def ffmpeg_path() -> str:
    global _FFMPEG
    if _FFMPEG:
        return _FFMPEG
    for c in _candidates():
        if os.path.isfile(c):
            _FFMPEG = c
            return c
    try:
        import imageio_ffmpeg  # type: ignore
        p = imageio_ffmpeg.get_ffmpeg_exe()
        if p and os.path.isfile(p):
            _FFMPEG = p
            return p
    except Exception:
        pass
    p = shutil.which("ffmpeg")
    if p:
        _FFMPEG = p
        return p
    raise FileNotFoundError("Không tìm thấy ffmpeg. Hãy cài lại phần mềm.")


def run(args, **kw):
    kw.setdefault("creationflags", CREATE_NO_WINDOW)
    return subprocess.run(args, **kw)


def popen(args, **kw):
    kw.setdefault("creationflags", CREATE_NO_WINDOW)
    return subprocess.Popen(args, **kw)


@dataclass
class VideoInfo:
    width: int = 0
    height: int = 0
    fps: float = 25.0
    fps_str: str = "25"
    duration: float = 0.0
    vcodec: str = ""
    pix_fmt: str = ""
    color_range: str = "tv"
    colorspace: str = ""
    color_primaries: str = ""
    color_trc: str = ""
    has_audio: bool = False
    acodec: str = ""
    bitrate_kbps: int = 0
    rotation: int = 0

    @property
    def frames(self) -> int:
        return max(1, int(round(self.duration * self.fps)))


_COLOR_WORDS = {
    "bt709", "smpte170m", "bt470bg", "bt2020nc", "bt2020c", "bt2020", "smpte240m",
    "bt470m", "fcc", "ycgco", "iec61966-2-1", "gamma22", "gamma28", "linear",
    "smpte2084", "arib-std-b67", "film", "unknown",
}


def probe(path: str) -> VideoInfo:
    p = run([ffmpeg_path(), "-hide_banner", "-i", path],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
    txt = p.stderr
    info = VideoInfo()
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", txt)
    if m:
        info.duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    m = re.search(r"bitrate:\s*(\d+)\s*kb/s", txt)
    if m:
        info.bitrate_kbps = int(m.group(1))
    vline = None
    for line in txt.splitlines():
        if "Video:" in line and vline is None and "attached pic" not in line:
            vline = line
        if "Audio:" in line and not info.has_audio:
            info.has_audio = True
            am = re.search(r"Audio:\s*([\w\-]+)", line)
            info.acodec = am.group(1) if am else ""
    if vline is None:
        raise ValueError("Tệp không có luồng video")
    m = re.search(r"Video:\s*([\w\-]+)", vline)
    info.vcodec = m.group(1) if m else ""
    m = re.search(r",\s*(\d{2,5})x(\d{2,5})[\s,\[]", vline)
    if m:
        info.width, info.height = int(m.group(1)), int(m.group(2))
    m = re.search(r"([\d.]+)\s*(?:k)?\s*fps", vline)
    tb = re.search(r"([\d.]+)\s*(k)?\s*tbr", vline)
    fps = None
    if m:
        fps = float(m.group(1))
    elif tb:
        fps = float(tb.group(1)) * (1000 if tb.group(2) else 1)
    if fps and fps > 0:
        info.fps = fps
        # common NTSC rates as exact fractions
        for num, den in ((24000, 1001), (30000, 1001), (60000, 1001)):
            if abs(fps - num / den) < 0.01:
                info.fps_str = f"{num}/{den}"
                break
        else:
            info.fps_str = ("%.3f" % fps).rstrip("0").rstrip(".")
    m = re.search(r"Video:[^,]*,\s*([a-z0-9_]+)(\(([^)]*)\))?", vline)
    if m:
        info.pix_fmt = m.group(1)
        if m.group(3):
            parts = [x.strip() for x in m.group(3).split(",")]
            if "pc" in parts or info.pix_fmt.startswith("yuvj"):
                info.color_range = "pc"
            cs = [x for x in parts if x not in ("tv", "pc", "progressive", "top first",
                                                 "bottom first", "top coded first (swapped)")]
            if cs:
                trip = cs[0].split("/")
                if len(trip) == 3:
                    info.colorspace, info.color_primaries, info.color_trc = trip
                elif len(trip) == 1 and trip[0] in _COLOR_WORDS:
                    info.colorspace = info.color_primaries = info.color_trc = trip[0]
        elif info.pix_fmt.startswith("yuvj"):
            info.color_range = "pc"
    m = re.search(r"rotate\s*:\s*(-?\d+)", txt) or re.search(r"rotation of (-?[\d.]+) degrees", txt)
    if m:
        info.rotation = int(float(m.group(1)))
    if info.width <= 0 or info.height <= 0:
        raise ValueError("Không đọc được kích thước video")
    return info


def encoders() -> set:
    p = run([ffmpeg_path(), "-hide_banner", "-encoders"], capture_output=True, text=True,
            encoding="utf-8", errors="replace")
    out = set()
    for line in p.stdout.splitlines():
        m = re.match(r"\s*[VAS][\.FSXBD]{5}\s+(\S+)", line)
        if m:
            out.add(m.group(1))
    return out
