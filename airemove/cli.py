"""Command line: python -m airemove.cli <files/folders> -o <output dir> [--keep-metadata]"""
from __future__ import annotations

import argparse
import sys
import time

from . import batch
from . import engine as E
from .imagefile import IMAGE_EXTS, scan_metadata
from .video import VIDEO_EXTS


def main(argv=None):
    ap = argparse.ArgumentParser(prog="airemove", description="AI Remove - thsky90 (CLI)")
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--keep-metadata", action="store_true")
    ap.add_argument("--box", help="manual box x,y,w,h")
    ap.add_argument("--inpaint", action="store_true")
    ap.add_argument("--no-recursive", action="store_true")
    a = ap.parse_args(argv)
    opts = E.Options()
    if a.box:
        opts.mode = "manual"
        opts.manual_box = tuple(float(v) for v in a.box.split(","))
    if a.inpaint:
        opts.method = "inpaint"
    files = batch.list_files(a.inputs, IMAGE_EXTS + VIDEO_EXTS, not a.no_recursive)
    if not files:
        print("Không có tệp phù hợp", file=sys.stderr)
        return 1
    for src, base in files:
        t = time.time()
        dst = batch.output_path(src, base, a.out)
        try:
            if batch.is_video(src):
                geo = batch.process_video_file(src, dst, opts, not a.keep_metadata,
                                               progress=lambda f, s: None, log=print)
            else:
                geo = batch.process_image_file(src, dst, opts, not a.keep_metadata)
            state = "đã gỡ logo" if geo.found else ("vá vùng thủ công" if opts.manual_box else "không thấy logo")
            print(f"[OK] {src} -> {dst} | {state} | x={geo.x:.1f} y={geo.y:.1f} "
                  f"size={geo.size:.1f} k={geo.strength:.2f} ({time.time() - t:.1f}s)")
            print("     metadata gốc:", ", ".join(scan_metadata(src)) or "-",
                  "| sau xử lý:", ", ".join(scan_metadata(dst)) or "sạch")
        except Exception as ex:  # noqa: BLE001
            print(f"[LỖI] {src}: {ex}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
