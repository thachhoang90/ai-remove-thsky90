"""MODULE 2 — Đóng logo hàng loạt chạy nền (Bulk Watermarking).

BulkWatermarkWorker là một QThread: toàn bộ việc đọc ảnh, chèn logo và ghi
file diễn ra ở luồng nền nên giao diện không bị "đơ". Tiến độ được báo về
giao diện qua tín hiệu (signal) của Qt. Có thể ngưng giữa chừng.
"""
from __future__ import annotations

import os
import time
import traceback

import numpy as np
from PySide6.QtCore import QThread, Signal

from .. import batch
from .. import imagefile as IF
from . import ops


class BulkWatermarkWorker(QThread):
    progress = Signal(int, int, str)        # (đã xong, tổng, tên tệp)
    log = Signal(str)
    finished_all = Signal(int, int, float)  # (thành công, lỗi, số giây)

    def __init__(self, files, out_dir, logo_rgba: np.ndarray, settings: ops.WatermarkSettings,
                 fmt: str = "keep", quality: int = 92):
        super().__init__()
        self.files = files                  # danh sách (đường dẫn, thư mục gốc)
        self.out_dir = out_dir
        self.logo = logo_rgba
        self.st = settings
        self.fmt = fmt                      # "keep" | "jpg" | "png" | "webp"
        self.quality = quality
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        t0 = time.time()
        ok = fail = 0
        total = len(self.files)
        for i, (src, base) in enumerate(self.files):
            if self._cancel:
                self.log.emit("— Đã ngưng đóng logo hàng loạt")
                break
            name = os.path.basename(src)
            self.progress.emit(i, total, name)
            try:
                li = IF.load(src)
                rgb = li.rgb
                alpha = li.alphas[0] if li.alphas else None
                out = ops.compose_watermark(rgb, self.logo, self.st)
                ext = os.path.splitext(src)[1].lower().lstrip(".")
                fmt = ext if self.fmt == "keep" else self.fmt
                if fmt in ("jpeg", "jfif"):
                    fmt = "jpg"
                if fmt not in ("jpg", "png", "webp"):
                    fmt = "png"                 # HEIC/GIF/BMP... -> PNG để không mất chất lượng
                dst = batch.output_path(src, base, self.out_dir, "." + fmt)
                data = ops.encode(out, fmt, self.quality, alpha if fmt != "jpg" else None)
                with open(dst, "wb") as fh:
                    fh.write(data)
                ok += 1
                self.log.emit(f"✔ {name} → {os.path.basename(dst)}")
            except Exception as ex:  # noqa: BLE001
                traceback.print_exc()
                fail += 1
                self.log.emit(f"✖ {name}: {ex}")
        self.progress.emit(total if not self._cancel else ok + fail, total, "")
        self.finished_all.emit(ok, fail, time.time() - t0)
