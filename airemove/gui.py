"""AI Remove - thsky90 : desktop interface (PySide6)."""
from __future__ import annotations

import os
import sys
import time
import traceback
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import (QObject, QPointF, QRectF, QSettings, Qt, QThread, QTimer, Signal,
                            QUrl)
from PySide6.QtGui import (QAction, QColor, QDesktopServices, QFont, QIcon, QImage, QPainter,
                           QPainterPath, QPalette, QPen, QPixmap)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QButtonGroup, QCheckBox,
                               QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow,
                               QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
                               QRadioButton, QSizePolicy, QSlider, QSplitter, QTabWidget,
                               QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from . import __version__
from . import batch
from . import engine as E
from . import imagefile as IF
from . import video as V
from .scrollview import ScrollFrame
from .editor import ops as EO

APP_NAME = "AI Remove - thsky90"
ACCENT = "#4F46E5"


# ==========================================================================
# helpers
# ==========================================================================

def np_to_qimage(rgb: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return img.copy()


def app_icon() -> QIcon:
    base = os.path.join(E._asset_dir(), "app_icon.png")
    if os.path.isfile(base):
        return QIcon(base)
    return QIcon()


@dataclass
class Item:
    path: str
    base: str | None
    manual_box: tuple | None = None      # (x, y, w, h) image px
    geo: E.Geometry | None = None        # cached auto detection
    status: str = "Chờ xử lý"
    out: str | None = None
    size: tuple | None = None            # (W, H)
    duration: float = 0.0
    tree: QTreeWidgetItem | None = None
    template: tuple | None = None        # ((W, H), geometry dict) của ảnh mẫu - v3.1


# ==========================================================================
# Workers
# ==========================================================================

class Worker(QThread):
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, *args, **kw):
        super().__init__()
        self.fn, self.args, self.kw = fn, args, kw

    def run(self):
        try:
            self.done.emit(self.fn(*self.args, **self.kw))
        except Exception as ex:  # noqa: BLE001
            traceback.print_exc()
            self.failed.emit(str(ex))


class BatchWorker(QThread):
    file_started = Signal(int)
    file_progress = Signal(float, str)
    file_done = Signal(int, str, str)     # index, status, output path
    log = Signal(str)
    finished_all = Signal(int, int, float)

    def __init__(self, kind, jobs, out_dir, strip, method, force_mp4, wm=None):
        super().__init__()
        self.wm = wm                # (logo RGBA, WatermarkSettings) hoặc None
        self.kind = kind
        self.jobs = jobs            # list of (index, Item snapshot dict)
        self.out_dir = out_dir
        self.strip = strip
        self.method = method
        self.force_mp4 = force_mp4
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        t0 = time.time()
        ok = fail = 0
        for idx, job in self.jobs:
            if self._cancel:
                break
            self.file_started.emit(idx)
            src = job["path"]
            opts = E.Options(method=self.method)
            if job.get("manual_box"):
                opts.mode = "manual"
                opts.manual_box = job["manual_box"]
            if job.get("template"):
                opts.extra["fallback_geo"] = job["template"]
            geo = job.get("geo") if not job.get("manual_box") else None
            if geo is not None and self.method == "inpaint":
                geo = None
            try:
                new_ext = ".mp4" if (self.kind == "video" and self.force_mp4) else None
                dst = batch.output_path(src, job.get("base"), self.out_dir, new_ext)
                before = IF.scan_metadata(src)
                if self.kind == "video":
                    geo = batch.process_video_file(
                        src, dst, opts, self.strip,
                        progress=lambda f, s: self.file_progress.emit(f, s),
                        cancel=lambda: self._cancel, log=self.log.emit, geo=geo)
                else:
                    self.file_progress.emit(0.3, "Đang xử lý…")
                    post = None
                    if self.wm:
                        post = lambda o, wm=self.wm: EO.compose_watermark(o, wm[0], wm[1])  # noqa: E731
                    geo = batch.process_image_file(src, dst, opts, self.strip, geo, post=post)
                    self.file_progress.emit(1.0, "Xong")
                after = IF.scan_metadata(dst)
                if geo.found:
                    state = "Đã gỡ logo"
                    detail = (f"vị trí ({geo.x:.0f}, {geo.y:.0f}), cỡ {geo.size:.0f}px, "
                              f"độ mạnh {geo.strength:.2f}")
                    if geo.note == "template":
                        detail += ", theo vị trí của ảnh mẫu"
                elif opts.manual_box:
                    state, detail = "Đã vá vùng", "theo khung thủ công"
                else:
                    state, detail = "Không thấy logo", "giữ nguyên ảnh/khung hình"
                if self.wm and self.kind == "image":
                    detail += ", đã chèn logo riêng"
                meta = ""
                if self.strip:
                    meta = (" | metadata: " + (", ".join(before) or "không có") + " → "
                            + (", ".join(after) or "sạch"))
                self.log.emit(f"✔ {os.path.basename(src)}: {state} ({detail}){meta}")
                self.file_done.emit(idx, state, dst)
                ok += 1
            except V.Cancelled:
                self.file_done.emit(idx, "Đã dừng", "")
                break
            except Exception as ex:  # noqa: BLE001
                traceback.print_exc()
                self.log.emit(f"✖ {os.path.basename(src)}: lỗi – {ex}")
                self.file_done.emit(idx, "Lỗi", "")
                fail += 1
        self.finished_all.emit(ok, fail, time.time() - t0)


class ScanWorker(QThread):
    """v3.0 - "Quét dò tìm logo tự động": dò logo Gemini trên mọi tệp của danh
    sách (không xử lý, không ghi tệp) để chỉ những tệp có logo được tích chọn."""
    item_done = Signal(int, object)        # chỉ số, Geometry (hoặc None nếu lỗi)
    finished_all = Signal(float)

    def __init__(self, kind, todo):
        super().__init__()
        self.kind = kind
        self.todo = todo                   # list of (index, path, geo đã biết hoặc None)
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        t0 = time.time()
        for idx, path, geo in self.todo:
            if self._cancel:
                break
            try:
                if geo is None:
                    if self.kind == "image":
                        geo = E.detect_image(IF.load(path).rgb)
                    else:
                        info = V.probe(path)
                        geo = V.detect_video(V.sample_frames(path, info, 40), info)
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                geo = None
            self.item_done.emit(idx, geo)
        self.finished_all.emit(time.time() - t0)


class LogoDialog(QDialog):
    """v3.0 - "Chèn logo" ngay trong tab Ảnh: chọn một logo (PNG trong suốt là
    tốt nhất), vị trí, độ đậm, kích thước.  Logo được chèn vào ảnh kết quả sau
    khi gỡ logo Gemini, cả ở bản xem trước lẫn khi xử lý hàng loạt."""

    def __init__(self, tab: "MediaTab"):
        super().__init__(tab)
        self.tab = tab
        self.setWindowTitle("Chèn logo vào ảnh")
        self.setMinimumWidth(460)
        wm = tab.wm
        st = wm["st"]
        lay = QGridLayout(self)
        b = QPushButton("Chọn logo…")
        b.clicked.connect(self._pick)
        self.lbl = QLabel(os.path.basename(wm["path"]) or "Chưa chọn logo")
        lay.addWidget(b, 0, 0)
        lay.addWidget(self.lbl, 0, 1, 1, 2)
        self.chk = QCheckBox("Chèn logo này vào ảnh khi xem trước và khi xử lý")
        self.chk.setChecked(wm["enabled"])
        self.chk.toggled.connect(self._apply)
        lay.addWidget(self.chk, 1, 0, 1, 3)
        lay.addWidget(QLabel("Vị trí"), 2, 0)
        self.cmb = QComboBox()
        for key, (label, _, _) in EO.POSITIONS.items():
            self.cmb.addItem(label, key)
        self.cmb.setCurrentIndex(max(0, self.cmb.findData(st.position)))
        self.cmb.currentIndexChanged.connect(self._apply)
        lay.addWidget(self.cmb, 2, 1, 1, 2)
        self.sliders = {}
        rows = (("opacity", "Độ đậm (độ mờ)", 5, 100, st.opacity),
                ("scale", "Kích thước (% chiều rộng ảnh)", 2, 60, st.scale),
                ("margin", "Cách mép (% cạnh ngắn)", 0, 20, st.margin))
        for r, (key, label, lo, hi, val) in enumerate(rows, start=3):
            sl = QSlider(Qt.Horizontal)
            sl.setRange(lo, hi)
            sl.setValue(int(round(val)))
            vl = QLabel(f"{int(round(val))}%")
            sl.valueChanged.connect(lambda v, vl=vl: (vl.setText(f"{v}%"), self._apply()))
            lay.addWidget(QLabel(label), r, 0)
            lay.addWidget(sl, r, 1)
            lay.addWidget(vl, r, 2)
            self.sliders[key] = sl
        tip = QLabel("Logo được chèn sau khi gỡ logo Gemini. Bấm \"Xem thử\" để thấy kết quả trên ảnh đang chọn; "
                     "khi bấm \"Bắt đầu xử lý ảnh\", logo được chèn vào mọi ảnh đã tích chọn.")
        tip.setObjectName("Hint")
        tip.setWordWrap(True)
        lay.addWidget(tip, 6, 0, 1, 3)
        row = QHBoxLayout()
        bp = QPushButton("Xem thử trên ảnh đang chọn")
        bp.setObjectName("Secondary")
        bp.clicked.connect(self.tab.make_after)
        bc = QPushButton("Đóng")
        bc.clicked.connect(self.accept)
        row.addWidget(bp)
        row.addStretch(1)
        row.addWidget(bc)
        lay.addLayout(row, 7, 0, 1, 3)

    def _pick(self):
        path, _ = QFileDialog.getOpenFileName(self, "Chọn logo", os.path.dirname(self.tab.wm["path"]),
                                              "Ảnh logo (*.png *.webp *.jpg *.jpeg *.bmp)")
        if not path:
            return
        try:
            logo = EO.load_logo(path)
        except Exception as ex:  # noqa: BLE001
            QMessageBox.warning(self, APP_NAME, f"Không đọc được logo:\n{ex}")
            return
        self.tab.wm.update(path=path, logo=logo)
        self.lbl.setText(os.path.basename(path))
        self.chk.setChecked(True)
        self._apply()

    def _apply(self, *_):
        wm = self.tab.wm
        wm["enabled"] = self.chk.isChecked() and wm["logo"] is not None
        wm["st"] = EO.WatermarkSettings(self.cmb.currentData(), self.sliders["opacity"].value(),
                                        self.sliders["scale"].value(), self.sliders["margin"].value())
        self.tab.wm_changed()


# ==========================================================================
# Preview canvas with an editable box
# ==========================================================================

class Canvas(QWidget):
    boxChanged = Signal(float, float, float)   # x, y, size (square box)

    HANDLE = 9

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(360, 200)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.before: QImage | None = None
        self.after: QImage | None = None
        self.show_after = False
        self.box = None            # (x, y, s) in image px
        self.editable = False
        self.box_ok = True
        self.scale = 1.0
        self.off = QPointF(0, 0)
        self.fitted = True
        self._drag = None
        self.message = "Chọn tệp để xem trước"

    # ---- data -----------------------------------------------------------
    def set_images(self, before: QImage | None, after: QImage | None = None, keep_view=False):
        same_size = (self.before is not None and before is not None
                     and self.before.size() == before.size())
        self.before, self.after = before, after
        if not (keep_view and same_size):
            self.fit()
        self.update()

    def set_after(self, after: QImage | None):
        self.after = after
        self.update()

    def set_box(self, box, ok=True):
        self.box = box
        self.box_ok = ok
        self.update()

    def image_size(self):
        img = self.before
        return (img.width(), img.height()) if img is not None else (0, 0)

    # ---- view -------------------------------------------------------------
    def fit(self):
        self.fitted = True
        W, H = self.image_size()
        if not W:
            return
        s = min((self.width() - 16) / W, (self.height() - 16) / H)
        self.scale = max(0.01, s)
        self.off = QPointF((self.width() - W * self.scale) / 2, (self.height() - H * self.scale) / 2)
        self.update()

    def zoom_to_box(self, factor=3.5):
        if not self.box or self.before is None:
            return
        x, y, s = self.box
        cx, cy = x + s / 2, y + s / 2
        span = s * factor
        self.scale = min(self.width(), self.height()) / span
        self.off = QPointF(self.width() / 2 - cx * self.scale, self.height() / 2 - cy * self.scale)
        self.fitted = False
        self.update()

    def resizeEvent(self, e):
        if self.fitted:
            self.fit()
        super().resizeEvent(e)

    def to_img(self, p: QPointF) -> QPointF:
        return QPointF((p.x() - self.off.x()) / self.scale, (p.y() - self.off.y()) / self.scale)

    def to_view(self, x, y) -> QPointF:
        return QPointF(x * self.scale + self.off.x(), y * self.scale + self.off.y())

    def wheelEvent(self, e):
        if self.before is None:
            return
        f = 1.2 if e.angleDelta().y() > 0 else 1 / 1.2
        p = e.position()
        ip = self.to_img(p)
        self.scale = max(0.02, min(40.0, self.scale * f))
        self.off = QPointF(p.x() - ip.x() * self.scale, p.y() - ip.y() * self.scale)
        self.fitted = False
        self.update()

    # ---- painting ---------------------------------------------------------
    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#1E1F24"))
        # checker hint
        img = self.after if (self.show_after and self.after is not None) else self.before
        if img is None:
            p.setPen(QColor("#9CA3AF"))
            p.setFont(QFont(self.font().family(), 11))
            p.drawText(self.rect(), Qt.AlignCenter, self.message)
            return
        p.setRenderHint(QPainter.SmoothPixmapTransform, self.scale < 2.0)
        W, H = img.width(), img.height()
        target = QRectF(self.off.x(), self.off.y(), W * self.scale, H * self.scale)
        p.drawImage(target, img)
        if self.box:
            x, y, s = self.box
            tl = self.to_view(x, y)
            r = QRectF(tl.x(), tl.y(), s * self.scale, s * self.scale)
            col = QColor("#22C55E") if self.box_ok else QColor("#F59E0B")
            if self.editable:
                col = QColor("#38BDF8")
            pen = QPen(col, 2)
            pen.setCosmetic(True)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            if not self.show_after or self.editable:
                p.drawRect(r)
                # sparkle guide
                path = QPainterPath()
                cx, cy = r.center().x(), r.center().y()
                hw = r.width() / 2
                path.moveTo(cx, cy - hw)
                path.quadTo(cx, cy, cx + hw, cy)
                path.quadTo(cx, cy, cx, cy + hw)
                path.quadTo(cx, cy, cx - hw, cy)
                path.quadTo(cx, cy, cx, cy - hw)
                pen2 = QPen(col, 1, Qt.DashLine)
                pen2.setCosmetic(True)
                p.setPen(pen2)
                p.drawPath(path)
                if self.editable:
                    p.setPen(Qt.NoPen)
                    p.setBrush(col)
                    for c in (r.topLeft(), r.topRight(), r.bottomLeft(), r.bottomRight()):
                        p.drawRect(QRectF(c.x() - self.HANDLE / 2, c.y() - self.HANDLE / 2,
                                          self.HANDLE, self.HANDLE))
        # label
        p.setPen(QColor("#E5E7EB"))
        tag = "SAU KHI XỬ LÝ" if (self.show_after and self.after is not None) else "ẢNH GỐC"
        p.fillRect(QRectF(8, 8, 120, 22), QColor(0, 0, 0, 150))
        p.drawText(QRectF(8, 8, 120, 22), Qt.AlignCenter, tag)

    # ---- editing ----------------------------------------------------------
    def _hit(self, pos):
        if not self.box:
            return None
        x, y, s = self.box
        tl = self.to_view(x, y)
        r = QRectF(tl.x(), tl.y(), s * self.scale, s * self.scale)
        h = self.HANDLE + 4
        corners = {"tl": r.topLeft(), "tr": r.topRight(), "bl": r.bottomLeft(), "br": r.bottomRight()}
        for k, c in corners.items():
            if abs(pos.x() - c.x()) <= h and abs(pos.y() - c.y()) <= h:
                return k
        if r.contains(pos):
            return "move"
        return None

    def mousePressEvent(self, e):
        if self.before is None:
            return
        pos = e.position()
        if e.button() in (Qt.MiddleButton, Qt.RightButton) or not self.editable:
            self._drag = ("pan", pos, QPointF(self.off))
            self.setCursor(Qt.ClosedHandCursor)
            return
        hit = self._hit(pos)
        ip = self.to_img(pos)
        if hit is None:
            # draw a new box from this point
            self.box = (ip.x(), ip.y(), 4.0)
            hit = "br"
        self._drag = (hit, ip, tuple(self.box))

    def mouseMoveEvent(self, e):
        pos = e.position()
        if self._drag is None:
            if self.editable:
                hit = self._hit(pos)
                cur = {"move": Qt.SizeAllCursor, "tl": Qt.SizeFDiagCursor, "br": Qt.SizeFDiagCursor,
                       "tr": Qt.SizeBDiagCursor, "bl": Qt.SizeBDiagCursor}.get(hit, Qt.CrossCursor)
                self.setCursor(cur)
            else:
                self.setCursor(Qt.OpenHandCursor)
            return
        kind, start, orig = self._drag
        if kind == "pan":
            d = pos - start
            self.off = orig + d
            self.fitted = False
            self.update()
            return
        ip = self.to_img(pos)
        x, y, s = orig
        dx, dy = ip.x() - start.x(), ip.y() - start.y()
        if kind == "move":
            nb = (x + dx, y + dy, s)
        else:
            # keep the box square, anchor the opposite corner
            if kind == "br":
                ns = max(6.0, max(ip.x() - x, ip.y() - y))
                nb = (x, y, ns)
            elif kind == "tl":
                ax, ay = x + s, y + s
                ns = max(6.0, max(ax - ip.x(), ay - ip.y()))
                nb = (ax - ns, ay - ns, ns)
            elif kind == "tr":
                ax, ay = x, y + s
                ns = max(6.0, max(ip.x() - ax, ay - ip.y()))
                nb = (ax, ay - ns, ns)
            else:  # bl
                ax, ay = x + s, y
                ns = max(6.0, max(ax - ip.x(), ip.y() - ay))
                nb = (ax - ns, ay, ns)
        self.box = nb
        self.update()
        self.boxChanged.emit(*nb)

    def mouseReleaseEvent(self, e):
        self._drag = None
        self.setCursor(Qt.ArrowCursor)
        if self.editable and self.box:
            self.boxChanged.emit(*self.box)

    def keyPressEvent(self, e):
        if not (self.editable and self.box):
            return super().keyPressEvent(e)
        step = 0.25 if e.modifiers() & Qt.ShiftModifier else 1.0
        x, y, s = self.box
        k = e.key()
        if k == Qt.Key_Left:
            x -= step
        elif k == Qt.Key_Right:
            x += step
        elif k == Qt.Key_Up:
            y -= step
        elif k == Qt.Key_Down:
            y += step
        elif k in (Qt.Key_Plus, Qt.Key_Equal):
            x, y, s = x - step / 2, y - step / 2, s + step
        elif k == Qt.Key_Minus:
            x, y, s = x + step / 2, y + step / 2, max(6, s - step)
        else:
            return super().keyPressEvent(e)
        self.box = (x, y, s)
        self.update()
        self.boxChanged.emit(x, y, s)


# ==========================================================================
# One tab (images or videos)
# ==========================================================================

class MediaTab(QWidget):
    status = Signal(str)

    def __init__(self, kind: str, main: "MainWindow"):
        super().__init__()
        self.kind = kind
        self.main = main
        self.items: list[Item] = []
        self.current: Item | None = None
        self.preview_frame = None   # for video: (y,u,v) not stored; we keep time
        self.preview_t = 0.0
        self._workers = []
        self._block = False
        self.exts = IF.IMAGE_EXTS if kind == "image" else V.VIDEO_EXTS
        self.apply_all_rel = None
        self._build()

    # ---- UI -----------------------------------------------------------------
    def _build(self):
        is_img = self.kind == "image"
        # v3.0: logo riêng của người dùng chèn vào ảnh kết quả (chỉ tab Ảnh)
        self.wm = {"enabled": False, "logo": None, "path": "", "st": EO.WatermarkSettings()}
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        split = QSplitter(Qt.Horizontal)
        lay.addWidget(split)

        # left: list
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Danh sách ảnh" if is_img else "Danh sách video")
        title.setObjectName("SectionTitle")
        ll.addWidget(title)
        row = QHBoxLayout()
        b1 = QPushButton("Chọn ảnh…" if is_img else "Chọn video…")
        b1.clicked.connect(self.pick_files)
        b2 = QPushButton("Chọn thư mục…")
        b2.clicked.connect(self.pick_folder)
        b3 = QPushButton("Xóa")
        b3.setToolTip("Xóa danh sách")
        b3.clicked.connect(self.clear)
        row.addWidget(b1)
        row.addWidget(b2)
        row.addWidget(b3)
        ll.addLayout(row)
        self.chk_sub = QCheckBox("Quét cả thư mục con")
        self.chk_sub.setChecked(True)
        ll.addWidget(self.chk_sub)
        selrow = QHBoxLayout()
        self.chk_all = QCheckBox("Chọn tất cả")
        self.chk_all.setTristate(False)
        self.chk_all.setChecked(True)
        self.chk_all.clicked.connect(self._check_all)
        self.lbl_sel = QLabel("")
        self.lbl_sel.setObjectName("Hint")
        selrow.addWidget(self.chk_all)
        selrow.addStretch(1)
        selrow.addWidget(self.lbl_sel)
        ll.addLayout(selrow)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Tên tệp", "Trạng thái"])
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tree.currentItemChanged.connect(self._on_select)
        self.tree.itemChanged.connect(lambda *_: self._update_sel())
        ll.addWidget(self.tree, 1)
        hint = QLabel("Mẹo: kéo thả tệp hoặc thư mục vào cửa sổ.\n"
                      + ("Định dạng: PNG, JPG, JPEG, JFIF, HEIC, HEIF, GIF, WEBP" if is_img else
                         "Định dạng: MP4, MOV, AVI, WEBM, WMV, MKV"))
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        ll.addWidget(hint)
        lt = QLabel("Nhật ký xử lý")
        lt.setObjectName("SectionTitle")
        ll.addWidget(lt)
        self.logbox = QPlainTextEdit()
        self.logbox.setReadOnly(True)
        self.logbox.setPlaceholderText("Kết quả từng tệp sẽ hiện ở đây")
        ll.addWidget(self.logbox, 1)
        ll.setStretch(ll.indexOf(self.tree), 2)
        split.addWidget(left)

        # right: preview + controls
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        t2 = QLabel("Xem trước & định vị")
        t2.setObjectName("SectionTitle")
        top.addWidget(t2)
        top.addStretch(1)
        self.btn_before = QToolButton()
        self.btn_before.setObjectName("Seg")
        self.btn_before.setText("Ảnh gốc")
        self.btn_before.setCheckable(True)
        self.btn_before.setChecked(True)
        self.btn_after = QToolButton()
        self.btn_after.setObjectName("Seg")
        self.btn_after.setText("Sau khi xử lý")
        self.btn_after.setCheckable(True)
        grp = QButtonGroup(self)
        grp.setExclusive(True)
        grp.addButton(self.btn_before)
        grp.addButton(self.btn_after)
        self.btn_after.toggled.connect(self._toggle_after)
        bz = QToolButton()
        bz.setObjectName("Tool")
        bz.setText("Phóng vùng logo")
        bz.clicked.connect(lambda: self.canvas.zoom_to_box())
        bf = QToolButton()
        bf.setObjectName("Tool")
        bf.setText("Vừa khung")
        bf.clicked.connect(lambda: self.canvas.fit())
        for b in (self.btn_before, self.btn_after, bz, bf):
            top.addWidget(b)
        rl.addLayout(top)

        self.canvas = Canvas()
        self.canvas.boxChanged.connect(self._on_box_changed)
        # v2.3: thanh trượt ngang + dọc để tới nhanh vùng cần định vị khi phóng to
        self.canvas_frame = ScrollFrame(self.canvas, self.canvas.image_size)
        rl.addWidget(self.canvas_frame, 1)

        if not is_img:
            vrow = QHBoxLayout()
            vrow.addWidget(QLabel("Khung hình:"))
            self.slider = QSlider(Qt.Horizontal)
            self.slider.setRange(0, 1000)
            self.slider.sliderReleased.connect(self._on_seek)
            self.slider.valueChanged.connect(self._on_slider_value)
            vrow.addWidget(self.slider, 1)
            self.lbl_time = QLabel("0.0 s")
            self.lbl_time.setMinimumWidth(70)
            vrow.addWidget(self.lbl_time)
            rl.addLayout(vrow)

        self.lbl_detect = QLabel("")
        self.lbl_detect.setObjectName("DetectLabel")
        self.lbl_detect.setWordWrap(True)
        rl.addWidget(self.lbl_detect)

        # positioning controls
        box = QFrame()
        box.setObjectName("Card")
        g = QGridLayout(box)
        g.setContentsMargins(12, 10, 12, 10)
        g.setHorizontalSpacing(10)
        self.rb_auto = QRadioButton("Tự động dò logo")
        self.rb_manual = QRadioButton("Định vị thủ công")
        self.rb_auto.setChecked(True)
        self.rb_manual.toggled.connect(self._on_mode)
        g.addWidget(self.rb_auto, 0, 0)
        g.addWidget(self.rb_manual, 0, 1)
        g.addWidget(QLabel("Cách xóa:"), 0, 2)
        self.cmb_method = QComboBox()
        self.cmb_method.addItem("Khử alpha (chính xác – khuyên dùng)", "alpha")
        self.cmb_method.addItem("Vá vùng (inpaint) – cho logo lạ", "inpaint")
        self.cmb_method.currentIndexChanged.connect(lambda *_: self._invalidate_after())
        g.addWidget(self.cmb_method, 0, 3, 1, 3)
        g.setColumnStretch(6, 1)

        self.sp_x = QDoubleSpinBox()
        self.sp_y = QDoubleSpinBox()
        self.sp_s = QDoubleSpinBox()
        for sp, lab, col in ((self.sp_x, "X", 0), (self.sp_y, "Y", 2), (self.sp_s, "Cỡ", 4)):
            sp.setDecimals(1)
            sp.setRange(0, 20000)
            sp.setFixedWidth(110)
            sp.setSingleStep(0.5)
            sp.valueChanged.connect(self._on_spin)
            l = QLabel(lab + ":")
            g.addWidget(l, 1, col)
            g.addWidget(sp, 1, col + 1)
        self.manual_widgets = [self.sp_x, self.sp_y, self.sp_s]
        row2 = QHBoxLayout()
        self.btn_snap = QPushButton("Khớp chính xác trong khung")
        self.btn_snap.setToolTip("Dò lại ngôi sao Gemini bên trong khung bạn đặt (tinh chỉnh dưới 1 pixel)")
        self.btn_snap.clicked.connect(self._snap)
        self.btn_reset = QPushButton("Về vị trí tự động")
        self.btn_reset.clicked.connect(self._reset_manual)
        self.btn_all = QPushButton("Áp dụng khung này cho tất cả")
        self.btn_all.setToolTip("Dùng cùng vị trí (tính từ góc phải-dưới, co giãn theo kích thước) cho mọi tệp trong danh sách")
        self.btn_all.clicked.connect(self._apply_all)
        self.btn_prev = QPushButton("Xem kết quả")
        self.btn_prev.setObjectName("Secondary")
        self.btn_prev.clicked.connect(self.make_after)
        for b in (self.btn_snap, self.btn_reset, self.btn_all):
            row2.addWidget(b)
            self.manual_widgets.append(b)
        row2.addStretch(1)
        if is_img:
            self.btn_logo = QPushButton("Chèn logo…")
            self.btn_logo.setToolTip("Chèn thêm logo của bạn vào ảnh kết quả (chọn tệp, vị trí, độ mờ, kích thước)")
            self.btn_logo.clicked.connect(lambda: LogoDialog(self).exec())
            row2.addWidget(self.btn_logo)
        row2.addWidget(self.btn_prev)
        g.addLayout(row2, 2, 0, 1, 6)
        tip = QLabel("Thủ công: kéo khung xanh vào đúng ngôi sao, kéo góc để đổi cỡ; phím mũi tên dịch 1 px "
                     "(Shift: 0,25 px); con lăn chuột để phóng to; chuột phải để kéo ảnh; khi đang phóng to, "
                     "dùng thanh trượt ngang/dọc quanh khung để tới nhanh vùng cần định vị.")
        tip.setObjectName("Hint")
        tip.setWordWrap(True)
        g.addWidget(tip, 3, 0, 1, 6)
        rl.addWidget(box)
        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 3)
        split.setSizes([420, 820])
        self._on_mode()

    # ---- list management --------------------------------------------------
    def pick_files(self):
        if self.kind == "image":
            flt = "Ảnh (*.png *.jpg *.jpeg *.jfif *.heic *.heif *.gif *.webp *.bmp *.tif *.tiff)"
        else:
            flt = "Video (*.mp4 *.mov *.m4v *.avi *.webm *.wmv *.mkv)"
        last = self.main.settings.value(f"last_in_{self.kind}", "")
        files, _ = QFileDialog.getOpenFileNames(self, "Chọn tệp", last, flt + ";;Tất cả (*.*)")
        if files:
            self.main.settings.setValue(f"last_in_{self.kind}", os.path.dirname(files[0]))
            self.add_paths(files)

    def pick_folder(self):
        last = self.main.settings.value(f"last_in_{self.kind}", "")
        d = QFileDialog.getExistingDirectory(self, "Chọn thư mục", last)
        if d:
            self.main.settings.setValue(f"last_in_{self.kind}", d)
            self.add_paths([d])

    def add_paths(self, paths):
        found = batch.list_files(paths, self.exts, self.chk_sub.isChecked())
        known = {it.path for it in self.items}
        added = 0
        for f, base in found:
            if f in known:
                continue
            it = Item(f, base)
            w = QTreeWidgetItem([os.path.relpath(f, base) if base else os.path.basename(f), it.status])
            w.setToolTip(0, f)
            w.setFlags(w.flags() | Qt.ItemIsUserCheckable)
            w.setCheckState(0, Qt.Checked)
            it.tree = w
            self.tree.addTopLevelItem(w)
            self.items.append(it)
            added += 1
        if added and self.current is None:
            self.tree.setCurrentItem(self.items[0].tree)
        self.status.emit(f"Đã thêm {added} tệp" if added else "Không có tệp mới phù hợp")
        self.main.update_counts()
        self._update_sel()

    def _checked(self, it: Item) -> bool:
        return it.tree is None or it.tree.checkState(0) == Qt.Checked

    def _check_all(self, on):
        self.tree.blockSignals(True)
        for it in self.items:
            if it.tree is not None:
                it.tree.setCheckState(0, Qt.Checked if on else Qt.Unchecked)
        self.tree.blockSignals(False)
        self._update_sel()

    def _update_sel(self):
        n = len(self.items)
        k = sum(1 for it in self.items if self._checked(it))
        self.chk_all.blockSignals(True)
        self.chk_all.setChecked(n > 0 and k == n)
        self.chk_all.blockSignals(False)
        self.lbl_sel.setText(f"Đã chọn {k}/{n}" if n else "")
        self.main.refresh_start_label()

    def selected_count(self):
        return sum(1 for it in self.items if self._checked(it))

    def clear(self):
        self.items.clear()
        self.tree.clear()
        self.current = None
        self.canvas.set_images(None)
        self.canvas.set_box(None)
        self.lbl_detect.setText("")
        self.main.update_counts()
        self._update_sel()

    def set_item_status(self, idx, text, out=""):
        it = self.items[idx]
        it.status = text
        if out:
            it.out = out
        if it.tree is not None:
            it.tree.setText(1, text)
            color = {"Đã gỡ logo": "#16A34A", "Đã vá vùng": "#16A34A", "Lỗi": "#DC2626",
                     "Không thấy logo": "#D97706"}.get(text)
            if color:
                it.tree.setForeground(1, QColor(color))

    # ---- selection & preview ----------------------------------------------
    def _on_select(self, cur, prev):
        if cur is None:
            return
        it = next((x for x in self.items if x.tree is cur), None)
        if it is None:
            return
        self.current = it
        self.btn_before.setChecked(True)
        self.canvas.set_after(None)
        self.canvas.message = "Đang tải…"
        self.canvas.set_images(None)
        if self.kind == "image":
            self._run(self._load_image, it.path, done=lambda r, it=it: self._show_loaded(it, r))
        else:
            self._run(self._load_video, it.path, done=lambda r, it=it: self._show_loaded(it, r))

    @staticmethod
    def _load_image(path):
        li = IF.load(path)
        return {"rgb": li.rgb}

    @staticmethod
    def _load_video(path):
        info = V.probe(path)
        t = min(max(0.0, info.duration * 0.35), max(0.0, info.duration - 0.1))
        rgb = V.grab_rgb(path, t, info)
        return {"rgb": rgb, "duration": info.duration, "t": t, "info": info}

    def _show_loaded(self, it: Item, r):
        if it is not self.current:
            return
        rgb = r["rgb"]
        it.size = (rgb.shape[1], rgb.shape[0])
        self.canvas.set_images(np_to_qimage(rgb))
        if self.kind == "video":
            it.duration = r["duration"]
            self.preview_t = r["t"]
            self._block = True
            self.slider.setValue(int(1000 * r["t"] / max(r["duration"], 0.01)))
            self._block = False
            self.lbl_time.setText(f"{r['t']:.1f} s")
            self._preview_info = r["info"]
        if it.manual_box:
            x, y, w, h = it.manual_box
            self.rb_manual.setChecked(True)
            self._set_box((x, y, min(w, h)), True)
        else:
            if self.rb_manual.isChecked() and self.apply_all_rel is None:
                pass
            self._show_auto(it)

    def _show_auto(self, it: Item):
        if it.geo is not None:
            self._display_geo(it, it.geo)
            return
        self.lbl_detect.setText("Đang dò logo Gemini…")
        self.canvas.set_box(None)
        if self.kind == "image":
            def job(path=it.path):
                li = IF.load(path)
                return E.detect_image(li.rgb)
        else:
            def job(path=it.path):
                info = V.probe(path)
                fr = V.sample_frames(path, info, 40)
                return V.detect_video(fr, info)
        self._run(job, done=lambda g, it=it: self._geo_ready(it, g))

    def _geo_ready(self, it: Item, geo: E.Geometry):
        it.geo = geo
        if it is self.current:
            self._display_geo(it, geo)

    def _display_geo(self, it, geo):
        if geo.found:
            msg = (f"Đã tìm thấy logo Gemini tại ({geo.x:.1f}, {geo.y:.1f}), cỡ {geo.size:.1f} px – "
                   f"độ khớp {geo.score * 100:.0f}%, độ mờ logo {geo.strength:.2f}.")
            if getattr(geo, "calib", None) is not None:
                msg += " Đã tự hiệu chuẩn theo video."
            self.lbl_detect.setStyleSheet("color:#16A34A")
        elif geo.size > 0:
            msg = ("Không chắc có logo Gemini ở góc phải-dưới. Nếu vẫn thấy ngôi sao, hãy bật "
                   "'Định vị thủ công' và kéo khung vào đúng vị trí.")
            self.lbl_detect.setStyleSheet("color:#D97706")
        else:
            msg = "Không tìm thấy logo."
            self.lbl_detect.setStyleSheet("color:#D97706")
        self.lbl_detect.setText(msg)
        if not (self.rb_manual.isChecked() and it.manual_box):
            if geo.found:
                self._set_box((geo.x, geo.y, geo.size), True)
            else:
                self.canvas.set_box(None)
            if self.rb_manual.isChecked() and it.manual_box is None:
                self._init_manual_box(it)

    def _set_box(self, b, ok=True):
        self.canvas.set_box(b, ok)
        self._block = True
        self.sp_x.setValue(b[0])
        self.sp_y.setValue(b[1])
        self.sp_s.setValue(b[2])
        self._block = False

    def _on_slider_value(self, v):
        if self.current and self.kind == "video" and not self._block:
            self.lbl_time.setText(f"{v / 1000 * self.current.duration:.1f} s")

    def _on_seek(self):
        it = self.current
        if not it or self.kind != "video":
            return
        t = self.slider.value() / 1000 * it.duration
        t = min(t, max(0.0, it.duration - 0.1))
        self.preview_t = t
        self.btn_before.setChecked(True)
        self.canvas.set_after(None)

        def job(path=it.path, t=t):
            return V.grab_rgb(path, t)

        self._run(job, done=lambda rgb, it=it: it is self.current and
                  self.canvas.set_images(np_to_qimage(rgb), None, keep_view=True))

    # ---- manual mode -----------------------------------------------------
    def _on_mode(self, *_):
        manual = self.rb_manual.isChecked()
        self.canvas.editable = manual
        for w in self.manual_widgets:
            w.setEnabled(manual)
        it = self.current
        if manual and it is not None and it.manual_box is None:
            self._init_manual_box(it)
        if manual and self.canvas.box:
            self.canvas.zoom_to_box(5)
        if not manual and it is not None:
            it.manual_box = None
            if it.geo is not None:
                self._display_geo(it, it.geo)
        self.canvas.update()
        self._invalidate_after()

    def _init_manual_box(self, it: Item):
        if it.size is None:
            return
        W, H = it.size
        if it.geo is not None and it.geo.size > 0:
            b = (it.geo.x, it.geo.y, it.geo.size)
        elif self.apply_all_rel:
            b = self._from_rel(self.apply_all_rel, W, H)
        else:
            s = 48.0 * min(W, H) / 1024 if min(W, H) <= 1100 else 96.0 * min(W, H) / 1536
            b = (W - s * 2.0 - s, H - s * 2.0 - s, s)
        it.manual_box = (b[0], b[1], b[2], b[2])
        self._set_box(b, True)

    def _on_box_changed(self, x, y, s):
        it = self.current
        if it is None:
            return
        it.manual_box = (x, y, s, s)
        self._set_box((x, y, s), True)
        self._invalidate_after()

    def _on_spin(self, *_):
        if self._block or not self.rb_manual.isChecked() or self.current is None:
            return
        x, y, s = self.sp_x.value(), self.sp_y.value(), max(6.0, self.sp_s.value())
        self.current.manual_box = (x, y, s, s)
        self.canvas.set_box((x, y, s), True)
        self._invalidate_after()

    def _snap(self):
        it = self.current
        if not it or not it.manual_box:
            return
        self.lbl_detect.setText("Đang khớp ngôi sao trong khung…")
        opts = E.Options(mode="manual", manual_box=it.manual_box)
        if self.kind == "image":
            def job(path=it.path):
                return E.detect_image(IF.load(path).rgb, opts)
        else:
            def job(path=it.path):
                info = V.probe(path)
                return V.detect_video(V.sample_frames(path, info, 24), info, opts)

        def done(g, it=it):
            if g.found:
                it.manual_box = (g.x, g.y, g.size, g.size)
                if it is self.current:
                    self._set_box((g.x, g.y, g.size), True)
                    self.lbl_detect.setStyleSheet("color:#16A34A")
                    self.lbl_detect.setText(
                        f"Đã khớp: ({g.x:.2f}, {g.y:.2f}), cỡ {g.size:.2f} px, độ mờ {g.strength:.2f}.")
            elif it is self.current:
                self.lbl_detect.setStyleSheet("color:#D97706")
                self.lbl_detect.setText("Không khớp được ngôi sao Gemini trong khung – khi xử lý sẽ "
                                        "dùng phương pháp vá vùng theo đúng khung này.")
            self._invalidate_after()
        self._run(job, done=done)

    def _reset_manual(self):
        it = self.current
        if not it:
            return
        it.manual_box = None
        if it.geo is not None and it.geo.size > 0:
            self._init_manual_box(it)
        self._invalidate_after()

    @staticmethod
    def _to_rel(box, W, H):
        x, y, s = box
        ref = float(min(W, H))
        return ((W - x) / ref, (H - y) / ref, s / ref)

    @staticmethod
    def _from_rel(rel, W, H):
        rx, ry, rs = rel
        ref = float(min(W, H))
        return (W - rx * ref, H - ry * ref, rs * ref)

    def _apply_all(self):
        it = self.current
        if not it or not it.manual_box or not it.size:
            return
        W, H = it.size
        rel = self._to_rel(it.manual_box[:3], W, H)
        self.apply_all_rel = rel
        n = 0
        for other in self.items:
            if other is it:
                continue
            other.manual_box = ("rel",) + rel
            n += 1
            if other.tree is not None:
                other.tree.setText(1, "Thủ công (theo mẫu)")
        self.status.emit(f"Đã áp dụng khung cho {n} tệp khác")

    def apply_manual_to_undetected(self):
        """v3.1 - nút "Áp dụng vị trí xóa logo thủ công": lấy khung thủ công của
        ảnh đang chọn làm mẫu cho MỌI ảnh mà bước dò tự động chưa tìm thấy logo
        (ảnh đã dò được giữ nguyên chế độ tự động), đồng thời tích chọn chúng.

        Ngôi sao trong khung mẫu được khớp chính xác một lần; ảnh cùng kích
        thước dùng lại đúng vị trí, cỡ và độ mạnh đó nếu không tự khớp được,
        nên vẫn khôi phục bằng phép tính ngược thay vì vá vùng."""
        it = self.current
        if not it or not it.manual_box or it.manual_box[0] == "rel" or not it.size:
            QMessageBox.information(self, APP_NAME,
                                    "Hãy chọn một ảnh trong danh sách, bật \"Định vị thủ công\" và đặt khung "
                                    "vào đúng logo trước, rồi bấm nút này.")
            return
        box = tuple(it.manual_box)
        W, H = it.size
        rel = self._to_rel(box[:3], W, H)

        def job(path=it.path, box=box):
            if self.kind != "image":
                return None
            return E.detect_image(IF.load(path).rgb, E.Options(mode="manual", manual_box=box))

        def done(g, it=it):
            tpl = None
            if g is not None and g.found:
                tpl = ((W, H), g.to_dict())
                it.manual_box = (g.x, g.y, g.size, g.size)
                if it is self.current:
                    self._set_box((g.x, g.y, g.size), True)
            it.template = tpl
            n = keep = 0
            self.tree.blockSignals(True)
            for o in self.items:
                if o is it:
                    continue
                if o.geo is not None and o.geo.found and not o.manual_box:
                    keep += 1                       # đã dò tự động được: giữ nguyên
                    continue
                o.manual_box = ("rel",) + rel
                o.template = tpl
                n += 1
                if o.tree is not None:
                    o.tree.setCheckState(0, Qt.Checked)
                    o.tree.setText(1, "Thủ công (theo mẫu)")
            if it.tree is not None:
                it.tree.setCheckState(0, Qt.Checked)
            self.tree.blockSignals(False)
            self._update_sel()
            how = ("đã khớp ngôi sao trong khung mẫu – ảnh cùng kích thước dùng đúng vị trí này"
                   if tpl else "không khớp được ngôi sao trong khung mẫu – các ảnh sẽ được vá vùng theo khung")
            self.main.log(f"— Áp dụng vị trí thủ công cho {n} tệp chưa dò được logo (đã tích chọn); "
                          f"{keep} tệp đã dò tự động giữ nguyên; {how}.")
            self.status.emit(f"Đã áp dụng vị trí thủ công cho {n} tệp")
        self._run(job, done=done)

    # ---- after preview ----------------------------------------------------
    def wm_active(self):
        """(logo RGBA, cài đặt) nếu đang bật chèn logo, ngược lại None."""
        return (self.wm["logo"], self.wm["st"]) if self.wm["enabled"] and self.wm["logo"] is not None else None

    def wm_changed(self):
        if hasattr(self, "btn_logo"):
            self.btn_logo.setText("Chèn logo: đang bật" if self.wm_active() else "Chèn logo…")
        self._invalidate_after()

    def _invalidate_after(self):
        self.canvas.set_after(None)
        if self.btn_after.isChecked():
            self.btn_before.setChecked(True)

    def _toggle_after(self, on):
        self.canvas.show_after = on
        if on and self.canvas.after is None:
            self.make_after()
        self.canvas.update()

    def current_options(self, it: Item) -> E.Options:
        opts = E.Options(method=self.cmb_method.currentData())
        box = self.resolve_box(it)
        if box:
            opts.mode = "manual"
            opts.manual_box = box
            if it.template:
                opts.extra["fallback_geo"] = it.template
        return opts

    def resolve_box(self, it: Item, size=None):
        mb = it.manual_box
        if not mb:
            return None
        if mb[0] == "rel":
            sz = size or it.size
            if sz is None:
                return ("rel",) + tuple(mb[1:])
            x, y, s = self._from_rel(mb[1:], *sz)
            return (x, y, s, s)
        return tuple(mb)

    def make_after(self):
        it = self.current
        if not it or self.canvas.before is None:
            return
        opts = self.current_options(it)
        self.btn_prev.setEnabled(False)
        self.btn_prev.setText("Đang xử lý…")
        if self.kind == "image":
            geo_cached = it.geo if (opts.mode == "auto" and opts.method == "alpha") else None

            def job(path=it.path, opts=opts, g=geo_cached, wm=self.wm_active()):
                rgb = IF.load(path).rgb
                out, g2 = (batch.apply_geometry(rgb, g, opts), g) if g is not None else E.process_image(rgb, opts)
                if wm:
                    out = EO.compose_watermark(out, wm[0], wm[1])
                return out, g2
        else:
            t = self.preview_t
            geo_cached = it.geo if (opts.mode == "auto") else None

            def job(path=it.path, opts=opts, g=geo_cached, t=t):
                info = V.probe(path)
                if g is None:
                    if opts.mode == "manual" and opts.method == "inpaint":
                        x, y, w, h = opts.manual_box
                        g = E.Geometry(x, y, min(w, h), found=False)
                    else:
                        g = V.detect_video(V.sample_frames(path, info, 24), info, opts)
                _, after = V.preview_pair(path, t, g, info,
                                          "inpaint" if opts.method == "inpaint" else "alpha")
                return after, g

        def done(r, it=it):
            self.btn_prev.setEnabled(True)
            self.btn_prev.setText("Xem kết quả")
            if it is not self.current:
                return
            out, g = r
            self.canvas.set_after(np_to_qimage(out))
            self.btn_after.setChecked(True)
            self.canvas.show_after = True
            self.canvas.update()

        def fail(msg):
            self.btn_prev.setEnabled(True)
            self.btn_prev.setText("Xem kết quả")
            QMessageBox.warning(self, APP_NAME, "Không tạo được bản xem trước:\n" + msg)
        self._run(job, done=done, fail=fail)

    # ---- worker helper ------------------------------------------------------
    def _run(self, fn, *args, done=None, fail=None):
        w = Worker(fn, *args)
        if done:
            w.done.connect(done)
        w.failed.connect(fail or (lambda m: self.status.emit("Lỗi: " + m)))
        w.finished.connect(lambda w=w: self._workers.remove(w) if w in self._workers else None)
        self._workers.append(w)
        w.start()

    # ---- batch ---------------------------------------------------------------
    def jobs(self):
        out = []
        for i, it in enumerate(self.items):
            if not self._checked(it):
                continue
            mb = it.manual_box
            if mb and mb[0] == "rel":
                if it.size is None:
                    try:
                        if self.kind == "image":
                            from PIL import Image
                            with Image.open(it.path) as im:
                                it.size = im.size
                        else:
                            inf = V.probe(it.path)
                            it.size = V._dims(inf)
                    except Exception:
                        it.size = None
                mb = self.resolve_box(it) if it.size else None
            out.append((i, {"path": it.path, "base": it.base, "manual_box": mb, "template": it.template if mb else None,
                            "geo": it.geo if it.geo is not None and it.geo.found else None}))
        return out


# ==========================================================================
# Main window
# ==========================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings("thsky90", "AI Remove")
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(app_icon())
        self.setAcceptDrops(True)
        self.worker: BatchWorker | None = None
        self._build()
        self.resize(1320, 820)

    def _build(self):
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # header
        header = QFrame()
        header.setObjectName("Header")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(20, 12, 20, 12)
        logo = QLabel()
        pm = app_icon().pixmap(40, 40)
        logo.setPixmap(pm)
        hl.addWidget(logo)
        tl = QVBoxLayout()
        t1 = QLabel("AI Remove <span style='color:#A5B4FC;font-weight:500'>– thsky90</span>")
        t1.setObjectName("AppTitle")
        t1.setTextFormat(Qt.RichText)
        t2 = QLabel("Gỡ logo ngôi sao Gemini khỏi ảnh và video, xóa metadata ẩn, chỉnh sửa ảnh cho mạng xã hội")
        t2.setObjectName("AppSub")
        tl.addWidget(t1)
        tl.addWidget(t2)
        hl.addLayout(tl)
        hl.addStretch(1)
        ver = QLabel(f"v{__version__}")
        ver.setObjectName("AppSub")
        hl.addWidget(ver)
        root.addWidget(header)

        self.tabs = QTabWidget()
        self.tab_img = MediaTab("image", self)
        self.tab_vid = MediaTab("video", self)
        self.tabs.addTab(self.tab_img, "  Ảnh  ")
        self.tabs.addTab(self.tab_vid, "  Video  ")
        from .editor.tab import EditorTab
        self.tab_edit = EditorTab(self)
        self.tabs.addTab(self.tab_edit, "  Chỉnh sửa ảnh  ")
        self.tabs.currentChanged.connect(self._on_tab)
        for t in (self.tab_img, self.tab_vid):
            t.status.connect(self.show_status)
        root.addWidget(self.tabs, 1)

        # bottom panel
        bottom = QFrame()
        self.bottom = bottom
        bottom.setObjectName("Bottom")
        bl = QGridLayout(bottom)
        bl.setContentsMargins(16, 10, 16, 12)
        bl.setHorizontalSpacing(10)
        bl.addWidget(QLabel("Thư mục xuất:"), 0, 0)
        self.ed_out = QLineEdit(self.settings.value("out_dir", ""))
        self.ed_out.setPlaceholderText("Chọn thư mục để lưu ảnh/video đã xử lý")
        bl.addWidget(self.ed_out, 0, 1, 1, 3)
        bo = QPushButton("Chọn thư mục đích…")
        bo.setObjectName("Folder")
        bo.clicked.connect(self.pick_out)
        bl.addWidget(bo, 0, 4)
        self.btn_open = QPushButton("Mở thư mục kết quả")
        self.btn_open.setObjectName("Open")
        self.btn_open.clicked.connect(self.open_out)
        bl.addWidget(self.btn_open, 0, 5)

        self.chk_meta = QCheckBox("Gỡ metadata ẩn (EXIF, XMP, IPTC, C2PA / Content Credentials, thông tin AI)")
        self.chk_meta.setChecked(self.settings.value("strip_meta", "true") in ("true", True))
        bl.addWidget(self.chk_meta, 1, 0, 1, 4)
        self.cmb_vfmt = QComboBox()
        self.cmb_vfmt.addItem("Video: giữ định dạng gốc", False)
        self.cmb_vfmt.addItem("Video: xuất MP4 (H.264)", True)
        self.cmb_vfmt.setCurrentIndex(1 if self.settings.value("force_mp4", "false") in ("true", True) else 0)
        bl.addWidget(self.cmb_vfmt, 1, 4)

        self.btn_start = QPushButton("Bắt đầu xử lý ảnh")
        self.btn_start.setObjectName("Primary")
        self.btn_start.setMinimumHeight(40)
        self.btn_start.clicked.connect(self.start)
        self.btn_stop = QPushButton("Ngưng tiến trình")
        self.btn_stop.setObjectName("Danger")
        self.btn_stop.setMinimumHeight(40)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop)
        sl = QHBoxLayout()
        # v3.0: quét trước toàn bộ danh sách, chỉ tích chọn tệp có logo
        self.btn_scan = QPushButton("Quét dò tìm logo tự động")
        self.btn_scan.setObjectName("Scan")
        self.btn_scan.setMinimumHeight(40)
        self.btn_scan.setToolTip("Dò logo Gemini trên mọi tệp trong danh sách: tệp có logo được tích chọn, "
                                 "tệp không thấy logo bị bỏ tích (tệp đã đặt khung thủ công giữ nguyên).")
        self.btn_scan.clicked.connect(self.scan)
        self.scanner = None
        # v3.1: dùng khung thủ công của ảnh đang chọn cho mọi ảnh chưa dò được logo
        self.btn_manual_all = QPushButton("Áp dụng vị trí xóa logo thủ công")
        self.btn_manual_all.setObjectName("ManualAll")
        self.btn_manual_all.setMinimumHeight(40)
        self.btn_manual_all.setToolTip("Đặt khung thủ công trên một ảnh rồi bấm nút này: vị trí và kích thước đó được áp dụng "
                                       "cho mọi ảnh chưa dò tự động được logo (ảnh đã dò được giữ nguyên) và tích chọn chúng.")
        self.btn_manual_all.clicked.connect(self.apply_manual_all)
        sl.addWidget(self.btn_start, 3)
        sl.addWidget(self.btn_scan, 2)
        sl.addWidget(self.btn_manual_all, 2)
        sl.addWidget(self.btn_stop, 2)
        bl.addLayout(sl, 1, 5, 1, 1)

        self.prog_all = QProgressBar()
        self.prog_all.setRange(0, 1)
        self.prog_all.setValue(0)
        self.prog_all.setFormat("Tổng: chưa chạy")
        self.prog_file = QProgressBar()
        self.prog_file.setRange(0, 1000)
        self.prog_file.setFormat("Tệp hiện tại")
        bl.addWidget(self.prog_all, 2, 0, 1, 3)
        bl.addWidget(self.prog_file, 2, 3, 1, 3)
        root.addWidget(bottom)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Sẵn sàng")
        self.update_counts()

    # ---- misc ------------------------------------------------------------------
    def show_status(self, msg):
        self.statusBar().showMessage(msg, 6000)

    def update_counts(self):
        ni, nv = len(self.tab_img.items), len(self.tab_vid.items)
        self.tabs.setTabText(0, f"  Ảnh ({ni})  " if ni else "  Ảnh  ")
        self.tabs.setTabText(1, f"  Video ({nv})  " if nv else "  Video  ")

    def _on_tab(self, i):
        # Tab "Chỉnh sửa ảnh" có nút xuất riêng -> ẩn thanh xử lý hàng loạt bên dưới.
        self.bottom.setVisible(self.tabs.widget(i) is not getattr(self, "tab_edit", None))
        self.refresh_start_label()

    def refresh_start_label(self):
        if not hasattr(self, "btn_start") or not hasattr(self, "tabs"):
            return
        tab = self.tabs.currentWidget()
        if not isinstance(tab, MediaTab):
            return
        what = "ảnh" if tab.kind == "image" else "video"
        n, k = len(tab.items), tab.selected_count()
        self.btn_start.setText(f"Xử lý {k} {what} đã chọn" if n and k < n else f"Bắt đầu xử lý {what}")

    def pick_out(self):
        d = QFileDialog.getExistingDirectory(self, "Chọn thư mục đích", self.ed_out.text())
        if d:
            self.ed_out.setText(d)
            self.settings.setValue("out_dir", d)

    def open_out(self):
        d = self.ed_out.text().strip()
        if d and os.path.isdir(d):
            QDesktopServices.openUrl(QUrl.fromLocalFile(d))

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if self.tabs.currentWidget() is getattr(self, "tab_edit", None):
            imgs = [p for p in paths if batch.is_image(p)]
            if imgs:
                self.tab_edit.open_image(imgs[0])
            return
        imgs = [p for p in paths if os.path.isdir(p) or batch.is_image(p)]
        vids = [p for p in paths if os.path.isdir(p) or batch.is_video(p)]
        if any(batch.is_video(p) for p in paths) and not any(batch.is_image(p) for p in paths):
            self.tabs.setCurrentIndex(1)
            self.tab_vid.add_paths(vids)
        elif any(batch.is_image(p) for p in paths) and not any(batch.is_video(p) for p in paths):
            self.tabs.setCurrentIndex(0)
            self.tab_img.add_paths(imgs)
        else:
            self.tab_img.add_paths(imgs)
            self.tab_vid.add_paths(vids)

    # ---- batch -------------------------------------------------------------------
    def start(self):
        tab = self.tabs.currentWidget()
        if not isinstance(tab, MediaTab):
            return
        if not tab.items:
            QMessageBox.information(self, APP_NAME, "Danh sách đang trống. Hãy chọn tệp hoặc thư mục.")
            return
        out = self.ed_out.text().strip()
        if not out:
            self.pick_out()
            out = self.ed_out.text().strip()
            if not out:
                return
        try:
            os.makedirs(out, exist_ok=True)
        except OSError as ex:
            QMessageBox.warning(self, APP_NAME, f"Không tạo được thư mục xuất:\n{ex}")
            return
        self.settings.setValue("out_dir", out)
        self.settings.setValue("strip_meta", self.chk_meta.isChecked())
        self.settings.setValue("force_mp4", bool(self.cmb_vfmt.currentData()))
        jobs = tab.jobs()
        if not jobs:
            QMessageBox.information(self, APP_NAME, "Chưa tích chọn tệp nào. Hãy tích vào ô cạnh tên tệp (hoặc 'Chọn tất cả').")
            return
        self.prog_all.setFormat("Tổng: %v/%m tệp")
        self.prog_all.setRange(0, len(jobs))
        self.prog_all.setValue(0)
        self.prog_file.setValue(0)
        self.worker = BatchWorker(tab.kind, jobs, out, self.chk_meta.isChecked(),
                                  tab.cmb_method.currentData(), bool(self.cmb_vfmt.currentData()),
                                  wm=tab.wm_active() if tab.kind == "image" else None)
        self._tab_running = tab
        self.worker.file_started.connect(lambda i: tab.set_item_status(i, "Đang xử lý…"))
        self.worker.file_progress.connect(self._on_file_progress)
        self.worker.file_done.connect(self._on_file_done)
        self.worker.log.connect(self.log)
        self.worker.finished_all.connect(self._on_all_done)
        self.btn_start.setEnabled(False)
        self.btn_scan.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.tabs.tabBar().setEnabled(False)
        self.log(f"— Bắt đầu {len(jobs)} tệp → {out}")
        self.worker.start()

    def apply_manual_all(self):
        tab = self.tabs.currentWidget()
        if isinstance(tab, MediaTab):
            self._tab_running = tab
            tab.apply_manual_to_undetected()

    # ---- v3.0: quét dò tìm logo tự động -----------------------------------
    def scan(self):
        tab = self.tabs.currentWidget()
        if not isinstance(tab, MediaTab):
            return
        if not tab.items:
            QMessageBox.information(self, APP_NAME, "Danh sách đang trống. Hãy chọn tệp hoặc thư mục.")
            return
        todo = [(i, it.path, it.geo) for i, it in enumerate(tab.items) if not it.manual_box]
        self._scan = {"tab": tab, "found": 0, "miss": 0, "manual": len(tab.items) - len(todo)}
        self.prog_all.setFormat("Đã quét: %v/%m tệp")
        self.prog_all.setRange(0, max(1, len(todo)))
        self.prog_all.setValue(0)
        self._tab_running = tab
        self.scanner = ScanWorker(tab.kind, todo)
        self.scanner.item_done.connect(self._on_scan_item)
        self.scanner.finished_all.connect(self._on_scan_done)
        self.btn_start.setEnabled(False)
        self.btn_scan.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.tabs.tabBar().setEnabled(False)
        self.log(f"— Quét dò tìm logo trên {len(todo)} tệp…")
        self.scanner.start()

    def _on_scan_item(self, i, geo):
        tab = self._scan["tab"]
        it = tab.items[i]
        found = geo is not None and geo.found
        if geo is not None:
            it.geo = geo
        self._scan["found" if found else "miss"] += 1
        if it.tree is not None:
            it.tree.setCheckState(0, Qt.Checked if found else Qt.Unchecked)
        tab.set_item_status(i, "Có logo – chờ xử lý" if found else "Không thấy logo")
        self.prog_all.setValue(self.prog_all.value() + 1)

    def _on_scan_done(self, secs):
        sc = self._scan
        sc["tab"]._update_sel()
        self.btn_start.setEnabled(True)
        self.btn_scan.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.tabs.tabBar().setEnabled(True)
        extra = f", {sc['manual']} tệp có khung thủ công giữ nguyên" if sc["manual"] else ""
        self.log(f"— Quét xong ({secs:.1f} giây): {sc['found']} tệp có logo (đã tích chọn), "
                 f"{sc['miss']} tệp không thấy logo (đã bỏ tích){extra}. "
                 "Tệp bỏ tích vẫn có thể đặt khung thủ công rồi tích lại.")
        self.show_status(f"Quét xong: {sc['found']} tệp có logo")
        self.scanner = None

    def stop(self):
        if self.scanner:
            self.scanner.cancel()
            self.btn_stop.setEnabled(False)
            self.show_status("Đang dừng quét…")
        if self.worker:
            self.worker.cancel()
            self.btn_stop.setEnabled(False)
            self.show_status("Đang dừng…")

    def _on_file_progress(self, f, text):
        self.prog_file.setValue(int(f * 1000))
        self.prog_file.setFormat(f"Tệp hiện tại: {text}")

    def _on_file_done(self, i, state, out):
        self._tab_running.set_item_status(i, state, out)
        self.prog_all.setValue(self.prog_all.value() + 1)

    def _on_all_done(self, ok, fail, secs):
        self.btn_start.setEnabled(True)
        self.btn_scan.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.tabs.tabBar().setEnabled(True)
        self.prog_file.setValue(1000)
        self.prog_file.setFormat("Hoàn tất")
        self.log(f"— Hoàn tất: {ok} thành công, {fail} lỗi, {secs:.1f} giây")
        self.show_status(f"Hoàn tất {ok} tệp")
        self.worker = None

    def log(self, msg):
        tab = getattr(self, "_tab_running", None) or self.tabs.currentWidget()
        tab.logbox.appendPlainText(msg)

    def closeEvent(self, e):
        if getattr(self, "tab_edit", None) is not None and self.tab_edit.bulk_running():
            if QMessageBox.question(self, APP_NAME, "Đang đóng logo hàng loạt. Dừng và thoát?") != QMessageBox.Yes:
                e.ignore()
                return
            self.tab_edit.stop_bulk()
            self.tab_edit._bulk.wait(5000)
        if self.worker and self.worker.isRunning():
            if QMessageBox.question(self, APP_NAME, "Đang xử lý. Dừng và thoát?") != QMessageBox.Yes:
                e.ignore()
                return
            self.worker.cancel()
            self.worker.wait(5000)
        e.accept()


STYLE = """
QWidget { font-size: 10pt; }
#Header { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #1E1B4B, stop:1 #312E81); }
#AppTitle { color: white; font-size: 17pt; font-weight: 700; }
#AppSub { color: #C7D2FE; }
#SectionTitle { font-weight: 700; font-size: 11.5pt; color: #111827; padding: 2px 0 4px 0; }
#Hint { color: #6B7280; font-size: 9.5pt; }
#DetectLabel { font-weight: 600; padding: 4px 2px; font-size: 10.5pt; }
#Card { background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 10px; }
#Bottom { background: #FFFFFF; border-top: 1px solid #E5E7EB; }
QMainWindow, QTabWidget::pane { background: #F3F4F8; }
QTabWidget::pane { border: none; border-top: 3px solid %(accent)s; }
QTabBar::tab { padding: 10px 30px; margin: 8px 4px 0 4px; border-top-left-radius: 10px;
  border-top-right-radius: 10px; background: #EEF2FF; color: %(accent)s; font-weight: 700;
  font-size: 12.5pt; border: 1px solid #C7D2FE; border-bottom: none; }
QTabBar::tab:hover { background: #E0E7FF; }
QTabBar::tab:selected { background: %(accent)s; color: white; border-color: %(accent)s; }
QPushButton, QToolButton { padding: 7px 14px; border: 1px solid #D1D5DB; border-radius: 8px;
  background: #FFFFFF; color: #111827; font-size: 10.5pt; font-weight: 600; }
QPushButton:hover, QToolButton:hover { border-color: %(accent)s; }
QPushButton:disabled, QToolButton:disabled { color: #9CA3AF; background: #F3F4F6; border-color: #E5E7EB; }
QToolButton#Seg { background: #EEF2FF; color: %(accent)s; border-color: #C7D2FE; font-size: 11pt; }
QToolButton#Seg:checked { background: %(accent)s; color: white; border-color: %(accent)s; }
QToolButton#Tool, QPushButton#Folder { background: #E0F2FE; color: #0369A1; border-color: #BAE6FD; font-size: 11pt; }
QToolButton#Tool:hover, QPushButton#Folder:hover { background: #BAE6FD; }
QPushButton#Open { background: #ECFDF5; color: #047857; border-color: #A7F3D0; font-size: 11pt; }
QPushButton#Open:hover { background: #D1FAE5; }
QPushButton#Primary { background: %(accent)s; color: white; border: none; font-weight: 700; font-size: 12pt; }
QPushButton#Primary:hover { background: #4338CA; }
QPushButton#Primary:disabled { background: #A5B4FC; color: #EEF2FF; }
QPushButton#Scan { background: #0891B2; color: white; border: none; font-weight: 700; font-size: 11.5pt; }
QPushButton#Scan:hover { background: #0E7490; }
QPushButton#Scan:disabled { background: #A5F3FC; color: #ECFEFF; }
QPushButton#ManualAll { background: #D97706; color: white; border: none; font-weight: 700; font-size: 11pt; }
QPushButton#ManualAll:hover { background: #B45309; }
QPushButton#ManualAll:disabled { background: #FDE68A; color: #FFFBEB; }
QPushButton#Danger { background: #DC2626; color: white; border: none; font-weight: 700; font-size: 12pt; }
QPushButton#Danger:hover { background: #B91C1C; }
QPushButton#Danger:disabled { background: #FCA5A5; color: #FEF2F2; }
QPushButton#Secondary { background: #EEF2FF; color: %(accent)s; border-color: #C7D2FE; font-weight: 600; }
QTreeWidget { font-size: 10.5pt; }
QTreeWidget::item { padding: 3px 0; }
QTreeWidget, QPlainTextEdit, QLineEdit, QComboBox, QDoubleSpinBox { border: 1px solid #D1D5DB;
  border-radius: 7px; background: white; padding: 3px; }
QProgressBar { border: 1px solid #D1D5DB; border-radius: 8px; text-align: center; background: #F9FAFB;
  min-height: 28px; font-size: 11pt; font-weight: 600; }
QProgressBar::chunk { background: %(accent)s; border-radius: 7px; }
""" % {"accent": ACCENT}


def main():
    if os.name == "nt":
        try:  # own taskbar icon/group instead of pythonw's
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("thsky90.AIRemove")
        except Exception:
            pass
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    pal = app.palette()
    pal.setColor(QPalette.Window, QColor("#F5F6FA"))
    pal.setColor(QPalette.Base, QColor("#FFFFFF"))
    pal.setColor(QPalette.AlternateBase, QColor("#F9FAFB"))
    pal.setColor(QPalette.Text, QColor("#111827"))
    pal.setColor(QPalette.WindowText, QColor("#111827"))
    pal.setColor(QPalette.ButtonText, QColor("#111827"))
    pal.setColor(QPalette.Highlight, QColor(ACCENT))
    app.setPalette(pal)
    app.setStyleSheet(STYLE)
    f = app.font()
    if os.name == "nt":
        f.setFamily("Segoe UI")
    app.setFont(f)
    w = MainWindow()
    w.show()
    args = [a for a in sys.argv[1:] if os.path.exists(a)]
    if args:
        w.dropEvent  # noqa
        imgs = [a for a in args if os.path.isdir(a) or batch.is_image(a)]
        vids = [a for a in args if batch.is_video(a)]
        if imgs:
            w.tab_img.add_paths(imgs)
        if vids:
            w.tab_vid.add_paths(vids)
            if not imgs:
                w.tabs.setCurrentIndex(1)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
