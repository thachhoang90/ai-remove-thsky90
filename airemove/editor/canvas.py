"""Vùng hiển thị ảnh của tab "Chỉnh sửa ảnh".

Ba chế độ:
    "view"  : xem ảnh — lăn chuột để phóng to/thu nhỏ, kéo chuột để di chuyển.
    "crop"  : khung cắt — kéo bên trong để di chuyển, kéo góc/cạnh để co giãn;
              nếu có tỉ lệ cố định (khung mẫu mạng xã hội) thì giữ đúng tỉ lệ.
    "brush" : cọ vẽ chọn vật thể cần xóa — chuột trái tô, chuột phải tẩy.

Ảnh hiển thị có thể là bản thu nhỏ (để xem trước nhanh), nhưng MỌI tọa độ
(khung cắt, mặt nạ cọ vẽ) đều tính theo kích thước THẬT của ảnh ("logical
size"), nên khi áp dụng không bị lệch.
"""
from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget


def np_to_qimage(rgb: np.ndarray) -> QImage:
    """numpy RGB uint8 -> QImage (sao chép dữ liệu để an toàn bộ nhớ)."""
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


class EditorCanvas(QWidget):
    cropChanged = Signal(float, float, float, float)
    maskChanged = Signal()

    HANDLE = 10

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(360, 260)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.img: QImage | None = None
        self.logical = (0, 0)             # kích thước thật (rộng, cao) của ảnh
        self.mode = "view"
        self.scale = 1.0                  # điểm màn hình / điểm ảnh thật
        self.off = QPointF(0, 0)
        self.fitted = True
        self._drag = None
        self.message = "Mở một ảnh để bắt đầu chỉnh sửa"
        self.tag = ""
        # khung cắt
        self.crop = None                  # (x, y, w, h) theo ảnh thật
        self.crop_ratio = None            # rộng/cao hoặc None (tự do)
        # cọ vẽ
        self.mask: np.ndarray | None = None
        self.brush = 40
        self._mask_img: QImage | None = None
        self._cursor = None
        self._last_pt = None

    # ------------------------------------------------------------------ dữ liệu
    def set_image(self, img: QImage | None, logical=None, keep_view=True):
        old = self.logical
        self.img = img
        self.logical = logical or ((img.width(), img.height()) if img is not None else (0, 0))
        if not keep_view or old != self.logical or self.fitted:
            self.fit()
        self.update()

    def set_mode(self, mode: str):
        self.mode = mode
        if mode == "brush" and self.logical[0]:
            W, H = self.logical
            if self.mask is None or self.mask.shape != (H, W):
                self.mask = np.zeros((H, W), np.uint8)
                self._mask_img = None
        self.setCursor(Qt.CrossCursor if mode != "view" else Qt.OpenHandCursor)
        self.update()

    def clear_mask(self):
        if self.mask is not None:
            self.mask[:] = 0
        self._mask_img = None
        self.update()
        self.maskChanged.emit()

    def set_crop(self, rect, ratio=None):
        self.crop = rect
        self.crop_ratio = ratio
        self.update()

    # ------------------------------------------------------------------ hiển thị
    def fit(self):
        self.fitted = True
        W, H = self.logical
        if not W:
            self.update()
            return
        self.scale = max(0.01, min((self.width() - 24) / W, (self.height() - 24) / H))
        self.off = QPointF((self.width() - W * self.scale) / 2, (self.height() - H * self.scale) / 2)
        self.update()

    def zoom_actual(self):
        """Hiển thị 100% (1 điểm ảnh thật = 1 điểm màn hình) quanh tâm."""
        W, H = self.logical
        if not W:
            return
        cx = (self.width() / 2 - self.off.x()) / self.scale
        cy = (self.height() / 2 - self.off.y()) / self.scale
        self.scale = 1.0
        self.off = QPointF(self.width() / 2 - cx, self.height() / 2 - cy)
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
        if self.img is None:
            return
        f = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        p = e.position()
        ip = self.to_img(p)
        self.scale = max(0.02, min(32.0, self.scale * f))
        self.off = QPointF(p.x() - ip.x() * self.scale, p.y() - ip.y() * self.scale)
        self.fitted = False
        self.update()

    def _mask_overlay(self) -> QImage | None:
        """Ảnh phủ màu đỏ bán trong suốt thể hiện vùng đã tô (dựng ở độ phân giải hiển thị)."""
        if self.mask is None or not self.mask.any():
            return None
        if self._mask_img is None:
            dw = max(1, min(self.mask.shape[1], 1600))
            dh = max(1, round(self.mask.shape[0] * dw / self.mask.shape[1]))
            m = cv2.resize(self.mask, (dw, dh), interpolation=cv2.INTER_AREA)
            rgba = np.zeros((dh, dw, 4), np.uint8)
            rgba[..., 0] = 255
            rgba[..., 1] = 64
            rgba[..., 2] = 64
            rgba[..., 3] = (m.astype(np.float32) * (120 / 255.0)).astype(np.uint8)
            self._mask_img = QImage(rgba.data, dw, dh, 4 * dw, QImage.Format_RGBA8888).copy()
        return self._mask_img

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0E1014"))
        if self.img is None:
            p.setPen(QColor("#8B93A7"))
            p.setFont(QFont(self.font().family(), 12))
            p.drawText(self.rect(), Qt.AlignCenter, self.message)
            return
        W, H = self.logical
        target = QRectF(self.off.x(), self.off.y(), W * self.scale, H * self.scale)
        p.setRenderHint(QPainter.SmoothPixmapTransform, self.scale * W / max(1, self.img.width()) < 2.5)
        p.drawImage(target, self.img)
        if self.mode == "brush":
            ov = self._mask_overlay()
            if ov is not None:
                p.drawImage(target, ov)
            if self._cursor is not None:
                pen = QPen(QColor(255, 255, 255, 220), 1.5)
                pen.setCosmetic(True)
                p.setPen(pen)
                p.setBrush(Qt.NoBrush)
                r = self.brush / 2 * self.scale
                p.drawEllipse(self._cursor, r, r)
        if self.mode == "crop" and self.crop:
            x, y, w, h = self.crop
            tl = self.to_view(x, y)
            r = QRectF(tl.x(), tl.y(), w * self.scale, h * self.scale)
            shade = QColor(0, 0, 0, 150)
            # làm tối phần nằm ngoài khung cắt
            p.fillRect(QRectF(target.left(), target.top(), target.width(), r.top() - target.top()), shade)
            p.fillRect(QRectF(target.left(), r.bottom(), target.width(), target.bottom() - r.bottom()), shade)
            p.fillRect(QRectF(target.left(), r.top(), r.left() - target.left(), r.height()), shade)
            p.fillRect(QRectF(r.right(), r.top(), target.right() - r.right(), r.height()), shade)
            pen = QPen(QColor("#FFFFFF"), 1.5)
            pen.setCosmetic(True)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawRect(r)
            thirds = QPen(QColor(255, 255, 255, 90), 1)
            thirds.setCosmetic(True)
            p.setPen(thirds)
            for i in (1, 2):                      # đường chia 1/3 (bố cục)
                p.drawLine(QPointF(r.left() + r.width() * i / 3, r.top()), QPointF(r.left() + r.width() * i / 3, r.bottom()))
                p.drawLine(QPointF(r.left(), r.top() + r.height() * i / 3), QPointF(r.right(), r.top() + r.height() * i / 3))
            p.setPen(Qt.NoPen)
            p.setBrush(QColor("#38BDF8"))
            for c in self._handles(r).values():
                p.drawRect(QRectF(c.x() - self.HANDLE / 2, c.y() - self.HANDLE / 2, self.HANDLE, self.HANDLE))
            p.setPen(QColor("#FFFFFF"))
            label = f"{int(round(w))} × {int(round(h))} px"
            p.fillRect(QRectF(r.left(), r.top() - 24, 130, 22), QColor(0, 0, 0, 170))
            p.drawText(QRectF(r.left(), r.top() - 24, 130, 22), Qt.AlignCenter, label)
        if self.tag:
            p.fillRect(QRectF(10, 10, 150, 24), QColor(0, 0, 0, 160))
            p.setPen(QColor("#E5E7EB"))
            p.drawText(QRectF(10, 10, 150, 24), Qt.AlignCenter, self.tag)

    def _handles(self, r: QRectF) -> dict:
        h = {"tl": r.topLeft(), "tr": r.topRight(), "bl": r.bottomLeft(), "br": r.bottomRight()}
        if self.crop_ratio is None:              # tự do: thêm điểm giữa các cạnh
            h.update({"t": QPointF(r.center().x(), r.top()), "b": QPointF(r.center().x(), r.bottom()),
                      "l": QPointF(r.left(), r.center().y()), "r": QPointF(r.right(), r.center().y())})
        return h

    # ------------------------------------------------------------------ chuột
    def _crop_hit(self, pos):
        if not self.crop:
            return None
        x, y, w, h = self.crop
        tl = self.to_view(x, y)
        r = QRectF(tl.x(), tl.y(), w * self.scale, h * self.scale)
        for k, c in self._handles(r).items():
            if abs(pos.x() - c.x()) <= self.HANDLE + 3 and abs(pos.y() - c.y()) <= self.HANDLE + 3:
                return k
        return "move" if r.contains(pos) else None

    def mousePressEvent(self, e):
        if self.img is None:
            return
        pos = e.position()
        pan = e.button() == Qt.MiddleButton or (self.mode != "brush" and e.button() == Qt.RightButton)
        if self.mode == "view" or pan:
            self._drag = ("pan", pos, QPointF(self.off))
            self.setCursor(Qt.ClosedHandCursor)
            return
        if self.mode == "crop":
            hit = self._crop_hit(pos)
            if hit:
                self._drag = ("crop", hit, self.to_img(pos), tuple(self.crop))
            return
        if self.mode == "brush":
            erase = e.button() == Qt.RightButton
            ip = self.to_img(pos)
            self._last_pt = (ip.x(), ip.y())
            self._paint(ip.x(), ip.y(), ip.x(), ip.y(), erase)
            self._drag = ("brush", erase)

    def mouseMoveEvent(self, e):
        pos = e.position()
        if self.mode == "brush":
            self._cursor = pos
            self.update()
        if self._drag is None:
            if self.mode == "crop":
                hit = self._crop_hit(pos)
                cur = {"move": Qt.SizeAllCursor, "tl": Qt.SizeFDiagCursor, "br": Qt.SizeFDiagCursor,
                       "tr": Qt.SizeBDiagCursor, "bl": Qt.SizeBDiagCursor, "t": Qt.SizeVerCursor,
                       "b": Qt.SizeVerCursor, "l": Qt.SizeHorCursor, "r": Qt.SizeHorCursor}.get(hit, Qt.ArrowCursor)
                self.setCursor(cur)
            return
        kind = self._drag[0]
        if kind == "pan":
            _, start, orig = self._drag
            self.off = orig + (pos - start)
            self.fitted = False
            self.update()
        elif kind == "crop":
            _, hit, start, orig = self._drag
            ip = self.to_img(pos)
            self.crop = self._resize_crop(hit, ip.x() - start.x(), ip.y() - start.y(), orig)
            self.update()
            self.cropChanged.emit(*self.crop)
        elif kind == "brush":
            ip = self.to_img(pos)
            x0, y0 = self._last_pt
            self._paint(x0, y0, ip.x(), ip.y(), self._drag[1])
            self._last_pt = (ip.x(), ip.y())

    def mouseReleaseEvent(self, e):
        if self._drag and self._drag[0] == "brush":
            self.maskChanged.emit()
        self._drag = None
        if self.mode == "view":
            self.setCursor(Qt.OpenHandCursor)

    def leaveEvent(self, e):
        self._cursor = None
        self.update()

    def _paint(self, x0, y0, x1, y1, erase):
        """Tô (hoặc tẩy) một nét cọ hình tròn lên mặt nạ ở độ phân giải thật."""
        if self.mask is None:
            return
        val = 0 if erase else 255
        cv2.line(self.mask, (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))),
                 val, thickness=max(1, int(self.brush)), lineType=cv2.LINE_AA)
        if erase:
            self.mask[self.mask < 255] = 0
        else:
            self.mask[self.mask > 0] = 255
        self._mask_img = None
        self.update()

    def _resize_crop(self, hit, dx, dy, orig):
        """Di chuyển / co giãn khung cắt, luôn nằm trong ảnh và giữ tỉ lệ nếu có."""
        W, H = self.logical
        x, y, w, h = orig
        ratio = self.crop_ratio
        minw = 16
        if hit == "move":
            nx = min(max(0.0, x + dx), W - w)
            ny = min(max(0.0, y + dy), H - h)
            return (nx, ny, w, h)
        # điểm neo = góc/cạnh đối diện
        left, top, right, bottom = x, y, x + w, y + h
        if "l" in hit:
            left = min(right - minw, max(0.0, left + dx))
        if "r" in hit:
            right = max(left + minw, min(float(W), right + dx))
        if hit.startswith("t") or hit == "t":
            top = min(bottom - minw, max(0.0, top + dy))
        if hit.startswith("b") or hit == "b":
            bottom = max(top + minw, min(float(H), bottom + dy))
        nw, nh = right - left, bottom - top
        if ratio:
            # giữ tỉ lệ: lấy theo cạnh thay đổi nhiều hơn, rồi giới hạn trong ảnh
            if abs(dx) >= abs(dy):
                nh = nw / ratio
            else:
                nw = nh * ratio
            ax = right if "l" in hit else left
            ay = bottom if hit.startswith("t") else top
            maxw = ax if "l" in hit else W - ax
            maxh = ay if hit.startswith("t") else H - ay
            if nw > maxw:
                nw, nh = maxw, maxw / ratio
            if nh > maxh:
                nh, nw = maxh, maxh * ratio
            left = ax - nw if "l" in hit else ax
            top = ay - nh if hit.startswith("t") else ay
        return (left, top, nw, nh)
