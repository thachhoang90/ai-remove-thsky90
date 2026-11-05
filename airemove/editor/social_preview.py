"""Giả lập xem trước ảnh trên mạng xã hội (Social Grid Preview).

Hai chế độ (chỉ là mô phỏng bố cục, không dùng logo hay giao diện thật của
các nền tảng):
    "instagram": trang cá nhân dạng lưới 3 cột. Từ 2025 lưới hồ sơ Instagram
                 hiển thị ô dọc 3:4 (có thể chuyển sang ô vuông 1:1). Ảnh
                 đang sửa đặt ở ô đầu tiên; tùy chọn "puzzle" chia ảnh thành
                 9 ô liền nhau để xem hiệu ứng ghép lưới.
    "facebook" : một bài đăng trên bảng tin. Ảnh dọc cao hơn 4:5 sẽ bị cắt
                 giữa còn 4:5 như cách bảng tin hiển thị trên điện thoại.
"""
from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget


class SocialPreview(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.img: QImage | None = None
        self.mode = "instagram"
        self.puzzle = False
        self.cell_ratio = 3 / 4          # rộng/cao của ô lưới Instagram

    def set_image(self, img: QImage | None):
        self.img = img
        self.update()

    def set_mode(self, mode: str, puzzle: bool = False, square: bool = False):
        self.mode = mode
        self.puzzle = puzzle
        self.cell_ratio = 1.0 if square else 3 / 4
        self.update()

    # ---------------------------------------------------------------- vẽ
    @staticmethod
    def _cover(p: QPainter, img: QImage, target: QRectF, src: QRectF | None = None):
        """Vẽ ảnh phủ kín ô 'target' (cắt giữa, giữ tỉ lệ) — kiểu 'object-fit: cover'."""
        sx, sy, sw, sh = (0.0, 0.0, float(img.width()), float(img.height())) if src is None else \
            (src.x(), src.y(), src.width(), src.height())
        tr = target.width() / target.height()
        if sw / sh > tr:
            nw = sh * tr
            sx += (sw - nw) / 2
            sw = nw
        else:
            nh = sw / tr
            sy += (sh - nh) / 2
            sh = nh
        p.drawImage(target, img, QRectF(sx, sy, sw, sh))

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor("#0E1014"))
        # Khung điện thoại
        H = self.height() - 40
        W = H * 9 / 19.5
        if W > self.width() - 40:
            W = self.width() - 40
            H = W * 19.5 / 9
        phone = QRectF((self.width() - W) / 2, (self.height() - H) / 2, W, H)
        path = QPainterPath()
        path.addRoundedRect(phone, 34, 34)
        p.fillPath(path, QColor("#000000"))
        p.setPen(QPen(QColor("#3A3F4B"), 3))
        p.drawPath(path)
        screen = phone.adjusted(10, 10, -10, -10)
        p.save()
        sp = QPainterPath()
        sp.addRoundedRect(screen, 26, 26)
        p.setClipPath(sp)
        p.fillRect(screen, QColor("#FFFFFF") if self.mode == "facebook" else QColor("#000000"))
        if self.mode == "facebook":
            self._paint_facebook(p, screen)
        else:
            self._paint_instagram(p, screen)
        p.restore()
        p.setPen(QColor("#8B93A7"))
        p.setFont(QFont(self.font().family(), 9))
        p.drawText(QRectF(0, self.height() - 22, self.width(), 20), Qt.AlignCenter,
                   "Mô phỏng bố cục hiển thị – không phải giao diện thật của nền tảng")

    def _placeholder(self, p, r: QRectF, i: int):
        cols = [("#334155", "#1E293B"), ("#3F3F46", "#27272A"), ("#1F2937", "#111827"), ("#374151", "#1F2937")]
        a, b = cols[i % len(cols)]
        g = QLinearGradient(r.topLeft(), r.bottomRight())
        g.setColorAt(0, QColor(a))
        g.setColorAt(1, QColor(b))
        p.fillRect(r, g)

    def _paint_instagram(self, p: QPainter, s: QRectF):
        f = self.font().family()
        x, y, w = s.x(), s.y(), s.width()
        p.setPen(QColor("#FFFFFF"))
        p.setFont(QFont(f, 11, QFont.Bold))
        p.drawText(QRectF(x + 16, y + 30, w - 32, 24), Qt.AlignLeft | Qt.AlignVCenter, "ten_cua_ban")
        # ảnh đại diện + số liệu
        av = QRectF(x + 16, y + 66, 64, 64)
        p.setBrush(QColor("#4F46E5"))
        p.setPen(QPen(QColor("#E5E7EB"), 2))
        p.drawEllipse(av)
        p.setFont(QFont(f, 10, QFont.Bold))
        for i, (num, lab) in enumerate((("9", "bài viết"), ("1.250", "theo dõi"), ("180", "đang theo dõi"))):
            cx = x + 96 + i * (w - 112) / 3
            p.setPen(QColor("#FFFFFF"))
            p.drawText(QRectF(cx, y + 74, (w - 112) / 3, 20), Qt.AlignCenter, num)
            p.setFont(QFont(f, 7))
            p.setPen(QColor("#A1A1AA"))
            p.drawText(QRectF(cx, y + 96, (w - 112) / 3, 18), Qt.AlignCenter, lab)
            p.setFont(QFont(f, 10, QFont.Bold))
        p.setPen(QColor("#FFFFFF"))
        p.setFont(QFont(f, 9))
        p.drawText(QRectF(x + 16, y + 140, w - 32, 18), Qt.AlignLeft, "Tên hiển thị")
        p.setPen(QColor("#A1A1AA"))
        p.drawText(QRectF(x + 16, y + 158, w - 32, 18), Qt.AlignLeft, "Mô tả ngắn về trang của bạn")
        # nút
        bw = (w - 40) / 2
        for i, t in enumerate(("Chỉnh sửa", "Chia sẻ")):
            r = QRectF(x + 16 + i * (bw + 8), y + 184, bw, 28)
            p.setBrush(QColor("#27272A"))
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(r, 8, 8)
            p.setPen(QColor("#FFFFFF"))
            p.drawText(r, Qt.AlignCenter, t)
        # lưới 3 cột
        top = y + 226
        gap = 2
        cw = (w - 2 * gap) / 3
        ch = cw / self.cell_ratio
        img = self.img
        for i in range(12):
            r, c = divmod(i, 3)
            cell = QRectF(x + c * (cw + gap), top + r * (ch + gap), cw, ch)
            if cell.top() > s.bottom():
                break
            if img is not None and self.puzzle and i < 9:
                # chia ảnh thành lưới 3×3 theo tỉ lệ tổng của 9 ô
                tot = QRectF(0, 0, 3 * cw + 2 * gap, 3 * ch + 2 * gap)
                k = max(tot.width() / img.width(), tot.height() / img.height())
                ox = (img.width() * k - tot.width()) / 2
                oy = (img.height() * k - tot.height()) / 2
                src = QRectF((c * (cw + gap) + ox) / k, (r * (ch + gap) + oy) / k, cw / k, ch / k)
                p.drawImage(cell, img, src)
            elif img is not None and i == 0:
                self._cover(p, img, cell)
            else:
                self._placeholder(p, cell, i)

    def _paint_facebook(self, p: QPainter, s: QRectF):
        f = self.font().family()
        x, y, w = s.x(), s.y(), s.width()
        p.fillRect(QRectF(x, y, w, 56), QColor("#FFFFFF"))
        p.setPen(QColor("#111827"))
        p.setFont(QFont(f, 13, QFont.Bold))
        p.drawText(QRectF(x + 14, y + 22, w - 28, 28), Qt.AlignLeft | Qt.AlignVCenter, "Bảng tin")
        p.fillRect(QRectF(x, y + 58, w, 8), QColor("#E5E7EB"))
        cy = y + 76
        p.setBrush(QColor("#4F46E5"))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QRectF(x + 12, cy, 40, 40))
        p.setPen(QColor("#111827"))
        p.setFont(QFont(f, 10, QFont.Bold))
        p.drawText(QRectF(x + 60, cy + 2, w - 70, 20), Qt.AlignLeft, "Trang của bạn")
        p.setPen(QColor("#6B7280"))
        p.setFont(QFont(f, 8))
        p.drawText(QRectF(x + 60, cy + 22, w - 70, 18), Qt.AlignLeft, "Vừa xong · Công khai")
        p.setPen(QColor("#111827"))
        p.setFont(QFont(f, 9))
        p.drawText(QRectF(x + 12, cy + 50, w - 24, 20), Qt.AlignLeft, "Nội dung bài viết của bạn…")
        top = cy + 76
        img = self.img
        if img is not None:
            ar = img.width() / img.height()
            shown = max(ar, 4 / 5)               # ảnh dọc hơn 4:5 bị cắt còn 4:5
            ih = w / shown
            self._cover(p, img, QRectF(x, top, w, ih))
            top += ih
        else:
            self._placeholder(p, QRectF(x, top, w, w), 0)
            top += w
        p.setPen(QColor("#6B7280"))
        p.setFont(QFont(f, 8))
        p.drawText(QRectF(x + 12, top + 6, w - 24, 18), Qt.AlignLeft, "Bạn và 248 người khác")
        p.drawText(QRectF(x + 12, top + 6, w - 24, 18), Qt.AlignRight, "32 bình luận")
        p.fillRect(QRectF(x + 12, top + 28, w - 24, 1), QColor("#E5E7EB"))
        p.setFont(QFont(f, 9, QFont.Bold))
        for i, t in enumerate(("Thích", "Bình luận", "Chia sẻ")):
            p.drawText(QRectF(x + i * w / 3, top + 32, w / 3, 26), Qt.AlignCenter, t)
        p.fillRect(QRectF(x, top + 64, w, 8), QColor("#E5E7EB"))
