"""Thanh trượt ngang + dọc cho khung xem trước (v2.3).

Các khung xem trước của ứng dụng (tab Ảnh, tab Video, tab Chỉnh sửa ảnh) tự vẽ
ảnh theo hai giá trị: ``scale`` (độ phóng) và ``off`` (vị trí góc trên-trái của
ảnh trong khung).  Lớp ``ScrollFrame`` bọc một khung như vậy và thêm hai thanh
trượt chuẩn của hệ điều hành:

* Phóng to ảnh  -> thanh trượt sáng lên, kéo để tới nhanh vùng cần xem / định vị.
* Ảnh nằm trọn trong khung -> thanh trượt mờ đi (không có gì để cuộn).
* Lăn chuột để phóng, giữ chuột phải/giữa để kéo ảnh vẫn dùng như trước; thanh
  trượt tự cập nhật theo.

Không cần sửa logic vẽ của khung: ``ScrollFrame`` nghe sự kiện vẽ lại/đổi kích
thước của khung để đồng bộ thanh trượt, và ghi ngược ``off`` khi người dùng kéo.
"""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QEvent, QPointF, Qt, QTimer
from PySide6.QtWidgets import QGridLayout, QScrollBar, QWidget


class ScrollFrame(QWidget):
    MARGIN = 12          # khoảng trống quanh ảnh khi cuộn hết cỡ (điểm màn hình)

    def __init__(self, canvas: QWidget, content_size: Callable[[], tuple], parent=None):
        """``canvas``: khung vẽ có thuộc tính ``scale``, ``off`` (QPointF), ``fitted``.
        ``content_size``: hàm trả về (rộng, cao) của ảnh theo điểm ảnh thật."""
        super().__init__(parent)
        self.canvas = canvas
        self._size = content_size
        self.hbar = QScrollBar(Qt.Horizontal)
        self.vbar = QScrollBar(Qt.Vertical)
        for bar in (self.hbar, self.vbar):
            bar.setEnabled(False)
            bar.setFocusPolicy(Qt.NoFocus)
            bar.setToolTip("Kéo để di chuyển vùng xem khi ảnh đang phóng to")
        g = QGridLayout(self)
        g.setContentsMargins(0, 0, 0, 0)
        g.setSpacing(2)
        g.addWidget(canvas, 0, 0)
        g.addWidget(self.vbar, 0, 1)
        g.addWidget(self.hbar, 1, 0)
        g.setRowStretch(0, 1)
        g.setColumnStretch(0, 1)
        canvas.installEventFilter(self)
        self.hbar.valueChanged.connect(self._scrolled)
        self.vbar.valueChanged.connect(self._scrolled)
        # lưới an toàn: đồng bộ định kỳ, phòng khi hệ thống gộp/bỏ qua sự kiện vẽ lại
        self._timer = QTimer(self)
        self._timer.setInterval(200)
        self._timer.timeout.connect(self.sync)
        self._timer.start()

    # ---- khung vẽ -> thanh trượt ---------------------------------------
    def eventFilter(self, obj, ev):
        if obj is self.canvas and ev.type() in (QEvent.Paint, QEvent.Resize):
            self.sync()
        return False

    def sync(self):
        """Cập nhật phạm vi và vị trí hai thanh trượt theo trạng thái hiện tại."""
        W, H = self._size()
        c = self.canvas
        m = self.MARGIN
        for bar, content, view, off in ((self.hbar, W * c.scale, c.width(), c.off.x()),
                                        (self.vbar, H * c.scale, c.height(), c.off.y())):
            bar.blockSignals(True)
            if not (W and H) or content <= view:
                # ảnh nằm trọn trong khung theo chiều này: không có gì để cuộn
                bar.setRange(0, 0)
                bar.setEnabled(False)
            else:
                val = int(round(m - off))
                rng = int(round(content + 2 * m - view))
                # nếu người dùng đã kéo ảnh ra ngoài phạm vi thường (ví dụ phóng vào
                # logo ở sát góc), nới phạm vi để thanh trượt đi tiếp từ đúng vị trí
                # hiện tại thay vì giật về mép
                bar.setEnabled(True)
                bar.setRange(min(0, val), max(rng, val))
                bar.setPageStep(max(1, int(view)))
                bar.setSingleStep(max(8, int(view) // 12))
                bar.setValue(val)
            bar.blockSignals(False)

    # ---- thanh trượt -> khung vẽ ---------------------------------------
    def _scrolled(self, *_):
        c = self.canvas
        x = self.MARGIN - self.hbar.value() if self.hbar.isEnabled() else c.off.x()
        y = self.MARGIN - self.vbar.value() if self.vbar.isEnabled() else c.off.y()
        c.off = QPointF(x, y)
        c.fitted = False
        c.update()
