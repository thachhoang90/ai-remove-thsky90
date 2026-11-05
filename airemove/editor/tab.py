"""Tab "Chỉnh sửa ảnh" — giao diện tối (Dark Theme) cho bản Windows.

Mô hình dữ liệu (quan trọng để hiểu luồng xử lý):
    history.current.image  : ảnh "đã chốt" (sau các thao tác đã áp dụng).
    self.adj               : các thông số ĐANG XEM TRƯỚC (bộ lọc, sáng, tương
                             phản, bão hòa, sắc nét, mịn da) — chưa ghi vào ảnh.
    self.logo + self.wm    : lớp logo xem trước, luôn vẽ trên cùng.

Khi kéo thanh trượt, chỉ bản thu nhỏ (cạnh dài ≤ 1600 px) được tính lại để
phản hồi tức thì. Khi bấm "Áp dụng" (hoặc trước một thao tác phá hủy như cắt,
xóa vật thể), thông số được tính trên ảnh gốc độ phân giải đầy đủ và đẩy vào
lịch sử (Undo/Redo).
"""
from __future__ import annotations

import io
import json
import os

import cv2
import numpy as np
from PySide6.QtCore import QSettings, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QIcon, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox, QFileDialog,
                               QFrame, QGridLayout, QHBoxLayout, QInputDialog, QLabel, QListView,
                               QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit,
                               QProgressBar, QPushButton, QRadioButton, QScrollArea, QSizePolicy,
                               QSlider, QStackedWidget, QToolBox, QToolButton, QVBoxLayout, QWidget)

from .. import batch
from .. import imagefile as IF
from . import ops
from .bulk import BulkWatermarkWorker
from .canvas import EditorCanvas, np_to_qimage
from .history import History
from .social_preview import SocialPreview

PREVIEW_MAX = 1600       # cạnh dài tối đa của ảnh xem trước thời gian thực

DARK_QSS = """
#EditorTab { background: #15171C; }
#EditorTab QWidget { color: #E5E7EB; font-size: 10pt; }
#EditorTab QWidget#ToolPage, #EditorTab QScrollArea, #EditorTab QScrollArea > QWidget > QWidget {
  background: transparent; border: none; }
#EditorTab QFrame#Panel { background: #1C1F26; border: 1px solid #2A2F3A; border-radius: 10px; }
#EditorTab QFrame#TopBar { background: #1C1F26; border: 1px solid #2A2F3A; border-radius: 10px; }
#EditorTab QToolBox::tab { background: #232733; color: #C7D2FE; border-radius: 8px; padding: 2px 10px;
  min-height: 30px; font-weight: 700; font-size: 11pt; }
#EditorTab QToolBox::tab:selected { background: #4F46E5; color: #FFFFFF; }
#EditorTab QToolBox::tab:hover { background: #2E3445; }
#EditorTab QPushButton, #EditorTab QToolButton { background: #262B36; color: #E5E7EB; border: 1px solid #343B48;
  border-radius: 8px; padding: 7px 12px; font-weight: 600; font-size: 10.5pt; }
#EditorTab QPushButton:hover, #EditorTab QToolButton:hover { border-color: #6366F1; background: #2D3340; }
#EditorTab QPushButton:checked, #EditorTab QToolButton:checked { background: #4F46E5; border-color: #4F46E5; color: #FFFFFF; }
#EditorTab QPushButton:disabled, #EditorTab QToolButton:disabled { color: #5B6273; background: #1E222B; border-color: #2A2F3A; }
#EditorTab QPushButton#Accent { background: #4F46E5; border: 1px solid #4F46E5; color: #FFFFFF; }
#EditorTab QPushButton#Accent:hover { background: #4338CA; }
#EditorTab QPushButton#Good { background: #059669; border: 1px solid #059669; color: #FFFFFF; }
#EditorTab QPushButton#Good:hover { background: #047857; }
#EditorTab QPushButton#Danger { background: #B91C1C; border: 1px solid #B91C1C; color: #FFFFFF; }
#EditorTab QPushButton#Accent:disabled, #EditorTab QPushButton#Good:disabled, #EditorTab QPushButton#Danger:disabled {
  background: #1E222B; border-color: #2A2F3A; color: #5B6273; }
#EditorTab QPushButton#Compare { background: #0E7490; border: 1px solid #0E7490; color: #FFFFFF; }
#EditorTab QPushButton#Compare:pressed { background: #F59E0B; border-color: #F59E0B; color: #111827; }
#EditorTab QSlider::groove:horizontal { height: 6px; background: #2A2F3A; border-radius: 3px; }
#EditorTab QSlider::sub-page:horizontal { background: #6366F1; border-radius: 3px; }
#EditorTab QSlider::handle:horizontal { background: #E5E7EB; width: 16px; height: 16px; margin: -6px 0; border-radius: 8px; }
#EditorTab QComboBox, #EditorTab QListWidget, #EditorTab QPlainTextEdit { background: #11141A; border: 1px solid #343B48;
  color: #E5E7EB; border-radius: 6px; padding: 3px; }
#EditorTab QComboBox QAbstractItemView { background: #11141A; color: #E5E7EB; selection-background-color: #4F46E5; }
#EditorTab QListWidget::item { padding: 3px; border-radius: 6px; }
#EditorTab QListWidget::item:selected { background: #4F46E5; color: #FFFFFF; }
#EditorTab QCheckBox, #EditorTab QRadioButton, #EditorTab QLabel { background: transparent; color: #D1D5DB; }
#EditorTab QLabel#Hint { color: #8B93A7; font-size: 9pt; }
#EditorTab QLabel#Section { color: #A5B4FC; font-weight: 700; font-size: 10.5pt; padding-top: 6px; }
#EditorTab QLabel#Status { color: #A7F3D0; font-weight: 600; }
#EditorTab QProgressBar { background: #11141A; border: 1px solid #343B48; border-radius: 6px; text-align: center;
  color: #E5E7EB; min-height: 20px; }
#EditorTab QProgressBar::chunk { background: #059669; border-radius: 5px; }
"""


def _section(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("Section")
    return lab


def _hint(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("Hint")
    lab.setWordWrap(True)
    return lab


class SliderRow(QWidget):
    """Một hàng thanh trượt: nhãn – thanh trượt – giá trị. Nhấp đúp nhãn để đặt lại."""
    changed = Signal(int)

    def __init__(self, label, mn, mx, value=0, suffix="", parent=None):
        super().__init__(parent)
        self.default = value
        self.suffix = suffix
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.lab = QLabel(label)
        self.lab.setMinimumWidth(86)
        self.lab.setToolTip("Nhấp đúp để đặt lại")
        self.s = QSlider(Qt.Horizontal)
        self.s.setRange(mn, mx)
        self.s.setValue(value)
        self.val = QLabel()
        self.val.setMinimumWidth(44)
        self.val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lay.addWidget(self.lab)
        lay.addWidget(self.s, 1)
        lay.addWidget(self.val)
        self.s.valueChanged.connect(self._on)
        self.lab.mouseDoubleClickEvent = lambda e: self.s.setValue(self.default)
        self._on(value)

    def _on(self, v):
        self.val.setText(f"{v}{self.suffix}")
        self.changed.emit(v)

    def value(self):
        return self.s.value()

    def set_silent(self, v):
        self.s.blockSignals(True)
        self.s.setValue(v)
        self.s.blockSignals(False)
        self.val.setText(f"{v}{self.suffix}")


class EditorTab(QWidget):
    def __init__(self, main=None):
        super().__init__()
        self.main = main
        self.setObjectName("EditorTab")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(DARK_QSS)
        self.settings = QSettings("thsky90", "AI Remove")
        self.path = None
        self.original = None            # ảnh lúc mở (để so sánh trước/sau)
        self.history = History()
        self.adj = ops.Adjustments()
        self.wm = ops.WatermarkSettings()
        self.logo = None
        self.logo_path = None
        self._prev_cache = None         # (chỉ số lịch sử, ảnh thu nhỏ, hệ số)
        self._skin_prev = None          # mặt nạ da của ảnh xem trước
        self._bulk = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self.refresh_preview)
        self._build()
        self._update_buttons()

    # ====================================================================== UI
    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 8)
        root.setSpacing(8)

        # ---- thanh công cụ trên
        top = QFrame()
        top.setObjectName("TopBar")
        tl = QHBoxLayout(top)
        tl.setContentsMargins(10, 6, 10, 6)
        b_open = QPushButton("Mở ảnh…")
        b_open.setObjectName("Accent")
        b_open.clicked.connect(self.pick_image)
        b_exp = QPushButton("Xuất ảnh…")
        b_exp.setObjectName("Good")
        b_exp.clicked.connect(self.export_image)
        self.b_undo = QPushButton("↶ Hoàn tác")
        self.b_undo.setToolTip("Ctrl+Z")
        self.b_undo.clicked.connect(self.undo)
        self.b_redo = QPushButton("↷ Làm lại")
        self.b_redo.setToolTip("Ctrl+Y")
        self.b_redo.clicked.connect(self.redo)
        self.b_cmp = QPushButton("Giữ để so sánh")
        self.b_cmp.setObjectName("Compare")
        self.b_cmp.setToolTip("Nhấn giữ để xem ảnh gốc, thả ra để quay lại ảnh đang sửa")
        self.b_cmp.pressed.connect(self._show_original)
        self.b_cmp.released.connect(self._hide_original)
        b_fit = QPushButton("Vừa khung")
        b_fit.clicked.connect(lambda: self.canvas.fit())
        b_100 = QPushButton("100%")
        b_100.clicked.connect(lambda: self.canvas.zoom_actual())
        self.lbl_info = QLabel("Chưa mở ảnh")
        self.lbl_info.setObjectName("Hint")
        for w in (b_open, b_exp, self.b_undo, self.b_redo, self.b_cmp, b_fit, b_100):
            tl.addWidget(w)
        tl.addStretch(1)
        tl.addWidget(self.lbl_info)
        root.addWidget(top)

        body = QHBoxLayout()
        body.setSpacing(10)
        root.addLayout(body, 1)

        # ---- cột công cụ (5 module) bên trái
        panel = QFrame()
        panel.setObjectName("Panel")
        panel.setMinimumWidth(390)
        panel.setMaximumWidth(450)
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(8, 8, 8, 8)
        self.toolbox = QToolBox()
        pl.addWidget(self.toolbox)
        self.toolbox.addItem(self._scroll(self._page_file()), "1. Tệp và Xuất bản")
        self.toolbox.addItem(self._scroll(self._page_logo()), "2. Đóng logo")
        self.toolbox.addItem(self._scroll(self._page_adjust()), "3. Bộ lọc và Điều chỉnh")
        self.toolbox.addItem(self._scroll(self._page_retouch()), "4. Làm đẹp và AI")
        self.toolbox.addItem(self._scroll(self._page_history()), "5. Lịch sử thao tác")
        self.toolbox.setCurrentIndex(2)
        self.toolbox.currentChanged.connect(self._on_page)
        body.addWidget(panel)

        # ---- vùng ảnh
        self.stack = QStackedWidget()
        self.canvas = EditorCanvas()
        self.canvas.maskChanged.connect(self._update_buttons)
        self.social = SocialPreview()
        self.stack.addWidget(self.canvas)
        self.stack.addWidget(self.social)
        body.addWidget(self.stack, 1)

        self.lbl_status = QLabel("Mở ảnh (hoặc kéo thả ảnh vào cửa sổ) để bắt đầu.")
        self.lbl_status.setObjectName("Status")
        root.addWidget(self.lbl_status)

        for seq, fn in ((QKeySequence.Undo, self.undo), (QKeySequence.Redo, self.redo),
                        (QKeySequence("Ctrl+Y"), self.redo), (QKeySequence.Open, self.pick_image),
                        (QKeySequence.Save, self.export_image)):
            sc = QShortcut(seq, self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)

    @staticmethod
    def _scroll(w: QWidget) -> QScrollArea:
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sa.setFrameShape(QFrame.NoFrame)
        w.setObjectName("ToolPage")
        sa.setWidget(w)
        return sa

    # ---------------------------------------------------------------- MODULE 1
    def _page_file(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(_section("Khung cắt mẫu một chạm"))
        grid = QGridLayout()
        self.crop_group = QButtonGroup(self)
        self.crop_group.setExclusive(True)
        short = {"free": "Tự do", "fb_cover": "FB Cover 851×315", "fb_post": "FB Post 1.91:1",
                 "ig_post": "IG Post 1:1", "ig_portrait": "IG dọc 4:5", "story": "Story/TikTok 9:16",
                 "yt_thumb": "YouTube 16:9"}
        for i, (key, label, ratio, size) in enumerate(ops.CROP_PRESETS):
            b = QPushButton(short.get(key, label))
            b.setToolTip(label + (f" – xuất {size[0]}×{size[1]} px" if size else ""))
            b.setCheckable(True)
            b.setProperty("preset", key)
            b.clicked.connect(lambda _=False, k=key: self.start_crop(k))
            self.crop_group.addButton(b)
            grid.addWidget(b, i // 2, i % 2)
        v.addLayout(grid)
        self.chk_resize = QCheckBox("Thu về kích thước chuẩn của nền tảng khi cắt")
        self.chk_resize.setChecked(True)
        v.addWidget(self.chk_resize)
        row = QHBoxLayout()
        self.b_crop_apply = QPushButton("Áp dụng cắt")
        self.b_crop_apply.setObjectName("Accent")
        self.b_crop_apply.clicked.connect(self.apply_crop)
        self.b_crop_cancel = QPushButton("Hủy cắt")
        self.b_crop_cancel.clicked.connect(self.cancel_crop)
        row.addWidget(self.b_crop_apply)
        row.addWidget(self.b_crop_cancel)
        v.addLayout(row)
        v.addWidget(_hint("Kéo bên trong khung để di chuyển, kéo góc để co giãn (giữ đúng tỉ lệ khung mẫu)."))

        v.addWidget(_section("Xuất file"))
        g = QGridLayout()
        g.addWidget(QLabel("Định dạng"), 0, 0)
        self.cmb_fmt = QComboBox()
        for t, d in (("Giữ như ảnh gốc", "keep"), ("JPG", "jpg"), ("PNG", "png"), ("WEBP", "webp")):
            self.cmb_fmt.addItem(t, d)
        g.addWidget(self.cmb_fmt, 0, 1)
        v.addLayout(g)
        self.sl_quality = SliderRow("Chất lượng", 40, 100, 92)
        v.addWidget(self.sl_quality)
        self.chk_smart = QCheckBox("Nén thông minh cho mạng xã hội")
        self.chk_smart.setChecked(True)
        self.chk_smart.toggled.connect(lambda on: self.sl_quality.setEnabled(not on))
        self.sl_quality.setEnabled(False)
        v.addWidget(self.chk_smart)
        g2 = QHBoxLayout()
        g2.addWidget(QLabel("Nền tảng"))
        self.cmb_platform = QComboBox()
        for k, rule in ops.PLATFORMS.items():
            self.cmb_platform.addItem(rule["label"], k)
        g2.addWidget(self.cmb_platform, 1)
        v.addLayout(g2)
        v.addWidget(_hint("Nén thông minh: tự thu về đúng kích thước nền tảng hiển thị, rồi tìm mức nén nhỏ nhất "
                          "mà mắt thường không phân biệt được (SSIM ≥ 0,985) để Facebook/Instagram không nén lại "
                          "làm mờ ảnh. File xuất không kèm metadata ẩn."))
        b = QPushButton("Xuất ảnh…")
        b.setObjectName("Good")
        b.clicked.connect(self.export_image)
        v.addWidget(b)

        v.addWidget(_section("Xem trước trên mạng xã hội"))
        self.chk_social = QCheckBox("Bật chế độ xem trước")
        self.chk_social.toggled.connect(self._toggle_social)
        v.addWidget(self.chk_social)
        row = QHBoxLayout()
        self.rb_ig = QRadioButton("Instagram – lưới 3×3")
        self.rb_fb = QRadioButton("Facebook – bảng tin")
        self.rb_ig.setChecked(True)
        for rb in (self.rb_ig, self.rb_fb):
            rb.toggled.connect(self._update_social)
            row.addWidget(rb)
        v.addLayout(row)
        self.chk_puzzle = QCheckBox("Chia ảnh thành 9 ô (puzzle)")
        self.chk_square = QCheckBox("Ô vuông 1:1 (mặc định ô dọc 3:4)")
        for c in (self.chk_puzzle, self.chk_square):
            c.toggled.connect(self._update_social)
            v.addWidget(c)
        v.addStretch(1)
        return w

    # ---------------------------------------------------------------- MODULE 2
    def _page_logo(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(_section("Logo"))
        row = QHBoxLayout()
        b = QPushButton("Chọn logo (PNG trong suốt)…")
        b.clicked.connect(self.pick_logo)
        row.addWidget(b, 1)
        self.lbl_logo = QLabel()
        self.lbl_logo.setFixedSize(64, 40)
        self.lbl_logo.setAlignment(Qt.AlignCenter)
        self.lbl_logo.setStyleSheet("background:#2A2F3A;border-radius:6px;")
        row.addWidget(self.lbl_logo)
        v.addLayout(row)
        self.chk_wm = QCheckBox("Hiện logo trên ảnh (xem trước)")
        self.chk_wm.setChecked(True)
        self.chk_wm.toggled.connect(self.schedule_preview)
        v.addWidget(self.chk_wm)

        v.addWidget(_section("Vị trí"))
        grid = QGridLayout()
        self.pos_group = QButtonGroup(self)
        arrows = {"tl": "↖", "tc": "↑", "tr": "↗", "ml": "←", "c": "●", "mr": "→", "bl": "↙", "bc": "↓", "br": "↘"}
        for i, key in enumerate(("tl", "tc", "tr", "ml", "c", "mr", "bl", "bc", "br")):
            tb = QToolButton()
            tb.setText(arrows[key])
            tb.setToolTip(ops.POSITIONS[key][0])
            tb.setCheckable(True)
            tb.setFixedSize(46, 34)
            tb.setChecked(key == self.wm.position)
            tb.clicked.connect(lambda _=False, k=key: self._set_wm(position=k))
            self.pos_group.addButton(tb)
            grid.addWidget(tb, i // 3, i % 3)
        grid.setColumnStretch(3, 1)
        v.addLayout(grid)
        self.sl_op = SliderRow("Độ đậm", 0, 100, int(self.wm.opacity), "%")
        self.sl_op.changed.connect(lambda val: self._set_wm(opacity=val))
        self.sl_scale = SliderRow("Kích thước", 2, 60, int(self.wm.scale), "%")
        self.sl_scale.setToolTip("Chiều rộng logo tính theo % chiều rộng ảnh")
        self.sl_scale.changed.connect(lambda val: self._set_wm(scale=val))
        self.sl_margin = SliderRow("Cách mép", 0, 15, int(self.wm.margin), "%")
        self.sl_margin.changed.connect(lambda val: self._set_wm(margin=val))
        for s in (self.sl_op, self.sl_scale, self.sl_margin):
            v.addWidget(s)

        v.addWidget(_section("Mẫu thiết lập (preset)"))
        row = QHBoxLayout()
        self.cmb_preset = QComboBox()
        self.cmb_preset.activated.connect(self._load_preset)
        row.addWidget(self.cmb_preset, 1)
        b = QPushButton("Lưu")
        b.clicked.connect(self._save_preset)
        row.addWidget(b)
        b = QPushButton("Xóa")
        b.clicked.connect(self._delete_preset)
        row.addWidget(b)
        v.addLayout(row)
        self._reload_presets()
        self.b_wm_apply = QPushButton("Đóng logo vào ảnh")
        self.b_wm_apply.setObjectName("Accent")
        self.b_wm_apply.clicked.connect(self.apply_watermark)
        v.addWidget(self.b_wm_apply)

        v.addWidget(_section("Đóng logo hàng loạt (chạy nền)"))
        row = QHBoxLayout()
        b = QPushButton("Thư mục ảnh…")
        b.clicked.connect(self._pick_bulk_in)
        row.addWidget(b)
        b = QPushButton("Thư mục xuất…")
        b.clicked.connect(self._pick_bulk_out)
        row.addWidget(b)
        v.addLayout(row)
        self.lbl_bulk_in = _hint("Chưa chọn thư mục ảnh")
        self.lbl_bulk_out = _hint("Chưa chọn thư mục xuất")
        v.addWidget(self.lbl_bulk_in)
        v.addWidget(self.lbl_bulk_out)
        self.chk_bulk_sub = QCheckBox("Quét cả thư mục con")
        self.chk_bulk_sub.setChecked(True)
        v.addWidget(self.chk_bulk_sub)
        row = QHBoxLayout()
        row.addWidget(QLabel("Định dạng"))
        self.cmb_bulk_fmt = QComboBox()
        for t, d in (("Giữ như ảnh gốc", "keep"), ("JPG", "jpg"), ("PNG", "png"), ("WEBP", "webp")):
            self.cmb_bulk_fmt.addItem(t, d)
        row.addWidget(self.cmb_bulk_fmt, 1)
        v.addLayout(row)
        row = QHBoxLayout()
        self.b_bulk = QPushButton("Bắt đầu đóng logo")
        self.b_bulk.setObjectName("Good")
        self.b_bulk.clicked.connect(self.start_bulk)
        self.b_bulk_stop = QPushButton("Ngưng")
        self.b_bulk_stop.setObjectName("Danger")
        self.b_bulk_stop.setEnabled(False)
        self.b_bulk_stop.clicked.connect(self.stop_bulk)
        row.addWidget(self.b_bulk, 2)
        row.addWidget(self.b_bulk_stop, 1)
        v.addLayout(row)
        self.bulk_prog = QProgressBar()
        self.bulk_prog.setRange(0, 1)
        self.bulk_prog.setValue(0)
        self.bulk_prog.setFormat("Chưa chạy")
        v.addWidget(self.bulk_prog)
        self.bulk_log = QPlainTextEdit()
        self.bulk_log.setReadOnly(True)
        self.bulk_log.setMaximumHeight(110)
        v.addWidget(self.bulk_log)
        v.addStretch(1)
        self.bulk_in = None
        self.bulk_out = self.settings.value("bulk_out", "")
        if self.bulk_out:
            self.lbl_bulk_out.setText("Xuất vào: " + self.bulk_out)
        return w

    # ---------------------------------------------------------------- MODULE 3
    def _page_adjust(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(_section("Bộ lọc màu"))
        self.lst_filters = QListWidget()
        self.lst_filters.setViewMode(QListView.IconMode)
        self.lst_filters.setIconSize(QSize(100, 64))
        self.lst_filters.setResizeMode(QListView.Adjust)
        self.lst_filters.setMovement(QListView.Static)
        self.lst_filters.setSpacing(4)
        self.lst_filters.setMinimumHeight(250)
        for key, label, _ in ops.FILTERS:
            it = QListWidgetItem(label)
            it.setData(Qt.UserRole, key)
            it.setSizeHint(QSize(110, 92))
            self.lst_filters.addItem(it)
        self.lst_filters.setCurrentRow(0)
        self.lst_filters.currentItemChanged.connect(self._on_filter)
        v.addWidget(self.lst_filters)
        self.sl_famount = SliderRow("Cường độ", 0, 100, 100, "%")
        self.sl_famount.changed.connect(lambda val: self._set_adj(filter_amount=val))
        v.addWidget(self.sl_famount)
        v.addWidget(_section("Điều chỉnh (xem trước tức thì)"))
        self.sl_b = SliderRow("Độ sáng", -100, 100, 0)
        self.sl_c = SliderRow("Tương phản", -100, 100, 0)
        self.sl_s = SliderRow("Bão hòa", -100, 100, 0)
        self.sl_sh = SliderRow("Sắc nét", 0, 100, 0)
        self.sl_b.changed.connect(lambda val: self._set_adj(brightness=val))
        self.sl_c.changed.connect(lambda val: self._set_adj(contrast=val))
        self.sl_s.changed.connect(lambda val: self._set_adj(saturation=val))
        self.sl_sh.changed.connect(lambda val: self._set_adj(sharpness=val))
        for s in (self.sl_b, self.sl_c, self.sl_s, self.sl_sh):
            v.addWidget(s)
        row = QHBoxLayout()
        b = QPushButton("Áp dụng")
        b.setObjectName("Accent")
        b.clicked.connect(lambda: self.commit_adjustments("Bộ lọc & điều chỉnh"))
        row.addWidget(b)
        b = QPushButton("Đặt lại")
        b.clicked.connect(self.reset_adjustments)
        row.addWidget(b)
        v.addLayout(row)
        v.addWidget(_hint("Sáng/tương phản/bão hòa được gộp thành một ma trận màu 3×4 và áp trong một lần tính, "
                          "nên thanh trượt phản hồi theo thời gian thực. Nhấp đúp tên thanh trượt để đặt lại."))
        v.addStretch(1)
        return w

    # ---------------------------------------------------------------- MODULE 4
    def _page_retouch(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(_section("Tự động tối ưu"))
        b = QPushButton("Tối ưu một chạm (Auto Enhance)")
        b.setObjectName("Accent")
        b.clicked.connect(self.auto_enhance)
        v.addWidget(b)
        v.addWidget(_hint("Cân bằng trắng, giãn mức đen/trắng và hiệu chỉnh sáng dựa trên histogram, "
                          "thêm tương phản cục bộ (CLAHE)."))
        v.addWidget(_section("Làm mịn da"))
        self.sl_skin = SliderRow("Mịn da", 0, 100, 0)
        self.sl_skin.changed.connect(lambda val: self._set_adj(skin=val))
        v.addWidget(self.sl_skin)
        b = QPushButton("Áp dụng làm mịn")
        b.clicked.connect(lambda: self.commit_adjustments("Làm mịn da"))
        v.addWidget(b)
        v.addWidget(_hint("Bilateral Filter + tách tần số: làm phẳng mụn, vết thâm nhưng giữ đường nét mắt, mũi, môi "
                          "và 30% kết cấu da để không bị bệt. Chỉ áp lên vùng da (tự nhận diện khuôn mặt)."))
        v.addWidget(_section("Xóa vật thể thừa"))
        self.b_brush = QPushButton("Bật cọ vẽ")
        self.b_brush.setCheckable(True)
        self.b_brush.toggled.connect(self._toggle_brush)
        v.addWidget(self.b_brush)
        self.sl_brush = SliderRow("Cỡ cọ", 4, 300, 40, " px")
        self.sl_brush.changed.connect(lambda val: setattr(self.canvas, "brush", val))
        v.addWidget(self.sl_brush)
        row = QHBoxLayout()
        row.addWidget(QLabel("Thuật toán"))
        self.cmb_inpaint = QComboBox()
        self.cmb_inpaint.addItem("Telea (Fast Marching)", "telea")
        self.cmb_inpaint.addItem("Navier–Stokes", "ns")
        row.addWidget(self.cmb_inpaint, 1)
        v.addLayout(row)
        row = QHBoxLayout()
        self.b_remove = QPushButton("Xóa vùng đã tô")
        self.b_remove.setObjectName("Danger")
        self.b_remove.clicked.connect(self.remove_objects)
        b = QPushButton("Xóa nét tô")
        b.clicked.connect(lambda: self.canvas.clear_mask())
        row.addWidget(self.b_remove)
        row.addWidget(b)
        v.addLayout(row)
        v.addWidget(_hint("Chuột trái: tô vùng cần xóa. Chuột phải: tẩy nét tô. Lăn chuột để phóng to, "
                          "giữ nút giữa để kéo ảnh. Vùng đã tô được lấp bằng cv2.inpaint."))
        v.addStretch(1)
        return w

    # ---------------------------------------------------------------- MODULE 5
    def _page_history(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        row = QHBoxLayout()
        b = QPushButton("↶ Hoàn tác")
        b.clicked.connect(self.undo)
        row.addWidget(b)
        b = QPushButton("↷ Làm lại")
        b.clicked.connect(self.redo)
        row.addWidget(b)
        v.addLayout(row)
        self.lst_hist = QListWidget()
        self.lst_hist.itemClicked.connect(lambda it: self.jump(self.lst_hist.row(it)))
        v.addWidget(self.lst_hist, 1)
        v.addWidget(_hint(f"Lưu tối đa {History.MAX_STEPS} bước. Bấm vào một dòng để quay lại trạng thái đó. "
                          "Phím tắt: Ctrl+Z hoàn tác, Ctrl+Y làm lại."))
        return w

    # ============================================================== tiện ích
    def has_image(self) -> bool:
        return self.history.current is not None

    @property
    def base(self) -> np.ndarray:
        return self.history.current.image

    @property
    def alpha(self):
        return self.history.current.alpha

    def status(self, msg: str):
        self.lbl_status.setText(msg)

    def _busy(self, on: bool):
        if on:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()
        else:
            QApplication.restoreOverrideCursor()

    def _preview_base(self):
        """Ảnh thu nhỏ của trạng thái hiện tại (được lưu đệm theo chỉ số lịch sử)."""
        key = (self.history.index, id(self.history.current))
        if self._prev_cache is None or self._prev_cache[0] != key:
            small, k = ops.resize_long_side(self.base, PREVIEW_MAX)
            self._prev_cache = (key, small, k)
            self._skin_prev = None
            self._refresh_thumbs(small)
        return self._prev_cache[1], self._prev_cache[2]

    def _update_buttons(self):
        has = self.has_image()
        self.b_undo.setEnabled(has and (self.history.can_undo() or not self.adj.is_identity()))
        self.b_redo.setEnabled(has and self.history.can_redo())
        self.b_cmp.setEnabled(has)
        self.b_remove.setEnabled(has and self.canvas.mask is not None and bool(self.canvas.mask.any()))
        crop_on = self.canvas.mode == "crop"
        self.b_crop_apply.setEnabled(crop_on)
        self.b_crop_cancel.setEnabled(crop_on)
        self.b_wm_apply.setEnabled(has and self.logo is not None)

    def _refresh_history(self):
        self.lst_hist.clear()
        for i, lab in enumerate(self.history.labels()):
            self.lst_hist.addItem(f"{i + 1:>2}. {lab}")
        if self.history.index >= 0:
            self.lst_hist.setCurrentRow(self.history.index)
        self._update_buttons()

    def _refresh_thumbs(self, small):
        """Ảnh thu nhỏ minh họa cho từng bộ lọc."""
        th, _ = ops.resize_long_side(small, 120)
        for i in range(self.lst_filters.count()):
            it = self.lst_filters.item(i)
            out = ops.apply_filter(th, it.data(Qt.UserRole), 100)
            it.setIcon(QIcon(QPixmap.fromImage(np_to_qimage(out))))

    # ============================================================== mở ảnh
    def pick_image(self):
        last = self.settings.value("editor_last_dir", "")
        f, _ = QFileDialog.getOpenFileName(self, "Mở ảnh để chỉnh sửa", last,
                                           "Ảnh (*.jpg *.jpeg *.jfif *.png *.webp *.heic *.heif *.bmp *.tif *.tiff *.gif)")
        if f:
            self.settings.setValue("editor_last_dir", os.path.dirname(f))
            self.open_image(f)

    @staticmethod
    def _to_srgb(rgb, icc):
        """Ảnh có hồ sơ màu khác sRGB (vd. Display P3 của iPhone) -> chuyển về sRGB,
        vì mạng xã hội hiển thị theo sRGB; không chuyển thì màu sẽ bị nhạt/lệch."""
        if not icc:
            return rgb
        try:
            from PIL import Image, ImageCms
            prof = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            desc = ImageCms.getProfileDescription(prof) or ""
            if "srgb" in desc.lower():
                return rgb
            im = ImageCms.profileToProfile(Image.fromarray(rgb), prof, ImageCms.createProfile("sRGB"),
                                           outputMode="RGB")
            return np.asarray(im).copy()
        except Exception:
            return rgb

    def open_image(self, path: str):
        try:
            self._busy(True)
            li = IF.load(path)
            rgb = self._to_srgb(li.rgb, li.icc)
            alpha = li.alphas[0] if li.alphas else None
        except Exception as ex:  # noqa: BLE001
            self._busy(False)
            QMessageBox.warning(self, "AI Remove - thsky90", f"Không mở được ảnh:\n{ex}")
            return
        self._busy(False)
        self.path = path
        self.original = rgb
        self.history.reset(rgb, alpha, "Mở ảnh")
        self._prev_cache = None
        self.canvas.mask = None
        self.cancel_crop()
        self.b_brush.setChecked(False)
        self.reset_adjustments(silent=True)
        self.canvas.fitted = True
        self.refresh_preview()
        self._refresh_history()
        h, w = rgb.shape[:2]
        size = os.path.getsize(path) / 1024
        self.lbl_info.setText(f"{os.path.basename(path)} — {w}×{h} px — {size:,.0f} KB")
        self.status("Đã mở ảnh. Chọn công cụ ở cột bên trái.")

    # ============================================================== xem trước
    def schedule_preview(self, *_):
        self._timer.start()
        self._update_buttons()

    def _render(self, img, scale, skin_mask=None, with_logo=True):
        out = ops.render_adjustments(img, self.adj, scale, skin_mask)
        if with_logo and self.logo is not None and self.chk_wm.isChecked():
            out = ops.compose_watermark(out, self.logo, self.wm)
        return out

    def refresh_preview(self):
        if not self.has_image():
            self.canvas.set_image(None)
            self.social.set_image(None)
            return
        small, k = self._preview_base()
        if self.adj.skin > 0 and self._skin_prev is None:
            self._skin_prev = ops.skin_mask(small)
        out = self._render(small, k, self._skin_prev)
        q = np_to_qimage(out)
        H, W = self.base.shape[:2]
        self.canvas.tag = ""
        self.canvas.set_image(q, (W, H))
        self.social.set_image(q)

    def _show_original(self):
        if self.original is None:
            return
        small, _ = ops.resize_long_side(self.original, PREVIEW_MAX)
        H, W = self.original.shape[:2]
        self.canvas.tag = "ẢNH GỐC"
        self.canvas.set_image(np_to_qimage(small), (W, H))
        self.social.set_image(np_to_qimage(small))

    def _hide_original(self):
        self.refresh_preview()

    def _on_page(self, i):
        if i != 3 and self.b_brush.isChecked():
            self.b_brush.setChecked(False)

    # ============================================================== MODULE 3
    def _set_adj(self, **kw):
        for k, val in kw.items():
            setattr(self.adj, k, val)
        self.schedule_preview()

    def _on_filter(self, cur, prev):
        if cur is not None:
            self._set_adj(filter=cur.data(Qt.UserRole))

    def reset_adjustments(self, silent=False):
        self.adj = ops.Adjustments()
        for s in (self.sl_b, self.sl_c, self.sl_s, self.sl_sh, self.sl_skin):
            s.set_silent(0)
        self.sl_famount.set_silent(100)
        self.lst_filters.blockSignals(True)
        self.lst_filters.setCurrentRow(0)
        self.lst_filters.blockSignals(False)
        if not silent:
            self.schedule_preview()

    def commit_adjustments(self, label="Điều chỉnh", quiet=False) -> bool:
        """Tính các thông số đang xem trước trên ảnh độ phân giải đầy đủ và ghi vào lịch sử."""
        if not self.has_image() or self.adj.is_identity():
            if not quiet:
                self.status("Không có thay đổi nào để áp dụng.")
            return False
        self._busy(True)
        try:
            mask = ops.skin_mask(self.base) if self.adj.skin > 0 else None
            out = ops.render_adjustments(self.base, self.adj, 1.0, mask)
            desc = self._adj_desc()
            self.history.push(out, self.alpha, f"{label}: {desc}" if desc else label)
        finally:
            self._busy(False)
        self.reset_adjustments(silent=True)
        self.refresh_preview()
        self._refresh_history()
        self.status(f"Đã áp dụng: {label.lower()}.")
        return True

    def _adj_desc(self):
        a = self.adj
        parts = []
        if a.filter != "none" and a.filter_amount > 0:
            name = dict((k, lab) for k, lab, _ in ops.FILTERS)[a.filter]
            parts.append(f"{name} {int(a.filter_amount)}%")
        for lab, val in (("sáng", a.brightness), ("tương phản", a.contrast), ("bão hòa", a.saturation),
                         ("sắc nét", a.sharpness), ("mịn da", a.skin)):
            if val:
                parts.append(f"{lab} {int(val):+d}" if lab not in ("sắc nét", "mịn da") else f"{lab} {int(val)}")
        return ", ".join(parts)

    # ============================================================== MODULE 4
    def auto_enhance(self):
        if not self.has_image():
            return
        self.commit_adjustments("Điều chỉnh", quiet=True)
        self._busy(True)
        try:
            out, notes = ops.auto_enhance(self.base)
            self.history.push(out, self.alpha, "Tối ưu một chạm")
        finally:
            self._busy(False)
        self.refresh_preview()
        self._refresh_history()
        self.status("Tối ưu một chạm: " + notes)

    def _toggle_brush(self, on):
        if on:
            if not self.has_image():
                self.b_brush.setChecked(False)
                return
            self.cancel_crop()
            self.canvas.brush = self.sl_brush.value()
            self.canvas.set_mode("brush")
            self.b_brush.setText("Tắt cọ vẽ")
            self.status("Tô lên vật thể cần xóa (chuột phải để tẩy), rồi bấm 'Xóa vùng đã tô'.")
        else:
            if self.canvas.mode == "brush":
                self.canvas.set_mode("view")
            self.b_brush.setText("Bật cọ vẽ")
        self._update_buttons()

    def remove_objects(self):
        m = self.canvas.mask
        if not self.has_image() or m is None or not m.any():
            self.status("Hãy bật cọ vẽ và tô lên vật thể cần xóa trước.")
            return
        self.commit_adjustments("Điều chỉnh", quiet=True)
        self._busy(True)
        try:
            out = ops.remove_objects(self.base, m, self.cmb_inpaint.currentData())
            self.history.push(out, self.alpha, "Xóa vật thể")
        finally:
            self._busy(False)
        self.canvas.clear_mask()
        self.refresh_preview()
        self._refresh_history()
        self.status("Đã xóa vật thể và lấp vùng trống bằng cv2.inpaint.")

    # ============================================================== MODULE 1
    def start_crop(self, key):
        if not self.has_image():
            for b in self.crop_group.buttons():
                b.setChecked(False)
            return
        self.b_brush.setChecked(False)
        label, ratio, size = ops.PRESET_MAP[key]
        H, W = self.base.shape[:2]
        self._crop_key = key
        self.canvas.set_crop(ops.max_rect_for_ratio(W, H, ratio), ratio)
        self.canvas.set_mode("crop")
        self.status(f"Khung cắt: {label}. Chỉnh khung rồi bấm 'Áp dụng cắt'.")
        self._update_buttons()

    def cancel_crop(self):
        if self.canvas.mode == "crop":
            self.canvas.set_mode("view")
        self.canvas.crop = None
        self.crop_group.setExclusive(False)
        for b in self.crop_group.buttons():
            b.setChecked(False)
        self.crop_group.setExclusive(True)
        self._update_buttons()

    def apply_crop(self):
        if self.canvas.mode != "crop" or not self.canvas.crop:
            return
        key = getattr(self, "_crop_key", "free")
        label, ratio, size = ops.PRESET_MAP[key]
        self.commit_adjustments("Điều chỉnh", quiet=True)
        out, a = ops.crop_image(self.base, self.canvas.crop, size if self.chk_resize.isChecked() else None, self.alpha)
        self.history.push(out, a, f"Cắt {label}")
        self.cancel_crop()
        self.canvas.mask = None
        self.canvas.fitted = True
        self.refresh_preview()
        self._refresh_history()
        self.status(f"Đã cắt: {out.shape[1]}×{out.shape[0]} px.")

    def _toggle_social(self, on):
        self.stack.setCurrentIndex(1 if on else 0)
        self._update_social()

    def _update_social(self, *_):
        self.social.set_mode("facebook" if self.rb_fb.isChecked() else "instagram",
                             self.chk_puzzle.isChecked(), self.chk_square.isChecked())

    def export_image(self):
        if not self.has_image():
            QMessageBox.information(self, "AI Remove - thsky90", "Chưa có ảnh để xuất.")
            return
        fmt = self.cmb_fmt.currentData()
        src_ext = os.path.splitext(self.path or "")[1].lower().lstrip(".")
        if fmt == "keep":
            fmt = {"jpeg": "jpg", "jfif": "jpg"}.get(src_ext, src_ext)
            if fmt not in ("jpg", "png", "webp"):
                fmt = "png"
        stem = os.path.splitext(os.path.basename(self.path or "anh"))[0]
        start = os.path.join(self.settings.value("editor_out_dir", os.path.dirname(self.path or "")),
                             f"{stem}_edit.{fmt}")
        flt = {"jpg": "JPEG (*.jpg)", "png": "PNG (*.png)", "webp": "WEBP (*.webp)"}[fmt]
        dst, _ = QFileDialog.getSaveFileName(self, "Xuất ảnh", start, flt)
        if not dst:
            return
        if not dst.lower().endswith("." + fmt) and not (fmt == "jpg" and dst.lower().endswith(".jpeg")):
            dst += "." + fmt
        self.settings.setValue("editor_out_dir", os.path.dirname(dst))
        self._busy(True)
        try:
            mask = ops.skin_mask(self.base) if self.adj.skin > 0 else None
            img = self._render(self.base, 1.0, mask)
            alpha = self.alpha
            if self.chk_smart.isChecked():
                data, info = ops.smart_compress(img, fmt, self.cmb_platform.currentData(), alpha)
            else:
                data = ops.encode(img, fmt, self.sl_quality.value(), alpha if fmt != "jpg" else None)
                info = {"width": img.shape[1], "height": img.shape[0], "size": len(data),
                        "quality": self.sl_quality.value() if fmt != "png" else None}
            with open(dst, "wb") as fh:
                fh.write(data)
        except Exception as ex:  # noqa: BLE001
            self._busy(False)
            QMessageBox.warning(self, "AI Remove - thsky90", f"Không xuất được ảnh:\n{ex}")
            return
        self._busy(False)
        orig = os.path.getsize(self.path) if self.path and os.path.exists(self.path) else 0
        msg = f"Đã xuất {os.path.basename(dst)} — {info['width']}×{info['height']} px, {info['size'] / 1024:,.0f} KB"
        if orig:
            msg += f" (ảnh gốc {orig / 1024:,.0f} KB)"
        if info.get("quality"):
            msg += f", chất lượng {info['quality']}"
        if info.get("ssim") and info.get("ssim") < 1:
            msg += f", SSIM {info['ssim']:.3f}"
        self.status(msg)

    # ============================================================== MODULE 2
    def pick_logo(self):
        last = self.settings.value("logo_dir", "")
        f, _ = QFileDialog.getOpenFileName(self, "Chọn logo", last, "Ảnh logo (*.png *.webp *.jpg *.jpeg)")
        if f:
            self.settings.setValue("logo_dir", os.path.dirname(f))
            self._use_logo(f)

    def _use_logo(self, path) -> bool:
        try:
            self.logo = ops.load_logo(path)
        except Exception as ex:  # noqa: BLE001
            QMessageBox.warning(self, "AI Remove - thsky90", f"Không đọc được logo:\n{ex}")
            return False
        self.logo_path = path
        th, _ = ops.resize_long_side(self.logo, 60)
        from PySide6.QtGui import QImage
        q = QImage(np.ascontiguousarray(th).data, th.shape[1], th.shape[0], 4 * th.shape[1],
                   QImage.Format_RGBA8888).copy()
        self.lbl_logo.setPixmap(QPixmap.fromImage(q))
        if self.logo[..., 3].min() == 255:
            self.status("Lưu ý: logo không có nền trong suốt – nên dùng PNG trong suốt để logo đẹp hơn.")
        self.schedule_preview()
        return True

    def _set_wm(self, **kw):
        for k, val in kw.items():
            setattr(self.wm, k, val)
        self.schedule_preview()

    def _sync_wm_controls(self):
        self.sl_op.set_silent(int(self.wm.opacity))
        self.sl_scale.set_silent(int(self.wm.scale))
        self.sl_margin.set_silent(int(self.wm.margin))
        for b in self.pos_group.buttons():
            b.setChecked(b.toolTip() == ops.POSITIONS[self.wm.position][0])

    def apply_watermark(self):
        if not self.has_image() or self.logo is None:
            return
        self.commit_adjustments("Điều chỉnh", quiet=True)
        out = ops.compose_watermark(self.base, self.logo, self.wm)
        self.history.push(out, self.alpha, f"Đóng logo ({ops.POSITIONS[self.wm.position][0].lower()})")
        self.chk_wm.setChecked(False)       # tránh vẽ logo hai lần
        self.refresh_preview()
        self._refresh_history()
        self.status("Đã đóng logo vào ảnh (tắt lớp xem trước để không bị chồng hai lần).")

    def _presets(self) -> dict:
        try:
            return json.loads(self.settings.value("wm_presets", "{}") or "{}")
        except Exception:
            return {}

    def _reload_presets(self):
        self.cmb_preset.clear()
        self.cmb_preset.addItem("— Chọn mẫu đã lưu —", None)
        for name in sorted(self._presets()):
            self.cmb_preset.addItem(name, name)

    def _save_preset(self):
        if self.logo_path is None:
            QMessageBox.information(self, "AI Remove - thsky90", "Hãy chọn logo trước khi lưu mẫu.")
            return
        name, ok = QInputDialog.getText(self, "Lưu mẫu logo", "Tên mẫu:")
        if not ok or not name.strip():
            return
        p = self._presets()
        p[name.strip()] = {"logo": self.logo_path, **self.wm.to_dict()}
        self.settings.setValue("wm_presets", json.dumps(p, ensure_ascii=False))
        self._reload_presets()
        self.cmb_preset.setCurrentIndex(self.cmb_preset.findData(name.strip()))
        self.status(f"Đã lưu mẫu logo '{name.strip()}'.")

    def _load_preset(self, idx):
        name = self.cmb_preset.itemData(idx)
        if not name:
            return
        d = self._presets().get(name)
        if not d:
            return
        if d.get("logo") and os.path.exists(d["logo"]):
            self._use_logo(d["logo"])
        else:
            self.status("Không tìm thấy file logo của mẫu này – hãy chọn lại logo.")
        self.wm = ops.WatermarkSettings.from_dict(d)
        self._sync_wm_controls()
        self.schedule_preview()

    def _delete_preset(self):
        name = self.cmb_preset.currentData()
        if not name:
            return
        p = self._presets()
        p.pop(name, None)
        self.settings.setValue("wm_presets", json.dumps(p, ensure_ascii=False))
        self._reload_presets()

    def _pick_bulk_in(self):
        d = QFileDialog.getExistingDirectory(self, "Thư mục ảnh cần đóng logo", self.settings.value("bulk_in", ""))
        if d:
            self.bulk_in = d
            self.settings.setValue("bulk_in", d)
            n = len(batch.list_files([d], IF.IMAGE_EXTS, self.chk_bulk_sub.isChecked()))
            self.lbl_bulk_in.setText(f"Ảnh: {d} ({n} tệp)")

    def _pick_bulk_out(self):
        d = QFileDialog.getExistingDirectory(self, "Thư mục xuất", self.bulk_out or "")
        if d:
            self.bulk_out = d
            self.settings.setValue("bulk_out", d)
            self.lbl_bulk_out.setText("Xuất vào: " + d)

    def start_bulk(self):
        if self.logo is None:
            QMessageBox.information(self, "AI Remove - thsky90", "Hãy chọn logo trước.")
            return
        if not self.bulk_in:
            self._pick_bulk_in()
            if not self.bulk_in:
                return
        if not self.bulk_out:
            self._pick_bulk_out()
            if not self.bulk_out:
                return
        files = batch.list_files([self.bulk_in], IF.IMAGE_EXTS, self.chk_bulk_sub.isChecked())
        if not files:
            QMessageBox.information(self, "AI Remove - thsky90", "Thư mục không có ảnh phù hợp.")
            return
        os.makedirs(self.bulk_out, exist_ok=True)
        self._bulk = BulkWatermarkWorker(files, self.bulk_out, self.logo.copy(),
                                         ops.WatermarkSettings(**self.wm.to_dict()),
                                         self.cmb_bulk_fmt.currentData(), self.sl_quality.value())
        self._bulk.progress.connect(self._bulk_progress)
        self._bulk.log.connect(self.bulk_log.appendPlainText)
        self._bulk.finished_all.connect(self._bulk_done)
        self.bulk_prog.setRange(0, len(files))
        self.bulk_prog.setValue(0)
        self.bulk_prog.setFormat("%v/%m ảnh")
        self.b_bulk.setEnabled(False)
        self.b_bulk_stop.setEnabled(True)
        self.bulk_log.appendPlainText(f"— Bắt đầu đóng logo {len(files)} ảnh (chạy nền, vẫn sửa ảnh khác được)")
        self._bulk.start()

    def stop_bulk(self):
        if self._bulk:
            self._bulk.cancel()
            self.b_bulk_stop.setEnabled(False)

    def _bulk_progress(self, i, total, name):
        self.bulk_prog.setValue(i)
        if name:
            self.status(f"Đóng logo hàng loạt: {i + 1}/{total} – {name}")

    def _bulk_done(self, ok, fail, secs):
        self.bulk_prog.setValue(self.bulk_prog.maximum() if not fail and self._bulk and not self._bulk._cancel else ok + fail)
        self.bulk_log.appendPlainText(f"— Xong: {ok} ảnh, {fail} lỗi, {secs:.1f} giây")
        self.status(f"Đóng logo hàng loạt xong: {ok} ảnh, {fail} lỗi.")
        self.b_bulk.setEnabled(True)
        self.b_bulk_stop.setEnabled(False)
        self._bulk = None

    def bulk_running(self) -> bool:
        return self._bulk is not None and self._bulk.isRunning()

    # ============================================================== MODULE 5
    def undo(self):
        if not self.has_image():
            return
        if not self.adj.is_identity():          # có thay đổi đang xem trước -> hủy chúng trước
            self.reset_adjustments()
            self.status("Đã hủy các điều chỉnh chưa áp dụng.")
            return
        if self.history.can_undo():
            self.history.undo()
            self._after_history("Hoàn tác")

    def redo(self):
        if self.has_image() and self.history.can_redo():
            self.history.redo()
            self._after_history("Làm lại")

    def jump(self, i):
        if self.has_image():
            self.reset_adjustments(silent=True)
            self.history.jump(i)
            self._after_history("Quay lại bước " + str(i + 1))

    def _after_history(self, what):
        self.canvas.mask = None
        self.cancel_crop()
        self.b_brush.setChecked(False)
        self.refresh_preview()
        self._refresh_history()
        self.status(f"{what}: {self.history.current.label}")
