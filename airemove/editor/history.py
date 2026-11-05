"""MODULE 5 — Lịch sử thao tác (History Stack) cho Undo/Redo.

Mỗi lần người dùng "áp dụng" một thao tác, ảnh kết quả được đẩy vào ngăn xếp.
    Hoàn tác (Undo): lùi con trỏ về trạng thái trước.
    Làm lại  (Redo): tiến con trỏ tới trạng thái sau.
Khi đã Undo rồi làm thao tác mới, các trạng thái "phía sau" bị bỏ (giống
Photoshop). Giữ tối đa MAX_STEPS trạng thái (mặc định 30 > 20 bước yêu cầu);
trạng thái cũ nhất bị loại khi vượt quá.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class State:
    image: np.ndarray            # ảnh RGB uint8
    alpha: np.ndarray | None     # kênh trong suốt (nếu có)
    label: str                   # mô tả thao tác, hiển thị trong danh sách lịch sử


class History:
    MAX_STEPS = 30

    def __init__(self, max_steps: int | None = None):
        self.max = max_steps or self.MAX_STEPS
        self.states: list[State] = []
        self.index = -1

    # ---- truy vấn ---------------------------------------------------------
    @property
    def current(self) -> State | None:
        return self.states[self.index] if 0 <= self.index < len(self.states) else None

    def can_undo(self) -> bool:
        return self.index > 0

    def can_redo(self) -> bool:
        return self.index < len(self.states) - 1

    def labels(self) -> list[str]:
        return [s.label for s in self.states]

    # ---- thay đổi ---------------------------------------------------------
    def reset(self, image: np.ndarray, alpha=None, label: str = "Mở ảnh"):
        """Bắt đầu lịch sử mới khi mở ảnh."""
        self.states = [State(image, alpha, label)]
        self.index = 0

    def push(self, image: np.ndarray, alpha=None, label: str = "Chỉnh sửa"):
        """Thêm trạng thái mới sau con trỏ hiện tại."""
        del self.states[self.index + 1:]
        self.states.append(State(image, alpha, label))
        if len(self.states) > self.max:
            # Giữ lại trạng thái đầu tiên (ảnh gốc) để luôn quay về được.
            del self.states[1]
        self.index = len(self.states) - 1

    def undo(self) -> State | None:
        if self.can_undo():
            self.index -= 1
        return self.current

    def redo(self) -> State | None:
        if self.can_redo():
            self.index += 1
        return self.current

    def jump(self, i: int) -> State | None:
        """Nhảy thẳng tới một trạng thái trong danh sách lịch sử."""
        if 0 <= i < len(self.states):
            self.index = i
        return self.current
