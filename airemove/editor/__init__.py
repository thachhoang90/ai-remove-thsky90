"""Tab "Chỉnh sửa ảnh" của AI Remove - thsky90.

Gói này gồm:
    ops.py            Lõi xử lý ảnh (bộ lọc, ma trận màu, làm đẹp, xóa vật thể,
                      đóng logo, cắt ảnh theo tỉ lệ mạng xã hội, nén thông minh).
    history.py        Ngăn xếp lịch sử Undo/Redo.
    canvas.py         Vùng hiển thị ảnh: phóng to/thu nhỏ, khung cắt, cọ vẽ.
    social_preview.py Giả lập lưới Instagram 3×3 và bảng tin Facebook.
    bulk.py           Đóng logo hàng loạt chạy nền (QThread).
    tab.py            Giao diện tab (giao diện tối – Dark Theme).
"""
