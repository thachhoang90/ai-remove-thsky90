"""Lõi xử lý ảnh cho tab "Chỉnh sửa ảnh".

Quy ước chung trong module này:
    * Ảnh luôn là mảng numpy RGB, kiểu uint8, hình dạng (cao, rộng, 3).
    * Kênh trong suốt (alpha), nếu có, được giữ riêng và chỉ bị cắt/đổi cỡ
      cùng với ảnh; các phép chỉnh màu không đụng tới alpha.
    * Mọi hàm đều "thuần": nhận ảnh vào, trả ảnh mới ra, không sửa ảnh gốc.
      Nhờ vậy lịch sử Undo/Redo chỉ cần lưu lại các ảnh kết quả.

Nội dung:
    MODULE 1  Cắt ảnh theo tỉ lệ mạng xã hội, xuất file, nén thông minh.
    MODULE 2  Đóng logo (watermark) với vị trí / độ mờ / tỉ lệ.
    MODULE 3  Bộ lọc màu và thanh trượt điều chỉnh (dựa trên ma trận màu).
    MODULE 4  Tự động tối ưu, làm mịn da, xóa vật thể (cv2.inpaint).
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass, asdict

import cv2
import numpy as np
from PIL import Image

# Trọng số độ sáng (luma) chuẩn Rec.601 — dùng để tính "độ xám" của điểm ảnh.
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


# ============================================================================
# Tiện ích chung
# ============================================================================

def to_float(img: np.ndarray) -> np.ndarray:
    """uint8 [0..255] -> float32 [0..1] để tính toán không bị tràn số."""
    return img.astype(np.float32) * (1.0 / 255.0)


def to_uint8(img: np.ndarray) -> np.ndarray:
    """float32 [0..1] -> uint8, có cắt ngưỡng và làm tròn."""
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


def luma(img_f: np.ndarray) -> np.ndarray:
    """Độ sáng của ảnh float (cao, rộng)."""
    return img_f @ LUMA


def resize_long_side(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Thu nhỏ ảnh sao cho cạnh dài nhất <= max_side. Trả về (ảnh, hệ số)."""
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return img, 1.0
    k = max_side / float(s)
    out = cv2.resize(img, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_AREA)
    return out, k


def _curve_lut(points) -> np.ndarray:
    """Tạo bảng tra (LUT) 256 phần tử từ các điểm điều khiển đường cong tông màu.

    points: danh sách (vào, ra) trong khoảng 0..255, nội suy tuyến tính."""
    xs, ys = zip(*points)
    lut = np.interp(np.arange(256), xs, ys)
    return np.clip(lut, 0, 255).astype(np.uint8)


def _apply_curves(img: np.ndarray, r=None, g=None, b=None, all_=None) -> np.ndarray:
    """Áp đường cong cho từng kênh (hoặc chung cho cả 3 kênh) bằng cv2.LUT — rất nhanh."""
    out = img
    if all_ is not None:
        out = cv2.LUT(out, _curve_lut(all_))
    if r is not None or g is not None or b is not None:
        ident = [(0, 0), (255, 255)]
        lut = np.stack([_curve_lut(r or ident), _curve_lut(g or ident), _curve_lut(b or ident)], axis=-1)
        out = cv2.LUT(out, lut.reshape(256, 1, 3))
    return out


# ============================================================================
# MODULE 3 — ĐIỀU CHỈNH THÔNG SỐ BẰNG MA TRẬN MÀU
# ============================================================================
#
# Độ sáng, độ tương phản và độ bão hòa đều là phép biến đổi TUYẾN TÍNH trên
# vector màu [R, G, B]. Ta gộp cả ba thành MỘT ma trận 3×4 (3×3 + cột dịch
# chuyển) rồi áp cho toàn ảnh bằng một lệnh cv2.transform duy nhất. Cách này
# nhanh hơn nhiều so với áp lần lượt từng phép, nên thanh trượt phản hồi
# theo thời gian thực.

def saturation_matrix(sat: float) -> np.ndarray:
    """Ma trận 3×3 thay đổi độ bão hòa.

    sat = 1: giữ nguyên; 0: ảnh xám; >1: rực màu hơn.
    Công thức: M = (1 - sat) * L + sat * I, với mỗi hàng của L là trọng số luma.
    Ma trận này giữ nguyên độ sáng (luma) của mọi điểm ảnh."""
    L = np.tile(LUMA, (3, 1))
    return ((1.0 - sat) * L + sat * np.eye(3, dtype=np.float32)).astype(np.float32)


def adjustment_matrix(brightness: float = 0, contrast: float = 0, saturation: float = 0) -> np.ndarray:
    """Gộp 3 thanh trượt (mỗi thanh -100..100) thành một ma trận 3×4.

    Thứ tự áp dụng: bão hòa -> độ sáng -> độ tương phản.
        x1 = S·x                    (bão hòa)
        x2 = x1 + b                 (độ sáng: cộng thêm một lượng)
        x3 = k·(x2 - 0.5) + 0.5     (tương phản: co/giãn quanh mức xám giữa)
    => x3 = (k·S)·x + (k·b + 0.5·(1 - k))
    """
    sat = 1.0 + saturation / 100.0                       # 0 .. 2
    b = brightness / 100.0 * 0.35                        # dịch tối đa ±35% thang sáng
    k = 1.0 + contrast / 100.0 if contrast < 0 else 1.0 + contrast / 100.0 * 1.5
    S = saturation_matrix(sat)
    M = np.zeros((3, 4), np.float32)
    M[:, :3] = k * S
    M[:, 3] = k * b + 0.5 * (1.0 - k)
    return M


def apply_matrix(img_f: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Áp ma trận màu 3×4 cho ảnh float [0..1] (cv2.transform xử lý cả cột dịch)."""
    return cv2.transform(img_f, M)


def sharpen(img_f: np.ndarray, amount: float, scale: float = 1.0) -> np.ndarray:
    """Tăng độ sắc nét bằng mặt nạ làm nét (Unsharp Mask).

    out = ảnh + a·(ảnh - ảnh_mờ). Chỉ làm nét kênh độ sáng để không sinh
    viền màu. 'scale' < 1 khi xem trước trên ảnh thu nhỏ (giữ cảm giác giống
    khi xuất ở kích thước thật)."""
    if amount <= 0:
        return img_f
    a = amount / 100.0 * 1.6
    sigma = max(0.5, 1.0 * scale)
    y = luma(img_f)
    detail = y - cv2.GaussianBlur(y, (0, 0), sigma)
    # Bỏ qua dao động rất nhỏ (nhiễu) để không làm nổi hạt nhiễu.
    detail = np.sign(detail) * np.maximum(np.abs(detail) - 0.004, 0)
    return img_f + (a * detail)[..., None]


# ----------------------------------------------------------------------------
# Bộ lọc màu thịnh hành. Mỗi bộ lọc là tổ hợp ma trận màu + đường cong tông.
# ----------------------------------------------------------------------------

def _vignette(h: int, w: int, strength: float) -> np.ndarray:
    """Mặt nạ tối dần về các góc (hiệu ứng vignette) — giá trị 1 ở tâm."""
    y, x = np.ogrid[:h, :w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r2 = ((x - cx) / (w / 2.0)) ** 2 + ((y - cy) / (h / 2.0)) ** 2
    return (1.0 - strength * np.clip(r2 / 2.0, 0, 1)).astype(np.float32)


def _f_vintage(img):
    # Ngả vàng nâu (sepia nhẹ), nâng điểm đen, nén vùng sáng, giảm bão hòa, tối góc.
    f = to_float(img)
    sepia = np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]], np.float32)
    M = 0.75 * saturation_matrix(0.8) + 0.25 * sepia
    f = cv2.transform(f, M)
    out = _apply_curves(to_uint8(f), r=[(0, 28), (128, 140), (255, 238)],
                        g=[(0, 20), (128, 128), (255, 228)], b=[(0, 38), (128, 112), (255, 205)])
    f = to_float(out) * _vignette(*img.shape[:2], 0.35)[..., None]
    return to_uint8(f)


def _f_cinematic(img):
    # "Teal & Orange": vùng tối ngả xanh mòng két, vùng sáng ngả cam; tương phản chữ S.
    f = to_float(img)
    y = luma(f)[..., None]
    shadows, highlights = (1.0 - y) ** 2, y ** 2
    f = f + shadows * np.array([-0.06, 0.015, 0.07], np.float32) \
          + highlights * np.array([0.07, 0.02, -0.07], np.float32)
    f = cv2.transform(f, saturation_matrix(0.9))
    out = _apply_curves(to_uint8(f), all_=[(0, 8), (64, 52), (128, 128), (192, 206), (255, 248)])
    return out


def _f_bright(img):
    # Sáng – tối giản: tăng phơi sáng, giảm tương phản, nâng vùng tối, màu nhạt trong trẻo.
    f = to_float(img)
    f = np.power(np.clip(f, 0, 1), 0.82)
    f = apply_matrix(f, adjustment_matrix(4, -18, -15))
    out = _apply_curves(to_uint8(f), all_=[(0, 22), (128, 140), (255, 255)],
                        b=[(0, 4), (255, 255)])
    return out


def _f_bw(img):
    # Đen trắng: trộn kênh (đỏ nhấn hơn một chút cho da sáng) + tương phản chữ S.
    f = to_float(img)
    y = f @ np.array([0.36, 0.53, 0.11], np.float32)
    out = to_uint8(np.repeat(y[..., None], 3, axis=2))
    return _apply_curves(out, all_=[(0, 0), (60, 46), (128, 128), (200, 214), (255, 255)])


def _f_warm(img):
    # Tông ấm: tăng đỏ/vàng, giảm xanh dương, bão hòa nhẹ.
    f = to_float(img)
    f = f * np.array([1.08, 1.02, 0.88], np.float32)
    f = cv2.transform(f, saturation_matrix(1.08))
    return _apply_curves(to_uint8(f), r=[(0, 6), (255, 255)], b=[(0, 0), (255, 236)])


def _f_cool(img):
    # Tông lạnh: ngả xanh dương, giảm đỏ.
    f = to_float(img) * np.array([0.92, 1.0, 1.08], np.float32)
    return _apply_curves(to_uint8(f), b=[(0, 10), (255, 255)])


def _f_film(img):
    # Phim bạc màu (fade): nâng điểm đen, giảm tương phản, vùng tối hơi xanh lục.
    f = apply_matrix(to_float(img), adjustment_matrix(0, -12, -10))
    return _apply_curves(to_uint8(f), r=[(0, 26), (255, 244)], g=[(0, 32), (255, 246)], b=[(0, 26), (255, 232)])


def _f_vivid(img):
    # Rực rỡ: tăng bão hòa và tương phản.
    return to_uint8(apply_matrix(to_float(img), adjustment_matrix(0, 12, 35)))


def _soft_focus(f, amount, sigma_frac=1 / 250.0):
    """Làm mềm ảo (soft focus): trộn với bản mờ Gauss — tạo cảm giác "mơ" cho chân dung."""
    s = max(1.0, min(f.shape[:2]) * sigma_frac)
    return f * (1 - amount) + cv2.GaussianBlur(f, (0, 0), s) * amount


def _f_portrait(img):
    # Chân dung trong trẻo: sáng hơn, da ấm hồng nhẹ, tương phản thấp, mềm ảo 12%.
    f = np.power(np.clip(to_float(img), 0, 1), 0.9)
    f = f * np.array([1.03, 1.0, 0.97], np.float32)
    f = apply_matrix(f, adjustment_matrix(3, -12, -4))
    f = _soft_focus(np.clip(f, 0, 1), 0.12)
    return _apply_curves(to_uint8(f), all_=[(0, 12), (128, 136), (255, 252)])


def _f_golden(img):
    # Giờ vàng: vùng sáng ngả cam vàng, nâng vùng tối, tối góc nhẹ.
    f = to_float(img)
    y = luma(f)[..., None]
    f = f + (y ** 1.5) * np.array([0.08, 0.035, -0.06], np.float32) + ((1 - y) ** 2) * np.array([0.02, 0.0, -0.02], np.float32)
    f = cv2.transform(np.clip(f, 0, 1), saturation_matrix(1.1))
    out = _apply_curves(to_uint8(f), all_=[(0, 16), (128, 134), (255, 250)])
    return to_uint8(to_float(out) * _vignette(*img.shape[:2], 0.22)[..., None])


def _f_pastel(img):
    # Pastel Hàn Quốc: tương phản thấp, nâng điểm đen, màu dịu, vùng sáng phớt hồng.
    f = apply_matrix(to_float(img), adjustment_matrix(6, -28, -22))
    y = luma(np.clip(f, 0, 1))[..., None]
    f = f + (y ** 2) * np.array([0.035, 0.0, 0.03], np.float32)
    return _apply_curves(to_uint8(f), all_=[(0, 34), (128, 140), (255, 250)])


def _f_moody(img):
    # Moody tối: giảm sáng, giảm màu, vùng tối ngả xanh lạnh, nén vùng sáng.
    f = apply_matrix(to_float(img), adjustment_matrix(-8, 8, -28))
    y = luma(np.clip(f, 0, 1))[..., None]
    f = f + ((1 - y) ** 2) * np.array([-0.03, 0.01, 0.045], np.float32)
    return _apply_curves(to_uint8(f), all_=[(0, 8), (90, 70), (128, 112), (255, 232)])


def _f_kodak(img):
    # Phim Kodak: ấm, đỏ/vàng đậm đà, điểm đen sâu, tương phản chữ S.
    f = to_float(img) * np.array([1.06, 1.02, 0.92], np.float32)
    f = cv2.transform(np.clip(f, 0, 1), saturation_matrix(1.15))
    return _apply_curves(to_uint8(f), all_=[(0, 6), (64, 56), (128, 132), (192, 204), (255, 250)],
                         b=[(0, 8), (255, 240)])


def _f_fuji(img):
    # Fuji cổ điển: màu trầm, xanh lá ngả lam, vùng tối hơi lạnh, tương phản vừa.
    f = to_float(img)
    M = np.array([[0.98, 0.04, -0.02], [0.0, 0.94, 0.06], [-0.02, 0.06, 0.96]], np.float32)
    f = cv2.transform(f, M)
    f = cv2.transform(np.clip(f, 0, 1), saturation_matrix(0.85))
    return _apply_curves(to_uint8(f), all_=[(0, 10), (64, 58), (128, 130), (192, 200), (255, 248)],
                         b=[(0, 14), (255, 250)])


def _f_sepia(img):
    # Sepia cổ điển: ảnh nâu đỏ hoài cổ + tối góc.
    f = to_float(img)
    sepia = np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]], np.float32)
    f = cv2.transform(f, sepia)
    out = _apply_curves(to_uint8(f), all_=[(0, 18), (255, 240)])
    return to_uint8(to_float(out) * _vignette(*img.shape[:2], 0.3)[..., None])


def _f_matte(img):
    # Matte lạnh (Bắc Âu): ngả lạnh, điểm đen nâng cao, tương phản thấp, màu nhạt.
    f = to_float(img) * np.array([0.95, 1.0, 1.05], np.float32)
    f = apply_matrix(np.clip(f, 0, 1), adjustment_matrix(0, -15, -12))
    return _apply_curves(to_uint8(f), all_=[(0, 38), (128, 132), (255, 240)])


# Danh sách bộ lọc: (khóa, tên hiển thị, hàm). "none" = ảnh gốc.
FILTERS = [
    ("none", "Gốc", None),
    ("vintage", "Vintage", _f_vintage),
    ("cinematic", "Cinematic", _f_cinematic),
    ("bright", "Sáng / Tối giản", _f_bright),
    ("bw", "Đen trắng", _f_bw),
    ("warm", "Tông ấm", _f_warm),
    ("cool", "Tông lạnh", _f_cool),
    ("film", "Phim bạc màu", _f_film),
    ("vivid", "Rực rỡ", _f_vivid),
    ("portrait", "Chân dung trong trẻo", _f_portrait),
    ("golden", "Giờ vàng", _f_golden),
    ("pastel", "Pastel Hàn Quốc", _f_pastel),
    ("moody", "Moody tối", _f_moody),
    ("kodak", "Phim Kodak", _f_kodak),
    ("fuji", "Fuji cổ điển", _f_fuji),
    ("sepia", "Sepia cổ điển", _f_sepia),
    ("matte", "Matte lạnh", _f_matte),
]
_FILTER_FN = {k: fn for k, _, fn in FILTERS}


def apply_filter(img: np.ndarray, name: str, intensity: float = 100) -> np.ndarray:
    """Áp bộ lọc 'name' với cường độ 0..100 (trộn với ảnh gốc theo tỉ lệ)."""
    fn = _FILTER_FN.get(name)
    if fn is None or intensity <= 0:
        return img
    out = fn(img)
    if intensity >= 100:
        return out
    t = intensity / 100.0
    return cv2.addWeighted(out, t, img, 1.0 - t, 0)


@dataclass
class Adjustments:
    """Các thông số chỉnh sửa "chưa áp dụng" (đang xem trước)."""
    filter: str = "none"
    filter_amount: float = 100
    brightness: float = 0
    contrast: float = 0
    saturation: float = 0
    sharpness: float = 0
    skin: float = 0          # mức làm mịn da (MODULE 4), 0 = tắt

    def is_identity(self) -> bool:
        return (self.filter == "none" or self.filter_amount <= 0) and not any(
            (self.brightness, self.contrast, self.saturation, self.sharpness, self.skin))

    def to_dict(self):
        return asdict(self)


def render_adjustments(img: np.ndarray, adj: Adjustments, scale: float = 1.0,
                       skin_mask_cache=None) -> np.ndarray:
    """Áp toàn bộ chuỗi chỉnh sửa theo thứ tự cố định:
    bộ lọc -> ma trận (sáng/tương phản/bão hòa) -> làm nét -> làm mịn da."""
    out = apply_filter(img, adj.filter, adj.filter_amount)
    if adj.brightness or adj.contrast or adj.saturation or adj.sharpness:
        f = to_float(out)
        if adj.brightness or adj.contrast or adj.saturation:
            f = apply_matrix(f, adjustment_matrix(adj.brightness, adj.contrast, adj.saturation))
        if adj.sharpness:
            f = sharpen(f, adj.sharpness, scale)
        out = to_uint8(f)
    if adj.skin > 0:
        out = smooth_skin(out, adj.skin, scale=scale, mask=skin_mask_cache)
    return out


# ============================================================================
# MODULE 4 — LÀM ĐẸP VÀ XỬ LÝ TỰ ĐỘNG
# ============================================================================

def guided_filter(I: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Guided Filter (He và cộng sự, 2010) — làm mịn GIỮ CẠNH, chi phí O(N).

    Trong mỗi cửa sổ bán kính r, đầu ra là hàm tuyến tính của ảnh dẫn I:
    q = a·I + b. Vùng phẳng (phương sai < eps) bị làm mịn mạnh; vùng có cạnh
    (phương sai lớn) giữ nguyên => không sinh quầng sáng (halo) quanh vật thể
    như CLAHE hay unsharp bán kính lớn."""
    k = (2 * r + 1, 2 * r + 1)
    mean = lambda x: cv2.boxFilter(x, -1, k, borderType=cv2.BORDER_REFLECT)
    mI, mp = mean(I), mean(p)
    var = mean(I * I) - mI * mI
    cov = mean(I * p) - mI * mp
    a = cov / (var + eps)
    b = mp - a * mI
    return mean(a) * I + mean(b)


def _rgb_to_lab(img_u8):
    """RGB uint8 -> Lab float32 (L: 0..100, a/b: khoảng -127..127)."""
    return cv2.cvtColor(to_float(img_u8), cv2.COLOR_RGB2LAB)


def _lab_to_rgb(lab):
    return to_uint8(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def _remove_blemishes(img: np.ndarray, skin: np.ndarray, face_w: int):
    """Xóa mụn / đốm thâm nhỏ trên vùng da.

    Một nốt mụn là một chấm NHỎ, TỐI hơn (hoặc ĐỎ hơn) vùng da xung quanh.
        1. Lọc trung vị (median) kích thước k ≈ 1/18 bề rộng khuôn mặt cho ra
           "da sạch" — chấm nhỏ hơn k bị xóa khỏi ảnh lọc.
        2. Điểm nào tối hơn ảnh lọc đáng kể, hoặc đỏ hơn hẳn xung quanh => ứng viên.
        3. Chỉ giữ các cụm có diện tích nhỏ (loại mắt, lông mày, lỗ mũi, viền môi).
        4. Lấp các cụm đó bằng cv2.inpaint (Telea) rồi trộn mềm vào ảnh.
    Trả về (ảnh, số đốm đã xóa)."""
    k = int(max(5, min(31, face_w / 18))) | 1
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)          # uint8: L 0..255, a/b quanh 128
    L, A = lab[..., 0], lab[..., 1].astype(np.float32)
    med = cv2.medianBlur(L, k)
    dark = (med.astype(np.int16) - L.astype(np.int16)) > 9
    red = (A - cv2.GaussianBlur(A, (0, 0), k / 2.0)) > 6
    red &= (med.astype(np.int16) - L.astype(np.int16)) > 2
    cand = ((dark | red) & (skin > 0.6)).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
    keep = np.zeros_like(cand)
    max_area = (k * 0.9) ** 2
    count = 0
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if 3 <= area <= max_area and max(w, h) <= k * 1.3 and area >= 0.3 * w * h:
            keep[labels == i] = 255
            count += 1
    if not count or count > 80:          # quá nhiều "đốm" => là kết cấu/hoa văn, không phải mụn
        return img, 0
    keep = cv2.dilate(keep, np.ones((3, 3), np.uint8), iterations=2)
    fixed = cv2.inpaint(img, keep, 3, cv2.INPAINT_TELEA)
    w = cv2.GaussianBlur(keep.astype(np.float32) / 255.0, (0, 0), 1.5)[..., None]
    out = img.astype(np.float32) * (1 - w) + fixed.astype(np.float32) * w
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), count


def is_graphic(img: np.ndarray) -> bool:
    """Ảnh đồ họa / hoạt hình (thumbnail, minh họa)? — nhiều vùng màu rất rực và cạnh sắc.
    Ảnh chụp thật hiếm khi có > 30% điểm ảnh bão hòa cao. Với ảnh đồ họa, không chạy các
    bước làm đẹp da (dễ xóa nhầm chi tiết vẽ) mà chỉ tối ưu nhẹ ánh sáng và màu."""
    small, _ = resize_long_side(img, 600)
    sat = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)[..., 1]
    g = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
    m = cv2.blur(g, (5, 5))
    sd = np.sqrt(np.maximum(cv2.blur(g * g, (5, 5)) - m * m, 0))
    return float((sat > 150).mean()) > 0.30 and float((sd > 12).mean()) > 0.22


def auto_enhance(img: np.ndarray, strength: float = 70) -> tuple[np.ndarray, str]:
    """TỐI ƯU MỘT CHẠM — làm đẹp tự nhiên như ảnh chụp từ máy ảnh chuyên nghiệp.

    Tự nhận diện khuôn mặt (chính diện và nghiêng). Nếu là ảnh chân dung thì
    ưu tiên làm đẹp da; nếu không thì chỉ tối ưu ánh sáng và màu sắc.
    Các bước (đều ở mức nhẹ, có giới hạn để ảnh không bị "quá tay"):
        1. Giảm nhiễu màu (chroma noise) — ảnh điện thoại sạch màu như máy ảnh.
        2. Cân bằng trắng theo các bề mặt gần trung tính (trời, tường, đường)
           và theo màu da của khuôn mặt (đưa sắc da về dải tự nhiên).
        3. Chỉnh độ sáng: chân dung lấy khuôn mặt làm chuẩn (đưa độ sáng da về
           ~L 66), ảnh thường lấy trung vị cả ảnh — bằng đường cong mềm, không cắt sáng.
        4. Giữ chi tiết vùng sáng/tối (giống HDR nhẹ, KHÔNG quầng sáng): tách lớp
           nền bằng Guided Filter, nén lớp nền, giữ nguyên lớp chi tiết.
        5. Tăng màu "thông minh" (vibrance): màu nhạt tăng nhiều, màu đã đậm và
           màu da tăng ít — tránh da bị cam/đỏ.
        6. Chân dung: xóa mụn nhỏ, làm mờ nếp nhăn + mịn da nhẹ (giữ kết cấu),
           làm sáng và đều màu da, giảm đỏ.
        7. Làm nét đầu ra cho chi tiết (tóc, mắt, quần áo, cảnh) nhưng KHÔNG làm
           nét vùng da.
    strength 0..100 điều chỉnh mức độ chung (mặc định 70)."""
    s = max(0.0, min(1.0, strength / 100.0))
    if s <= 0:
        return img, "không thay đổi"
    H, W = img.shape[:2]
    notes = []
    faces = _faces(img)
    graphic = is_graphic(img)
    portrait = len(faces) > 0 and not graphic
    face_w = int(np.median([f[2] for f in faces])) if portrait else 0
    if graphic:
        notes.append("ảnh đồ họa/hoạt hình – chỉ tối ưu nhẹ ánh sáng và màu")
    else:
        notes.append(f"nhận diện {len(faces)} khuôn mặt" if portrait else "không thấy khuôn mặt – tối ưu ánh sáng/màu")

    # ---- 1. giảm nhiễu màu (chỉ kênh màu Cr/Cb, không làm mờ chi tiết sáng tối)
    ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    sig = max(1.0, min(H, W) / 700.0)
    for c in (1, 2):
        ch = ycc[..., c]
        sm = cv2.bilateralFilter(ch, 0, 10, sig * 2)
        ycc[..., c] = cv2.addWeighted(sm, 0.7 * s + 0.2, ch, 0.8 - 0.7 * s, 0)
    work = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)

    lab = _rgb_to_lab(work)
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    skin = skin_mask(work) if portrait else None
    face_sel = None
    if portrait:
        face_sel = np.zeros((H, W), bool)
        for (x, y, w, h) in faces:
            face_sel[max(0, y + h // 6):min(H, y + h * 5 // 6), max(0, x + w // 6):min(W, x + w * 5 // 6)] = True
        face_sel &= skin > 0.5
        if face_sel.sum() < 50:
            face_sel = None

    # ---- 2. cân bằng trắng (ảnh đồ họa giữ nguyên màu chủ ý của người vẽ)
    chroma = np.sqrt(A * A + B * B)
    neutral = (chroma < 14) & (L > 20) & (L < 92)
    da = db = 0.0
    if neutral.sum() > 0.02 * H * W and not graphic:
        da, db = -float(np.median(A[neutral])), -float(np.median(B[neutral]))
    if face_sel is not None:
        # màu da tự nhiên: góc màu (hue) khoảng 45–62°; đẩy nhẹ về dải này
        fa, fb = float(np.median(A[face_sel])), float(np.median(B[face_sel]))
        hue = math.degrees(math.atan2(fb, fa))
        target = min(max(hue, 45.0), 62.0)
        if abs(target - hue) > 1:
            c = math.hypot(fa, fb)
            ta, tb = c * math.cos(math.radians(target)), c * math.sin(math.radians(target))
            da = 0.5 * da + 0.5 * (ta - fa)
            db = 0.5 * db + 0.5 * (tb - fb)
    lim = 7.0
    da, db = float(np.clip(da, -lim, lim)) * (0.5 + 0.4 * s), float(np.clip(db, -lim, lim)) * (0.5 + 0.4 * s)
    if abs(da) > 0.3 or abs(db) > 0.3:
        A += da
        B += db
        notes.append(f"cân bằng trắng (a {da:+.1f}, b {db:+.1f})")

    # ---- 3. độ sáng theo khuôn mặt (chân dung) hoặc theo cả ảnh
    #      Chân dung: chỉ nâng sáng toàn ảnh một phần (giữ trời/cảnh tự nhiên), phần
    #      còn lại "đánh đèn" riêng vùng mặt–cổ bằng mặt nạ mềm (giống đèn fill-flash).
    face_light = None
    if face_sel is not None:
        ref = float(np.median(L[face_sel]))
        target = 64.0
    else:
        ref = float(np.median(L))
        # chỉ làm sáng ảnh tối, không làm tối ảnh sáng; ảnh đồ họa giữ nguyên độ sáng chủ ý
        target = ref if graphic else (50.0 if ref < 50 else ref)
    ref = min(max(ref, 5.0), 95.0)
    g = math.log(target / 100.0) / math.log(ref / 100.0)
    g = float(np.clip(g, 0.78, 1.12))
    g_all = 1.0 + (g - 1.0) * ((0.45 if face_sel is not None else 0.6) + 0.4 * s)
    if abs(g_all - 1.0) > 0.015:
        L = 100.0 * np.power(np.clip(L / 100.0, 0, 1), g_all)
        notes.append(f"chỉnh sáng (gamma {g_all:.2f})")
    if face_sel is not None:
        now = float(np.median(L[face_sel]))
        gap = min(max(target - now, 0.0), 12.0) * (0.5 + 0.4 * s)
        if gap > 0.5:
            sel = (cv2.dilate((skin > 0.35).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0).astype(np.float32)
            face_light = cv2.GaussianBlur(sel, (0, 0), max(3.0, face_w / 6.0))
            face_light = face_light / max(float(face_light.max()), 1e-6)
            L = L + gap * face_light * (1.0 - L / 115.0)
            notes.append(f"đánh sáng khuôn mặt +{gap:.1f}")

    # ---- 4. giữ chi tiết vùng sáng/tối (tone mapping cục bộ không quầng)
    Ln = L / 100.0
    r = max(8, int(min(H, W) / 25))
    base = guided_filter(Ln, Ln, r, 0.02)
    detail = Ln - base
    hi = np.clip((base - 0.72) / 0.28, 0, 1)        # vùng rất sáng (trời): nén xuống
    lo = np.clip((0.28 - base) / 0.28, 0, 1)        # vùng rất tối: nâng lên
    base = base - 0.10 * s * hi * (base - 0.72) * 2.2 + 0.08 * s * lo * (0.28 - base) * 2.5
    Ln = base + detail * (1.0 + 0.06 * s)
    # đường cong S rất nhẹ cho "độ trong" của ảnh
    Ln = np.clip(Ln, 0, 1)
    Ln = Ln + 0.06 * s * np.sin(2 * np.pi * (Ln - 0.5)) / (2 * np.pi) * 4 * (Ln * (1 - Ln))
    L = np.clip(Ln, 0, 1) * 100.0
    notes.append("giữ chi tiết vùng sáng/tối")

    # ---- 5. vibrance: màu nhạt tăng nhiều hơn, bảo vệ màu da
    chroma = np.sqrt(A * A + B * B)
    boost = 1.0 + 0.22 * s * np.clip(1.0 - chroma / 55.0, 0, 1)
    if skin is not None:
        boost = 1.0 + (boost - 1.0) * (1.0 - 0.75 * skin)
    A *= boost
    B *= boost
    lab = np.dstack([L, A, B]).astype(np.float32)
    out = _lab_to_rgb(lab)

    # ---- 6. làm đẹp chân dung
    if portrait and skin is not None and skin.max() > 0.3:
        if face_w >= 110:                                    # mặt quá nhỏ thì không đủ chi tiết để xóa mụn
            out, nb = _remove_blemishes(out, skin, face_w)
            if nb:
                notes.append(f"xóa {nb} đốm mụn/thâm")
        out = smooth_skin(out, 22 + 30 * s, mask=skin)       # mờ nếp nhăn + mịn nhẹ, giữ 30% kết cấu
        lab = _rgb_to_lab(out)
        lift = (3.0 + 3.0 * s) * skin                        # sáng da nhẹ
        lab[..., 0] = np.clip(lab[..., 0] + lift * (1.0 - lab[..., 0] / 100.0) * 1.4, 0, 100)
        lab[..., 1] *= 1.0 - 0.10 * s * skin                 # giảm đỏ, da đều màu
        out = _lab_to_rgb(lab)
        notes.append("làm mịn nhẹ, mờ nếp nhăn, sáng da")

    # ---- 7. làm nét đầu ra, trừ vùng da
    f = to_float(out)
    y = luma(f)
    det = y - cv2.GaussianBlur(y, (0, 0), max(0.8, min(H, W) / 1600.0))
    det = np.sign(det) * np.maximum(np.abs(det) - 0.006, 0)
    wsharp = 0.45 * s * (np.ones_like(y) if skin is None else (1.0 - 0.9 * skin))
    out = to_uint8(f + (det * wsharp)[..., None])
    notes.append("làm nét chi tiết")
    return out, ", ".join(notes)


_CASCADES = None
_YUNET = None


def _yunet_faces(small: np.ndarray):
    """Nhận diện khuôn mặt bằng mạng nơ-ron YuNet (OpenCV Zoo, giấy phép MIT, 230 KB).
    Nhận được cả mặt nghiêng, đeo kính, ánh sáng yếu — tốt hơn nhiều so với Haar."""
    global _YUNET
    import os
    from ..engine import _asset_dir
    if _YUNET is None:
        path = os.path.join(_asset_dir(), "face_detection_yunet_2023mar.onnx")
        if not hasattr(cv2, "FaceDetectorYN") or not os.path.isfile(path):
            _YUNET = False
        else:
            _YUNET = cv2.FaceDetectorYN.create(path, "", (320, 320), 0.6, 0.3, 50)
    if _YUNET is False:
        return None
    h, w = small.shape[:2]
    _YUNET.setInputSize((w, h))
    _, res = _YUNET.detect(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
    if res is None:
        return []
    return [[int(r[0]), int(r[1]), int(r[2]), int(r[3])] for r in res if r[2] > 12 and r[3] > 12]


def _faces(img: np.ndarray):
    """Tìm khuôn mặt: ưu tiên YuNet; dự phòng Haar cascade (chính diện + nghiêng).
    Trả về list (x, y, w, h) theo tọa độ ảnh gốc."""
    global _CASCADES
    try:
        small, k = resize_long_side(img, 900)
        found = None
        try:
            found = _yunet_faces(small)
        except Exception:
            found = None
        if not found:
            if _CASCADES is None:
                base = cv2.data.haarcascades
                _CASCADES = [cv2.CascadeClassifier(base + n) for n in
                             ("haarcascade_frontalface_default.xml", "haarcascade_profileface.xml")]
                _CASCADES = [c for c in _CASCADES if not c.empty()]
            gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY))
            mins = max(24, int(min(gray.shape) / 14))
            found = []
            for i, cas in enumerate(_CASCADES):
                for flip in ((False, True) if i == 1 else (False,)):   # mặt nghiêng: thử cả 2 hướng
                    g = cv2.flip(gray, 1) if flip else gray
                    for (x, y, w, h) in cas.detectMultiScale(g, 1.1, 5, minSize=(mins, mins)):
                        if flip:
                            x = g.shape[1] - x - w
                        found.append([int(x), int(y), int(w), int(h)])
        boxes = []                                   # gộp các khung trùng nhau
        for b in sorted(found, key=lambda b: -b[2] * b[3]):
            if all(_iou(b, o) < 0.3 for o in boxes):
                boxes.append(b)
        return [tuple(int(v / k) for v in b) for b in boxes]
    except Exception:
        return []


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / float(aw * ah + bw * bh - inter + 1e-9)


def skin_mask(img: np.ndarray) -> np.ndarray:
    """Mặt nạ vùng da (float 0..1, mép mềm).

    Dựa trên ngưỡng màu da trong không gian YCrCb (ít phụ thuộc ánh sáng).
    Nếu tìm thấy khuôn mặt, chỉ giữ vùng da quanh mặt/cổ để tránh làm mờ
    nhầm gỗ, cát, tường có màu giống da."""
    ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    Y, Cr, Cb = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    m = ((Cr >= 133) & (Cr <= 178) & (Cb >= 80) & (Cb <= 132) & (Y > 35)).astype(np.uint8) * 255
    k = max(3, int(min(img.shape[:2]) / 250) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k * 2 + 1, k * 2 + 1), np.uint8))
    faces = _faces(img)
    if faces:
        region = np.zeros_like(m)
        for (x, y, w, h) in faces:
            cx, cy = x + w / 2, y + h / 2
            cv2.ellipse(region, (int(cx), int(cy + h * 0.25)), (int(w * 0.95), int(h * 1.35)), 0, 0, 360, 255, -1)
        m = cv2.bitwise_and(m, region)
    blur = max(3, int(min(img.shape[:2]) / 120))
    return cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), blur)


def smooth_skin(img: np.ndarray, strength: float, scale: float = 1.0, mask=None) -> np.ndarray:
    """Làm mịn da nhưng giữ khối mặt và kết cấu tự nhiên.

    Kỹ thuật "tách tần số" (frequency separation):
        * Lớp nền mịn  = Bilateral Filter — bộ lọc làm mờ có GIỮ CẠNH: chỉ trộn
          các điểm ảnh gần nhau VÀ có màu gần giống nhau, nên mụn/vết thâm bị
          làm phẳng còn đường viền mắt, mũi, môi vẫn sắc.
        * Lớp chi tiết = ảnh - ảnh mờ Gauss rất nhỏ (lỗ chân lông, sợi tóc).
        * Kết quả = nền mịn + 30% chi tiết  -> da mịn nhưng không bị "bệt nhựa".
    Sau đó trộn với ảnh gốc theo mặt nạ da và cường độ 0..100.
    Bilateral chạy trên ảnh thu nhỏ (cạnh dài <= 1400) cho nhanh rồi phóng lại."""
    if strength <= 0:
        return img
    s = strength / 100.0
    if mask is None or mask.shape != img.shape[:2]:
        mask = skin_mask(img)
    work, k = resize_long_side(img, 1400)
    sigma_space = max(3.0, min(work.shape[:2]) / 120.0)
    sigma_color = 18 + 55 * s
    base = cv2.bilateralFilter(work, d=0, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    base = cv2.bilateralFilter(base, d=0, sigmaColor=sigma_color * 0.7, sigmaSpace=sigma_space * 0.6)
    if k != 1.0:
        base = cv2.resize(base, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    f = img.astype(np.float32)
    fine = f - cv2.GaussianBlur(f, (0, 0), max(0.8, 1.2 * scale))
    smoothed = base.astype(np.float32) + 0.3 * fine
    w = (mask * min(1.0, s * 1.1))[..., None]
    return np.clip(f + (smoothed - f) * w + 0.5, 0, 255).astype(np.uint8)


def remove_objects(img: np.ndarray, mask: np.ndarray, method: str = "telea",
                   radius: int | None = None) -> np.ndarray:
    """Xóa vật thể theo vùng người dùng tô (mask != 0) bằng cv2.inpaint.

    cv2.inpaint "chảy" màu và đường nét từ viền vào trong vùng trống:
        * TELEA (Fast Marching): lấp từ ngoài vào trong theo trung bình có trọng số.
        * NS (Navier–Stokes): lan truyền theo đường đẳng sáng, giữ đường nét tốt hơn.
    Cải tiến thêm:
        1. Nới rộng mặt nạ 2–3 px để không sót viền vật thể.
        2. Vùng lớn: lấp ở độ phân giải thấp trước (giữ cấu trúc tổng thể), phóng
           lên rồi lấp lại ở độ phân giải gốc — nhanh và ít vệt kéo dài hơn.
        3. Thêm nhiễu hạt khớp với độ nhiễu của vùng xung quanh để mảng vá không
           "mịn bất thường" so với phần ảnh còn lại."""
    m = (mask > 0).astype(np.uint8) * 255
    if not m.any():
        return img
    h, w = img.shape[:2]
    grow = max(2, int(min(h, w) / 500))
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
    flag = cv2.INPAINT_NS if method == "ns" else cv2.INPAINT_TELEA
    r = radius or max(3, int(min(h, w) / 300))
    area = float((m > 0).mean())
    src = img
    if area > 0.004 and max(h, w) > 1200:
        small, k = resize_long_side(img, 1000)
        msmall = cv2.resize(m, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
        msmall = cv2.dilate(msmall, np.ones((3, 3), np.uint8))
        low = cv2.inpaint(small, msmall, max(3, int(r * k)), flag)
        low = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)
        sel = m > 0
        src = img.copy()
        src[sel] = low[sel]
        # Lấp lại ở độ phân giải gốc chỉ trên dải viền để khớp chi tiết.
        inner = cv2.erode(m, np.ones((2 * r + 1, 2 * r + 1), np.uint8))
        band = cv2.subtract(m, inner)
        out = cv2.inpaint(src, band, r, flag) if band.any() else src
    else:
        out = cv2.inpaint(img, m, r, flag)
    # Thêm nhiễu hạt theo độ nhiễu xung quanh.
    ring = cv2.subtract(cv2.dilate(m, np.ones((15, 15), np.uint8)), m) > 0
    if ring.sum() > 50:
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        hp = g - cv2.GaussianBlur(g, (0, 0), 1.0)
        sigma = float(np.clip(hp[ring].std(), 0, 8))
        if sigma > 0.6:
            rng = np.random.default_rng(12345)
            noise = rng.normal(0, sigma * 0.8, (h, w)).astype(np.float32)
            sel = m > 0
            o = out.astype(np.float32)
            o[sel] += noise[sel][:, None]
            out = np.clip(o + 0.5, 0, 255).astype(np.uint8)
    return out


# ============================================================================
# MODULE 2 — ĐÓNG LOGO (WATERMARK)
# ============================================================================

# 9 vị trí đặt logo (lưới 3×3): khóa -> (tên hiển thị, hệ số ngang, hệ số dọc)
POSITIONS = {
    "tl": ("Góc trên trái", 0.0, 0.0), "tc": ("Trên giữa", 0.5, 0.0), "tr": ("Góc trên phải", 1.0, 0.0),
    "ml": ("Giữa trái", 0.0, 0.5), "c": ("Chính giữa", 0.5, 0.5), "mr": ("Giữa phải", 1.0, 0.5),
    "bl": ("Góc dưới trái", 0.0, 1.0), "bc": ("Dưới giữa", 0.5, 1.0), "br": ("Góc dưới phải", 1.0, 1.0),
}


@dataclass
class WatermarkSettings:
    position: str = "br"     # một khóa trong POSITIONS
    opacity: float = 80      # độ đậm 0..100 (%)
    scale: float = 18        # chiều rộng logo = scale% chiều rộng ảnh
    margin: float = 3        # khoảng cách tới mép = margin% cạnh ngắn của ảnh

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: d[k] for k in ("position", "opacity", "scale", "margin") if k in d})


def load_logo(path: str) -> np.ndarray:
    """Đọc logo thành RGBA uint8. Ảnh không có kênh trong suốt -> alpha = 255."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGBA")).copy()


def watermark_rect(img_shape, logo_shape, st: WatermarkSettings):
    """Tính vị trí và kích thước logo (x, y, w, h) trên ảnh."""
    H, W = img_shape[:2]
    lh0, lw0 = logo_shape[:2]
    lw = max(1, int(round(W * st.scale / 100.0)))
    lh = max(1, int(round(lw * lh0 / float(lw0))))
    if lh > H:                                     # logo quá cao so với ảnh: co lại
        lh = H
        lw = max(1, int(round(lh * lw0 / float(lh0))))
    margin = int(round(min(W, H) * st.margin / 100.0))
    _, fx, fy = POSITIONS.get(st.position, POSITIONS["br"])
    x = int(round(margin + fx * (W - lw - 2 * margin)))
    y = int(round(margin + fy * (H - lh - 2 * margin)))
    return max(0, min(x, W - lw)), max(0, min(y, H - lh)), lw, lh


def compose_watermark(img: np.ndarray, logo_rgba: np.ndarray, st: WatermarkSettings) -> np.ndarray:
    """Chồng logo lên ảnh bằng phép trộn alpha:
        kết_quả = α·logo + (1 - α)·ảnh,  với α = alpha_logo × độ_đậm."""
    if logo_rgba is None or st.opacity <= 0:
        return img
    x, y, lw, lh = watermark_rect(img.shape, logo_rgba.shape, st)
    interp = cv2.INTER_AREA if lw < logo_rgba.shape[1] else cv2.INTER_CUBIC
    logo = cv2.resize(logo_rgba, (lw, lh), interpolation=interp)
    a = (logo[..., 3:4].astype(np.float32) / 255.0) * (st.opacity / 100.0)
    out = img.copy()
    roi = out[y:y + lh, x:x + lw].astype(np.float32)
    roi = a * logo[..., :3].astype(np.float32) + (1.0 - a) * roi
    out[y:y + lh, x:x + lw] = np.clip(roi + 0.5, 0, 255).astype(np.uint8)
    return out


# ============================================================================
# MODULE 1 — CẮT ẢNH THEO TỈ LỆ MẠNG XÃ HỘI & XUẤT FILE
# ============================================================================

# Khung mẫu một chạm: khóa, tên, tỉ lệ (rộng/cao) hoặc None (tự do), kích thước chuẩn khi xuất.
CROP_PRESETS = [
    ("free", "Tự do", None, None),
    ("fb_cover", "Facebook Cover (851×315)", 851 / 315, (1702, 630)),
    ("fb_post", "Facebook Post 1.91:1", 1.91, (1200, 628)),
    ("ig_post", "Instagram Post 1:1", 1.0, (1080, 1080)),
    ("ig_portrait", "Instagram dọc 4:5", 4 / 5, (1080, 1350)),
    ("story", "Stories / TikTok 9:16", 9 / 16, (1080, 1920)),
    ("yt_thumb", "YouTube Thumbnail 16:9", 16 / 9, (1280, 720)),
]
PRESET_MAP = {k: (label, ratio, size) for k, label, ratio, size in CROP_PRESETS}


def max_rect_for_ratio(W: int, H: int, ratio: float | None):
    """Khung lớn nhất có tỉ lệ 'ratio' nằm giữa ảnh (x, y, w, h)."""
    if not ratio:
        return 0.0, 0.0, float(W), float(H)
    if W / H > ratio:
        h = float(H)
        w = h * ratio
    else:
        w = float(W)
        h = w / ratio
    return (W - w) / 2.0, (H - h) / 2.0, w, h


def crop_image(img: np.ndarray, rect, target_size=None, alpha=None):
    """Cắt ảnh theo khung (x, y, w, h) và (tùy chọn) đổi về kích thước chuẩn.

    Khi thu nhỏ đáng kể sẽ làm nét nhẹ để bù độ mềm do nội suy."""
    H, W = img.shape[:2]
    x, y, w, h = rect
    x0, y0 = int(round(max(0, x))), int(round(max(0, y)))
    x1, y1 = int(round(min(W, x + w))), int(round(min(H, y + h)))
    out = img[y0:y1, x0:x1].copy()
    a = alpha[y0:y1, x0:x1].copy() if alpha is not None else None
    if target_size and out.size:
        tw, th = target_size
        if out.shape[1] > tw:                               # chỉ thu nhỏ, không phóng to làm vỡ ảnh
            k = tw / float(out.shape[1])
            out = cv2.resize(out, (tw, th), interpolation=cv2.INTER_AREA)
            if a is not None:
                a = cv2.resize(a, (tw, th), interpolation=cv2.INTER_AREA)
            if k < 0.8:
                out = to_uint8(sharpen(to_float(out), 25, 0.6))
    return out, a


# Quy tắc của từng nền tảng khi "nén thông minh".
PLATFORMS = {
    "facebook": {"label": "Facebook", "max_long": 2048, "ssim": 0.985, "max_bytes": None},
    "instagram": {"label": "Instagram", "max_w": 1080, "ssim": 0.985, "max_bytes": None},
    "youtube": {"label": "YouTube Thumbnail", "max_long": 1280, "ssim": 0.98, "max_bytes": 2_000_000},
    "general": {"label": "Giữ kích thước", "ssim": 0.99, "max_bytes": None},
}


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Chỉ số tương đồng cấu trúc SSIM (1.0 = giống hệt) trên kênh độ sáng.

    SSIM đo theo cách mắt người cảm nhận (độ sáng, tương phản, cấu trúc) nên
    phù hợp để quyết định "nén tới đâu thì vẫn chưa thấy khác biệt"."""
    x = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    y = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda z: cv2.GaussianBlur(z, (11, 11), 1.5)
    mx, my = blur(x), blur(y)
    sxx = blur(x * x) - mx * mx
    syy = blur(y * y) - my * my
    sxy = blur(x * y) - mx * my
    m = ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx * mx + my * my + C1) * (sxx + syy + C2))
    return float(m.mean())


def _chroma_busy(img: np.ndarray) -> bool:
    """Ảnh có nhiều cạnh màu sắc nét (chữ, đồ họa)? -> nên giữ màu đủ phân giải (4:4:4)."""
    ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    e = 0.0
    for c in (1, 2):
        gx = cv2.Sobel(ycc[..., c], cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(ycc[..., c], cv2.CV_32F, 0, 1)
        e += float(np.mean(np.abs(gx) + np.abs(gy)))
    return e > 18.0


def encode(img: np.ndarray, fmt: str, quality: int = 92, alpha=None, subsampling=None) -> bytes:
    """Mã hóa ảnh ra bytes theo định dạng 'jpg' | 'png' | 'webp' (không kèm metadata)."""
    fmt = fmt.lower()
    if alpha is not None and fmt in ("png", "webp"):
        im = Image.fromarray(np.dstack([img, alpha]), "RGBA")
    else:
        im = Image.fromarray(img, "RGB")
    buf = io.BytesIO()
    if fmt in ("jpg", "jpeg"):
        if subsampling is None:
            subsampling = 0 if _chroma_busy(img) else 2
        im.save(buf, "JPEG", quality=int(quality), optimize=True, progressive=True, subsampling=subsampling)
    elif fmt == "webp":
        im.save(buf, "WEBP", quality=int(quality), method=6)
    else:
        if alpha is None:
            # Ảnh ít màu (logo, đồ họa) -> PNG bảng màu, vẫn không mất dữ liệu.
            colors = im.getcolors(256)
            if colors is not None:
                im = im.quantize(colors=len(colors), method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE)
        im.save(buf, "PNG", optimize=True, compress_level=9)
    return buf.getvalue()


def prepare_for_platform(img: np.ndarray, platform: str, alpha=None):
    """Đưa ảnh về kích thước nền tảng hiển thị (để mạng xã hội không phải thu nhỏ lại)."""
    rule = PLATFORMS.get(platform, PLATFORMS["general"])
    h, w = img.shape[:2]
    k = 1.0
    if "max_long" in rule and max(h, w) > rule["max_long"]:
        k = rule["max_long"] / float(max(h, w))
    if "max_w" in rule and w > rule["max_w"]:
        k = min(k, rule["max_w"] / float(w))
    if k < 1.0:
        size = (max(1, round(w * k)), max(1, round(h * k)))
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        if alpha is not None:
            alpha = cv2.resize(alpha, size, interpolation=cv2.INTER_AREA)
        # Làm nét đầu ra nhẹ: bù phần mềm đi do thu nhỏ, chống bị "mờ" sau khi nền tảng nén lại.
        img = to_uint8(sharpen(to_float(img), 22, 0.6))
    return img, alpha, k


def smart_compress(img: np.ndarray, fmt: str = "jpg", platform: str = "facebook", alpha=None):
    """NÉN THÔNG MINH: dung lượng nhỏ nhất mà vẫn giữ độ sắc nét.

    Vì sao ảnh bị mờ trên Facebook/Instagram? Nền tảng sẽ THU NHỎ và NÉN LẠI
    mọi ảnh quá lớn (Facebook: cạnh dài > 2048 px; Instagram: rộng > 1080 px)
    hoặc dung lượng cao. Nén hai lần chồng lên nhau làm ảnh nhòe, vỡ khối.
    Cách khắc phục:
        1. Tự thu nhỏ đúng kích thước hiển thị của nền tảng bằng nội suy chất
           lượng cao (INTER_AREA) + làm nét đầu ra nhẹ.
        2. Tìm nhị phân chất lượng JPEG/WEBP THẤP NHẤT mà SSIM so với ảnh
           chưa nén vẫn >= ngưỡng (≈0.985 – mắt thường không phân biệt được).
           File nhỏ, đúng kích thước => nền tảng gần như không nén lại nữa.
        3. JPEG lũy tiến (progressive) + tối ưu Huffman; giữ màu 4:4:4 khi ảnh
           có chữ/đồ họa để chữ không bị lem màu.
    Trả về (bytes, thông tin)."""
    fmt = fmt.lower()
    rule = PLATFORMS.get(platform, PLATFORMS["general"])
    ref, alpha2, k = prepare_for_platform(img, platform, alpha)
    info = {"width": ref.shape[1], "height": ref.shape[0], "scale": k, "format": fmt}
    if fmt == "png":
        data = encode(ref, "png", alpha=alpha2)
        info.update(size=len(data), quality=None, ssim=1.0)
        return data, info
    target = rule["ssim"]
    lo, hi = 55, 96
    best = None
    sub = (0 if _chroma_busy(ref) else 2) if fmt in ("jpg", "jpeg") else None
    while lo <= hi:                                    # tìm nhị phân chất lượng
        q = (lo + hi) // 2
        data = encode(ref, fmt, q, alpha2, sub)
        dec = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
        s = ssim(ref, dec)
        if s >= target:
            best = (q, data, s)
            hi = q - 1
        else:
            lo = q + 1
    if best is None:
        q = 96
        data = encode(ref, fmt, q, alpha2, sub)
        best = (q, data, ssim(ref, np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))))
    q, data, s = best
    # Giới hạn dung lượng (vd. YouTube Thumbnail <= 2 MB).
    cap = rule.get("max_bytes")
    while cap and len(data) > cap and q > 40:
        q -= 5
        data = encode(ref, fmt, q, alpha2, sub)
    info.update(size=len(data), quality=q, ssim=s, subsampling=sub)
    return data, info
