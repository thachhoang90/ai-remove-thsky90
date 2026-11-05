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

def auto_enhance(img: np.ndarray) -> tuple[np.ndarray, str]:
    """Tối ưu một chạm dựa trên biểu đồ histogram.

    1. Cân bằng trắng "thế giới xám" (gray-world) chỉ trên các điểm trung tính
       vừa phải (bỏ vùng quá tối/quá sáng), giới hạn hệ số để không đổi màu quá đà.
    2. Kéo giãn mức (auto levels): lấy phân vị 0.5% và 99.5% của độ sáng làm
       điểm đen/trắng mới — áp cùng một hệ số cho cả 3 kênh để không lệch màu.
    3. Hiệu chỉnh gamma để độ sáng trung bình tiến về mức xám giữa.
    4. CLAHE nhẹ trên kênh L (không gian Lab) để tăng tương phản cục bộ.
    Trả về (ảnh, mô tả các bước đã làm)."""
    f = to_float(img)
    notes = []
    # --- 1. Cân bằng trắng
    y = luma(f)
    mid = (y > 0.12) & (y < 0.92)
    if mid.sum() > 500:
        means = f[mid].mean(axis=0)
        gray = means.mean()
        gains = np.clip(gray / np.maximum(means, 1e-4), 0.8, 1.25)
        gains = 1.0 + 0.7 * (gains - 1.0)            # chỉ sửa 70% để giữ "không khí" ảnh
        f = f * gains.astype(np.float32)
        notes.append("cân bằng trắng R×%.2f G×%.2f B×%.2f" % tuple(gains))
    # --- 2. Kéo giãn mức theo histogram độ sáng
    y = luma(np.clip(f, 0, 1))
    lo, hi = np.percentile(y, [0.5, 99.5])
    lo = min(lo, 0.10)
    hi = max(hi, 0.80)
    if hi - lo > 0.05:
        f = (f - lo) / (hi - lo)
        notes.append("giãn mức đen/trắng %.0f→0, %.0f→255" % (lo * 255, hi * 255))
    f = np.clip(f, 0, 1)
    # --- 3. Gamma
    m = float(luma(f).mean())
    if 0.05 < m < 0.95:
        g = math.log(0.48) / math.log(m)
        g = float(np.clip(g, 0.75, 1.35))
        g = 1.0 + 0.6 * (g - 1.0)
        if abs(g - 1.0) > 0.02:
            f = np.power(f, g)
            notes.append("gamma %.2f" % g)
    out = to_uint8(f)
    # --- 4. CLAHE trên kênh L
    lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8))
    L2 = clahe.apply(lab[..., 0])
    lab[..., 0] = cv2.addWeighted(L2, 0.5, lab[..., 0], 0.5, 0)
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    notes.append("tương phản cục bộ (CLAHE)")
    # Tăng bão hòa nhẹ 8%
    out = to_uint8(apply_matrix(to_float(out), adjustment_matrix(0, 0, 8)))
    return out, ", ".join(notes)


_FACE_CASCADE = None


def _faces(img: np.ndarray):
    """Tìm khuôn mặt (Haar cascade có sẵn trong OpenCV). Trả về list (x,y,w,h)."""
    global _FACE_CASCADE
    try:
        if _FACE_CASCADE is None:
            path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            _FACE_CASCADE = cv2.CascadeClassifier(path)
        small, k = resize_long_side(img, 900)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        found = _FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        return [tuple(int(v / k) for v in f) for f in found]
    except Exception:
        return []


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
