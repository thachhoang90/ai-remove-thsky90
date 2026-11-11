# AI Remove - thsky90

Gỡ logo ngôi sao Gemini (✦) khỏi ảnh và video do Gemini / Veo tạo ra, kèm tùy chọn xóa metadata ẩn (EXIF, XMP, IPTC, C2PA / Content Credentials), và tab **Chỉnh sửa ảnh** cho mạng xã hội.

- **Bản Windows**: tải `AI-Remove-thsky90-<phiên bản>-Setup.exe` ở mục **Releases**, cài đặt không cần quyền admin.
- **Bản web**: chạy trên GitHub Pages, xử lý 100% trong trình duyệt (tệp không được tải lên máy chủ).

## Tính năng

| | Bản Windows | Bản web |
|---|---|---|
| Ảnh: PNG, JPG, JPEG, JFIF, WEBP, BMP, TIFF | Có | Có (trừ TIFF) |
| Ảnh: HEIC, HEIF | Có | Không (trình duyệt chưa đọc được) |
| GIF động | Có, giữ đủ khung | Chỉ khung đầu, xuất PNG |
| Video: MP4, MOV, AVI, WEBM, WMV, MKV | Có, giữ định dạng gốc hoặc xuất MP4 | Có, xuất MP4 |
| Chọn 1 tệp hoặc cả thư mục (kể cả thư mục con) | Có | Có |
| Tự dò logo / định vị thủ công có xem trước | Có | Có |
| Gỡ metadata ẩn | Có | Có |
| Chọn thư mục xuất | Có | Chrome/Edge: có; trình duyệt khác: tải .zip |

## Tab "Chỉnh sửa ảnh" (từ bản 2.1)

Giao diện tối, có ở cả bản Windows và bản web:

| Module | Chức năng |
|---|---|
| 1. Tệp và Xuất bản | Đọc/ghi JPG, PNG, WEBP (bản Windows đọc thêm HEIC); chuyển định dạng khi xuất; **nén thông minh** (thu về kích thước nền tảng + tìm mức nén nhỏ nhất có SSIM ≥ 0,985); khung cắt một chạm: Facebook Cover, FB Post 1.91:1, Instagram 1:1, 4:5, Stories/TikTok 9:16, YouTube 16:9 (khung di chuyển, co giãn được); xem trước lưới Instagram 3×3 / bảng tin Facebook |
| 2. Đóng logo | Logo PNG trong suốt, 9 vị trí, độ đậm 0–100%, kích thước theo % ảnh, lề; lưu/xóa mẫu (preset); đóng logo hàng loạt cả thư mục chạy nền, có nút ngưng |
| 3. Bộ lọc và Điều chỉnh | 16 bộ lọc (Vintage, Cinematic, Sáng/Tối giản, Đen trắng, Tông ấm, Tông lạnh, Phim bạc màu, Rực rỡ, Chân dung trong trẻo, Giờ vàng, Pastel Hàn Quốc, Moody tối, Phim Kodak, Fuji cổ điển, Sepia cổ điển, Matte lạnh) có cường độ; độ sáng, tương phản, bão hòa (gộp một ma trận màu 3×4), độ sắc nét (unsharp mask) — xem trước tức thì |
| 4. Làm đẹp và AI | Tối ưu một chạm cho chân dung và ảnh thường (AI YuNet nhận diện khuôn mặt trên bản Windows; cân bằng trắng theo màu da, đánh sáng khuôn mặt, giữ chi tiết vùng sáng/tối không quầng, vibrance, giảm nhiễu màu, xóa mụn, mờ nếp nhăn, sáng da; ảnh đồ họa/hoạt hình chỉ chỉnh nhẹ); làm mịn da (Bilateral Filter trên Windows / Guided Filter trên web + tách tần số, tự nhận vùng da); xóa vật thể bằng cọ vẽ + inpainting (cv2.inpaint Telea/Navier–Stokes; bản web dùng Telea viết bằng JavaScript) |
| 5. Giao diện | Undo/Redo 30 bước (Ctrl+Z / Ctrl+Y), cột lịch sử thao tác riêng bên phải khung ảnh; nút "Giữ để so sánh" trước/sau; giao diện tối (QSS trên Windows, CSS trên web) |

Mã nguồn có chú thích tiếng Việt chi tiết: `airemove/editor/` (Windows) và `web/js/editor/` (web).

## Thuật toán

Gemini đóng dấu bằng phép trộn alpha thông thường: `ảnh_có_logo = (1 − a)·ảnh_gốc + a·255`, với `a = k·A(x, y)`.
`A` là bản đồ alpha đã hiệu chuẩn của ngôi sao (chụp trên nền đen), `k` là độ mạnh.
Khi biết chính xác vị trí, kích thước và `k`, ảnh gốc được khôi phục bằng phép tính ngược
`ảnh_gốc = (ảnh_có_logo − a·255) / (1 − a)` — không "vẽ bù" như các công cụ AI inpaint.

Phần mềm tập trung vào việc **tìm đúng logo đến dưới 1 pixel**. Với **ảnh tĩnh** (từ bản 2.3):

1. **Dò theo "bậc sáng quanh viền"** (`airemove/detect.py`). Một lớp trắng bán trong suốt bắt buộc làm mọi điểm ngay bên trong viền sáng hơn điểm ngay bên ngoài, bất kể bên dưới là gì. Mỗi cặp điểm vắt qua viền cho một ước lượng riêng của độ mạnh `k`; trên logo thật đa số các cặp đồng thuận một giá trị, ở chỗ khác chúng phân tán. Trung vị có trọng số là `k`, tỉ lệ đồng thuận là độ tin cậy. Nét chữ, khuôn mặt hay cảnh rối nằm dưới logo chỉ làm hỏng vài cặp nên không còn làm trượt kết quả như cách so khớp độ sáng (NCC) của bản cũ.
2. Ứng viên lấy từ bản đồ "dấu của bậc sáng" đa tỉ lệ và từ các bố cục Gemini dùng (48 px cách mép 96 px; 48/32 px; 96/64 px, kể cả khi ảnh đã bị thu phóng). Ngoài tỉ lệ đồng thuận còn kiểm tra bậc sáng có **chỉ nằm đúng trên viền** hay không, để loại tia sáng, đốm sáng do họa sĩ vẽ. Ngưỡng được hiệu chỉnh trên ~200 góc ảnh có logo và ~200 góc ảnh sạch (PNG, WebP/JPEG đã thu nhỏ): không báo nhầm, nhận ~96 % logo.
3. Tinh chỉnh dưới 1 pixel vị trí, kích thước, độ sắc viền bằng hàm chi phí L1 trên dải viền; file gốc từ Gemini được khóa về đúng lưới 48/96 px để khôi phục không sai số.
4. **Khôi phục** (`airemove/restore.py`) bằng phép tính ngược ở trên: chữ, mặt nhân vật, họa tiết dưới logo được lấy lại từ chính file, không vẽ bù. Kênh màu của JPEG/WebP (4:2:0) dùng alpha đã làm mềm tương ứng.
5. **Khớp với vùng xung quanh**: file đã qua chuyển đổi màu, bảng màu (GIF/PNG-8) hoặc ảnh từng đi qua Gemini nhiều lần thường để lại một "hình thoi" hơi xỉn. Sai lệch được đo ở ngay hai bên viền rồi trừ đi cho vùng bên trong; chỉ áp dụng khi sai lệch chắc chắn về mặt thống kê nên file chuẩn không bị đụng tới.
6. **Viền**: điểm ảnh trên viền chỉ được thay bằng phép nội suy cắt ngang viền khi tranh ở đó phẳng, điểm đó lệch hơn độ nhiễu của chính bức tranh, và độ lệch nằm trong giới hạn mà sai số alpha có thể gây ra. Nét chữ, nét vẽ cắt qua viền được giữ nguyên.
7. Chế độ thủ công dùng đúng bộ dò này trong khung người dùng vẽ (khung lệch hoặc rộng hơn logo vẫn tự khớp). Chỉ khi trong khung không có logo Gemini thì mới vá vùng theo khung. Ảnh không có logo được giữ nguyên.

Riêng **video**: logo đứng yên nên phần mềm lấy mẫu ~40 khung, cộng dồn bản đồ NCC để khóa vị trí, rồi **tự hiệu chuẩn bản đồ alpha từ chính video** (trung vị có trọng số của `(I − nền)/(trắng − nền)` qua nhiều khung). Việc khử thực hiện trực tiếp trên mặt phẳng Y/U/V đã giải mã, phần còn lại của khung hình không bị chuyển đổi màu.

Giới hạn: watermark ẩn **SynthID** nằm trong điểm ảnh và không thể gỡ bằng phần mềm này.

## Dùng thử từ mã nguồn

```bash
pip install numpy opencv-python-headless pillow pillow-heif PySide6-Essentials imageio-ffmpeg
python main.py                          # giao diện
python main.py --cli anh_hoac_thu_muc -o thu_muc_xuat   # dòng lệnh
```

Bản web: phục vụ thư mục `web/` bằng bất kỳ máy chủ tĩnh nào, ví dụ `python -m http.server -d web`.

## Build bộ cài Windows

Bộ cài được build trên Linux (không cần máy Windows):

```bash
sudo apt-get install nsis gcc-mingw-w64-x86-64 zip unzip
bash packaging/windows/build_on_linux.sh
```

Kết quả nằm trong `dist/`: `AI-Remove-thsky90-<ver>-Setup.exe` và bản portable `.zip`.
Workflow `.github/workflows/build-windows.yml` làm việc này tự động: vào **Releases → Draft a new release**, tạo tag `v1.0.0`, bấm **Publish release**; khoảng 10–15 phút sau file Setup.exe và bản portable được gắn vào Release. Hoặc vào **Actions → Build Windows installer → Run workflow** để lấy bản build dạng artifact.
Workflow `.github/workflows/pages.yml` triển khai thư mục `web/` lên GitHub Pages
(bật trong Settings → Pages → Source: GitHub Actions).

## Cấu trúc

```
airemove/engine.py        lõi dùng chung (mẫu alpha, hình học) + luồng xử lý ảnh/video
airemove/detect.py        dò logo trên ảnh tĩnh bằng bậc sáng quanh viền (2.3)
airemove/restore.py       khôi phục vùng dưới logo, khớp màu xung quanh, làm sạch viền (2.3)
airemove/scrollview.py    thanh trượt ngang/dọc cho khung xem trước (2.3)
airemove/video.py         xử lý video (ffmpeg, Y/U/V, tự hiệu chuẩn alpha)
airemove/imagefile.py     đọc/ghi ảnh, metadata
airemove/gui.py           giao diện PySide6
airemove/editor/          tab "Chỉnh sửa ảnh" (ops, canvas, lịch sử, xem trước MXH, đóng logo hàng loạt)
web/                      bản web (JavaScript + ffmpeg.wasm); web/js/logo.js = dò + khôi phục ảnh tĩnh (2.3); web/js/editor/ = tab chỉnh sửa ảnh
packaging/windows/        launcher .exe, kịch bản NSIS, script build
```

## Lịch sử phiên bản

- **2.3.0**: viết lại thuật toán gỡ logo trên ảnh tĩnh. Dò logo bằng bậc sáng quanh viền nên tìm được logo nằm đè lên chữ, mặt nhân vật, cảnh phức tạp (bản 2.2 bỏ sót); khôi phục bằng phép tính ngược kèm khớp màu với vùng xung quanh và làm sạch viền có bảo vệ nét vẽ, không còn tự rơi về kiểu vá vùng gây lem màu; chế độ thủ công tự khớp ngôi sao trong khung. Thêm thanh trượt ngang/dọc cho khung xem trước (cả 3 tab, bản Windows và bản web). Bản web dùng cùng thuật toán (`web/js/logo.js`) và dò logo nhanh hơn nhiều (1–2 giây thay cho ~12 giây trên ảnh 2000 px). Xử lý video giữ nguyên như 2.2.
- **2.2.0**: cột lịch sử thao tác riêng bên phải khung ảnh; thêm 8 bộ lọc (tổng 16); viết lại "Tối ưu một chạm" theo hướng làm đẹp chân dung tự nhiên (nhận diện khuôn mặt YuNet, đánh sáng mặt, xóa mụn, mờ nếp nhăn, sáng da, giữ chi tiết trời không quầng) và tự nhận biết ảnh đồ họa/hoạt hình.
- **2.1.0**: thêm tab "Chỉnh sửa ảnh" (5 module ở trên) cho cả bản Windows và bản web; giữ nguyên toàn bộ tính năng gỡ logo/metadata của bản 1.1.
- **1.1.0**: sửa lỗi build bộ cài trên GitHub Actions (đường dẫn `dist` tương đối); giao diện mới: tab Ảnh/Video nổi bật, nút to hơn và có màu, thanh trạng thái lớn hơn, cột danh sách rộng hơn; thêm cột tích chọn (chỉ xử lý/tải các mục đã chọn), nút "Tải các mục đã chọn" và "Ngưng tiến trình" (bản web; bản Windows có cột tích chọn và nút "Ngưng tiến trình"); sửa lỗi thanh chọn khung hình video hiện cả ở tab Ảnh trên bản web.
- **1.0.0**: phát hành đầu tiên.

## Bản quyền

MIT. Bản đồ alpha của logo Gemini lấy từ GeminiWatermarkTool (© 2024 AllenK) và gemini-watermark-remover (© 2025 Jad), giấy phép MIT. Xem `LICENSE`.
