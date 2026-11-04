# AI Remove - thsky90

Gỡ logo ngôi sao Gemini (✦) khỏi ảnh và video do Gemini / Veo tạo ra, kèm tùy chọn xóa metadata ẩn (EXIF, XMP, IPTC, C2PA / Content Credentials).

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

## Thuật toán

Gemini đóng dấu bằng phép trộn alpha thông thường: `ảnh_có_logo = (1 − a)·ảnh_gốc + a·255`, với `a = k·A(x, y)`.
`A` là bản đồ alpha đã hiệu chuẩn của ngôi sao (chụp trên nền đen), `k` là độ mạnh.
Khi biết chính xác vị trí, kích thước và `k`, ảnh gốc được khôi phục bằng phép tính ngược
`ảnh_gốc = (ảnh_có_logo − a·255) / (1 − a)` — không "vẽ bù" như các công cụ AI inpaint.

Phần mềm tập trung vào việc **tìm đúng logo đến dưới 1 pixel**:

1. Dò thô bằng tương quan chuẩn hóa (NCC) đa tỉ lệ ở góc phải-dưới, nên vẫn tìm được khi ảnh đã bị resize hoặc vị trí logo thay đổi (Gemini 2026 đặt logo 48 px cách mép 96 px, bản cũ 48/32 px và 96/64 px).
2. Tinh chỉnh theo từng tọa độ (x, y, kích thước, độ mạnh, độ sắc viền) bằng cách tối thiểu năng lượng gradient còn sót sau khi khử.
3. Ước lượng độ mạnh thứ hai bằng cách so độ sáng bên trong logo với phần nền nội suy điều hòa (harmonic) từ xung quanh; hai ước lượng được kết hợp theo độ phẳng của nền.
4. Xử lý màu trong không gian YCbCr: kênh màu của JPEG/WebP/H.264 bị lấy mẫu 4:2:0 nên dùng alpha đã làm mềm tương ứng.
5. Làm sạch đường viền 1–2 px còn sót do nén.
6. Chốt kiểm tra: nếu kết quả vẫn còn vết trên nền phẳng (ví dụ GIF bị lượng tử màu) thì tự chuyển sang vá vùng. Ảnh không có logo được giữ nguyên.

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
airemove/engine.py        thuật toán dò + khử logo
airemove/video.py         xử lý video (ffmpeg, Y/U/V, tự hiệu chuẩn alpha)
airemove/imagefile.py     đọc/ghi ảnh, metadata
airemove/gui.py           giao diện PySide6
web/                      bản web (JavaScript + ffmpeg.wasm)
packaging/windows/        launcher .exe, kịch bản NSIS, script build
```

## Bản quyền

MIT. Bản đồ alpha của logo Gemini lấy từ GeminiWatermarkTool (© 2024 AllenK) và gemini-watermark-remover (© 2025 Jad), giấy phép MIT. Xem `LICENSE`.
