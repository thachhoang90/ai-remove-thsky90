// AI Remove - thsky90 — thanh trượt ngang + dọc cho khung xem trước (v2.3)
//
// Các khung xem trước của bản web tự vẽ ảnh lên <canvas> theo ba giá trị:
// scale (độ phóng), ox / oy (vị trí góc trên-trái của ảnh).  Mô-đun này thêm
// hai thanh trượt phủ lên mép phải và mép dưới của khung:
//   - phóng to ảnh  -> thanh trượt hiện rõ, kéo để tới nhanh vùng cần định vị;
//   - ảnh nằm trọn trong khung -> thanh trượt mờ đi (không có gì để cuộn);
//   - lăn chuột / kéo ảnh vẫn dùng như cũ, thanh trượt tự chạy theo.

const MARGIN = 12;          // khoảng trống quanh ảnh khi cuộn hết cỡ (px màn hình)

/**
 * @param stage  phần tử bao (position: relative) chứa canvas
 * @param view   đối tượng khung nhìn có scale, ox, oy, fitted, draw(), size()
 * @param size   hàm trả về [rộng, cao] của ảnh (điểm ảnh thật) hoặc null khi chưa có ảnh
 */
export function attachScrollbars(stage, view, size) {
  const mk = cls => {
    const bar = document.createElement('div'); bar.className = 'sb ' + cls + ' off';
    bar.title = 'Kéo để di chuyển vùng xem khi ảnh đang phóng to';
    const thumb = document.createElement('i'); bar.appendChild(thumb); stage.appendChild(bar);
    return { bar, thumb, lo: 0, hi: 0, val: 0, track: 0, len: 0 };
  };
  const H = mk('sb-h'), V = mk('sb-v');

  function syncAxis(a, content, viewLen, off, horizontal) {
    if (!content || content <= viewLen) { a.bar.classList.add('off'); return; }
    a.bar.classList.remove('off');
    const val = MARGIN - off, rng = content + 2 * MARGIN - viewLen;
    // nếu ảnh đã bị kéo ra ngoài phạm vi thường (ví dụ phóng vào logo ở sát góc),
    // nới phạm vi để thanh trượt đi tiếp từ đúng vị trí hiện tại
    a.lo = Math.min(0, val); a.hi = Math.max(rng, val); a.val = val;
    const r = a.bar.getBoundingClientRect();
    a.track = horizontal ? r.width : r.height;
    a.len = Math.max(28, a.track * viewLen / (a.hi - a.lo + viewLen));
    const pos = (a.hi > a.lo ? (val - a.lo) / (a.hi - a.lo) : 0) * (a.track - a.len);
    if (horizontal) { a.thumb.style.width = a.len + 'px'; a.thumb.style.left = pos + 'px'; }
    else { a.thumb.style.height = a.len + 'px'; a.thumb.style.top = pos + 'px'; }
  }

  function sync() {
    const dim = size(), [w, h] = view.size();
    syncAxis(H, dim ? dim[0] * view.scale : 0, w, view.ox, true);
    syncAxis(V, dim ? dim[1] * view.scale : 0, h, view.oy, false);
  }

  function setVal(a, val, horizontal) {
    val = Math.max(a.lo, Math.min(a.hi, val));
    if (horizontal) view.ox = MARGIN - val; else view.oy = MARGIN - val;
    view.fitted = false;
    view.draw();
  }

  for (const [a, horizontal] of [[H, true], [V, false]]) {
    // kéo con trượt
    a.thumb.addEventListener('pointerdown', e => {
      e.preventDefault(); e.stopPropagation();
      a.thumb.setPointerCapture(e.pointerId); a.thumb.classList.add('drag');
      const start = horizontal ? e.clientX : e.clientY, v0 = a.val;
      const move = ev => {
        const d = (horizontal ? ev.clientX : ev.clientY) - start;
        setVal(a, v0 + d * (a.hi - a.lo) / Math.max(1, a.track - a.len), horizontal);
      };
      const up = () => {
        a.thumb.classList.remove('drag');
        a.thumb.removeEventListener('pointermove', move); a.thumb.removeEventListener('pointerup', up); a.thumb.removeEventListener('pointercancel', up);
      };
      a.thumb.addEventListener('pointermove', move); a.thumb.addEventListener('pointerup', up); a.thumb.addEventListener('pointercancel', up);
    });
    // bấm vào rãnh: nhảy một trang
    a.bar.addEventListener('pointerdown', e => {
      if (e.target !== a.bar) return;
      e.preventDefault();
      const r = a.thumb.getBoundingClientRect(), p = horizontal ? e.clientX : e.clientY, t0 = horizontal ? r.left : r.top;
      const [w, h] = view.size(), page = (horizontal ? w : h) * 0.9;
      setVal(a, a.val + (p < t0 ? -page : page), horizontal);
    });
    a.bar.addEventListener('wheel', e => e.stopPropagation(), { passive: true });
  }

  // đồng bộ sau mỗi lần khung vẽ lại (phóng, kéo, đổi ảnh, đổi kích thước cửa sổ)
  const draw = view.draw.bind(view);
  view.draw = function () { draw(); sync(); };
  return { sync, h: H, v: V };
}
