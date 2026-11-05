"""
image_processing/mudguard_filter.py - Tách dè xe, bô xe và bóng đổ dính vào mép biển số
Giúp khôi phục tỷ lệ khung hình chuẩn (Aspect Ratio) cho biển số 2 dòng bị dè dính liền.
"""

from typing import Tuple, Optional
import numpy as np
import cv2


def detect_and_separate_mudguard(
    img_np: np.ndarray,
    bbox: Tuple[int, int, int, int],
    scale_factor: float = 1.0,
    min_ar_trigger: float = 1.15
) -> Tuple[Tuple[int, int, int, int], bool]:
    """
    Kiểm tra và cắt bỏ phần dè xe nhựa đen / gầm xe / bô xe dính ở đáy hộp bao (Bounding Box).

    Hiện tượng thực tế:
    Biển 2 dòng xe máy thường có tỷ lệ w/h ~ 1.20 - 1.60.
    Khi chụp, phần dè nhựa đen bên dưới (hoặc chữ viết phấn trên dè như '301')
    dễ bị phép Closing nối liền vào biển, làm chiều cao h tăng vọt,
    tỷ lệ w/h tụt xuống 0.70 - 1.10 (hình chữ nhật đứng), dẫn tới bị bộ lọc loại bỏ nhầm.

    Thuật toán:
    1. Chỉ kích hoạt khi w/h < min_ar_trigger (nghi vấn bị dính dè ở đáy).
    2. Cắt ROI từ ảnh gốc hoặc ảnh xử lý.
    3. Phân tích lược đồ cường độ sáng theo chiều dọc (Vertical Brightness Profile)
       và đạo hàm biên ngang Sobel-Y (Horizontal Edge Projection).
    4. Tìm đường viền đáy biển (nơi cường độ sáng tụt dốc từ nền biển sáng sang nhựa đen,
       hoặc nơi có gờ viền ngang mạnh).
    5. Cắt ngắn chiều cao BBox tại đường viền đáy, đưa tỷ lệ w/h về vùng chuẩn (1.15 - 1.85).

    Args:
        img_np: Ảnh gốc RGB hoặc ảnh xám.
        bbox: (x, y, w, h) của ứng viên.
        scale_factor: Hệ số tỉ lệ nếu bbox trên ảnh resize.
        min_ar_trigger: Ngưỡng w/h để kích hoạt kiểm tra tách dè (mặc định 1.15).

    Returns:
        (new_bbox, was_separated): BBox mới sau khi tách dè và cờ đánh dấu có tách hay không.
    """
    x, y, w, h = bbox
    ar = float(w) / float(max(1, h))

    # Nếu tỷ lệ đã chuẩn (ar >= 1.18), không cần tách dè
    if ar >= min_ar_trigger:
        return bbox, False

    try:
        H, W = img_np.shape[:2]
        # Chuyển đổi tọa độ nếu cần
        if scale_factor > 0 and scale_factor != 1.0:
            rx = int(round(x / scale_factor))
            ry = int(round(y / scale_factor))
            rw = int(round(w / scale_factor))
            rh = int(round(h / scale_factor))
        else:
            rx, ry, rw, rh = x, y, w, h

        rx1 = max(0, min(rx, W - 1))
        ry1 = max(0, min(ry, H - 1))
        rx2 = max(0, min(rx + rw, W))
        ry2 = max(0, min(ry + rh, H))

        if rx2 - rx1 < 20 or ry2 - ry1 < 25:
            return bbox, False

        roi = img_np[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return bbox, False

        if len(roi.shape) == 3:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        else:
            roi_gray = roi

        roi_h, roi_w = roi_gray.shape

        # Tính profile cường độ trung bình theo từng dòng
        row_means = np.mean(roi_gray, axis=1)

        # Tính Sobel-Y để tìm các đường gờ ngang mạnh
        sobel_y = cv2.Sobel(roi_gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_y = np.mean(np.abs(sobel_y), axis=1)

        # Biển số 2 dòng thật sự thường nằm ở nửa trên của ROI bị dính dè
        # Tìm điểm cắt tối ưu trong khoảng từ 40% đến 80% chiều cao ROI
        min_cut = int(0.40 * roi_h)
        max_cut = int(0.85 * roi_h)

        best_cut = None
        max_score = -1.0

        for cut_y in range(min_cut, max_cut):
            top_mean = float(np.mean(row_means[max(0, cut_y - 25):cut_y]))
            bottom_mean = float(np.mean(row_means[cut_y:min(roi_h, cut_y + 25)]))

            # Độ chênh lệch sáng giữa vùng trên (biển số) và vùng dưới (dè xe đen)
            contrast_drop = top_mean - bottom_mean

            # Gờ biên ngang tại vị trí cắt
            edge_val = float(edge_y[cut_y])

            # Kiểm tra tỷ lệ khung hình kết quả nếu cắt tại đây
            candidate_h = cut_y
            candidate_ar = float(roi_w) / float(max(1, candidate_h))

            # Ưu tiên tỷ lệ chuẩn của biển 2 dòng (1.20 - 1.65)
            ar_fit = 1.0 - abs(candidate_ar - 1.40) / 0.50
            if candidate_ar < 1.05 or candidate_ar > 2.20:
                continue

            score = contrast_drop * 1.5 + edge_val * 2.0 + max(0.0, ar_fit) * 40.0

            if score > max_score and (contrast_drop > 15.0 or edge_val > 40.0):
                max_score = score
                best_cut = cut_y

        if best_cut is not None:
            # Quy đổi ngược lại hệ tọa độ ban đầu
            if scale_factor > 0 and scale_factor != 1.0:
                new_h = int(round(best_cut * scale_factor))
            else:
                new_h = best_cut

            new_bbox = (x, y, w, max(15, new_h))
            return new_bbox, True

        return bbox, False
    except Exception:
        return bbox, False


def check_edge_cut_and_substitution(
    cand: dict,
    ocr_res: dict,
    cropped_w: int,
    cropped_h: int
) -> Tuple[bool, bool]:
    """
    Phân biệt rạch ròi 2 loại lỗi:
    1. edge_cut: Ký tự bị mép ảnh xén mất một phần do sai lệch định vị hoặc cắt BBox quá sát.
    2. ocr_substitution: Vùng cắt có đủ lề đệm an toàn, ký tự nguyên vẹn nhưng CRNN đọc nhầm.

    Args:
        cand: Dictionary ứng viên.
        ocr_res: Kết quả OCR chi tiết.
        cropped_w: Chiều rộng ảnh cắt biển số.
        cropped_h: Chiều cao ảnh cắt biển số.

    Returns:
        (is_edge_cut, is_ocr_substitution)
    """
    characters = ocr_res.get('characters', [])
    if not characters:
        return False, False

    edge_cut = False
    ocr_substitution = False

    # Ngưỡng lề an toàn (pixels)
    border_margin = max(3, int(min(cropped_w, cropped_h) * 0.035))

    for ch in characters:
        cb = ch.get('bbox')
        if cb is not None:
            cx, cy, cw, ch_h = cb
            # Kiểm tra xem ký tự có bị chạm hoặc xén sát mép ảnh hay không
            if cx <= border_margin or (cx + cw) >= (cropped_w - border_margin):
                edge_cut = True
            if cy <= border_margin or (cy + ch_h) >= (cropped_h - border_margin):
                edge_cut = True

    # Nếu không bị cắt mép nhưng ký tự có điểm thấp hoặc xung đột cú pháp
    syntax_conflict = any(c.get('match_details', {}).get('syntax_conflict', False) for c in characters)
    low_conf = any(c.get('score', 1.0) < 0.68 for c in characters)
    needs_review = ocr_res.get('needs_review', False)

    if not edge_cut and (syntax_conflict or low_conf or needs_review):
        ocr_substitution = True

    return edge_cut, ocr_substitution
