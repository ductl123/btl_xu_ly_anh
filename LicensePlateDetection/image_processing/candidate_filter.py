"""
image_processing/candidate_filter.py - Lọc ứng viên biển số xe (Candidate Filtering)
Bao gồm:
- Lọc theo tỷ lệ khung hình (Aspect Ratio w/h):
  + Biển 1 dòng (dài): khoảng 4.5 - 5.0 (cho phép trong đoạn 2.6 - 6.2).
  + Biển 2 dòng (vuông): khoảng 1.3 - 1.5 (cho phép trong đoạn 1.05 - 2.2).
- Lọc theo tỷ lệ diện tích và kích thước tương đối so với toàn bộ bức ảnh.
- Lọc theo mật độ điểm trắng (White Density / Fill Ratio).
- Tính điểm tin cậy (Confidence Score).
- Áp dụng Non-Maximum Suppression (NMS) để loại bỏ các hộp bao chồng chéo.
- Cho phép phát hiện một hoặc nhiều biển số trong cùng một ảnh.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np


def compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """
    Tính chỉ số giao trên hợp (Intersection over Union - IoU) giữa 2 Bounding Box.
    Mỗi box có định dạng: (x, y, w, h).
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return float(inter_area) / float(union_area)


def apply_nms(candidates: List[Dict[str, Any]], iou_threshold: float = 0.35) -> List[Dict[str, Any]]:
    """
    Non-Maximum Suppression (NMS):
    Loại bỏ các hộp bao bị chồng chéo lên nhau, giữ lại hộp có điểm tin cậy (score) cao nhất.
    """
    if not candidates:
        return []

    # Sắp xếp các ứng viên giảm dần theo điểm tin cậy
    sorted_candidates = sorted(candidates, key=lambda c: c['score'], reverse=True)
    kept: List[Dict[str, Any]] = []

    for cand in sorted_candidates:
        should_keep = True
        for kept_cand in kept:
            iou = compute_iou(cand['bbox'], kept_cand['bbox'])
            if iou > iou_threshold:
                should_keep = False
                break
        if should_keep:
            kept.append(cand)

    return kept


def filter_license_plate_candidates(
    components: List[Dict[str, Any]],
    image_shape: Tuple[int, int],
    config_params: Dict[str, Any],
    sobel_np: Optional[np.ndarray] = None
) -> List[Dict[str, Any]]:
    """
    Lọc các vùng liên thông để chọn ra những ứng viên phù hợp với biển số xe Việt Nam.

    Args:
        components: Danh sách các thành phần liên thông từ bước trước.
        image_shape: (H, W) của ảnh xử lý.
        config_params: Dictionary chứa các ngưỡng từ config.py.
        sobel_np: Ma trận gradient Sobel-X để tính mật độ năng lượng cạnh.

    Returns:
        Danh sách ứng viên biển số đạt tất cả tiêu chí lọc, kèm phân loại và điểm tin cậy.
    """
    img_h, img_w = image_shape
    total_img_area = img_h * img_w

    # Đọc tham số lọc từ cấu hình
    single_min_ar = config_params.get("SINGLE_ROW_ASPECT_RATIO_MIN", 2.6)
    single_max_ar = config_params.get("SINGLE_ROW_ASPECT_RATIO_MAX", 6.2)
    double_min_ar = config_params.get("DOUBLE_ROW_ASPECT_RATIO_MIN", 0.90)
    double_max_ar = config_params.get("DOUBLE_ROW_ASPECT_RATIO_MAX", 2.20)

    min_area_ratio = config_params.get("MIN_AREA_RATIO", 0.0006)
    max_area_ratio = config_params.get("MAX_AREA_RATIO", 0.25)
    min_w_ratio = config_params.get("MIN_WIDTH_RATIO", 0.035)
    max_w_ratio = config_params.get("MAX_WIDTH_RATIO", 0.75)
    min_h_ratio = config_params.get("MIN_HEIGHT_RATIO", 0.012)
    max_h_ratio = config_params.get("MAX_HEIGHT_RATIO", 0.45)

    min_density = config_params.get("MIN_WHITE_DENSITY", 0.14)
    max_density = config_params.get("MAX_WHITE_DENSITY", 0.95)
    iou_thresh = config_params.get("NMS_IOU_THRESHOLD", 0.35)

    valid_candidates: List[Dict[str, Any]] = []

    for comp in components:
        x, y, w, h = comp['bbox']
        aspect_ratio = comp['aspect_ratio']
        density = comp['density']
        area = comp['bbox_area']

        # 0. Loại bỏ biên ảnh cực đoan (đèn trần, viền giá đèo hàng ở mép trên)
        if y < int(img_h * 0.035) or (y + h) > int(img_h * 0.985):
            continue

        # 1. Kiểm tra kích thước tương đối so với toàn ảnh
        area_ratio = area / float(total_img_area)
        w_ratio = w / float(img_w)
        h_ratio = h / float(img_h)

        if not (min_area_ratio <= area_ratio <= max_area_ratio):
            continue
        if not (min_w_ratio <= w_ratio <= max_w_ratio):
            continue
        if not (min_h_ratio <= h_ratio <= max_h_ratio):
            continue

        # 2. Kiểm tra mật độ điểm trắng (sau khi closing thành mảng đặc)
        if not (min_density <= density <= max_density):
            continue

        # 3. Phân loại theo tỷ lệ hình học biển số (biển 1 dòng hoặc 2 dòng)
        plate_type = None
        ideal_ar = None

        if single_min_ar <= aspect_ratio <= single_max_ar:
            plate_type = "single_row"
            ideal_ar = 4.7
        elif double_min_ar <= aspect_ratio <= double_max_ar:
            plate_type = "double_row"
            ideal_ar = 1.4

        if plate_type is None:
            continue

        # 4. Tính điểm tin cậy tổng hợp đa tiêu chí (Multi-factor Confidence Score)
        # a. Độ khớp tỷ lệ khung hình
        ar_deviation = abs(aspect_ratio - ideal_ar) / ideal_ar
        ar_score = max(0.0, 1.0 - ar_deviation)

        # b. Mật độ điểm trắng đặc trưng ký tự
        density_score = 1.0 - abs(density - 0.50) / 0.50
        density_score = max(0.0, min(1.0, density_score))

        # c. Điểm độ nổi bật diện tích (biển số thật có diện tích vượt trội so với đốm nhiễu)
        area_score = min(1.0, area_ratio / 0.012)

        # d. Mật độ năng lượng cạnh Sobel (biển số xe luôn có nhiều nét cạnh thẳng đứng sắc nét)
        edge_score = 0.5
        if sobel_np is not None:
            patch_sobel = sobel_np[y : y + h, x : x + w]
            if patch_sobel.size > 0:
                mean_energy = float(np.mean(patch_sobel))
                edge_score = min(1.0, mean_energy / 25.0)

        # Trọng số kết hợp
        total_score = (
            0.40 * ar_score +
            0.25 * density_score +
            0.20 * area_score +
            0.15 * edge_score
        )

        candidate_record = {
            'id': comp['id'],
            'bbox': (x, y, w, h),
            'area': comp['area'],
            'bbox_area': area,
            'aspect_ratio': aspect_ratio,
            'density': density,
            'plate_type': plate_type,
            'score': round(float(total_score), 4),
            'w_ratio': round(float(w_ratio), 4),
            'h_ratio': round(float(h_ratio), 4),
            'area_ratio': round(float(area_ratio), 6),
        }
        valid_candidates.append(candidate_record)

    # 5. Áp dụng NMS để loại trừ các khung hình trùng nhau
    filtered_candidates = apply_nms(valid_candidates, iou_threshold=iou_thresh)

    return filtered_candidates

