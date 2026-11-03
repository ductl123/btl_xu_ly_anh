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


def compute_iou_and_ios(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """
    Tính chỉ số giao trên hợp (IoU) và giao trên hộp nhỏ hơn (IoS - Intersection over Smaller).
    Mỗi box có định dạng: (x, y, w, h).
    IoS giúp phát hiện trường hợp hộp bao khổng lồ (ví dụ cả đuôi xe) chứa trọn biển số con.
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
        return 0.0, 0.0
    iou = float(inter_area) / float(union_area)
    ios = float(inter_area) / float(min(area1, area2)) if min(area1, area2) > 0 else 0.0
    return iou, ios


def merge_vertical_split_boxes(components: List[Dict[str, Any]], max_dy_ratio: float = 0.65, min_x_overlap: float = 0.45) -> List[Dict[str, Any]]:
    """
    Thuật toán Gộp Bounding Box (Bounding Box Merging):
    Tự động hàn gắn 2 contour thuộc dòng trên và dòng dưới của biển số 2 dòng
    khi chụp góc nghiêng/chéo làm đứt gãy kết nối giữa 2 dòng.
    
    Điều kiện gộp:
    - Trùng phùng theo phương ngang (Horizontal Overlap >= 45%).
    - Khoảng cách trục dọc giữa 2 hộp nhỏ hơn ngưỡng (dy <= 0.65 * chiều cao hộp).
    - Tỷ lệ khung hình sau khi gộp nằm trong khoảng biển 2 dòng hợp lệ (1.05 - 1.85).
    """
    if len(components) < 2:
        return components

    merged_list = []
    used_indices = set()

    for i in range(len(components)):
        if i in used_indices:
            continue
        comp_a = components[i]
        xa, ya, wa, ha = comp_a['bbox']
        merged_with_other = False

        # Không gộp nếu thành phần đã là một biển 2 dòng hoàn chỉnh (giữ nguyên không merge)
        if comp_a.get('aspect_ratio', 1.0) <= 1.70 and comp_a.get('bbox_area', 0) > 1000:
            merged_list.append(comp_a)
            continue

        for j in range(len(components)):
            if i == j or j in used_indices:
                continue
            comp_b = components[j]
            # Không gộp nếu thành phần B đã là một biển 2 dòng hoàn chỉnh
            if comp_b.get('aspect_ratio', 1.0) <= 1.70 and comp_b.get('bbox_area', 0) > 1000:
                continue
            xb, yb, wb, hb = comp_b['bbox']

            # Xác định hộp trên (top) và hộp dưới (bottom)
            top_comp, bot_comp = (comp_a, comp_b) if ya < yb else (comp_b, comp_a)
            xt, yt, wt, ht = top_comp['bbox']
            xbt, ybt, wbt, hbt = bot_comp['bbox']

            # Tính độ giao theo chiều ngang
            x_left = max(xt, xbt)
            x_right = min(xt + wt, xbt + wbt)
            overlap_w = max(0, x_right - x_left)
            overlap_ratio = overlap_w / float(min(wt, wbt)) if min(wt, wbt) > 0 else 0.0

            # Khoảng cách dọc từ đáy hộp trên đến đỉnh hộp dưới
            dy = ybt - (yt + ht)

            # Kiểm tra tương quan kích thước chiều rộng (không lệch quá 50%)
            w_diff = abs(wt - wbt) / float(max(wt, wbt)) if max(wt, wbt) > 0 else 1.0

            if overlap_ratio >= min_x_overlap and w_diff <= 0.50 and (-ht * 0.35 <= dy <= ht * max_dy_ratio):
                # Tạo hộp bao mới gộp cả 2
                new_x = min(xt, xbt)
                new_y = yt
                new_w = max(xt + wt, xbt + wbt) - new_x
                new_h = (ybt + hbt) - new_y
                new_ar = float(new_w) / float(new_h) if new_h > 0 else 1.0

                if 0.90 <= new_ar <= 1.95:
                    comb_area = comp_a['area'] + comp_b['area']
                    comb_bbox_area = new_w * new_h
                    comb_density = float(comb_area) / float(comb_bbox_area) if comb_bbox_area > 0 else 0.5

                    # Kế thừa góc nghiêng từ thành phần có diện tích lớn hơn
                    merged_angle = comp_a.get('angle', 0.0) if comp_a.get('area', 0) >= comp_b.get('area', 0) else comp_b.get('angle', 0.0)
                    merged_quad = comp_a.get('quad_corners') if comp_a.get('area', 0) >= comp_b.get('area', 0) else comp_b.get('quad_corners')

                    # Nếu có góc nghiêng đáng kể, tạo quad_corners cho hộp bao đã gộp
                    if abs(merged_angle) >= 1.5:
                        try:
                            import cv2
                            rect = ((new_x + new_w / 2.0, new_y + new_h / 2.0), (new_w, new_h), merged_angle)
                            box = cv2.boxPoints(rect)
                            s = box.sum(axis=1)
                            diff = np.diff(box, axis=1)
                            rect_ord = np.zeros((4, 2), dtype="float32")
                            rect_ord[0] = box[np.argmin(s)]
                            rect_ord[2] = box[np.argmax(s)]
                            rect_ord[1] = box[np.argmin(diff)]
                            rect_ord[3] = box[np.argmax(diff)]
                            merged_quad = rect_ord
                        except Exception:
                            pass

                    merged_comp = {
                        'id': comp_a['id'],
                        'bbox': (new_x, new_y, new_w, new_h),
                        'quad_corners': merged_quad,
                        'angle': merged_angle,
                        'area': comb_area,
                        'bbox_area': comb_bbox_area,
                        'density': comb_density,
                        'aspect_ratio': new_ar,
                        'is_merged': True
                    }
                    merged_list.append(merged_comp)
                    used_indices.add(i)
                    used_indices.add(j)
                    merged_with_other = True
                    break

        if not merged_with_other:
            merged_list.append(comp_a)

    return merged_list


def apply_nms(candidates: List[Dict[str, Any]], iou_threshold: float = 0.35, ios_threshold: float = 0.65) -> List[Dict[str, Any]]:
    """
    Non-Maximum Suppression (NMS) cải tiến:
    Loại bỏ các hộp bao bị chồng chéo (IoU > iou_threshold) hoặc bị bao trùm (IoS > ios_threshold),
    giữ lại hộp có điểm tin cậy (score) cao nhất.
    """
    if not candidates:
        return []

    # Sắp xếp các ứng viên giảm dần theo điểm tin cậy
    sorted_candidates = sorted(candidates, key=lambda c: c['score'], reverse=True)
    kept: List[Dict[str, Any]] = []

    for cand in sorted_candidates:
        should_keep = True
        for kept_cand in kept:
            iou, ios = compute_iou_and_ios(cand['bbox'], kept_cand['bbox'])
            if iou > iou_threshold or ios > ios_threshold:
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

    # Tự động gộp các bounding box bị xé lẻ do góc chụp nghiêng
    merged_components = merge_vertical_split_boxes(components)

    for comp in merged_components:
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
            'quad_corners': comp.get('quad_corners'),
            'angle': comp.get('angle', 0.0),
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


def refine_quad_corners(
    orig_np: np.ndarray,
    bbox: Tuple[int, int, int, int],
    scale_factor: float = 1.0,
    plate_type: str = "double_row"
) -> Tuple[Optional[np.ndarray], float]:
    """
    Tìm chính xác 4 góc nghiêng (Quad Corners) của biển số xe trên ảnh gốc độ phân giải cao:
    - Bóc tách mảng biển số màu sáng trên nền thân xe/dè xe tối màu.
    - Tìm đường viền chữ nhật nghiêng (cv2.minAreaRect) bao khít 4 cạnh biển số.
    - Sắp xếp 4 đỉnh chuẩn theo thứ tự: [Top-Left, Top-Right, Bottom-Right, Bottom-Left].
    - Tính góc xoay nghiêng thực tế của biển số.

    Args:
        orig_np: Mảng RGB/Grayscale của ảnh gốc.
        bbox: (x, y, w, h) trên ảnh xử lý (scaled).
        scale_factor: Tỷ lệ co dãn giữa ảnh xử lý và ảnh gốc.
        plate_type: 'double_row' hoặc 'single_row'.

    Returns:
        Tuple (quad_corners_on_scaled [4x2 float32], angle_degrees float).
    """
    try:
        import cv2
        H, W = orig_np.shape[:2]
        x, y, w, h = bbox

        ox = int(round(x / scale_factor))
        oy = int(round(y / scale_factor))
        ow = int(round(w / scale_factor))
        oh = int(round(h / scale_factor))

        margin_w = int(ow * 0.15)
        margin_h = int(oh * 0.15)
        rx1 = max(0, ox - margin_w)
        ry1 = max(0, oy - margin_h)
        rx2 = min(W, ox + ow + margin_w)
        ry2 = min(H, oy + oh + margin_h)

        roi = orig_np[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return None, 0.0

        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        else:
            gray = roi.copy()

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        best_quad = None
        best_angle = 0.0
        best_score = -1.0
        roi_area = (rx2 - rx1) * (ry2 - ry1)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < roi_area * 0.12 or area > roi_area * 0.98:
                continue
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (rw, rh), ang = rect
            if rw <= 0 or rh <= 0:
                continue
            ar = rw / rh if rw > rh else rh / rw
            if plate_type == 'double_row' and not (1.0 <= ar <= 1.95):
                continue
            if plate_type == 'single_row' and not (2.4 <= ar <= 6.0):
                continue

            box = cv2.boxPoints(rect)
            if area > best_score:
                best_score = area
                best_quad_orig = box + np.array([rx1, ry1], dtype='float32')

                # Sắp xếp 4 đỉnh: TL, TR, BR, BL
                s = best_quad_orig.sum(axis=1)
                diff = np.diff(best_quad_orig, axis=1)
                ordered_orig = np.zeros((4, 2), dtype='float32')
                ordered_orig[0] = best_quad_orig[np.argmin(s)]       # Top-Left
                ordered_orig[2] = best_quad_orig[np.argmax(s)]       # Bottom-Right
                ordered_orig[1] = best_quad_orig[np.argmin(diff)]    # Top-Right
                ordered_orig[3] = best_quad_orig[np.argmax(diff)]    # Bottom-Left

                best_quad = ordered_orig * scale_factor

                if rw < rh:
                    best_angle = ang - 90.0 if ang > 0 else ang + 90.0
                else:
                    best_angle = ang

        return best_quad, best_angle
    except Exception:
        return None, 0.0


