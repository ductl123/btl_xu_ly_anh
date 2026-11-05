"""
image_processing/candidate_filter.py - Lọc ứng viên biển số xe (Candidate Filtering)
Bao gồm:
- Lọc theo độ sáng nền động (Adaptive Background Brightness Verification) để loại bỏ nhựa đen, nan hoa, dè xe.
- Xác thực ký tự bên trong BBox (Character Verification & Baseline Variance) bằng Inverse Otsu & Morphology.
- Lọc theo tỷ lệ khung hình (Aspect Ratio w/h):
  + Biển 1 dòng (dài): khoảng 3.0 - 5.5 (cho phép 2.45 - 6.0).
  + Biển 2 dòng (vuông): chuẩn 1.05 - 1.85 (loại bỏ hoàn toàn vật thể hình chữ nhật đứng w/h < 1.0 như gương xe w/h=0.76 hay bô xe w/h=0.88).
- Lọc theo tỷ lệ diện tích và kích thước tương đối so với toàn bộ bức ảnh.
- Lọc theo mật độ điểm trắng (White Density / Fill Ratio).
- Tính điểm tin cậy (Multi-factor Confidence Score).
- Áp dụng Non-Maximum Suppression (NMS) để loại bỏ các hộp bao chồng chéo.
- Cho phép phát hiện một hoặc nhiều biển số trong cùng một ảnh.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from utils.image_utils import order_quad_points
from .mudguard_filter import detect_and_separate_mudguard



def is_tail_light_or_red_artifact(orig_img: np.ndarray, bbox: Tuple[int, int, int, int], scale_factor: float = 1.0) -> bool:
    """
    Kiểm tra và loại bỏ đèn hậu xe máy / chóa nhựa đỏ / đèn xi-nhan đỏ:
    Biển số xe Việt Nam (trắng, vàng, xanh) không bao giờ có màu đỏ chiếm ưu thế mạnh.
    """
    if orig_img is None or orig_img.size == 0 or len(orig_img.shape) != 3:
        return False
    try:
        H, W = orig_img.shape[:2]
        if scale_factor > 0 and scale_factor != 1.0:
            ox = int(round(bbox[0] / scale_factor))
            oy = int(round(bbox[1] / scale_factor))
            ow = int(round(bbox[2] / scale_factor))
            oh = int(round(bbox[3] / scale_factor))
        else:
            ox, oy, ow, oh = bbox

        x1 = max(0, min(ox, W - 1))
        y1 = max(0, min(oy, H - 1))
        x2 = max(0, min(ox + ow, W))
        y2 = max(0, min(oy + oh, H))
        if x2 <= x1 or y2 <= y1:
            return False

        roi = orig_img[y1:y2, x1:x2]
        if roi.size == 0:
            return False

        r_mean = float(np.mean(roi[:, :, 0]))
        g_mean = float(np.mean(roi[:, :, 1]))
        b_mean = float(np.mean(roi[:, :, 2]))

        # Nếu kênh R vượt trội so với G và B (chóa nhựa đèn hậu màu đỏ):
        if r_mean >= 75.0 and (r_mean - g_mean) > 20.0 and (r_mean - b_mean) > 20.0:
            if (r_mean - g_mean) / (r_mean + 1.0) >= 0.22:
                return True
        return False
    except Exception:
        return False

def is_valid_background_brightness(
    orig_img: np.ndarray,
    bbox: Tuple[int, int, int, int],
    min_brightness: Optional[float] = None,
    scale_factor: float = 1.0
) -> bool:
    """
    Kiểm tra độ sáng nền của vùng ROI ứng viên biển số (Background Brightness Verification):
    Trích xuất vùng ROI từ ảnh gốc, chuyển sang ảnh xám và tính giá trị sáng trung bình.
    Sử dụng Ngưỡng độ sáng nền động (Adaptive Brightness Threshold):
        min_brightness = max(60.0, Mean_image * 0.80)
    giúp chống dính nhựa đen / dè xe hiệu quả nhưng không loại nhầm biển số chụp trong hầm tối/ban đêm.

    Args:
        orig_img: Mảng numpy ảnh gốc (RGB/Grayscale/BGR).
        bbox: Hộp bao (x, y, w, h) của ứng viên.
        min_brightness: Ngưỡng độ sáng trung bình tối thiểu (nếu None sẽ tự động tính theo ngưỡng động).
        scale_factor: Tỷ lệ co dãn nếu bbox tính trên ảnh đã resize.

    Returns:
        True nếu vùng nền đạt độ sáng hợp lệ, False nếu là vùng tối/rác.
    """
    if orig_img is None or orig_img.size == 0:
        return True

    try:
        import cv2
        H, W = orig_img.shape[:2]

        # Chuyển ảnh xám nếu cần để tính độ sáng toàn cảnh
        if len(orig_img.shape) == 3:
            full_gray = cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)
        else:
            full_gray = orig_img

        # Tính ngưỡng độ sáng nền động nếu chưa được chỉ định
        if min_brightness is None:
            mean_img = float(np.mean(full_gray))
            min_brightness = max(60.0, mean_img * 0.80)

        if scale_factor > 0 and scale_factor != 1.0:
            ox = int(round(bbox[0] / scale_factor))
            oy = int(round(bbox[1] / scale_factor))
            ow = int(round(bbox[2] / scale_factor))
            oh = int(round(bbox[3] / scale_factor))
        else:
            ox, oy, ow, oh = bbox

        x1 = max(0, min(ox, W - 1))
        y1 = max(0, min(oy, H - 1))
        x2 = max(0, min(ox + ow, W))
        y2 = max(0, min(oy + oh, H))

        if x2 <= x1 or y2 <= y1:
            return False

        roi_gray = full_gray[y1:y2, x1:x2]
        if roi_gray.size == 0:
            return False

        mean_val = float(np.mean(roi_gray))
        p75 = float(np.percentile(roi_gray, 75))

        # Đạt yêu cầu nếu độ sáng trung bình >= min_brightness, hoặc p75 >= min_brightness kết hợp mean >= 55.0
        return (mean_val >= min_brightness) or (p75 >= min_brightness and mean_val >= 55.0)
    except Exception:
        return True


def verify_characters_inside(
    orig_img: np.ndarray,
    bbox: Tuple[int, int, int, int],
    min_chars: int = 4,
    scale_factor: float = 1.0,
    angle: float = 0.0
) -> bool:
    """
    Xác thực ký tự bên trong Bounding Box (Character Verification & Baseline Variance):
    1. Trích xuất ROI từ ảnh gốc, chuẩn hóa tương phản và nhị phân hóa bằng Inverse Otsu.
    2. Dùng Morphology Open (kernel 2x2) để khử nhiễu đốm nhỏ.
    3. Dùng cv2.findContours tìm các contour bên trong.
    4. Đếm số lượng contour con thỏa mãn:
       - Chiều cao h_char = 0.18 - 0.62 h_ROI (với biển 1 dòng cho phép đến 0.85).
       - Tỷ lệ w_char / h_char nằm trong khoảng 0.16 đến 0.85.
    5. Đánh giá Độ lệch chuẩn đường cơ sở ký tự (Baseline Variance):
       - Tâm Y của các ký tự con phải nằm trên đường ngang song song (Std Dev_Y < 0.14 * h_ROI).
       - Với biển 2 dòng: đánh giá theo từng dòng (trên và dưới).

    Args:
        orig_img: Mảng numpy ảnh gốc (RGB/Grayscale/BGR).
        bbox: Hộp bao (x, y, w, h) của ứng viên.
        min_chars: Số lượng ký tự tối thiểu cần đạt (mặc định 4).
        scale_factor: Tỷ lệ co dãn nếu bbox tính trên ảnh đã resize.
        angle: Góc nghiêng của ứng viên (để tự xoay ROI khi kiểm tra biển nghiêng).

    Returns:
        True nếu vùng ROI chứa đủ ký tự và nằm trên đường ngang thẳng hàng, False nếu là rác.
    """
    if orig_img is None or orig_img.size == 0:
        return True

    try:
        import cv2
        H, W = orig_img.shape[:2]
        if scale_factor > 0 and scale_factor != 1.0:
            ox = int(round(bbox[0] / scale_factor))
            oy = int(round(bbox[1] / scale_factor))
            ow = int(round(bbox[2] / scale_factor))
            oh = int(round(bbox[3] / scale_factor))
        else:
            ox, oy, ow, oh = bbox

        x1 = max(0, min(ox, W - 1))
        y1 = max(0, min(oy, H - 1))
        x2 = max(0, min(ox + ow, W))
        y2 = max(0, min(oy + oh, H))

        if x2 <= x1 or y2 <= y1:
            return False

        roi = orig_img[y1:y2, x1:x2]
        if roi.size == 0:
            return False

        if len(roi.shape) == 3:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        else:
            roi_gray = roi

        roi_h, roi_w = roi_gray.shape[:2]

        # Chuẩn hóa độ phân giải tối thiểu (nếu chiều cao < 60px) để Morphology Open (2x2) ổn định
        if roi_h < 60:
            scale_up = 60.0 / float(roi_h)
            target_w = max(10, int(round(roi_w * scale_up)))
            roi_gray = cv2.resize(roi_gray, (target_w, 60), interpolation=cv2.INTER_LINEAR)
            roi_h, roi_w = roi_gray.shape[:2]

        # Danh sách các góc xoay thử nghiệm để nắn thẳng phẳng ROI (tránh bỏ sót biển số xoay nghiêng/dọc)
        norm_angle = angle
        while norm_angle < -45.0:
            norm_angle += 90.0
        while norm_angle > 45.0:
            norm_angle -= 90.0

        candidate_angles = [angle]
        if abs(norm_angle - angle) > 1e-2:
            candidate_angles.append(norm_angle)
        if 0.0 not in candidate_angles:
            candidate_angles.append(0.0)

        # Morphology kernel 2x2
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

        for test_ang in candidate_angles:
            curr_gray = roi_gray.copy()
            if abs(test_ang) >= 2.0:
                M_rot = cv2.getRotationMatrix2D((roi_w / 2.0, roi_h / 2.0), test_ang, 1.0)
                curr_gray = cv2.warpAffine(curr_gray, M_rot, (roi_w, roi_h), borderMode=cv2.BORDER_REPLICATE)

            # Chuẩn hóa độ tương phản thích nghi (giúp Inverse Otsu tách nét ký tự rõ kể cả khi tối)
            p_low, p_high = np.percentile(curr_gray, (5, 95))
            if p_high > p_low:
                norm_gray = np.clip((curr_gray.astype(np.float32) - p_low) * 255.0 / (p_high - p_low), 0, 255).astype(np.uint8)
            else:
                norm_gray = curr_gray

            # Nhị phân hóa Inverse Otsu (ký tự đen trên nền trắng biến thành màu trắng 255)
            _, bin_roi = cv2.threshold(norm_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            opened = cv2.morphologyEx(bin_roi, cv2.MORPH_OPEN, kernel)

            # Tìm tất cả contours con bên trong ROI
            cnts, _ = cv2.findContours(opened, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            is_single_row = (float(roi_w) / float(roi_h) >= 2.0)
            max_h_ratio = 0.85 if is_single_row else 0.65
            min_h_ratio = 0.15

            valid_chars = []
            for cnt in cnts:
                cx, cy, cw, ch = cv2.boundingRect(cnt)
                if ch < 8 or cw < 3:
                    continue

                h_ratio = float(ch) / float(roi_h)
                char_ar = float(cw) / float(ch) if ch > 0 else 0.0

                if (min_h_ratio <= h_ratio <= max_h_ratio) and (0.15 <= char_ar <= 0.90):
                    center_y = cy + ch / 2.0
                    valid_chars.append((cx, cy, cw, ch, center_y))

            if len(valid_chars) >= min_chars:
                cys = [c[4] for c in valid_chars]
                threshold_std = 0.15 * roi_h

                if np.std(cys) < threshold_std:
                    return True

                mid_y = roi_h * 0.50
                row1_cys = [y for y in cys if y < mid_y]
                row2_cys = [y for y in cys if y >= mid_y]
                std1 = np.std(row1_cys) if len(row1_cys) > 1 else 0.0
                std2 = np.std(row2_cys) if len(row2_cys) > 1 else 0.0

                if (len(row1_cys) >= 2 and len(row2_cys) >= 2) and (std1 <= threshold_std * 1.5) and (std2 <= threshold_std * 1.5):
                    return True

        return False
    except Exception:
        # P15: Tránh nuốt lỗi im lặng và không đánh đồng lỗi với việc đã xác thực đạt
        return False


def compute_iou_and_ios(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """
    Tính chỉ số giao trên hợp (IoU) và giao trên hộp nhỏ hơn (IoS - Intersection over Smaller).
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
        return 0.0, 0.0
    iou = float(inter_area) / float(union_area)
    ios = float(inter_area) / float(min(area1, area2)) if min(area1, area2) > 0 else 0.0
    return iou, ios


def merge_vertical_split_boxes(components: List[Dict[str, Any]], max_dy_ratio: float = 0.85, min_x_overlap: float = 0.20) -> List[Dict[str, Any]]:
    """
    Thuật toán Gộp Bounding Box (Bounding Box Merging):
    Tự động hàn gắn 2 contour thuộc dòng trên và dòng dưới của biển số 2 dòng
    khi chụp góc nghiêng/chéo làm đứt gãy kết nối giữa 2 dòng.
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

        # Chỉ bỏ qua nếu thành phần đã là một biển 2 dòng hoàn chỉnh có chiều cao lớn (ha >= 110)
        if ha >= 110 and comp_a.get('aspect_ratio', 1.0) <= 1.40:
            merged_list.append(comp_a)
            continue

        for j in range(len(components)):
            if i == j or j in used_indices:
                continue
            comp_b = components[j]
            xb, yb, wb, hb = comp_b['bbox']
            if hb >= 110 and comp_b.get('aspect_ratio', 1.0) <= 1.40:
                continue

            # Xác định hộp trên (top) và hộp dưới (bottom)
            top_comp, bot_comp = (comp_a, comp_b) if ya < yb else (comp_b, comp_a)
            xt, yt, wt, ht = top_comp['bbox']
            xbt, ybt, wbt, hbt = bot_comp['bbox']

            # Cả hai thành phần phải là dạng dòng chữ ngang (ar >= 0.80) và không được vượt quá giới hạn cận cảnh (ht <= 260, hbt <= 260)
            if ht > 260 or hbt > 260:
                continue
            if (wt / float(ht) < 0.75) or (wbt / float(hbt) < 0.75):
                continue

            # Khoảng cách dọc từ đáy hộp trên đến đỉnh hộp dưới (cho phép thích nghi với biển cận cảnh)
            dy = ybt - (yt + ht)
            max_allowed_dy = max(60, int(ht * 1.30))
            if not (-15 <= dy <= max_allowed_dy):
                continue

            # Tính độ giao theo chiều ngang
            x_left = max(xt, xbt)
            x_right = min(xt + wt, xbt + wbt)
            overlap_w = max(0, x_right - x_left)
            min_w = min(wt, wbt)
            max_w = max(wt, wbt)
            overlap_ratio = overlap_w / float(min_w) if min_w > 0 else 0.0

            # Kiểm tra tương quan kích thước chiều rộng (không lệch quá 65%)
            w_diff = abs(wt - wbt) / float(max_w) if max_w > 0 else 1.0

            # Tâm ngang của 2 hàng phải tương đối thẳng hàng
            cx_t = xt + wt / 2.0
            cx_b = xbt + wbt / 2.0
            if abs(cx_t - cx_b) > 0.70 * max_w:
                continue

            if (overlap_ratio >= min_x_overlap or overlap_w >= 15) and w_diff <= 0.65:
                # Tạo hộp bao mới gộp cả 2
                new_x = min(xt, xbt)
                new_y = yt
                new_w = max(xt + wt, xbt + wbt) - new_x
                new_h = (ybt + hbt) - new_y
                new_ar = float(new_w) / float(new_h) if new_h > 0 else 1.0

                if 0.98 <= new_ar <= 2.25:
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
                            merged_quad = order_quad_points(box)
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


def merge_horizontal_split_boxes(components: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Thuật toán Gộp Bounding Box theo chiều ngang (Horizontal Split Merging):
    Tự động hàn gắn 2 contour thuộc nửa trái và nửa phải của cùng một biển số
    khi ký tự có khoảng cách rộng hoặc bị đứt đoạn ở giữa (ví dụ: khoảng cách giữa mã tỉnh và số xe).
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

        for j in range(len(components)):
            if i == j or j in used_indices:
                continue
            comp_b = components[j]
            xb, yb, wb, hb = comp_b['bbox']

            # Xác định hộp bên trái và hộp bên phải
            left_comp, right_comp = (comp_a, comp_b) if xa < xb else (comp_b, comp_a)
            xl, yl, wl, hl = left_comp['bbox']
            xr, yr, wr, hr = right_comp['bbox']

            # Chiều cao tương đương (không lệch quá 45%)
            max_h = max(hl, hr)
            min_h = min(hl, hr)
            if (max_h - min_h) / float(max_h) > 0.45:
                continue

            # Độ giao theo chiều dọc (phải cùng nằm trên cùng một độ cao)
            y_overlap = min(yl + hl, yr + hr) - max(yl, yr)
            if y_overlap < 0.50 * min_h:
                continue

            # Khoảng cách ngang giữa mép phải hộp trái và mép trái hộp phải
            dx = xr - (xl + wl)
            max_allowed_dx = max(45, int(0.70 * max_h))
            if not (-15 <= dx <= max_allowed_dx):
                continue

            new_x = xl
            new_y = min(yl, yr)
            new_w = (xr + wr) - new_x
            new_h = max(yl + hl, yr + hr) - new_y
            new_ar = float(new_w) / float(new_h) if new_h > 0 else 1.0

            if 1.00 <= new_ar <= 5.80:
                comb_area = comp_a['area'] + comp_b['area']
                comb_bbox_area = new_w * new_h
                comb_density = float(comb_area) / float(comb_bbox_area) if comb_bbox_area > 0 else 0.5

                merged_angle = comp_a.get('angle', 0.0) if comp_a.get('area', 0) >= comp_b.get('area', 0) else comp_b.get('angle', 0.0)
                merged_quad = comp_a.get('quad_corners') if comp_a.get('area', 0) >= comp_b.get('area', 0) else comp_b.get('quad_corners')

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
    sobel_np: Optional[np.ndarray] = None,
    orig_img_np: Optional[np.ndarray] = None,
    scale_factor: float = 1.0
) -> List[Dict[str, Any]]:
    """
    Lọc các vùng liên thông để chọn ra những ứng viên phù hợp với biển số xe Việt Nam.

    Args:
        components: Danh sách các thành phần liên thông từ bước trước.
        image_shape: (H, W) của ảnh xử lý.
        config_params: Dictionary chứa các ngưỡng từ config.py.
        sobel_np: Ma trận gradient Sobel-X để tính mật độ năng lượng cạnh.
        orig_img_np: Mảng numpy ảnh gốc (tuỳ chọn) để kiểm tra độ sáng nền và xác thực ký tự.
        scale_factor: Tỷ lệ thu phóng giữa ảnh xử lý và ảnh gốc.

    Returns:
        Danh sách ứng viên biển số đạt tất cả tiêu chí lọc, kèm phân loại và điểm tin cậy.
    """
    img_h, img_w = image_shape
    total_img_area = img_h * img_w

    # Đọc tham số lọc từ cấu hình
    single_min_ar = config_params.get("SINGLE_ROW_ASPECT_RATIO_MIN", 2.45)
    single_max_ar = config_params.get("SINGLE_ROW_ASPECT_RATIO_MAX", 6.0)
    double_min_ar = config_params.get("DOUBLE_ROW_ASPECT_RATIO_MIN", 1.05)
    double_max_ar = config_params.get("DOUBLE_ROW_ASPECT_RATIO_MAX", 1.85)

    min_area_ratio = config_params.get("MIN_AREA_RATIO", 0.0005)
    max_area_ratio = config_params.get("MAX_AREA_RATIO", 0.22)
    min_w_ratio = config_params.get("MIN_WIDTH_RATIO", 0.030)
    max_w_ratio = config_params.get("MAX_WIDTH_RATIO", 0.85)
    min_h_ratio = config_params.get("MIN_HEIGHT_RATIO", 0.015)
    max_h_ratio = config_params.get("MAX_HEIGHT_RATIO", 0.45)

    min_density = config_params.get("MIN_WHITE_DENSITY", 0.12)
    max_density = config_params.get("MAX_WHITE_DENSITY", 0.95)
    iou_thresh = config_params.get("NMS_IOU_THRESHOLD", 0.35)

    # Tính ngưỡng độ sáng nền thích nghi từ ảnh gốc
    dynamic_min_brightness = 75.0
    if orig_img_np is not None and orig_img_np.size > 0:
        try:
            import cv2
            if len(orig_img_np.shape) == 3:
                full_gray = cv2.cvtColor(orig_img_np, cv2.COLOR_RGB2GRAY)
            else:
                full_gray = orig_img_np
            mean_img = float(np.mean(full_gray))
            dynamic_min_brightness = max(60.0, mean_img * 0.80)
        except Exception:
            dynamic_min_brightness = 75.0

    valid_candidates: List[Dict[str, Any]] = []

    # Tự động gộp các bounding box bị xé lẻ theo cả chiều dọc và chiều ngang
    merged_components = merge_horizontal_split_boxes(merge_vertical_split_boxes(components))

    for comp in merged_components:
        x, y, w, h = comp['bbox']
        aspect_ratio = comp['aspect_ratio']
        density = comp['density']
        area = comp['bbox_area']

        # TÁCH DÈ XE / BÔ XE (MUDGUARD SEPARATION):
        # Nếu tỷ lệ w/h < 1.15 nhưng chiều cao lớn (h >= 30), nghi vấn bị dính dè nhựa đen/chữ phấn ở đáy
        if aspect_ratio < 1.15 and h >= 30 and orig_img_np is not None:
            sep_bbox, was_sep = detect_and_separate_mudguard(
                orig_img_np, (x, y, w, h), scale_factor=scale_factor, min_ar_trigger=1.15
            )
            if was_sep:
                x, y, w, h = sep_bbox
                aspect_ratio = float(w) / float(max(1, h))
                area = w * h
                comp['bbox'] = sep_bbox
                comp['aspect_ratio'] = aspect_ratio
                comp['bbox_area'] = area
                comp['mudguard_separated'] = True

        # 0. Loại bỏ biên ảnh cực đoan chỉ khi là dải nhiễu quá mỏng sát mép (h < 25 hoặc w < 25)
        if (y <= 5 or (y + h) >= img_h - 5) and (h < 25 or w < 25):
            continue

        # 1. Kiểm tra kích thước tương đối so với toàn ảnh (hỗ trợ cả biển xoay ngang lẫn xoay dọc)
        area_ratio = area / float(total_img_area)
        w_ratio = w / float(img_w)
        h_ratio = h / float(img_h)

        max_dim = max(w, h)
        min_dim = min(w, h)
        max_dim_ratio = max_dim / float(max(img_w, img_h))
        min_dim_ratio = min_dim / float(min(img_w, img_h))

        if not (min_area_ratio <= area_ratio <= max_area_ratio):
            continue
        if max_dim_ratio < (min_w_ratio * 0.70) or max_dim_ratio > (max_w_ratio * 1.15):
            continue
        if min_dim_ratio < (min_h_ratio * 0.70) or min_dim_ratio > (max_h_ratio * 1.30):
            continue

        # 2. Kiểm tra mật độ điểm trắng (sau khi closing thành mảng đặc)
        if not (min_density <= density <= max_density):
            continue

        # 3. Phân loại theo tỷ lệ hình học biển số:
        plate_type = None
        ideal_ar = None

        # [P05]: Biển nghiêng làm BBox bị nở to, nên ưu tiên OBB AR khi có góc nghiêng lớn
        cand_angle = abs(comp.get('angle', 0.0))
        is_tilted = (cand_angle >= 12.0 and cand_angle <= 78.0)
        obb_ar = comp.get('obb_aspect_ratio', aspect_ratio)

        if is_tilted:
            if 2.45 <= obb_ar <= 6.0:
                plate_type = "single_row"
                ideal_ar = 4.7
            elif 0.95 <= obb_ar <= 2.25:
                plate_type = "double_row"
                ideal_ar = 1.4

        if plate_type is None:
            rounded_ar = round(aspect_ratio, 2)
            if single_min_ar <= aspect_ratio <= single_max_ar:
                plate_type = "single_row"
                ideal_ar = 4.7
            elif (double_min_ar - 0.02) <= rounded_ar <= (double_max_ar + 0.02):
                plate_type = "double_row"
                ideal_ar = 1.4
            elif 0.98 <= obb_ar <= 2.25:
                plate_type = "double_row"
                ideal_ar = 1.4
            elif 2.45 <= obb_ar <= 6.0:
                plate_type = "single_row"
                ideal_ar = 4.7
            elif aspect_ratio < (double_min_ar - 0.01) and orig_img_np is not None:
                # Nếu hộp bao dính dè xe/lốp xe phía dưới khiến w/h < 1.05:
                # Thử trích xuất đường viền chữ nhật sáng màu bên trong
                try:
                    cand_quad, cand_ang = refine_quad_corners(orig_img_np, (x, y, w, h), scale_factor=scale_factor, plate_type="double_row")
                    if cand_quad is not None:
                        # Tính AR thực tế của quad
                        q_tl, q_tr, q_br, q_bl = cand_quad
                        qw = max(np.linalg.norm(q_tr - q_tl), np.linalg.norm(q_br - q_bl))
                        qh = max(np.linalg.norm(q_tl - q_bl), np.linalg.norm(q_tr - q_br))
                        quad_ar = qw / qh if qh > 0 else 0.0
                        if 0.98 <= round(quad_ar, 2) <= 2.25:
                            plate_type = "double_row"
                            ideal_ar = 1.4
                            comp['quad_corners'] = cand_quad
                            comp['angle'] = cand_ang
                except Exception:
                    pass

        if plate_type is None:
            continue

        # Kiểm tra độ sáng nền và xác thực ký tự bên trong Bounding Box để loại bỏ False Positives
        if orig_img_np is not None:
            curr_scale = scale_factor
            if curr_scale == 1.0 and orig_img_np.shape[1] > int(img_w * 1.15):
                curr_scale = float(img_w) / float(orig_img_np.shape[1])

            # Loại bỏ ngay đèn hậu xe máy màu đỏ / chóa phản quang đỏ
            if is_tail_light_or_red_artifact(orig_img_np, (x, y, w, h), scale_factor=curr_scale):
                continue

            if not is_valid_background_brightness(orig_img_np, (x, y, w, h), min_brightness=dynamic_min_brightness, scale_factor=curr_scale):
                continue
            cand_angle = comp.get('angle', 0.0)
            is_near_border = (x <= 30 or y <= 30 or (x + w) >= (img_w - 30) or (y + h) >= (img_h - 30))
            is_special_cand = comp.get('mudguard_separated', False) or comp.get('is_merged', False)
            required_min_chars = 2 if (is_near_border or is_special_cand) else 3
            if not verify_characters_inside(orig_img_np, (x, y, w, h), min_chars=required_min_chars, scale_factor=curr_scale, angle=cand_angle):
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


def refine_quad_corners(orig_np, bbox, scale_factor=1.0, plate_type="double_row"):
    """Find an actual plate boundary, not the rotated box of the whole vehicle.

    Return None when no convex quadrilateral with plausible ink is available.
    Coordinates are returned in the detection image coordinate system.
    """
    import cv2
    if scale_factor <= 0:
        raise ValueError("scale_factor must be positive")
    H, W = orig_np.shape[:2]
    x, y, w, h = np.asarray(bbox, dtype=float) / scale_factor
    x0, y0 = max(0, int(x - w*.12)), max(0, int(y - h*.12))
    x1, y1 = min(W, int(x+w*1.12)), min(H, int(y+h*1.12))
    roi = orig_np[y0:y1, x0:x1]
    if roi.size == 0:
        return None, 0.0
    # Bound cost while keeping original-image coordinates for the final warp.
    r = min(1.0, 1000.0 / max(roi.shape[:2]))
    if r < 1:
        roi = cv2.resize(roi, None, fx=r, fy=r, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) if roi.ndim == 3 else roi
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    kernel = np.ones((3,3), np.uint8)
    masks = [cv2.morphologyEx(cv2.Canny(gray, lo, hi), cv2.MORPH_CLOSE,
                              kernel, iterations=2) for lo,hi in [(20,60),(40,120),(70,180)]]
    _, otsu = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    masks.append(otsu)
    masks.extend(cv2.threshold(gray,t,255,cv2.THRESH_BINARY)[1]
                 for t in [60,80,100,120,140,160,180])
    best, best_score = None, -1.0
    rh, rw = gray.shape
    for mask in masks:
        for contour in cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]:
            area = cv2.contourArea(contour)
            if not .035*rw*rh < area < .88*rw*rh:
                continue
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 0 or area/hull_area < .80:
                continue
            for eps in [.015,.025,.04]:
                poly=cv2.approxPolyDP(hull,eps*cv2.arcLength(hull,True),True)
                if len(poly)!=4 or not cv2.isContourConvex(poly):
                    continue
                q=order_quad_points(poly.reshape(4,2))
                # Reject wedges that include bumper/background corners. Those
                # can contain text but destroy glyphs under a projective warp.
                prev=np.roll(q,1,axis=0)-q
                nxt=np.roll(q,-1,axis=0)-q
                cosines=(prev*nxt).sum(axis=1)/np.maximum(np.linalg.norm(prev,axis=1)*np.linalg.norm(nxt,axis=1),1e-6)
                corner_angles=np.degrees(np.arccos(np.clip(cosines,-1,1)))
                if corner_angles.min()<45 or corner_angles.max()>135:
                    continue
                qw=max(np.linalg.norm(q[1]-q[0]),np.linalg.norm(q[2]-q[3]))
                qh=max(np.linalg.norm(q[3]-q[0]),np.linalg.norm(q[2]-q[1]))
                if qh<20:
                    continue
                ar=qw/qh
                amin,amax,target=(2.35,6.,4.7) if plate_type=='single_row' else (1.,2.30,1.38)
                if not amin <= ar <= amax:
                    continue
                th,tw=120,max(20,int(120*ar))
                dst=np.float32([[0,0],[tw-1,0],[tw-1,th-1],[0,th-1]])
                flat=cv2.warpPerspective(gray,cv2.getPerspectiveTransform(q,dst),(tw,th))
                inner=flat[8:-8,8:-8]
                _, ink=cv2.threshold(inner,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                density=np.mean(ink>0)
                if not .06<density<.55:
                    continue
                n,_,stats,centers=cv2.connectedComponentsWithStats(ink,8)
                chars=[]
                for st,ct in zip(stats[1:],centers[1:]):
                    cx,cy,cw,ch,ca=st
                    if .15*inner.shape[0]<ch<.92*inner.shape[0] and .06<cw/ch<1.2 and ca>15:
                        chars.append(ct)
                if len(chars)<3 or np.ptp(np.array(chars)[:,0])<inner.shape[1]*.35:
                    continue
                ar_score=max(0.,1.-abs(ar-target)/target)
                score=.35*ar_score+.35*min(len(chars)/8.,1.)+.15*area/hull_area+.15*min(area/(rw*rh*.4),1.)
                if score>best_score:
                    best_score=score
                    best=q/r+np.float32([x0,y0])
    if best is None:
        return None,0.0
    edge=best[1]-best[0]
    angle=float(np.degrees(np.arctan2(edge[1],edge[0])))
    return (best*scale_factor).astype(np.float32),angle
