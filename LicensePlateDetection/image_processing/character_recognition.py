"""
image_processing/character_recognition.py - Nhận dạng ký tự biển số xe (Optical Character Recognition - OCR)
Phương pháp thuần xử lý ảnh số kinh điển:
1. Phân đoạn ký tự (Character Segmentation):
   - Tính ngưỡng Otsu tự động trên bề mặt lòng biển số (loại trừ khung viền xe).
   - Tách các nét ký tự tối màu trên nền sáng: char_bin = (gray < t).
   - Khử nhiễu biên mép ngoài (khung viền kim loại, ốc vít).
   - Trích xuất ký tự bằng Connected Component Labeling và lọc theo tỷ lệ hình học.
   - Sắp xếp thứ tự đọc (biển 1 dòng xếp trái->phải; biển 2 dòng tách dòng trên và dòng dưới).
2. Khớp mẫu tương quan đa phông (Multi-Font Tight Template Matching) thuần NumPy:
   - Cắt khung bao sát (tight crop) từng ký tự và chuẩn hóa về kích thước 20x32.
   - Kho mẫu đa phông chữ giao thông (Bahnschrift DIN 1451, Arial Bold, Segoe UI).
   - Kết hợp chỉ số IoU, tương quan chuẩn hóa và phân biệt cấu trúc hình học (lỗ rỗng, thanh ngang, thanh đứng).
"""

from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .preprocessing import rgb_to_grayscale, unsharp_mask, gamma_correction, enhance_contrast_dynamic
from .thresholding import otsu_threshold, adaptive_local_threshold
from .morphology import bridge_broken_characters, black_hat_transform, create_structuring_element
from .connected_components import find_connected_components


def crop_tight(binary_array: np.ndarray) -> np.ndarray:
    """
    Cắt sát viền thật của ký tự (loại bỏ toàn bộ lề trống xung quanh).
    """
    rows = np.any(binary_array, axis=1)
    cols = np.any(binary_array, axis=0)
    if not np.any(rows) or not np.any(cols):
        return binary_array
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return binary_array[rmin : rmax + 1, cmin : cmax + 1]


def normalize_char_patch(arr: np.ndarray, target_size=(20, 32)) -> np.ndarray:
    """
    Chuẩn hóa ảnh nhị phân ký tự về kích thước chuẩn (20, 32) sau khi đã cắt sát biên.
    """
    tight = crop_tight(arr > 0)
    if tight.size == 0 or tight.shape[0] < 2 or tight.shape[1] < 2:
        return np.zeros((target_size[1], target_size[0]), dtype=bool)

    im = Image.fromarray((tight.astype(np.uint8) * 255), mode="L")
    resized = im.resize(target_size, Image.Resampling.BILINEAR)
    return (np.array(resized) > 100)


class TemplateMatcher:
    """
    Quản lý kho mẫu chữ số và chữ cái đa phông, kết hợp giải thuật đối chứng cấu trúc.
    """
    TEMPLATE_W = 20
    TEMPLATE_H = 32
    CHARS = "0123456789ABCDEFGHKLMNPSTUVXYZ"

    def __init__(self):
        self.templates: Dict[str, List[np.ndarray]] = {ch: [] for ch in self.CHARS}
        self._init_templates()

    def _init_templates(self):
        """
        Sinh mẫu nhị phân từ các phông chữ chuẩn biển số giao thông.
        """
        font_paths = [
            "C:/Windows/Fonts/bahnschrift.ttf",  # DIN 1451 chuẩn biển số quốc tế
            "C:/Windows/Fonts/arialbd.ttf",     # Arial Bold
            "C:/Windows/Fonts/consola.ttf",     # Consolas
        ]
        loaded_fonts = []
        for fp in font_paths:
            if Path(fp).is_file():
                try:
                    loaded_fonts.append(ImageFont.truetype(fp, 42))
                except Exception:
                    continue

        if not loaded_fonts:
            loaded_fonts.append(ImageFont.load_default())

        for font in loaded_fonts:
            for ch in self.CHARS:
                im = Image.new("L", (50, 60), 0)
                draw = ImageDraw.Draw(im)
                draw.text((5, 5), ch, fill=255, font=font)
                arr = (np.array(im) > 128)
                norm = normalize_char_patch(arr, (self.TEMPLATE_W, self.TEMPLATE_H))
                self.templates[ch].append(norm)

    def match(self, patch: np.ndarray, is_digit_only: bool = False, is_letter_only: bool = False) -> Tuple[str, float]:
        """
        So khớp một mảng nhị phân ký tự với kho mẫu và tinh chỉnh bằng đặc trưng cấu trúc.
        """
        p_norm = normalize_char_patch(patch, (self.TEMPLATE_W, self.TEMPLATE_H))
        scores: Dict[str, float] = {}

        for ch, tmpl_list in self.templates.items():
            if is_digit_only and not ch.isdigit():
                continue
            if is_letter_only and ch.isdigit():
                continue

            max_sim = 0.0
            for tmpl in tmpl_list:
                intersection = np.sum(p_norm & tmpl)
                union = np.sum(p_norm | tmpl)
                iou = intersection / float(union) if union > 0 else 0.0

                # Normalized Cross-Correlation
                denom = np.sqrt(np.sum(p_norm.astype(float)) * np.sum(tmpl.astype(float)))
                ncc = intersection / denom if denom > 0 else 0.0

                sim = 0.65 * iou + 0.35 * ncc
                if sim > max_sim:
                    max_sim = sim

            scores[ch] = max_sim

        if not scores:
            return "?", 0.0

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_ch = ranked[0][0]
        best_score = ranked[0][1]

        # -------------------------------------------------------------
        # QUY TẮC TINH CHỈNH ĐẶC TRƯNG HÌNH HỌC (STRUCTURAL DISAMBIGUATION)
        # -------------------------------------------------------------
        mid_center_val = p_norm[14:18, 7:13].mean()  # Tâm ký tự

        # Quy tắc 1: Nhận diện chữ số 4 (thanh ngang đậm và góc trên bên phải rỗng)
        if is_digit_only and (best_ch in ['4', '0', '5', '6', '8', '9']):
            crossbar = max([p_norm[r, :].sum() for r in range(16, 28)])
            top_right_gap = p_norm[:14, 12:].mean()
            if crossbar >= 14 and top_right_gap < 0.15:
                best_ch = '4'

        # Quy tắc 2: Phân biệt 1 vs 7 và 1 vs 3 (số 1 luôn mảnh ar <= 0.42, số 7 có thanh ngang trên, số 3 có thanh giữa)
        if is_digit_only and best_ch == '1':
            h_p, w_p = patch.shape
            ar_p = w_p / float(h_p) if h_p > 0 else 0.5
            if ar_p <= 0.42:
                best_ch = '1'
            else:
                top_bar = p_norm[:4, :].sum(axis=1).mean()
                mid_bar = p_norm[14:18, 5:15].mean()
                if (ar_p >= 0.48 and top_bar >= 15.5) and scores.get('7', 0.0) >= 0.60:
                    best_ch = '7'
                elif ar_p >= 0.48 and mid_bar > 0.65 and scores.get('3', 0.0) >= 0.60:
                    best_ch = '3'

        # Quy tắc 3: Phân biệt 0 và 8 (0 có lỗ rỗng ở giữa, 8 có nét gạch ngang ở giữa)
        # Chỉ áp dụng cho 0 và 8, tuyệt đối không áp dụng cho 9 hay 6
        if is_digit_only and best_ch in ['0', '8']:
            if mid_center_val < 0.55:
                best_ch = '0'
            elif mid_center_val >= 0.65 and ranked[0][0] == '8':
                best_ch = '8'

        # Quy tắc 4: Phân biệt D và B
        if is_letter_only and best_ch in ['B', 'D']:
            right_mid_edge = p_norm[14:18, 16:].mean()
            if right_mid_edge >= 0.55:
                best_ch = 'D'
            elif mid_center_val >= 0.65 and right_mid_edge < 0.35:
                best_ch = 'B'

        # Quy tắc 5: Chữ M (2 cột trụ thẳng đứng 2 bên cạnh phân biệt với chữ X, N)
        if is_letter_only and best_ch in ['X', 'M', 'N', 'H']:
            left_col_solid = (p_norm[6:26, :5].mean() > 0.45)
            right_col_solid = (p_norm[6:26, -5:].mean() > 0.45)
            if left_col_solid and right_col_solid:
                if best_ch in ['X', 'H']:
                    best_ch = 'M'
                elif best_ch == 'N':
                    left_leg = p_norm[20:31, 2:6].mean()
                    right_leg = p_norm[20:31, 14:18].mean()
                    bot_mid = p_norm[24:31, 8:12].mean()
                    if left_leg > 0.70 and right_leg > 0.70 and bot_mid < 0.35 and scores.get('M', 0.0) >= 0.48:
                        best_ch = 'M'

        # Quy tắc 6: Phân biệt 7, 2, 3
        if is_digit_only and best_ch in ['7', '2', '3', '8', 'Z'] and best_ch != '4':
            top_bar = p_norm[:4, :].sum(axis=1).mean()
            bot_bar = p_norm[-4:, :].sum(axis=1).mean()
            left_indent = (p_norm[8:14, :5].mean() < 0.20 and p_norm[20:26, :5].mean() < 0.20)
            mid_bar = p_norm[14:18, 5:14].mean() > 0.50

            if ranked[0][0] == "3" and left_indent:
                best_ch = "3"
            elif top_bar > 15.0 and bot_bar < 8.0:
                best_ch = '7'
            elif ranked[0][0] == '2':
                best_ch = '2'
            elif left_indent and mid_bar:
                best_ch = '3'
            elif bot_bar > 14 and not left_indent and best_ch in ['2', '7']:
                best_ch = '2'

        return best_ch, round(float(best_score), 2)


_GLOBAL_MATCHER = None

def get_template_matcher() -> TemplateMatcher:
    global _GLOBAL_MATCHER
    if _GLOBAL_MATCHER is None:
        _GLOBAL_MATCHER = TemplateMatcher()
    return _GLOBAL_MATCHER


def binarize_cropped_plate(cropped_pil: Image.Image) -> np.ndarray:
    """
    Tiền xử lý và nhị phân hóa ảnh biển số xe kết hợp các kỹ thuật xử lý ảnh:
    - Kích thước quá nhỏ / Resolution thấp: Tự động nội suy Bicubic và Unsharp Masking.
    - Thiếu sáng / Tối: Kéo giãn phân vị động và Gamma correction.
    - Loá sáng / Phản xạ: Cắt đỉnh chói và phân ngưỡng thích nghi cục bộ Sauvola.
    - Dính bẩn / Che khuất: Hàn gắn nét ký tự bằng Morphology Closing & Black-Hat.
    """
    # 1. Xử lý ảnh kích thước quá nhỏ / Resolution thấp
    cw, ch = cropped_pil.size
    if ch < 120:
        scale_up = 120.0 / float(ch)
        new_cw = max(1, int(round(cw * scale_up)))
        cropped_pil = cropped_pil.resize((new_cw, 120), Image.Resampling.BICUBIC)
        gray = rgb_to_grayscale(np.array(cropped_pil))
        gray = unsharp_mask(gray, amount=1.2, kernel_size=3, sigma=0.8)
    else:
        gray = rgb_to_grayscale(np.array(cropped_pil))

    # 2. Xử lý ảnh thiếu sáng / tối hoặc loá sáng bằng CLAHE
    try:
        from image_processing.preprocessing import apply_clahe_contrast
        gray = apply_clahe_contrast(gray, clip_limit=2.5, tile_grid_size=(8, 8))
    except Exception:
        gray = enhance_contrast_dynamic(gray, p_low=2.0, p_high=98.0)

    h, w = gray.shape

    # 3. Phân đoạn ảnh (Image Segmentation) bóc tách ký tự
    try:
        import cv2
        char_bin_adapt = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 8
        )
    except Exception:
        char_bin_adapt = None

    y1, y2 = max(0, int(h * 0.08)), min(h, int(h * 0.92))
    x1, x2 = max(0, int(w * 0.06)), min(w, int(w * 0.94))

    inner_gray = gray[y1:y2, x1:x2]
    if inner_gray.size == 0:
        inner_gray = gray

    _, t = otsu_threshold(inner_gray)
    char_bin_otsu = (gray < t).astype(np.uint8) * 255

    if char_bin_adapt is not None:
        otsu_density = float(np.mean(char_bin_otsu > 0))
        if otsu_density < 0.10 or otsu_density > 0.55:
            char_bin = char_bin_adapt
        else:
            char_bin = char_bin_otsu
    else:
        char_bin = char_bin_otsu

    # 4. Xử lý biển số bị dính bẩn / vết trầy xước: Hàn gắn nét ký tự bị đứt
    char_bin = bridge_broken_characters(char_bin, gap_size=3)

    # 5. Triệt tiêu khung viền kim loại và ốc vít ở mép ngoài
    margin_y = max(1, int(h * 0.06))
    margin_x = max(1, int(w * 0.05))
    char_bin[:margin_y, :] = 0
    char_bin[-margin_y:, :] = 0
    char_bin[:, :margin_x] = 0
    char_bin[:, -margin_x:] = 0

    return char_bin


def estimate_tilt_angle(candidates: List[Dict[str, Any]], h: int) -> float:
    """
    Ước lượng góc nghiêng (Tilt/Deskew Angle) của biển số xe bằng độ dốc trục hoành
    của các ký tự trong dòng.
    """
    if len(candidates) < 3:
        return 0.0

    mid_y = h * 0.50
    top_row = [c for c in candidates if c["cy"] < mid_y]
    bot_row = [c for c in candidates if c["cy"] >= mid_y]

    target_row = top_row if len(top_row) >= 3 else (bot_row if len(bot_row) >= 3 else candidates)
    if len(target_row) < 3:
        return 0.0

    target_row.sort(key=lambda c: c["cx"])
    dx = target_row[-1]["cx"] - target_row[0]["cx"]
    dy = target_row[-1]["cy"] - target_row[0]["cy"]
    if abs(dx) < 1e-3:
        return 0.0

    angle = np.degrees(np.arctan2(dy, dx))
    return float(angle)


def segment_characters(char_bin: np.ndarray, plate_type: str = "double_row") -> List[Dict[str, Any]]:
    """
    Phân đoạn từng ký tự từ ảnh nhị phân bằng thuật toán Connected Component Labeling.
    """
    h, w = char_bin.shape
    components, _ = find_connected_components(char_bin, connectivity=8, compute_quad=False)

    char_candidates = []
    for c in components:
        bx, by, bw, bh = c['bbox']
        aspect_ratio = bw / float(bh) if bh > 0 else 0.0
        h_ratio = bh / float(h)
        w_ratio = bw / float(w)
        density = c['density']

        if plate_type == "single_row":
            if not (0.28 <= h_ratio <= 0.88):
                continue
            if not (0.015 <= w_ratio <= 0.22):
                continue
        else: # double_row
            if not (0.15 <= h_ratio <= 0.52):
                continue
            if not (0.015 <= w_ratio <= 0.25):
                continue

        if not (0.10 <= aspect_ratio <= 1.15):
            continue
        if not (0.10 <= density <= 0.88):
            continue

        patch = char_bin[by : by + bh, bx : bx + bw]
        char_candidates.append({
            "bbox": (bx, by, bw, bh),
            "patch": patch,
            "cx": bx + bw / 2.0,
            "cy": by + bh / 2.0,
        })

    # Lọc các hộp trùng hoặc lồng bên trong nhau
    filtered = []
    for i, c1 in enumerate(char_candidates):
        x1, y1, w1, h1 = c1["bbox"]
        is_inside = False
        for j, c2 in enumerate(char_candidates):
            if i == j: continue
            x2, y2, w2, h2 = c2["bbox"]
            if (x2 <= x1 and y2 <= y1 and x2 + w2 >= x1 + w1 and y2 + h2 >= y1 + h1 and (w2*h2 > w1*h1 * 1.3)):
                is_inside = True
                break
        if not is_inside:
            filtered.append(c1)

    return filtered


def recognize_license_plate(cropped_pil: Image.Image, plate_type: str = "double_row") -> Dict[str, Any]:
    """
    Hàm nhận diện toàn diện biển số xe:
    1. Nhị phân hóa chính xác lòng biển số (hỗ trợ chống tối, chống lóa)
    2. Tự động hiệu chỉnh góc nghiêng (Auto-Deskew) cho ảnh chụp xiên/nghiêng
    3. Tách từng ký tự
    4. Nhận dạng theo chuẩn cấu trúc biển số Việt Nam
    """
    # 1. Nhị phân hoá ảnh ban đầu ở góc 0 độ
    char_bin = binarize_cropped_plate(cropped_pil)
    h, w = char_bin.shape
    candidates = segment_characters(char_bin, plate_type=plate_type)

    # 2. Xử lý trường hợp ảnh chụp nghiêng / biến dạng góc (Auto-Deskewing):
    # Ước lượng góc nghiêng từ dãy ký tự ban đầu
    tilt_cands = estimate_tilt_angle(candidates, h)

    # Ước lượng góc nghiêng từ mô-men trung tâm bậc 2 (2nd-order Central Moments)
    # trên vùng nền sáng của biển số (theo giáo trình Xử lý ảnh số - Đặc trưng hình học đối tượng)
    arr_c = np.array(cropped_pil)
    if arr_c.ndim == 3:
        lum_c = 0.299 * arr_c[..., 0] + 0.587 * arr_c[..., 1] + 0.114 * arr_c[..., 2]
    else:
        lum_c = arr_c.astype(float)
    bright_mask = lum_c > np.percentile(lum_c, 55)
    ys_m, xs_m = np.where(bright_mask)
    if len(xs_m) > 100:
        xm_c, ym_c = np.mean(xs_m), np.mean(ys_m)
        u20 = np.mean((xs_m - xm_c)**2)
        u02 = np.mean((ys_m - ym_c)**2)
        u11 = np.mean((xs_m - xm_c) * (ys_m - ym_c))
        tilt_moments = float(np.degrees(0.5 * np.arctan2(2 * u11, u20 - u02)))
    else:
        tilt_moments = 0.0

    # Nếu có dấu hiệu nghiêng xiên hoặc số ký tự thu được ban đầu < 6, tiến hành tìm góc nắn thẳng tối ưu
    needs_deskew = (abs(tilt_cands) >= 2.5) or (abs(tilt_moments) >= 4.0) or (len(candidates) < 6)
    if needs_deskew:
        test_angles = [0.0]
        candidate_pool = [-tilt_cands, tilt_cands, -tilt_moments, tilt_moments, -18.0, -15.0, -12.0, 12.0, 15.0, 18.0]
        for a in candidate_pool:
            if abs(a) >= 2.5 and not any(abs(a - ta) < 1.0 for ta in test_angles):
                test_angles.append(a)

        best_score = -1.0
        best_cands = candidates
        best_bin = char_bin
        best_pil = cropped_pil

        for ang in test_angles:
            rot_pil = cropped_pil.rotate(ang, resample=Image.Resampling.BICUBIC) if abs(ang) > 0.5 else cropped_pil
            rot_bin = binarize_cropped_plate(rot_pil)
            cands_rot = segment_characters(rot_bin, plate_type=plate_type)
            rh, rw = rot_bin.shape

            if plate_type == "double_row" and len(cands_rot) >= 4:
                all_cys = [c['cy'] for c in cands_rot]
                s_y = (min(all_cys) + float(np.median(all_cys))) / 2.0
                top_c = [c for c in cands_rot if c['cy'] < s_y]
                bot_c = [c for c in cands_rot if c['cy'] >= s_y]
                sc = 0.0
                if 3 <= len(top_c) <= 5: sc += 3.5
                if 4 <= len(bot_c) <= 6: sc += 4.5
                if len(top_c) >= 3:
                    sc += max(0.0, 2.0 - np.var([c['cy'] for c in top_c]) / 40.0)
                if len(bot_c) >= 3:
                    sc += max(0.0, 2.0 - np.var([c['cy'] for c in bot_c]) / 40.0)
            else:
                sc = float(len(cands_rot))
                if len(cands_rot) >= 5:
                    sc += max(0.0, 2.0 - np.var([c['cy'] for c in cands_rot]) / 40.0)

            if sc > best_score:
                best_score = sc
                best_cands = cands_rot
                best_bin = rot_bin
                best_pil = rot_pil

        if best_score >= 5.0 or len(best_cands) > len(candidates):
            candidates = best_cands
            char_bin = best_bin
            cropped_pil = best_pil
            h, w = char_bin.shape

    matcher = get_template_matcher()

    if not candidates or len(candidates) < 2:
        return {
            "plate_text": "KXD",
            "char_count": len(candidates),
            "avg_confidence": 0.0,
            "characters": []
        }

    recognized_chars = []

    if plate_type == "double_row":
        # Tự động chia dòng 1 và dòng 2 bằng tâm phân vị dọc (Adaptive Vertical Clustering)
        # đảm bảo chính xác tuyệt đối kể cả khi ảnh cắt có lề hoặc biển số bị nghiêng
        if len(candidates) >= 4:
            all_cys = [c['cy'] for c in candidates]
            mid_cand = (min(all_cys) + max(all_cys)) / 2.0
            top_group = [cy for cy in all_cys if cy < mid_cand]
            bot_group = [cy for cy in all_cys if cy >= mid_cand]
            if top_group and bot_group:
                split_y = (max(top_group) + min(bot_group)) / 2.0
            else:
                split_y = float(np.median(all_cys))
        else:
            split_y = h * 0.48

        top_row = [c for c in candidates if c["cy"] < split_y]
        bot_row = [c for c in candidates if c["cy"] >= split_y]

        # Khử các đốm nhiễu viền khung / ốc vít có chiều cao hoặc tọa độ y bất thường ở dòng 2
        if len(bot_row) > 5:
            med_h = float(np.median([c['bbox'][3] for c in bot_row]))
            med_y = float(np.median([c['cy'] for c in bot_row]))
            bot_row = [
                c for c in bot_row
                if abs(c['bbox'][3] - med_h) <= 0.35 * med_h and abs(c['cy'] - med_y) <= 0.18 * h
            ]

        # Biển số xe máy Việt Nam dòng 2 có tối đa 5 chữ số:
        # Nếu dư ký tự do viền mép / ốc vít, loại bỏ ký tự nhiễu biên để giữ đúng 5 số
        while len(bot_row) > 5:
            first_c = bot_row[0]
            last_c = bot_row[-1]
            if last_c['bbox'][2] < 26 or last_c['bbox'][0] > w * 0.85:
                bot_row.pop(-1)
            elif first_c['bbox'][2] < 26 or first_c['bbox'][0] < w * 0.08:
                bot_row.pop(0)
            else:
                # Loại ký tự có điểm tin cậy thấp nhất
                min_idx = min(range(len(bot_row)), key=lambda i: matcher.match(bot_row[i]['patch'], is_digit_only=True)[1])
                bot_row.pop(min_idx)

        top_row.sort(key=lambda c: c["cx"])
        bot_row.sort(key=lambda c: c["cx"])

        top_text = ""
        top_scores = []
        for idx, c in enumerate(top_row):
            # Biển 2 dòng:
            # - Ký tự 0, 1: Luôn là số tỉnh (ví dụ 18, 37, 90)
            # - Ký tự 2: Luôn là chữ cái series (ví dụ M, A, B)
            # - Nếu dòng trên có 5 ký tự (ví dụ 18-MD2): Ký tự 3 là chữ cái (D), ký tự 4 là số (2)
            # - Nếu dòng trên có 4 ký tự: Ký tự 3 có thể là chữ cái (37-AC) hoặc chữ số (90-B3)
            if len(top_row) >= 5:
                is_digit = (idx in [0, 1, 4])
                is_letter = (idx in [2, 3])
            else:
                is_digit = (idx in [0, 1])
                is_letter = (idx == 2)

            ch, score = matcher.match(c["patch"], is_digit_only=is_digit, is_letter_only=is_letter)
            
            # Post-processing quy đổi ký tự nhầm lẫn:
            if is_digit:
                digit_map = {'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'L': '1', 'T': '1', 'Z': '2', 'S': '5', 'B': '8'}
                ch = digit_map.get(ch, ch)
            elif is_letter:
                letter_map = {'0': 'D', '1': 'I', '8': 'B'}
                ch = letter_map.get(ch, ch)

            top_text += ch
            top_scores.append(score)
            recognized_chars.append({"char": ch, "score": score, "row": 1, "bbox": c["bbox"]})

        bot_text = ""
        bot_scores = []
        for idx, c in enumerate(bot_row):
            # Dòng dưới luôn là các chữ số
            ch, score = matcher.match(c["patch"], is_digit_only=True)
            
            # Post-processing chuẩn hóa ký tự nhầm lẫn sang chữ số
            digit_map = {'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'L': '1', 'T': '1', 'Z': '2', 'S': '5', 'B': '8'}
            ch = digit_map.get(ch, ch)

            bot_text += ch
            bot_scores.append(score)
            recognized_chars.append({"char": ch, "score": score, "row": 2, "bbox": c["bbox"]})

        # Định dạng chuẩn biển số 2 dòng: ví dụ 18-MD2 074.73
        if len(top_text) >= 3 and len(bot_text) >= 4:
            if len(bot_text) == 5:
                formatted_bot = f"{bot_text[:3]}.{bot_text[3:]}"
            else:
                formatted_bot = bot_text
            formatted_top = f"{top_text[:2]}-{top_text[2:]}"
            plate_text = f"{formatted_top} {formatted_bot}"
        elif top_text and bot_text:
            plate_text = f"{top_text}-{bot_text}"
        elif top_text:
            plate_text = top_text
        elif bot_text:
            plate_text = bot_text
        else:
            plate_text = "KXD"

        all_scores = top_scores + bot_scores
        avg_score = float(np.mean(all_scores)) if all_scores else 0.0

    else: # single_row
        candidates.sort(key=lambda c: c["cx"])
        plate_text_chars = []
        scores = []
        for idx, c in enumerate(candidates):
            is_digit = (idx < 2)
            is_letter = (idx == 2)
            ch, score = matcher.match(c["patch"], is_digit_only=is_digit, is_letter_only=is_letter)
            
            # Post-processing quy đổi ký tự nhầm lẫn:
            if is_digit or idx > 2:
                digit_map = {'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'L': '1', 'T': '1', 'Z': '2', 'S': '5', 'B': '8'}
                ch = digit_map.get(ch, ch)
            elif is_letter:
                letter_map = {'0': 'D', '1': 'I', '8': 'B'}
                ch = letter_map.get(ch, ch)

            plate_text_chars.append(ch)
            scores.append(score)
            recognized_chars.append({"char": ch, "score": score, "row": 1, "bbox": c["bbox"]})

        raw_text = "".join(plate_text_chars)
        if len(raw_text) >= 6:
            # Ví dụ: 30F-123.45
            prefix = raw_text[:3]
            body = raw_text[3:]
            if len(body) == 5:
                plate_text = f"{prefix}-{body[:3]}.{body[3:]}"
            else:
                plate_text = f"{prefix}-{body}"
        elif len(raw_text) > 3:
            plate_text = f"{raw_text[:3]}-{raw_text[3:]}"
        else:
            plate_text = raw_text if raw_text else "KXD"

        avg_score = float(np.mean(scores)) if scores else 0.0

    return {
        "plate_text": plate_text,
        "char_count": len(recognized_chars),
        "avg_confidence": round(avg_score, 3),
        "characters": recognized_chars,
        "deskewed_image": cropped_pil
    }
