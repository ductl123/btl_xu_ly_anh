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

from .preprocessing import rgb_to_grayscale
from .thresholding import otsu_threshold
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

        # Quy tắc 1: Chữ số 1 (thanh dọc mảnh, không bị nhầm sang 3 hay 7)
        if is_digit_only and ranked[0][0] == '1':
            best_ch = '1'

        # Quy tắc 2: Nhận diện chữ số 4 (có thanh ngang phía dưới và góc trên bên phải hoàn toàn rỗng)
        elif is_digit_only and (best_ch in ['4', '0', '5', '6', '8', '9']):
            crossbar = max([p_norm[r, :].sum() for r in range(20, 30)]) if p_norm.shape[0] >= 30 else 0
            top_right_gap = p_norm[:12, 12:].mean() if p_norm.shape[0] >= 12 and p_norm.shape[1] >= 16 else 1.0
            if crossbar >= 14 and top_right_gap < 0.20:
                best_ch = '4'

        # Quy tắc 3: Phân biệt 0 và 8
        if is_digit_only and best_ch in ['0', '8', '6', '9']:
            if mid_center_val < 0.58:
                best_ch = '0'
            elif mid_center_val >= 0.70 and ranked[0][0] == '8':
                best_ch = '8'

        # Quy tắc 3: Phân biệt D và B
        if is_letter_only and best_ch in ['B', 'D']:
            if mid_center_val < 0.38:
                best_ch = 'D'
            elif mid_center_val > 0.65:
                best_ch = 'B'

        # Quy tắc 4: Chữ M (2 cột trụ thẳng đứng 2 bên cạnh)
        if is_letter_only and best_ch in ['X', 'B', 'M', 'N', 'H']:
            left_col_solid = (p_norm[6:26, :5].mean() > 0.55)
            right_col_solid = (p_norm[6:26, -5:].mean() > 0.55)
            if left_col_solid and right_col_solid:
                best_ch = 'M'

        # Quy tắc 5: Phân biệt 7, 2, 3 (chỉ khi không phải số 1)
        if is_digit_only and best_ch in ['7', '2', '3', '8'] and ranked[0][0] != '1':
            top_bar = p_norm[:4, :].sum(axis=1).mean()
            bot_bar = p_norm[-4:, :].sum(axis=1).mean()
            left_indent = (p_norm[8:14, :5].mean() < 0.20 and p_norm[20:26, :5].mean() < 0.20)
            mid_bar = p_norm[14:18, 5:14].mean() > 0.50

            if top_bar > 15.0 and bot_bar < 8.0:
                best_ch = '7'
            elif ranked[0][0] == '2':
                best_ch = '2'
            elif left_indent and mid_bar:
                best_ch = '3'
            elif bot_bar > 14 and not left_indent and best_ch in ['2', '7']:
                best_ch = '2'

        # Quy tắc 6: Nhận diện chữ số 4
        if is_digit_only and (best_ch in ['4', '5', '6', '8', '9']):
            crossbar = p_norm[21:27, :].sum(axis=1).max()
            top_right_gap = p_norm[:12, 12:].mean()
            if crossbar >= 13 and top_right_gap < 0.35:
                best_ch = '4'

        return best_ch, round(float(best_score), 2)


_GLOBAL_MATCHER = None

def get_template_matcher() -> TemplateMatcher:
    global _GLOBAL_MATCHER
    if _GLOBAL_MATCHER is None:
        _GLOBAL_MATCHER = TemplateMatcher()
    return _GLOBAL_MATCHER


def binarize_cropped_plate(cropped_pil: Image.Image) -> np.ndarray:
    """
    Tiền xử lý và nhị phân hóa ảnh biển số xe:
    Hỗ trợ xử lý cho:
    - Ảnh chụp tối (underexposed/low-light): Kéo giãn dải động độ tương phản.
    - Ảnh chụp loá (glare/reflection): Cắt ngưỡng thích nghi hạn chế cháy sáng.
    Tính toán ngưỡng Otsu trên vùng lòng biển số xe để tránh bị ảnh hưởng bởi khung viền tối màu.
    Các nét chữ và số tối màu được chuyển thành 255 (trắng).
    """
    gray = rgb_to_grayscale(np.array(cropped_pil))
    
    # Kéo giãn độ tương phản động Percentile để chống tối và chống lóa
    p2, p98 = np.percentile(gray, (2.0, 98.0))
    if p98 > p2:
        gray = np.clip((gray.astype(np.float32) - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
        
    h, w = gray.shape

    # Lấy vùng trung tâm lòng biển số (8% - 92% chiều cao, 6% - 94% chiều rộng)
    y1, y2 = max(0, int(h * 0.08)), min(h, int(h * 0.92))
    x1, x2 = max(0, int(w * 0.06)), min(w, int(w * 0.94))

    inner_gray = gray[y1:y2, x1:x2]
    if inner_gray.size == 0:
        inner_gray = gray

    _, t = otsu_threshold(inner_gray)

    # Ký tự màu tối trên nền sáng biển số
    char_bin = (gray < t).astype(np.uint8) * 255

    # Triệt tiêu khung viền kim loại và ốc vít ở mép ngoài
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
    components, _ = find_connected_components(char_bin, connectivity=8)

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
    char_bin = binarize_cropped_plate(cropped_pil)
    h, w = char_bin.shape
    candidates = segment_characters(char_bin, plate_type=plate_type)

    # Hiệu chỉnh góc nghiêng tự động (Deskewing) nếu phát hiện góc xiên > 2.5 độ
    tilt_angle = estimate_tilt_angle(candidates, h)
    if 2.5 <= abs(tilt_angle) <= 35.0:
        rotated_pil = cropped_pil.rotate(tilt_angle, resample=Image.Resampling.BICUBIC)
        char_bin_rot = binarize_cropped_plate(rotated_pil)
        cands_rot = segment_characters(char_bin_rot, plate_type=plate_type)
        if len(cands_rot) >= len(candidates) - 1:
            cropped_pil = rotated_pil
            char_bin = char_bin_rot
            candidates = cands_rot
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
        mid_y = h * 0.50
        top_row = [c for c in candidates if c["cy"] < mid_y]
        bot_row = [c for c in candidates if c["cy"] >= mid_y]

        top_row.sort(key=lambda c: c["cx"])
        bot_row.sort(key=lambda c: c["cx"])

        top_text = ""
        top_scores = []
        for idx, c in enumerate(top_row):
            # Biển 2 dòng: 2 ký tự đầu là số tỉnh (18), tiếp theo là chữ cái (M, D), ký tự thứ 5 (nếu có 5 ký tự) thường là số seri (ví dụ MD1, MD2, A1)
            is_digit = (idx in [0, 1]) or (idx == 4 and len(top_row) >= 5)
            is_letter = (idx in [2, 3]) and not is_digit
            ch, score = matcher.match(c["patch"], is_digit_only=is_digit, is_letter_only=is_letter)
            top_text += ch
            top_scores.append(score)
            recognized_chars.append({"char": ch, "score": score, "row": 1, "bbox": c["bbox"]})

        bot_text = ""
        bot_scores = []
        for idx, c in enumerate(bot_row):
            # Dòng dưới luôn là các chữ số
            ch, score = matcher.match(c["patch"], is_digit_only=True)
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
        "characters": recognized_chars
    }
