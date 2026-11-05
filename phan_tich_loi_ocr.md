# BÁO CÁO LỖI OCR — HỆ THỐNG NHẬN DẠNG KÝ TỰ BIỂN SỐ XE
> **Dành cho Agent IDE**: Toàn bộ lỗi đều nằm trong file chính
> `LicensePlateDetection/image_processing/character_recognition.py`
> Mỗi mục ghi rõ: Số dòng · Code lỗi thực tế · Hậu quả · Cách sửa.

---

## LỖI 1 (🔴 NGHIÊM TRỌNG) — Luồng `if/elif` chặn logic phân biệt `7/3/2`

### File & Dòng
`character_recognition.py` — hàm `disambiguate_character_v2()`, dòng **214–250**

### Code Lỗi Thực Tế
```python
# --- Bẫy 6 vs 8 --- (dòng 214)
if best_ch in ['6', '8'] or (best_ch == '8' and scores.get('6', 0.0) >= 0.65):
    if d_top_right_gap < 0.30:
        return '6'          # ← return sớm ở đây
    elif d_top_right_gap >= 0.45:
        return '8'          # ← hoặc return sớm ở đây

# --- Bẫy 7 vs 3 vs 2 vs Z vs 1 --- (dòng 224)  ← KHÔNG BAO GIỜ CHẠY ĐƯỢC NẾU best_ch='8'
if best_ch in ['7', '3', '2', 'Z', '1']:
    ...
```

### Hậu Quả
Khi template matching trả về `best_ch = '8'` nhưng ký tự thực là `3`:
- Điều kiện dòng 214 khớp (`'8'` nằm trong list)
- Nếu `d_top_right_gap < 0.30` → trả về `'6'` **sai hoàn toàn**
- Nhánh `7 vs 3 vs 2` ở dòng 224 không bao giờ được thực thi

### Cách Sửa
Đảo thứ tự: Kiểm tra `7 vs 3 vs 2` **TRƯỚC**, rồi mới đến `6 vs 8`:
```python
# ĐẶT TRƯỚC — Bẫy 7 vs 3 vs 2 (không bị chặn)
if best_ch in ['7', '3', '2', 'Z', '1']:
    ...  # (giữ nguyên logic hiện tại)

# ĐẶT SAU — Bẫy 6 vs 8
if best_ch in ['6', '8']:
    ...
```

---

## LỖI 2 (🔴 NGHIÊM TRỌNG) — Ký tự `1` bị lọc bỏ do Aspect Ratio

### File & Dòng
`character_recognition.py` — hàm `segment_characters()`, dòng **610–612**

### Code Lỗi Thực Tế
```python
ar = float(bw) / float(bh)
if not (0.10 <= ar <= 1.15):   # ← Ngưỡng dưới 0.10 quá cao
    continue
```

### Hậu Quả
Ký tự `1` trên biển số thực tế có AR ≈ **0.06–0.10** (rất mảnh). Với ngưỡng `ar >= 0.10`, đa số chữ `1` **bị lọc bỏ hoàn toàn** → Hệ thống không nhận ra được số `1`, đọc sai cả biển số.

### Cách Sửa
```python
ar = float(bw) / float(bh)
if not (0.07 <= ar <= 1.20):   # Hạ từ 0.10 → 0.07 để giữ ký tự '1'
    continue
```

---

## LỖI 3 (🔴 NGHIÊM TRỌNG) — Ngưỡng Otsu Lấp Lỗ Ký Tự `8`, `B`, `0`

### File & Dòng
`character_recognition.py` — hàm `binarize_cropped_plate()`, dòng **547–553**

### Code Lỗi Thực Tế
```python
p25 = float(np.percentile(inner, 25))
t_opt = int(round(0.75 * t_otsu + 0.25 * p25))
t_opt = max(35, min(160, t_opt))  # ← Clip cứng không thích nghi

char_bin = (enhanced < t_opt)
```

### Hậu Quả
Trên biển vàng (xe máy) và ảnh thiếu sáng:
- `t_otsu` bị kéo lên cao → `t_opt` cao → Pixel xám vùng nền bị nhị phân thành `True` (ký tự)
- Lỗ bên trong `8`, `0`, `B` bị lấp đầy → Euler = 2 → 1 → Template matching nhầm sang `0`, `6`, `D`
- Clip `min(160, ...)` ngăn không xử lý được ảnh chói sáng mạnh

### Cách Sửa
```python
# Phát hiện nền tối/sáng để chọn chiến lược ngưỡng
mean_inner = float(np.mean(inner))
if mean_inner > 160:  # Nền sáng (biển trắng/vàng bình thường)
    t_opt = int(round(0.80 * t_otsu + 0.20 * p25))
    t_opt = max(40, min(200, t_opt))
else:  # Nền tối / thiếu sáng
    t_opt = int(round(0.65 * t_otsu + 0.35 * p25))
    t_opt = max(25, min(140, t_opt))

char_bin = (enhanced < t_opt)
```
Hoặc tốt hơn: Dùng thêm **Adaptive Thresholding** song song để lấy union:
```python
import cv2
# Adaptive để bảo toàn lỗ ký tự
adaptive_bin = cv2.adaptiveThreshold(
    enhanced, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, 15, 4
) > 0
# Kết hợp: Union của Otsu và Adaptive
char_bin = (enhanced < t_opt) | adaptive_bin
```

---

## LỖI 4 (🟡 TRUNG BÌNH) — `crossbar_max` Và `ar_p` Tính Trên Patch Gốc, Không Phải 20×32

### File & Dòng
`character_recognition.py` — hàm `disambiguate_character_v2()`, dòng **198–212**

### Code Lỗi Thực Tế
```python
h_p, w_p = patch_shape          # ← Kích thước patch GỐC (trước khi normalize)
ar_p = float(w_p) / float(h_p)  # ← AR của patch gốc
crossbar_max = int(P[16:24, :].sum(axis=1).max())  # ← Trên P đã normalize 20×32

# Dòng 209-212:
if crossbar_max >= 13 and d_top_right_gap < 0.25:   # 13/20 = 65% chiều rộng (chuẩn)
    return '4'
if (ar_p <= 0.40 or vp_center >= 0.58) and crossbar_max < 13:  # ar_p dùng kích thước GỐC ← Sai
    return '1'
```

### Hậu Quả
- `ar_p` tính từ patch gốc (ví dụ 15×40 pixel) → `ar_p = 0.375 ≤ 0.40` → Trả về `'1'`
- Nhưng cùng ký tự đó nếu được cắt to hơn (25×40) → `ar_p = 0.625 > 0.40` → Không trả về `'1'`
- Cùng một ký tự `1` nhưng kết quả phụ thuộc vào kích thước cắt → **Không ổn định**

### Cách Sửa
Luôn tính AR trên kích thước chuẩn hóa 20×32:
```python
# Xóa dòng: h_p, w_p = patch_shape
# Thay bằng:
ar_p = 20.0 / 32.0   # = 0.625 cố định sau normalize, không dùng ar_p nữa
# Chỉ dùng vp_center và crossbar_max (đã trên 20×32) để phân biệt 4 vs 1
if crossbar_max >= 13:
    return '4'
elif vp_center >= 0.60:
    return '1'
```

---

## LỖI 5 (🟡 TRUNG BÌNH) — Logic `B vs D` Bị Điều Kiện `elif` Che Khuất

### File & Dòng
`character_recognition.py` — hàm `disambiguate_character_v2()`, dòng **303–312**

### Code Lỗi Thực Tế
```python
if bot_right < 0.20 and mid_right < 0.35:
    return 'P'
elif right_notch >= 0.55 and core_hole < 0.35 and holes <= 1:
    return 'D'
elif mid_center >= 0.62 or right_notch < 0.45 or holes == 2:   # ← Điều kiện rộng
    return 'B'
elif mid_center < 0.45:   # ← Không bao giờ chạy được!
    return 'D'             #    Vì elif trước đã bắt mọi case mid_center < 0.62
```

### Hậu Quả
Nhánh cuối `elif mid_center < 0.45: return 'D'` **không bao giờ được thực thi** vì nhánh trước `right_notch < 0.45` đã bắt tất cả các case mà `mid_center < 0.45`.
Kết quả: Một số ký tự `D` bị nhầm thành `B`.

### Cách Sửa
```python
if bot_right < 0.20 and mid_right < 0.35:
    return 'P'
elif holes == 2 or (mid_center >= 0.62 and right_notch < 0.45):
    return 'B'
elif right_notch >= 0.50 and core_hole < 0.35:
    return 'D'
elif right_notch >= 0.45 and mid_center < 0.50:
    return 'D'
else:
    return 'B'
```

---

## LỖI 6 (🟡 TRUNG BÌNH) — Font Consolas Làm Nhiễu Kho Template

### File & Dòng
`character_recognition.py` — hàm `TemplateMatcher._init_templates()`, dòng **397–401**

### Code Lỗi Thực Tế
```python
font_paths = [
    "C:/Windows/Fonts/bahnschrift.ttf",   # DIN 1451 — Phù hợp
    "C:/Windows/Fonts/arialbd.ttf",        # Arial Bold — Khá khác biển số
    "C:/Windows/Fonts/consola.ttf",        # Consolas — Font monospace, SAI HOÀN TOÀN
]
```

### Hậu Quả
Biển số Việt Nam dùng font **DIN 1451 Mittelschrift** (tương đương Highway Gothic). Consolas là font monospace lập trình, có tỷ lệ nét chữ hoàn toàn khác:
- Template Consolas của `1`, `I`, `l` trông giống nhau → IoU tương đồng cao → Nhiều false match
- Template Consolas của `0` trông khác `0` biển số (Consolas có oval hẹp) → Làm loãng điểm đúng

### Cách Sửa
```python
font_paths = [
    "C:/Windows/Fonts/bahnschrift.ttf",   # Giữ — Gần DIN 1451 nhất
    "C:/Windows/Fonts/arialbd.ttf",        # Giữ — Cover biển số cũ trước 2015
    # "C:/Windows/Fonts/consola.ttf",      # XÓA — Không phù hợp
]
```

---

## LỖI 7 (🟢 NHỎ) — `DIGIT_MAP` Áp Dụng 2 Lần Liên Tiếp

### File & Dòng
`character_recognition.py` — hàm `_recognize_single_orientation()`, dòng **802–805**

### Code Lỗi Thực Tế
```python
# Lần 1: BÊN TRONG disambiguate_character_v2() — dòng 168-172
DIGIT_MAP = {'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'L': '1', 'T': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6'}
if is_digit_only:
    best_ch = DIGIT_MAP.get(best_ch, best_ch)  # ← Áp dụng lần 1

# Lần 2: SAU KHI GỌI matcher.match() — dòng 802
ch, score = matcher.match(c["patch"], is_digit_only=is_digit, is_letter_only=is_letter)
if is_digit:
    ch = DIGIT_MAP.get(ch, ch)   # ← Áp dụng lại lần 2 (THỪA)
```

### Hậu Quả
Không gây lỗi kết quả (vì mapping idempotent) nhưng:
- Code không nhất quán, khó maintain
- Nếu sau này `disambiguate_character_v2` trả về ký tự ngoài DIGIT_MAP, lần 2 có thể ghi đè sai

### Cách Sửa
Xóa toàn bộ `DIGIT_MAP` / `LETTER_MAP` và 2 vòng `if is_digit: ch = DIGIT_MAP.get(...)` trong `_recognize_single_orientation()`. Chỉ giữ mapping **bên trong** `disambiguate_character_v2()`.

---

## LỖI 8 (🟢 NHỎ) — `EULER_GROUPS` Chứa Ký Tự Không Tồn Tại Trong Bộ 30 Ký Tự

### File & Dòng
`character_recognition.py` — class `TemplateMatcher`, dòng **386–390**

### Code Lỗi Thực Tế
```python
CHARS_LETTERS = "ABCDEFGHKLMNPSTUVXYZ"   # ← Không có Q, J

EULER_GROUPS = {
    2: set("8B"),
    1: set("0469ADPQ"),    # ← Q không có trong CHARS_LETTERS
    0: set("12357CEFGHJKLMNSTUVXYZ")  # ← J không có trong CHARS_ALL
}
```

### Hậu Quả
- `Q` trong nhóm Euler-1 nhưng không có trong `templates` → Phần thưởng `+0.015` cho nhóm sẽ không được tính cho `Q`
- `J` trong nhóm Euler-0 nhưng không được match → Debug gây nhầm lẫn

### Cách Sửa
```python
EULER_GROUPS = {
    2: set("8B"),
    1: set("0469ADP"),          # Bỏ Q (không dùng)
    0: set("12357CEFGHKLMNSTUVXYZ")  # Bỏ J (không dùng)
}
```

---

## TÓM TẮT ƯU TIÊN SỬA

| Thứ tự | Lỗi | Mức độ | Hàm cần sửa | Dòng |
|--------|-----|--------|-------------|------|
| 1 | Thứ tự if/elif chặn `7/3/2` | 🔴 Cao | `disambiguate_character_v2` | 214–250 |
| 2 | AR filter loại `'1'` | 🔴 Cao | `segment_characters` | 611 |
| 3 | Ngưỡng Otsu lấp lỗ `8/0/B` | 🔴 Cao | `binarize_cropped_plate` | 547–553 |
| 4 | `crossbar_max`/`ar_p` sai đơn vị | 🟡 TB | `disambiguate_character_v2` | 198–212 |
| 5 | B vs D elif bị che | 🟡 TB | `disambiguate_character_v2` | 303–312 |
| 6 | Font Consolas nhiễu template | 🟡 TB | `TemplateMatcher._init_templates` | 399 |
| 7 | DIGIT_MAP áp dụng 2 lần | 🟢 Nhỏ | `_recognize_single_orientation` | 802–805 |
| 8 | EULER_GROUPS ký tự không tồn tại | 🟢 Nhỏ | `TemplateMatcher` class | 386–390 |

---

> **Ghi chú cho Agent**: Tất cả 8 lỗi trên đều chỉ nằm trong 1 file duy nhất:
> `d:\hoc_tap\hoc_tap\nam_4\xu_li_anh\btl\LicensePlateDetection\image_processing\character_recognition.py`
> Không cần chỉnh sửa bất kỳ file nào khác.
