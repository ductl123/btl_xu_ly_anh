"""
image_processing/thresholding.py - Nhị phân hóa ảnh
Bao gồm:
- Tự cài đặt thuật toán xác định ngưỡng tự động Otsu (Otsu's Thresholding) bằng NumPy.
- Tự cài đặt phân ngưỡng cố định (Fixed Thresholding) để so sánh thực nghiệm.
"""

from typing import Tuple
import numpy as np


def otsu_threshold(image_gray: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Thuật toán phân ngưỡng Otsu tự cài đặt hoàn chỉnh:
    Tự động tìm ngưỡng t* sao cho phương sai giữa hai lớp (between-class variance) đạt giá trị cực đại.

    Nguyên lý toán học:
    1. Histogram h(i) cho i từ 0 đến 255 mức xám.
    2. Xác suất p(i) = h(i) / N với N là tổng số pixel.
    3. Trọng số lớp 0 (nền): w0(t) = sum_{i=0}^{t} p(i)
       Trọng số lớp 1 (tiền cảnh): w1(t) = 1 - w0(t)
    4. Trung bình tích lũy: mu(t) = sum_{i=0}^{t} i * p(i)
       Trung bình toàn ảnh: mu_T = sum_{i=0}^{255} i * p(i)
    5. Phương sai giữa 2 lớp:
       sigma_B^2(t) = [mu_T * w0(t) - mu(t)]^2 / [w0(t) * w1(t)]
    6. Ngưỡng tối ưu t* = argmax sigma_B^2(t).

    Args:
        image_gray: Mảng 2D ảnh mức xám (H, W), kiểu uint8 hoặc float.

    Returns:
        Tuple (ảnh nhị phân uint8 với các giá trị 0 và 255, ngưỡng tối ưu t_star).
    """
    img_uint8 = np.clip(image_gray, 0, 255).astype(np.uint8)
    total_pixels = img_uint8.size

    # 1. Tính histogram 256 mức xám bằng bincount
    hist = np.bincount(img_uint8.ravel(), minlength=256).astype(np.float64)

    # 2. Xác suất phân phối của từng mức xám
    prob = hist / total_pixels

    # 3. Trọng số tích lũy w0(t)
    omega = np.cumsum(prob)

    # 4. Kỳ vọng tích lũy mu(t) và kỳ vọng toàn thể mu_T
    indices = np.arange(256, dtype=np.float64)
    mu_accum = np.cumsum(indices * prob)
    mu_total = mu_accum[-1]

    # 5. Tính phương sai giữa 2 lớp sigma_B^2(t)
    # Tránh chia cho 0 khi omega = 0 hoặc omega = 1
    valid_mask = (omega > 0.0) & (omega < 1.0)
    variance_between = np.zeros(256, dtype=np.float64)

    if np.any(valid_mask):
        numerator = (mu_total * omega[valid_mask] - mu_accum[valid_mask]) ** 2
        denominator = omega[valid_mask] * (1.0 - omega[valid_mask])
        variance_between[valid_mask] = numerator / denominator

    # Nếu ảnh hằng hoặc không có tương phản (phương sai toàn 0)
    if not np.any(valid_mask) or np.max(variance_between) == 0.0:
        return np.zeros_like(img_uint8), 0

    # 6. Tìm ngưỡng tối ưu t*
    t_star = int(np.argmax(variance_between))

    # 7. Nhị phân hóa (P01): Phù hợp với phân lớp histogram (Lớp 0: <= t, Lớp 1: > t)
    # pixel > t_star gán 255 (tiền cảnh), pixel <= t_star gán 0 (nền)
    binary_img = np.where(img_uint8 > t_star, np.uint8(255), np.uint8(0))

    return binary_img, t_star


def fixed_threshold(image_gray: np.ndarray, threshold: int = 120) -> np.ndarray:
    """
    Phân ngưỡng cố định (Fixed Thresholding):
    Các pixel >= threshold chuyển thành 255, nhỏ hơn chuyển thành 0.
    Dùng để so sánh với phương pháp Otsu tự thích nghi.
    """
    binary_img = np.where(image_gray >= threshold, np.uint8(255), np.uint8(0))
    return binary_img


def adaptive_local_threshold(
    image_gray: np.ndarray,
    block_size: int = 25,
    k: float = 0.2,
    r: float = 128.0,
    method: str = "sauvola"
) -> np.ndarray:
    """
    [Mục 3.4: Phân đoạn ảnh (Image Segmentation)]
    Phân ngưỡng cục bộ thích nghi (Adaptive Local Thresholding - Sauvola / Mean):
    Đặc trị cho các trường hợp:
    - Loá sáng (Overexposure/Glare): Điểm chói phản quang chỉ ảnh hưởng cục bộ, không làm mù toàn ảnh.
    - Thiếu sáng / Chụp tối (Uneven illumination/Shadow): Vùng tối vẫn có ngưỡng cục bộ riêng để bóc tách nét chữ.
    
    Nguyên lý thuật toán Sauvola:
    T(x, y) = m(x, y) * [ 1 + k * ( s(x, y) / R - 1 ) ]
    Trong đó:
    - m(x, y): Giá trị kỳ vọng trung bình trong cửa sổ lân cận block_size x block_size.
    - s(x, y): Độ lệch chuẩn cục bộ.
    - R: Khoảng động độ lệch chuẩn cực đại (thường là 128).
    - k: Hệ số kiểm soát (thường từ 0.1 đến 0.3).
    
    Cài đặt thuần NumPy sử dụng bảng diện tích tích phân (Integral Image) để tính toán siêu tốc O(1) mỗi pixel.
    """
    if block_size % 2 == 0:
        block_size += 1

    img = image_gray.astype(np.float64)
    h, w = img.shape
    pad = block_size // 2

    # Bảng ảnh tích phân (Integral Image) cho giá trị và bình phương giá trị
    padded = np.pad(img, ((pad, pad), (pad, pad)), mode='edge')
    
    # Tính tổng cục bộ bằng Integral Image
    integral = np.pad(np.cumsum(np.cumsum(padded, axis=0), axis=1), ((1, 0), (1, 0)), mode='constant')
    integral_sq = np.pad(np.cumsum(np.cumsum(padded ** 2, axis=0), axis=1), ((1, 0), (1, 0)), mode='constant')

    # Tọa độ 4 góc của cửa sổ
    y1 = np.arange(h)
    y2 = y1 + block_size
    x1 = np.arange(w)
    x2 = x1 + block_size

    # Grid 2D
    Y1, X1 = np.meshgrid(y1, x1, indexing='ij')
    Y2, X2 = np.meshgrid(y2, x2, indexing='ij')

    area = float(block_size * block_size)

    # Tổng giá trị trong cửa sổ O(1)
    win_sum = (integral[Y2, X2] - integral[Y1, X2] - integral[Y2, X1] + integral[Y1, X1])
    win_sq_sum = (integral_sq[Y2, X2] - integral_sq[Y1, X2] - integral_sq[Y2, X1] + integral_sq[Y1, X1])

    mean = win_sum / area
    variance = np.maximum(0.0, (win_sq_sum / area) - (mean ** 2))
    std = np.sqrt(variance)

    if method == "sauvola":
        threshold = mean * (1.0 + k * ((std / r) - 1.0))
    else:  # Adaptive Mean
        threshold = mean - 7.0

    binary = np.where(img >= threshold, np.uint8(255), np.uint8(0))
    return binary


