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

    numerator = (mu_total * omega[valid_mask] - mu_accum[valid_mask]) ** 2
    denominator = omega[valid_mask] * (1.0 - omega[valid_mask])
    variance_between[valid_mask] = numerator / denominator

    # 6. Tìm ngưỡng tối ưu t*
    t_star = int(np.argmax(variance_between))

    # 7. Nhị phân hóa: pixel >= t_star gán 255 (trắng), ngược lại gán 0 (đen)
    binary_img = np.where(img_uint8 >= t_star, np.uint8(255), np.uint8(0))

    return binary_img, t_star


def fixed_threshold(image_gray: np.ndarray, threshold: int = 120) -> np.ndarray:
    """
    Phân ngưỡng cố định (Fixed Thresholding):
    Các pixel >= threshold chuyển thành 255, nhỏ hơn chuyển thành 0.
    Dùng để so sánh với phương pháp Otsu tự thích nghi.
    """
    binary_img = np.where(image_gray >= threshold, np.uint8(255), np.uint8(0))
    return binary_img

