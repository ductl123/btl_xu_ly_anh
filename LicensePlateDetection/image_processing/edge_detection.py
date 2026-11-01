"""
image_processing/edge_detection.py - Phát hiện cạnh
Bao gồm:
- Tự cài đặt toán tử Sobel theo phương X (đạo hàm bậc nhất theo phương ngang để làm nổi bật cạnh dọc của ký tự biển số xe).
- Tự cài đặt toán tử Sobel theo phương Y.
- Chuẩn hóa trị tuyệt đối về thang độ xám [0, 255].
- Tự cài đặt thuật toán Canny (Gaussian -> Gradient -> NMS -> Hysteresis) để so sánh thực nghiệm.
"""

from typing import Tuple
import numpy as np
from .preprocessing import convolve2d, gaussian_blur


def sobel_x(image_gray: np.ndarray) -> np.ndarray:
    """
    Toán tử Sobel theo phương X (Sobel-X operator):
    Dùng mặt nạ đạo hàm bậc nhất theo trục x:
        K_x = [[-1,  0,  1],
               [-2,  0,  2],
               [-1,  0,  1]]

    Lý do trọng tâm của đề tài:
    Các ký tự trên biển số xe (số và chữ cái) tạo ra sự tương phản cực lớn theo phương thẳng đứng.
    Đạo hàm theo x bắt chính xác các cạnh thẳng đứng này và làm suy giảm các đường ngang (thân xe, vạch kẻ đường ngang).

    Args:
        image_gray: Ảnh mức xám (H, W).

    Returns:
        Ảnh gradient dọc sau khi lấy trị tuyệt đối và chuẩn hóa về [0, 255], kiểu uint8.
    """
    kernel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    grad_x = convolve2d(image_gray.astype(np.float32), kernel_x)
    abs_grad_x = np.abs(grad_x)

    # Chuẩn hóa về [0, 255]
    max_val = np.max(abs_grad_x)
    if max_val > 0:
        normalized = (abs_grad_x / max_val) * 255.0
    else:
        normalized = abs_grad_x

    return np.clip(np.round(normalized), 0, 255).astype(np.uint8)


def sobel_y(image_gray: np.ndarray) -> np.ndarray:
    """
    Toán tử Sobel theo phương Y:
        K_y = [[-1, -2, -1],
               [ 0,  0,  0],
               [ 1,  2,  1]]
    """
    kernel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float32)

    grad_y = convolve2d(image_gray.astype(np.float32), kernel_y)
    abs_grad_y = np.abs(grad_y)

    max_val = np.max(abs_grad_y)
    if max_val > 0:
        normalized = (abs_grad_y / max_val) * 255.0
    else:
        normalized = abs_grad_y

    return np.clip(np.round(normalized), 0, 255).astype(np.uint8)


def compute_gradient_magnitude_and_direction(image_gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tính độ lớn gradient (magnitude) và góc hướng gradient (direction tính theo độ [0, 180)).
    """
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    gx = convolve2d(image_gray.astype(np.float32), kernel_x)
    gy = convolve2d(image_gray.astype(np.float32), kernel_y)

    magnitude = np.hypot(gx, gy)
    max_mag = np.max(magnitude)
    if max_mag > 0:
        magnitude = (magnitude / max_mag) * 255.0

    angle = np.rad2deg(np.arctan2(gy, gx)) % 180.0
    return magnitude, angle


def non_maximum_suppression(magnitude: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """
    Bước 2 trong Canny: Triệt tiêu không cực đại (Non-Maximum Suppression - NMS)
    Làm mảnh các cạnh bằng cách chỉ giữ lại pixel có độ lớn cực đại cục bộ theo hướng gradient.
    """
    h, w = magnitude.shape
    nms = np.zeros((h, w), dtype=np.float32)

    # Lượng tử hóa góc thành 4 hướng chính: 0°, 45°, 90°, 135°
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            theta = angle[r, c]
            val = magnitude[r, c]

            # Hướng 0° (ngang) -> so sánh lân cận trái phải
            if (0 <= theta < 22.5) or (157.5 <= theta <= 180):
                p1, p2 = magnitude[r, c - 1], magnitude[r, c + 1]
            # Hướng 45° (chéo chính)
            elif 22.5 <= theta < 67.5:
                p1, p2 = magnitude[r - 1, c + 1], magnitude[r + 1, c - 1]
            # Hướng 90° (thẳng đứng)
            elif 67.5 <= theta < 112.5:
                p1, p2 = magnitude[r - 1, c], magnitude[r + 1, c]
            # Hướng 135° (chéo phụ)
            else:
                p1, p2 = magnitude[r - 1, c - 1], magnitude[r + 1, c + 1]

            if val >= p1 and val >= p2:
                nms[r, c] = val
            else:
                nms[r, c] = 0

    return nms


def double_threshold_and_hysteresis(nms: np.ndarray, low_thresh: float, high_thresh: float) -> np.ndarray:
    """
    Bước 3 & 4 trong Canny: Phân ngưỡng kép và nối cạnh bằng vết trễ (Hysteresis Tracking).
    """
    h, w = nms.shape
    strong = 255
    weak = 75

    output = np.zeros((h, w), dtype=np.uint8)
    strong_rows, strong_cols = np.where(nms >= high_thresh)
    weak_rows, weak_cols = np.where((nms >= low_thresh) & (nms < high_thresh))

    output[strong_rows, strong_cols] = strong
    output[weak_rows, weak_cols] = weak

    # Vết trễ: nếu điểm weak nằm cạnh điểm strong thì giữ lại thành strong
    from collections import deque
    queue = deque(zip(strong_rows, strong_cols))

    while queue:
        r, c = queue.popleft()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if output[nr, nc] == weak:
                        output[nr, nc] = strong
                        queue.append((nr, nc))

    # Những điểm weak còn sót lại mà không nối với strong thì loại bỏ (về 0)
    output[output == weak] = 0
    return output


def canny_edge_detection(image_gray: np.ndarray, low_thresh: float = 50.0, high_thresh: float = 150.0) -> np.ndarray:
    """
    Thuật toán phát hiện biên Canny hoàn chỉnh tự viết bằng NumPy:
    1. Làm mờ nhẹ bằng Gaussian
    2. Tính gradient magnitude & direction
    3. Triệt tiêu không cực đại (NMS)
    4. Ngưỡng kép và nối cạnh (Hysteresis)

    Dùng cho phần so sánh thực nghiệm (Sobel-X vs Canny) theo đúng yêu cầu đề tài.
    """
    smoothed = gaussian_blur(image_gray, kernel_size=5, sigma=1.0)
    magnitude, angle = compute_gradient_magnitude_and_direction(smoothed)
    nms = non_maximum_suppression(magnitude, angle)
    edges = double_threshold_and_hysteresis(nms, low_thresh=low_thresh, high_thresh=high_thresh)
    return edges

