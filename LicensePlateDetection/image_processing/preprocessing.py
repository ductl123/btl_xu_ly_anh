"""
image_processing/preprocessing.py - Tiền xử lý ảnh
Bao gồm:
- Chuyển đổi RGB sang ảnh Grayscale bằng công thức phối hợp quang độ.
- Chuẩn hóa kích thước giữ nguyên tỷ lệ (aspect ratio).
- Tự cài đặt tích chập (Convolution 2D / Separable 1D) bằng NumPy.
- Tự cài đặt bộ lọc Gaussian (Gaussian Blur) để khử nhiễu.
- Tự cài đặt bộ lọc trung vị (Median Filter) phục vụ so sánh thực nghiệm.
"""

from typing import Tuple
import numpy as np
from PIL import Image


def rgb_to_grayscale(image_np: np.ndarray) -> np.ndarray:
    """
    Chuyển đổi ảnh màu RGB sang ảnh mức xám Grayscale theo công thức ITU-R 601-2:
    Y = 0.299*R + 0.587*G + 0.114*B

    Args:
        image_np: Mảng NumPy kích thước (H, W, 3) hoặc (H, W, 4) chứa giá trị pixel [0, 255].

    Returns:
        Mảng NumPy 2 chiều (H, W) kiểu dữ liệu uint8 với giá trị trong đoạn [0, 255].
    """
    if image_np.ndim == 2:
        return image_np.astype(np.uint8)

    # Lấy 3 kênh màu đầu tiên (bỏ qua kênh Alpha nếu là RGBA)
    rgb = image_np[..., :3].astype(np.float32)
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gray = np.dot(rgb, weights)
    return np.clip(np.round(gray), 0, 255).astype(np.uint8)


def enhance_contrast_dynamic(image_np: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    """
    Kéo giãn độ tương phản động theo phân vị (Percentile Dynamic Stretching).
    Xử lý đặc biệt cho:
    - Ảnh chụp tối (thiếu sáng/underexposed): Nâng sáng và làm rõ các chi tiết chìm.
    - Ảnh chụp loá (cháy sáng/glare/specular reflection): Cắt bỏ điểm chói cực đoan, chống bão hòa.
    
    Args:
        image_np: Mảng 2D mức xám uint8.
        p_low: Phân vị dưới (mặc định 2%).
        p_high: Phân vị trên (mặc định 98%).
        
    Returns:
        Mảng 2D uint8 với dải động trải đều trên [0, 255].
    """
    v_min, v_max = np.percentile(image_np, (p_low, p_high))
    if v_max <= v_min:
        return image_np.copy()
    stretched = np.clip((image_np.astype(np.float32) - v_min) / (v_max - v_min) * 255.0, 0, 255)
    return stretched.astype(np.uint8)



def resize_image(image_pil: Image.Image, target_max_dim: int = 900) -> Tuple[Image.Image, float]:
    """
    Chuẩn hóa kích thước ảnh nếu kích thước vượt quá target_max_dim.
    Giữ nguyên tỷ lệ khung hình (aspect ratio).

    Args:
        image_pil: Đối tượng PIL Image.
        target_max_dim: Chiều dài tối đa cho phép của cạnh dài nhất.

    Returns:
        Tuple gồm (ảnh PIL sau khi resize, tỷ lệ scale_factor = new_size / old_size).
    """
    orig_w, orig_h = image_pil.size
    max_dim = max(orig_w, orig_h)

    if max_dim <= target_max_dim:
        return image_pil.copy(), 1.0

    scale_factor = target_max_dim / float(max_dim)
    new_w = max(1, int(round(orig_w * scale_factor)))
    new_h = max(1, int(round(orig_h * scale_factor)))

    # Sử dụng bộ lấy mẫu chuẩn LANCZOS (hoặc BILINEAR) từ Pillow
    resized = image_pil.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    return resized, scale_factor


def create_gaussian_kernel_1d(size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Tạo nhân lọc Gaussian 1 chiều:
    G(x) = (1 / (sqrt(2*pi)*sigma)) * exp(-x^2 / (2*sigma^2))
    và chuẩn hóa tổng các trọng số bằng 1.
    """
    if size % 2 == 0:
        raise ValueError("Kích thước kernel phải là số lẻ (3, 5, 7, ...).")

    radius = size // 2
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel


def convolve_1d(image: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """
    Tích chập 1 chiều (Separable 1D convolution) dọc theo trục chỉ định (axis=0 cho hàng, axis=1 cho cột).
    Sử dụng kỹ thuật vectorization của NumPy với biên 'edge' (replicate padding).
    """
    k_len = len(kernel)
    pad_size = k_len // 2

    # Đệm biên dạng phản chiếu hoặc lặp biên để không làm tối góc ảnh
    if axis == 0:
        pad_width = ((pad_size, pad_size), (0, 0))
    else:
        pad_width = ((0, 0), (pad_size, pad_size))

    padded = np.pad(image.astype(np.float32), pad_width, mode='edge')
    output = np.zeros_like(image, dtype=np.float32)

    # Tích chập bằng cách trượt lát cắt (slices)
    h, w = image.shape
    for i, w_coeff in enumerate(kernel):
        if axis == 0:
            output += w_coeff * padded[i : i + h, :]
        else:
            output += w_coeff * padded[:, i : i + w]

    return output


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Tích chập 2D tổng quát bằng NumPy (dùng cho kernel không khả tách).
    """
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(image.astype(np.float32), ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    output = np.zeros_like(image, dtype=np.float32)

    h, w = image.shape
    for r in range(kh):
        for c in range(kw):
            w_coeff = kernel[r, c]
            if w_coeff != 0:
                output += w_coeff * padded[r : r + h, c : c + w]

    return output


def gaussian_blur(image_gray: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Làm mờ ảnh bằng bộ lọc Gaussian 2D phân rã (Separable 2D Gaussian Blur):
    Thực hiện lọc 1D theo phương ngang rồi lọc 1D theo phương dọc.
    Độ phức tạp giảm từ O(K^2 * N) xuống O(2 * K * N), tối ưu hiệu năng tối đa bằng pure NumPy.

    Args:
        image_gray: Ảnh mức xám 2D (H, W).
        kernel_size: Kích thước ma trận lọc (số lẻ, ví dụ 5).
        sigma: Độ lệch chuẩn của phân phối Gauss.

    Returns:
        Ảnh xám sau khi khử nhiễu (H, W), kiểu uint8.
    """
    k1d = create_gaussian_kernel_1d(size=kernel_size, sigma=sigma)
    # Lọc theo phương ngang (trục 1)
    blurred_h = convolve_1d(image_gray, k1d, axis=1)
    # Lọc theo phương dọc (trục 0)
    blurred_v = convolve_1d(blurred_h, k1d, axis=0)

    return np.clip(np.round(blurred_v), 0, 255).astype(np.uint8)


def median_blur(image_gray: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Bộ lọc trung vị (Median Filter) tự cài đặt:
    Thay thế mỗi pixel bằng giá trị trung vị của lân cận kích thước kernel_size x kernel_size.
    Rất hiệu quả trong khử nhiễu muối tiêu (salt-and-pepper noise).
    Dùng cho phần so sánh thực nghiệm.
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size phải là số lẻ.")

    pad = kernel_size // 2
    padded = np.pad(image_gray, ((pad, pad), (pad, pad)), mode='edge')
    h, w = image_gray.shape

    # Thu thập tất cả các cửa sổ con bằng NumPy stride / sliding window
    # Với kernel nhỏ (3x3), xếp chồng các lát cắt (slices)
    patches = []
    for r in range(kernel_size):
        for c in range(kernel_size):
            patches.append(padded[r : r + h, c : c + w])

    # patches stack: shape (kernel_size*kernel_size, H, W)
    stacked = np.stack(patches, axis=0)
    # Tính trung vị theo trục 0
    median_val = np.median(stacked, axis=0)
    return np.clip(np.round(median_val), 0, 255).astype(np.uint8)

