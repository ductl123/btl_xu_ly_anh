"""
image_processing/morphology.py - Xử lý hình thái học (Morphological Operations)
Bao gồm các thuật toán tự cài đặt hoàn toàn bằng NumPy:
- Tạo phần tử cấu trúc (Structuring Element - SE) hình chữ nhật hoặc elip/chữ thập.
- Giãn ảnh nhị phân (Binary Dilation).
- Xói ảnh nhị phân (Binary Erosion).
- Phép Đóng (Morphological Closing): A • B = (A ⊕ B) ⊖ B
  Sử dụng SE hình chữ nhật nằm ngang để gộp các cạnh dọc của ký tự biển số thành một khối đặc liền nét.
- Phép Mở (Morphological Opening): A ∘ B = (A ⊖ B) ⊕ B
  Sử dụng SE nhỏ để triệt tiêu các đốm nhiễu li ti xung quanh.
"""

import numpy as np


def create_structuring_element(height: int, width: int, shape: str = 'rect') -> np.ndarray:
    """
    Tạo ma trận phần tử cấu trúc (Structuring Element - SE):
    - 'rect': Hình chữ nhật (toàn số 1)
    - 'cross': Hình chữ thập
    - 'ellipse': Hình elip xấp xỉ

    Args:
        height: Chiều cao (số hàng, thường là số lẻ).
        width: Chiều rộng (số cột, thường là số lẻ).
        shape: Kiểu hình dạng ('rect', 'cross', 'ellipse').

    Returns:
        Mảng 2D boolean biểu diễn phần tử cấu trúc.
    """
    if shape == 'rect':
        return np.ones((height, width), dtype=bool)

    se = np.zeros((height, width), dtype=bool)
    center_r = height // 2
    center_c = width // 2

    if shape == 'cross':
        se[center_r, :] = True
        se[:, center_c] = True
    elif shape == 'ellipse':
        r_radius = max(1, height / 2.0)
        c_radius = max(1, width / 2.0)
        for r in range(height):
            for c in range(width):
                norm_dist = ((r - center_r) / r_radius) ** 2 + ((c - center_c) / c_radius) ** 2
                if norm_dist <= 1.0:
                    se[r, c] = True
    else:
        # Mặc định hình chữ nhật nếu không khớp
        se = np.ones((height, width), dtype=bool)

    return se


def binary_dilation(image_bin: np.ndarray, se: np.ndarray) -> np.ndarray:
    """
    Phép Giãn (Dilation) ảnh nhị phân:
    Định nghĩa toán học: A ⊕ B = U_{b in B} (A + b)
    Cài đặt thuần NumPy vectorization bằng cách trượt lát cắt (slices) theo từng tọa độ tích cực của SE.
    Nhanh gấp hàng trăm lần vòng lặp lồng pixel-by-pixel.

    Args:
        image_bin: Ảnh nhị phân (2D uint8 chứa 0/255 hoặc mảng boolean).
        se: Mảng 2D boolean phần tử cấu trúc.

    Returns:
        Ảnh sau phép giãn (uint8 với 0 và 255).
    """
    bool_img = (image_bin > 0)
    kh, kw = se.shape
    pad_h = kh // 2
    pad_w = kw // 2

    # Đệm biên 0
    padded = np.pad(bool_img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=False)
    h, w = bool_img.shape

    output = np.zeros((h, w), dtype=bool)

    # Duyệt qua các tọa độ mà phần tử cấu trúc = True
    r_indices, c_indices = np.where(se)
    for r, c in zip(r_indices, c_indices):
        dr = r - pad_h
        dc = c - pad_w
        output |= padded[pad_h + dr : pad_h + dr + h, pad_w + dc : pad_w + dc + w]

    return (output.astype(np.uint8) * 255)


def binary_erosion(image_bin: np.ndarray, se: np.ndarray) -> np.ndarray:
    """
    Phép Xói (Erosion) ảnh nhị phân:
    Định nghĩa toán học: A ⊖ B = {z | (B)_z subseteq A}
    Cài đặt thuần NumPy: Khởi tạo toàn True, sau đó AND với tất cả các phép dịch chuyển tương ứng.

    Args:
        image_bin: Ảnh nhị phân (2D uint8 chứa 0/255 hoặc mảng boolean).
        se: Mảng 2D boolean phần tử cấu trúc.

    Returns:
        Ảnh sau phép xói (uint8 với 0 và 255).
    """
    bool_img = (image_bin > 0)
    kh, kw = se.shape
    pad_h = kh // 2
    pad_w = kw // 2

    # Đệm biên 0
    padded = np.pad(bool_img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=False)
    h, w = bool_img.shape

    output = np.ones((h, w), dtype=bool)

    r_indices, c_indices = np.where(se)
    for r, c in zip(r_indices, c_indices):
        # Điểm A tại (r0, c0) phải khớp với phần tử đối ngẫu
        dr = r - pad_h
        dc = c - pad_w
        output &= padded[pad_h + dr : pad_h + dr + h, pad_w + dc : pad_w + dc + w]

    return (output.astype(np.uint8) * 255)


def binary_closing(image_bin: np.ndarray, se: np.ndarray) -> np.ndarray:
    """
    Phép Đóng (Closing):
    A • B = (A ⊕ B) ⊖ B (Giãn trước, xói sau)

    Tác dụng trong bài toán biển số:
    Các ký tự biển số sau khi bắt cạnh Sobel sẽ gồm nhiều nét dọc đứt quãng.
    Khi sử dụng Structuring Element hình chữ nhật nằm ngang (ví dụ 3x19), phép Closing sẽ
    nối liền các nét thẳng đứng lại với nhau, lấp kín các khoảng hở giữa các ký tự,
    biến toàn bộ vùng biển số thành một khối đặc duy nhất.
    """
    dilated = binary_dilation(image_bin, se)
    closed = binary_erosion(dilated, se)
    return closed


def binary_opening(image_bin: np.ndarray, se: np.ndarray) -> np.ndarray:
    """
    Phép Mở (Opening):
    A ∘ B = (A ⊖ B) ⊕ B (Xói trước, giãn sau)

    Tác dụng trong bài toán biển số:
    Sau khi closing, có thể xuất hiện các vệt nhiễu nhỏ hoặc đường nét mảnh trên thân xe, nền đường.
    Phép Opening với phần tử cấu trúc nhỏ (ví dụ 3x3) sẽ xóa bỏ hoàn toàn các chấm nhiễu này
    mà không làm thay đổi đáng kể hình dạng khối đặc của biển số xe.
    """
    eroded = binary_erosion(image_bin, se)
    opened = binary_dilation(eroded, se)
    return opened

