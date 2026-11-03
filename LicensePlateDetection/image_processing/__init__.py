"""
Package image_processing - Cung cấp toàn bộ các thuật toán xử lý ảnh số tự cài đặt.
Không sử dụng các hàm có sẵn của OpenCV/scikit-image/Deep Learning.
"""

from .preprocessing import rgb_to_grayscale, resize_image, gaussian_blur, median_blur
from .edge_detection import sobel_x, sobel_y, canny_edge_detection
from .thresholding import otsu_threshold, fixed_threshold
from .morphology import (
    create_structuring_element,
    binary_dilation,
    binary_erosion,
    binary_closing,
    binary_opening
)
from .connected_components import find_connected_components
from .candidate_filter import filter_license_plate_candidates

__all__ = [
    "rgb_to_grayscale",
    "resize_image",
    "gaussian_blur",
    "median_blur",
    "sobel_x",
    "sobel_y",
    "canny_edge_detection",
    "otsu_threshold",
    "fixed_threshold",
    "create_structuring_element",
    "binary_dilation",
    "binary_erosion",
    "binary_closing",
    "binary_opening",
    "find_connected_components",
    "filter_license_plate_candidates",
]

