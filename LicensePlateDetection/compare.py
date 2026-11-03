"""
compare.py - Module thực nghiệm so sánh các phương pháp xử lý ảnh số
Phục vụ mục 8 theo yêu cầu đề tài:
1. So sánh bộ lọc Gaussian Filter và Median Filter.
2. So sánh toán tử phát hiện cạnh Sobel (đặc biệt Sobel-X) và Canny.
3. So sánh các kích thước phần tử cấu trúc (Structuring Element) trong phép Closing.
4. So sánh phương pháp nhị phân hóa tự động Otsu và các ngưỡng cố định.
"""

from pathlib import Path
from typing import Union, Optional
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Headless backend an toàn cho đa luồng
import matplotlib.pyplot as plt

from image_processing.preprocessing import (
    rgb_to_grayscale,
    resize_image,
    gaussian_blur,
    median_blur,
)
from image_processing.edge_detection import (
    sobel_x,
    sobel_y,
    canny_edge_detection,
    compute_gradient_magnitude_and_direction,
)
from image_processing.thresholding import (
    otsu_threshold,
    fixed_threshold,
)
from image_processing.morphology import (
    create_structuring_element,
    binary_closing,
    binary_opening,
)
from utils.image_utils import load_image, save_image


def run_filter_comparison(image_gray: np.ndarray, output_path: Path) -> None:
    """
    Thực nghiệm 1: So sánh Gaussian Filter và Median Filter trong việc làm mờ và khử nhiễu.
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    img_gauss_3 = gaussian_blur(image_gray, kernel_size=3, sigma=0.8)
    img_gauss_5 = gaussian_blur(image_gray, kernel_size=5, sigma=1.2)
    img_median_3 = median_blur(image_gray, kernel_size=3)

    plots = [
        (image_gray, "Ảnh gốc mức xám"),
        (img_gauss_3, "Gaussian Blur (k=3, σ=0.8)"),
        (img_gauss_5, "Gaussian Blur (k=5, σ=1.2)"),
        (img_median_3, "Median Filter (k=3)"),
    ]

    for ax, (img, title) in zip(axes, plots):
        ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')

    plt.suptitle("So sánh thực nghiệm: Gaussian Filter vs Median Filter", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [+] Đã lưu so sánh bộ lọc khử nhiễu tại: {output_path.name}")


def run_edge_comparison(image_gray: np.ndarray, output_path: Path) -> None:
    """
    Thực nghiệm 2: So sánh Sobel-X, Sobel-Y, Sobel Magnitude và Canny.
    Làm rõ lý do tại sao Sobel-X là lựa chọn tối ưu nhất cho bài toán biển số xe.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    blurred = gaussian_blur(image_gray, kernel_size=5, sigma=1.0)

    # 1. Sobel-X (Cạnh dọc)
    edge_sobel_x = sobel_x(blurred)

    # 2. Sobel-Y (Cạnh ngang)
    edge_sobel_y = sobel_y(blurred)

    # 3. Sobel Magnitude (Cả 2 phương)
    mag, _ = compute_gradient_magnitude_and_direction(blurred)
    edge_mag = np.clip(mag, 0, 255).astype(np.uint8)

    # 4. Canny
    edge_canny = canny_edge_detection(blurred, low_thresh=40, high_thresh=120)

    plots = [
        (edge_sobel_x, "Sobel-X (Ưu tiên cạnh dọc - Đề xuất chính)"),
        (edge_sobel_y, "Sobel-Y (Cạnh ngang - Dễ bắt viền thân xe)"),
        (edge_mag, "Sobel Gradient Magnitude (Cả hai hướng)"),
        (edge_canny, "Canny Edge Detection (Đường biên mảnh, rời rạc)"),
    ]

    for ax, (img, title) in zip(axes, plots):
        ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')

    plt.suptitle("So sánh thực nghiệm: Sobel-X vs Sobel-Y vs Magnitude vs Canny", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [+] Đã lưu so sánh phát hiện biên tại: {output_path.name}")


def run_structuring_element_comparison(binary_img: np.ndarray, output_path: Path) -> None:
    """
    Thực nghiệm 3: So sánh hiệu quả của các kích thước Structuring Element (SE) trong phép Closing.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    se_configs = [
        (3, 9, "SE Nhỏ (3x9) - Chưa nối kín các ký tự"),
        (3, 19, "SE Chuẩn (3x19) - Tạo khối đặc tối ưu cho biển số"),
        (3, 31, "SE Dài (3x31) - Dễ dính liền các chi tiết lân cận"),
        (7, 19, "SE Cao (7x19) - Lan tỏa theo chiều dọc, tăng nhiễu"),
    ]

    for ax, (h, w, title) in zip(axes, se_configs):
        se = create_structuring_element(height=h, width=w, shape='rect')
        closed = binary_closing(binary_img, se)
        ax.imshow(closed, cmap='gray')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')

    plt.suptitle("So sánh thực nghiệm: Các kích thước Structuring Element (SE) trong phép Closing", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [+] Đã lưu so sánh Structuring Element tại: {output_path.name}")


def run_thresholding_comparison(image_sobel: np.ndarray, output_path: Path) -> None:
    """
    Thực nghiệm 4: So sánh phương pháp xác định ngưỡng Otsu và các ngưỡng cố định.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    otsu_bin, t_star = otsu_threshold(image_sobel)
    fixed_60 = fixed_threshold(image_sobel, threshold=60)
    fixed_120 = fixed_threshold(image_sobel, threshold=120)
    fixed_180 = fixed_threshold(image_sobel, threshold=180)

    plots = [
        (otsu_bin, f"Otsu Tự Thích Nghi (Ngưỡng tự động T*={t_star})"),
        (fixed_60, "Ngưỡng Cố Định Thấp (T=60) - Nhiều nhiễu"),
        (fixed_120, "Ngưỡng Cố Định Trung Bình (T=120)"),
        (fixed_180, "Ngưỡng Cố Định Cao (T=180) - Mất nét ký tự"),
    ]

    for ax, (img, title) in zip(axes, plots):
        ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')

    plt.suptitle("So sánh thực nghiệm: Phân ngưỡng Otsu vs Phân ngưỡng cố định", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [+] Đã lưu so sánh phân ngưỡng tại: {output_path.name}")


def run_all_comparisons(image_path: Union[str, Path], output_dir: Union[str, Path]) -> None:
    """
    Thực hiện toàn bộ 4 thực nghiệm so sánh trên ảnh mẫu và lưu đồ thị vào output_dir.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[*] Bắt đầu chạy chuỗi thực nghiệm so sánh trên ảnh: {Path(image_path).name}")
    orig_pil, orig_np = load_image(image_path)
    resized_pil, _ = resize_image(orig_pil, target_max_dim=800)
    gray = rgb_to_grayscale(np.array(resized_pil))

    # 1. So sánh bộ lọc
    run_filter_comparison(gray, out_dir / "so_sanh_1_bo_loc_gaussian_vs_median.png")

    # 2. So sánh phát hiện cạnh
    run_edge_comparison(gray, out_dir / "so_sanh_2_sobel_vs_canny.png")

    # 3 & 4. Chuẩn bị ảnh Sobel và Nhị phân cho thực nghiệm SE và Thresholding
    blurred = gaussian_blur(gray, kernel_size=5, sigma=1.0)
    sobel_img = sobel_x(blurred)
    otsu_bin, _ = otsu_threshold(sobel_img)

    run_thresholding_comparison(sobel_img, out_dir / "so_sanh_4_phan_nguong_otsu_vs_co_dinh.png")
    run_structuring_element_comparison(otsu_bin, out_dir / "so_sanh_3_kich_thuoc_structuring_element.png")

    print("[*] Hoàn tất 4 bài so sánh thực nghiệm!\n")

