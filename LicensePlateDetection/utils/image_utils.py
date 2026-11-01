"""
utils/image_utils.py - Các hàm hỗ trợ đọc, ghi, cắt và trực quan hóa ảnh
Sử dụng Pillow (PIL) và Matplotlib theo đúng yêu cầu công nghệ bắt buộc.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Union, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


def load_image(file_path: Union[str, Path]) -> Tuple[Image.Image, np.ndarray]:
    """
    Đọc ảnh từ file bằng Pillow và chuyển thành RGB NumPy array.

    Args:
        file_path: Đường dẫn tới file ảnh (JPG, PNG, JPEG, ...).

    Returns:
        Tuple (ảnh gốc PIL Image dạng RGB, mảng NumPy uint8).
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file ảnh: {file_path}")

    with Image.open(path) as img:
        img_rgb = img.convert("RGB")
        img_np = np.array(img_rgb)
        return img_rgb.copy(), img_np


def save_image(image_data: Union[Image.Image, np.ndarray], output_path: Union[str, Path]) -> None:
    """
    Lưu ảnh vào đường dẫn đích. Hỗ trợ cả PIL Image và NumPy array (2D hoặc 3D).

    Args:
        image_data: Đối tượng PIL Image hoặc mảng NumPy (uint8 / bool / float).
        output_path: Đường dẫn file xuất.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(image_data, Image.Image):
        image_data.save(path)
        return

    # Nếu là NumPy array
    arr = image_data
    if arr.dtype == bool:
        arr = (arr.astype(np.uint8) * 255)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        pil_img = Image.fromarray(arr, mode='L')
    elif arr.ndim == 3 and arr.shape[2] == 3:
        pil_img = Image.fromarray(arr, mode='RGB')
    elif arr.ndim == 3 and arr.shape[2] == 4:
        pil_img = Image.fromarray(arr, mode='RGBA')
    else:
        raise ValueError(f"Định dạng mảng NumPy không hợp lệ để lưu ảnh: shape={arr.shape}")

    pil_img.save(path)


def draw_bounding_boxes(
    original_pil: Image.Image,
    candidates: List[Dict[str, Any]],
    scale_factor: float = 1.0,
    outline_color: str = "#00FF00",
    line_width: int = 4
) -> Image.Image:
    """
    Vẽ Bounding Box quanh các biển số phát hiện được trên ảnh gốc.
    Quy đổi tọa độ từ ảnh thu nhỏ về kích thước ảnh gốc thông qua scale_factor.

    Args:
        original_pil: Ảnh gốc kích thước đầy đủ (PIL Image).
        candidates: Danh sách các ứng viên biển số từ candidate_filter.
        scale_factor: Tỷ lệ co dãn (new_size / orig_size) đã dùng ở bước resize.
        outline_color: Màu viền hộp bao (mặc định xanh lục sáng).
        line_width: Độ dày nét vẽ viền.

    Returns:
        Bản sao ảnh gốc có vẽ bounding box và nhãn chú thích.
    """
    canvas = original_pil.copy()
    draw = ImageDraw.Draw(canvas)
    orig_w, orig_h = original_pil.size

    for idx, cand in enumerate(candidates, start=1):
        x, y, w, h = cand['bbox']

        # Quy đổi ngược lại tọa độ gốc
        orig_x = int(round(x / scale_factor))
        orig_y = int(round(y / scale_factor))
        orig_w_box = int(round(w / scale_factor))
        orig_h_box = int(round(h / scale_factor))

        # Đảm bảo không vượt quá biên ảnh
        x1 = max(0, min(orig_x, orig_w - 1))
        y1 = max(0, min(orig_y, orig_h - 1))
        x2 = max(0, min(orig_x + orig_w_box, orig_w - 1))
        y2 = max(0, min(orig_y + orig_h_box, orig_h - 1))

        # Chọn màu: Biển 1 dòng màu xanh lá, biển 2 dòng màu vàng cam
        color = "#00FF00" if cand.get("plate_type") == "single_row" else "#FFD700"

        # Vẽ hình chữ nhật
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        # Chuẩn bị nội dung nhãn (hiển thị biển số nhận diện nếu có)
        plate_type_vn = "1-Dong" if cand.get("plate_type") == "single_row" else "2-Dong"
        plate_text_display = cand.get("plate_text", plate_type_vn)
        score = cand.get('score', 0.0)
        label_text = f"#{idx} [{plate_text_display}] AR={cand.get('aspect_ratio', 0):.2f} ({score:.2f})"

        # Vẽ nền nhãn để chữ rõ ràng
        text_y = max(0, y1 - 22)
        draw.rectangle([x1, text_y, x1 + len(label_text) * 9, text_y + 20], fill=color)
        draw.text((x1 + 4, text_y + 2), label_text, fill="#000000")

    return canvas


def crop_license_plate(
    original_pil: Image.Image,
    bbox: Tuple[int, int, int, int],
    scale_factor: float = 1.0,
    margin_ratio: float = 0.05
) -> Image.Image:
    """
    Cắt riêng vùng biển số được phát hiện từ ảnh gốc với độ phân giải cao nhất,
    có thêm một lề đệm nhỏ (margin) để không bị cắt xén sát mép ký tự.

    Args:
        original_pil: Ảnh gốc PIL.
        bbox: (x, y, w, h) tính trên tọa độ đã xử lý.
        scale_factor: Tỷ lệ co dãn.
        margin_ratio: Tỷ lệ mở rộng lề lân cận.

    Returns:
        Ảnh PIL của vùng biển số đã cắt.
    """
    orig_w, orig_h = original_pil.size
    x, y, w, h = bbox

    # Tọa độ thực trên ảnh gốc
    rx = x / scale_factor
    ry = y / scale_factor
    rw = w / scale_factor
    rh = h / scale_factor

    # Thêm lề mở rộng
    margin_x = rw * margin_ratio
    margin_y = rh * margin_ratio

    crop_x1 = max(0, int(round(rx - margin_x)))
    crop_y1 = max(0, int(round(ry - margin_y)))
    crop_x2 = min(orig_w, int(round(rx + rw + margin_x)))
    crop_y2 = min(orig_h, int(round(ry + rh + margin_y)))

    cropped = original_pil.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    return cropped


def create_pipeline_summary_grid(
    steps_dict: Dict[str, Any],
    detected_pil: Image.Image,
    candidates: List[Dict[str, Any]],
    output_path: Union[str, Path],
    show_plot: bool = False
) -> None:
    """
    Sử dụng Matplotlib vẽ lưới trực quan hóa tiến trình 8 giai đoạn từ đầu vào đến đầu ra:
    1. Ảnh màu gốc
    2. Ảnh xám (Grayscale)
    3. Làm mờ Gaussian (Gaussian Blur)
    4. Gradient Sobel-X
    5. Nhị phân hóa Otsu
    6. Phép Đóng (Morphological Closing)
    7. Phép Mở (Morphological Opening)
    8. Kết quả phát hiện Bounding Box trên ảnh gốc
    """
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    titles = [
        "1. Ảnh màu gốc",
        "2. Ảnh mức xám (Grayscale)",
        f"3. Lọc Gaussian (k={steps_dict.get('gaussian_k', 5)})",
        "4. Cạnh dọc Sobel-X",
        f"5. Nhị phân Otsu (T={steps_dict.get('otsu_t', 0)})",
        f"6. Phép Đóng (Closing {steps_dict.get('closing_se', '')})",
        f"7. Phép Mở (Opening {steps_dict.get('opening_se', '')})",
        f"8. Kết quả ({len(candidates)} biển số)"
    ]

    images = [
        steps_dict.get('original_resized'),
        steps_dict.get('grayscale'),
        steps_dict.get('blurred'),
        steps_dict.get('sobel_x'),
        steps_dict.get('binary'),
        steps_dict.get('closing'),
        steps_dict.get('opening'),
        detected_pil
    ]

    cmaps = [
        None, 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', None
    ]

    for ax, title, img, cmap in zip(axes, titles, images, cmaps):
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
        if img is not None:
            ax.imshow(img, cmap=cmap)
        ax.axis('off')

    plt.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200, bbox_inches='tight')

    if show_plot:
        plt.show()

    plt.close(fig)

