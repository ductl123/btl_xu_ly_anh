"""
utils/image_utils.py - Các hàm hỗ trợ đọc, ghi, cắt và trực quan hóa ảnh
Sử dụng Pillow (PIL) và Matplotlib theo đúng yêu cầu công nghệ bắt buộc.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Union, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

import matplotlib
matplotlib.use('Agg')  # Headless backend an toàn cho đa luồng và tránh xung đột với Tkinter
import matplotlib.pyplot as plt


def order_quad_points(pts):
    """Order a convex quad cyclically, assigning the longer edge to width.

    Keeps winding at near-vertical angles; never pairs corners independently.
    The remaining 180-degree reading direction is resolved by OCR.
    """
    points=np.asarray(pts,dtype=np.float32)
    if points.shape!=(4,2) or not np.isfinite(points).all() or len(np.unique(points,axis=0))!=4:
        raise ValueError("Expected four distinct finite corners")
    center=points.mean(axis=0)
    angles=np.arctan2(points[:,1]-center[1],points[:,0]-center[0])
    cycle=points[np.argsort(angles)]
    edges=np.roll(cycle,-1,axis=0)-cycle
    cross=edges[:,0]*np.roll(edges[:,1],-1)-edges[:,1]*np.roll(edges[:,0],-1)
    if not (np.all(cross>1e-3) or np.all(cross< -1e-3)):
        raise ValueError("Corners must form a convex quadrilateral")
    lengths=np.linalg.norm(edges,axis=1)
    eligible=[0,2] if lengths[[0,2]].mean()>=lengths[[1,3]].mean() else [1,3]
    start=min(eligible,key=lambda i:(float((cycle[i]+cycle[(i+1)%4])[1]),float((cycle[i]+cycle[(i+1)%4])[0])))
    return np.roll(cycle,-start,axis=0).copy()


def load_image(file_path: Union[str, Path]) -> Tuple[Image.Image, np.ndarray]:
    """
    Đọc ảnh từ file bằng Pillow, chuẩn hóa hướng chụp EXIF (P14) và chuyển thành RGB NumPy array.

    Args:
        file_path: Đường dẫn tới file ảnh (JPG, PNG, JPEG, ...).

    Returns:
        Tuple (ảnh gốc PIL Image dạng RGB, mảng NumPy uint8).
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file ảnh: {file_path}")

    with Image.open(path) as img:
        # Chuẩn hóa hướng chụp theo thông tin EXIF (chống xoay ngược trên ảnh điện thoại - P14)
        transposed = ImageOps.exif_transpose(img)
        img_rgb = (transposed if transposed is not None else img).convert("RGB")
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
        quad = cand.get('quad_corners')
        angle = cand.get('angle', 0.0)

        # Chọn màu: Biển 1 dòng màu xanh lá, biển 2 dòng màu vàng cam
        color = "#00FF00" if cand.get("plate_type") == "single_row" else "#FFD700"

        # Nếu có 4 góc nghiêng và góc nghiêng đáng kể (|angle| >= 1.5 độ), vẽ đa giác 4 đỉnh
        if quad is not None and len(quad) == 4 and abs(angle) >= 0.8:
            quad_orig = [(float(pt[0]) / scale_factor, float(pt[1]) / scale_factor) for pt in quad]
            # Vẽ đa giác 4 cạnh bám sát theo độ nghiêng thực tế của biển số
            draw.polygon(quad_orig, outline=color, width=line_width)

            # Tọa độ nhãn đặt ở đỉnh cao nhất
            top_pt = min(quad_orig, key=lambda pt: pt[1])
            text_x = max(0, int(top_pt[0]))
            text_y = max(0, int(top_pt[1]) - 22)
        else:
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

            # Vẽ hình chữ nhật
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
            text_x = x1
            text_y = max(0, y1 - 22)

        # Chuẩn bị nội dung nhãn (hiển thị biển số nhận diện nếu có)
        plate_type_vn = "1-Dong" if cand.get("plate_type") == "single_row" else "2-Dong"
        plate_text_display = cand.get("plate_text", plate_type_vn)
        score = cand.get('score', 0.0)
        label_text = f"#{idx} [{plate_text_display}] AR={cand.get('aspect_ratio', 0):.2f} ({score:.2f})"

        # Vẽ nền nhãn để chữ rõ ràng
        draw.rectangle([text_x, text_y, text_x + len(label_text) * 9, text_y + 20], fill=color)
        draw.text((text_x + 4, text_y + 2), label_text, fill="#000000")

    return canvas


def crop_license_plate(
    original_pil: Image.Image,
    bbox: Tuple[int, int, int, int],
    scale_factor: float = 1.0,
    margin_ratio: float = 0.05,
    quad_corners: Optional[Any] = None,
    trim_bottom_ratio: float = 0.0,
    ordered_quad: bool = False,
    target_aspect_ratio: Optional[float] = None
) -> Image.Image:
    """
    Cắt riêng vùng biển số được phát hiện từ ảnh gốc với độ phân giải cao nhất.
    Nếu phát hiện biển số bị nghiêng (có quad_corners), tự động gọi hàm order_quad_points
    để sắp thứ tự góc theo chu vi rồi nắn bằng phép biến đổi phối cảnh.
    ordered_quad=True giữ các trục đã được xác nhận từ bố cục dòng chữ;
    ảnh có thể còn ngược 180 độ và được kiểm tra ở bước nhận dạng.

    Args:
        original_pil: Ảnh gốc PIL.
        bbox: (x, y, w, h) tính trên tọa độ đã xử lý.
        scale_factor: Tỷ lệ co dãn.
        margin_ratio: Tỷ lệ mở rộng lề lân cận.
        quad_corners: Mảng 4x2 chứa tọa độ 4 góc nghiêng [TL, TR, BR, BL] trên ảnh xử lý.
        trim_bottom_ratio: Tỷ lệ chiều cao ở mép dưới cần cắt bỏ (loại trừ chữ phấn 301, 302, 601).

    Returns:
        Ảnh PIL của vùng biển số đã cắt và nắn phẳng đúng chiều.
    """
    orig_w, orig_h = original_pil.size
    x, y, w, h = bbox

    # 1. CẮT THEO ĐÚNG 4 GÓC NGHIÊNG NẾU CÓ THÔNG TIN QUAD CORNERS
    if quad_corners is not None and len(quad_corners) == 4:
        try:
            import cv2
            corners_orig = np.array(quad_corners, dtype="float32") / scale_factor

            # Giữ thứ tự trục do bộ phát hiện xác nhận; trường hợp còn lại sắp theo chu vi.
            if not ordered_quad:
                corners_orig = order_quad_points(corners_orig)

            # Mở rộng các góc nghiêng ra ngoài theo margin_ratio để không bị cắt xén mép chữ
            if margin_ratio > 0.0:
                center_pt = np.mean(corners_orig, axis=0)
                corners_orig = center_pt + (corners_orig - center_pt) * (1.0 + margin_ratio)

            tl, tr, br, bl = corners_orig

            # Tính chiều dài và chiều cao thực tế của biển nghiêng từ các cạnh chuẩn hóa (P02)
            w_top = np.linalg.norm(tr - tl)
            w_bot = np.linalg.norm(br - bl)
            target_w = int(round(max(w_top, w_bot)))

            h_left = np.linalg.norm(tl - bl)
            h_right = np.linalg.norm(tr - br)
            target_h = int(round(max(h_left, h_right)))
            if target_aspect_ratio is not None:
                if not .5 <= target_aspect_ratio <= 8.:
                    raise ValueError("Invalid target plate aspect ratio")
                target_w=max(20,int(round(target_h*target_aspect_ratio)))

            if target_w >= 20 and target_h >= 15:
                orig_np = np.array(original_pil)
                dst_pts = np.array([
                    [0, 0],
                    [target_w - 1, 0],
                    [target_w - 1, target_h - 1],
                    [0, target_h - 1]
                ], dtype="float32")

                # Tính ma trận biến đổi phối cảnh và warp về hình chữ nhật chuẩn
                M = cv2.getPerspectiveTransform(corners_orig, dst_pts)
                warped_np = cv2.warpPerspective(
                    orig_np, M, (target_w, target_h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE
                )

                # Cắt bỏ mép dưới nếu có yêu cầu loại bỏ chữ phấn/nhiễu dè xe
                if trim_bottom_ratio > 0.0:
                    keep_h = max(10, int(round(target_h * (1.0 - trim_bottom_ratio))))
                    warped_np = warped_np[:keep_h, :]

                return Image.fromarray(warped_np)
        except Exception:
            pass

    # 2. FALLBACK CẮT THEO HỘP CHỮ NHẬT NẰM NGANG
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

    # Cắt bỏ mép dưới nếu có yêu cầu loại bỏ chữ phấn
    if trim_bottom_ratio > 0.0:
        box_h_orig = crop_y2 - crop_y1
        crop_y2 = max(crop_y1 + 10, int(round(crop_y2 - box_h_orig * trim_bottom_ratio)))

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
    if not show_plot:
        plt.close(fig)
