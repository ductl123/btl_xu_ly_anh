"""
image_processing/connected_components.py - Gán nhãn thành phần liên thông (Connected Component Labeling)
Bao gồm:
- Tự cài đặt thuật toán tìm kiếm theo chiều rộng (BFS - Breadth-First Search) để phát hiện các vùng trắng liên thông.
- Hỗ trợ 8-connectivity (8 hướng) hoặc 4-connectivity (4 hướng).
- Tính toán Bounding Box: (x, y, w, h).
- Tính toán diện tích (Area - số lượng pixel trắng) và mật độ điểm trắng (White Density / Fill Ratio).
- Tạo ảnh màu trực quan hóa các thành phần liên thông phục vụ báo cáo.
"""

from collections import deque
from typing import List, Dict, Any, Tuple
import numpy as np


def find_connected_components(image_bin: np.ndarray, connectivity: int = 8) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Thuật toán Connected Component Labeling tự cài đặt bằng BFS:
    Quét qua các điểm ảnh tiền cảnh (màu trắng > 0), lan truyền theo 8 (hoặc 4) lân cận
    để gom thành từng đối tượng riêng biệt.

    Args:
        image_bin: Ảnh nhị phân (2D uint8 với giá trị 0 và 255).
        connectivity: Số hướng liên thông (8 hoặc 4).

    Returns:
        Tuple gồm:
        - Danh sách các dictionary chứa thông tin từng thành phần:
          {
              'id': int,
              'bbox': (x, y, w, h), # Tọa độ góc trên trái (x, y) và kích thước (w, h)
              'area': int,          # Số lượng pixel trắng thực tế
              'bbox_area': int,     # Diện tích hộp bao (w * h)
              'density': float,     # Mật độ lấp đầy (area / bbox_area)
              'aspect_ratio': float # Tỷ lệ khung hình (w / h)
          }
        - Ma trận nhãn label_map (2D int32), mỗi pixel mang ID của thành phần tương ứng (0 là nền).
    """
    h, w = image_bin.shape
    binary_mask = (image_bin > 0)
    visited = np.zeros((h, w), dtype=bool)
    label_map = np.zeros((h, w), dtype=np.int32)

    # Định nghĩa các hướng dịch chuyển lân cận
    if connectivity == 8:
        neighbor_offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
    else:
        neighbor_offsets = [
            (-1, 0),
            (0, -1), (0, 1),
            (1, 0)
        ]

    # Tìm danh sách tất cả các điểm trắng
    white_rows, white_cols = np.where(binary_mask)
    components: List[Dict[str, Any]] = []
    current_label = 0

    # Duyệt qua các điểm trắng chưa thăm
    for start_r, start_c in zip(white_rows, white_cols):
        if visited[start_r, start_c]:
            continue

        current_label += 1
        queue = deque([(start_r, start_c)])
        visited[start_r, start_c] = True
        label_map[start_r, start_c] = current_label

        min_r, max_r = start_r, start_r
        min_c, max_c = start_c, start_c
        pixel_count = 0

        # Lan truyền BFS
        while queue:
            r, c = queue.popleft()
            pixel_count += 1

            if r < min_r: min_r = r
            if r > max_r: max_r = r
            if c < min_c: min_c = c
            if c > max_c: max_c = c

            for dr, dc in neighbor_offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if binary_mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        label_map[nr, nc] = current_label
                        queue.append((nr, nc))

        # Tính toán thông số hộp bao
        box_x = int(min_c)
        box_y = int(min_r)
        box_w = int(max_c - min_c + 1)
        box_h = int(max_r - min_r + 1)
        bbox_area = box_w * box_h
        density = float(pixel_count) / float(bbox_area) if bbox_area > 0 else 0.0
        aspect_ratio = float(box_w) / float(box_h) if box_h > 0 else 0.0

        comp_info = {
            'id': current_label,
            'bbox': (box_x, box_y, box_w, box_h),
            'area': pixel_count,
            'bbox_area': bbox_area,
            'density': density,
            'aspect_ratio': aspect_ratio,
        }
        components.append(comp_info)

    return components, label_map


def create_colored_label_image(label_map: np.ndarray, num_labels: int) -> np.ndarray:
    """
    Tạo ảnh màu RGB từ bản đồ nhãn liên thông để trực quan hóa các vùng khác nhau bằng các màu ngẫu nhiên.
    """
    h, w = label_map.shape
    if num_labels == 0:
        return np.zeros((h, w, 3), dtype=np.uint8)

    # Sinh bảng màu giả lập (bỏ màu 0 là nền đen)
    np.random.seed(42)  # Cố định seed để màu ổn định giữa các lần chạy
    colors = np.random.randint(50, 256, size=(num_labels + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Nền màu đen

    colored = colors[label_map]
    return colored

