"""
image_processing/connected_components.py - Gán nhãn thành phần liên thông (Connected Component Labeling)
Bao gồm:
- Tự cài đặt thuật toán tìm kiếm theo chiều rộng (BFS - Breadth-First Search) để phát hiện các vùng trắng liên thông.
- Hỗ trợ 8-connectivity (8 hướng) hoặc 4-connectivity (4 hướng).
- Tính toán Bounding Box: (x, y, w, h).
- Tính toán diện tích (Area - số lượng pixel trắng) và mật độ điểm trắng (White Density / Fill Ratio).
- Sử dụng cv2.minAreaRect tính tỷ lệ aspect_ratio thực tế (Oriented Bounding Box - OBB):
  aspect_ratio = max(rw, rh) / min(rw, rh) giúp nhận diện chuẩn xác ngay cả khi biển số bị xoay nghiêng hoặc xoay dọc (45° đến 90°).
- Tạo ảnh màu trực quan hóa các thành phần liên thông phục vụ báo cáo.
"""

from collections import deque
from typing import List, Dict, Any, Tuple
import numpy as np


def find_connected_components(
    image_bin: np.ndarray,
    connectivity: int = 8,
    compute_quad: bool = True
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Gán nhãn thành phần liên thông CCL tối ưu tốc độ cao với cv2.connectedComponentsWithStats
    kèm dự phòng thuật toán BFS chuẩn DIP.
    Sử dụng cv2.minAreaRect để tính tỷ lệ w/h thực tế bất biến với góc xoay:
        aspect_ratio = max(rw, rh) / min(rw, rh)

    Args:
        image_bin: Ảnh nhị phân (mảng numpy 2D uint8/bool).
        connectivity: 8 hoặc 4 hướng lân cận.
        compute_quad: Có tính góc nghiêng và tọa độ 4 đỉnh OBB hay không.

    Returns:
        Tuple (Danh sách các thành phần liên thông, Ma trận bản đồ nhãn 2D int32).
    """
    try:
        import cv2
        from utils.image_utils import order_quad_points

        bin_u8 = (image_bin > 0).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_u8, connectivity=connectivity)

        components = []
        for label_idx in range(1, num_labels):
            box_x = int(stats[label_idx, cv2.CC_STAT_LEFT])
            box_y = int(stats[label_idx, cv2.CC_STAT_TOP])
            box_w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
            box_h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
            pixel_count = int(stats[label_idx, cv2.CC_STAT_AREA])

            bbox_area = box_w * box_h
            density = float(pixel_count) / float(bbox_area) if bbox_area > 0 else 0.0

            # Tỷ lệ Bounding Box nằm ngang mặc định
            box_ar = float(box_w) / float(box_h) if box_h > 0 else 1.0

            quad_corners = None
            angle = 0.0
            obb_ar = box_ar

            if compute_quad and pixel_count >= 15:
                try:
                    roi = (labels[box_y:box_y + box_h, box_x:box_x + box_w] == label_idx)
                    local_y, local_x = np.nonzero(roi)
                    if len(local_x) >= 5:
                        pts = np.column_stack((local_x + box_x, local_y + box_y)).astype(np.int32)
                        rect = cv2.minAreaRect(pts)
                        (cx, cy), (rw, rh), angle = rect

                        # Tính tỷ lệ aspect_ratio thực tế theo OBB:
                        # max(rw, rh) / min(rw, rh) đảm bảo dù biển xoay 90 độ thì tỷ lệ vẫn đạt ~1.2 - 1.5 thay vì < 1.0
                        if min(rw, rh) > 0:
                            obb_ar = float(max(rw, rh)) / float(min(rw, rh))

                        box = cv2.boxPoints(rect)
                        quad_corners = order_quad_points(box)
                except Exception:
                    quad_corners = None

            # Mặc định giữ nguyên tỷ lệ nằm ngang chuẩn cũ (box_ar).
            # BỔ SUNG THÊM: Khi vùng liên thông THỰC SỰ bị xoay nghiêng góc lớn (15° <= |angle| <= 75°),
            # tự động sử dụng tỷ lệ thực tế obb_ar để không bị loại nhầm biển số xe bị nghiêng.
            # Với vật thể đứng thẳng trục (gương xe, ống bô, đèn hậu có angle ~ 0° hoặc ~ -90°), giữ nguyên box_ar < 1.0 để bộ lọc loại bỏ.
            is_truly_tilted = (abs(angle) >= 15.0 and abs(angle + 90.0) >= 15.0 and abs(angle - 90.0) >= 15.0)
            if compute_quad and pixel_count >= 15 and is_truly_tilted:
                actual_ar = obb_ar
            else:
                actual_ar = box_ar

            components.append({
                'id': label_idx,
                'bbox': (box_x, box_y, box_w, box_h),
                'quad_corners': quad_corners,
                'angle': angle,
                'area': pixel_count,
                'bbox_area': bbox_area,
                'density': density,
                'aspect_ratio': actual_ar,
                'obb_aspect_ratio': obb_ar,
                'box_aspect_ratio': box_ar,
            })

        return components, labels

    except Exception:
        # Thuật toán dự phòng tự cài đặt bằng BFS
        h, w = image_bin.shape
        binary_mask = (image_bin > 0)
        visited = np.zeros((h, w), dtype=bool)
        label_map = np.zeros((h, w), dtype=np.int32)

        neighbor_offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ] if connectivity == 8 else [(-1, 0), (0, -1), (0, 1), (1, 0)]

        white_rows, white_cols = np.where(binary_mask)
        components = []
        current_label = 0

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

            while queue:
                r, c = queue.popleft()
                pixel_count += 1

                if r < min_r:
                    min_r = r
                if r > max_r:
                    max_r = r
                if c < min_c:
                    min_c = c
                if c > max_c:
                    max_c = c

                for dr, dc in neighbor_offsets:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if binary_mask[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            label_map[nr, nc] = current_label
                            queue.append((nr, nc))

            box_x = int(min_c)
            box_y = int(min_r)
            box_w = int(max_c - min_c + 1)
            box_h = int(max_r - min_r + 1)

            bbox_area = box_w * box_h
            density = float(pixel_count) / float(bbox_area) if bbox_area > 0 else 0.0

            # Tính aspect_ratio bất biến theo max/min kích thước
            actual_ar = float(max(box_w, box_h)) / float(min(box_w, box_h)) if min(box_w, box_h) > 0 else 1.0

            components.append({
                'id': current_label,
                'bbox': (box_x, box_y, box_w, box_h),
                'quad_corners': None,
                'angle': 0.0,
                'area': pixel_count,
                'bbox_area': bbox_area,
                'density': density,
                'aspect_ratio': actual_ar,
                'obb_aspect_ratio': actual_ar,
                'box_aspect_ratio': float(box_w) / float(box_h) if box_h > 0 else 1.0,
            })

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
