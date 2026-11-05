"""
image_processing/archive_detector_adapter.py - Bộ đề xuất ứng viên từ mô hình CNN đã huấn luyện
Đóng vai trò là nguồn ứng viên thứ 2 (Secondary Proposal Source) kết hợp cùng nhánh DIP truyền thống.
Hỗ trợ cả suy luận toàn ảnh và Overlapping Tiling cho các biển ở xa trong ảnh độ phân giải cao.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np
import torch

_CACHED_DETECTOR = None
_CACHED_CHECKPOINT = None


def get_detector_model(model_path: Optional[Path] = None):
    """Nạp mô hình archive detector (cached singleton)."""
    global _CACHED_DETECTOR, _CACHED_CHECKPOINT
    if _CACHED_DETECTOR is not None:
        return _CACHED_DETECTOR, _CACHED_CHECKPOINT

    if model_path is None:
        model_path = Path(__file__).resolve().parents[1] / "models" / "archive_detector" / "best.pth"

    if not model_path.exists():
        return None, None

    try:
        from learning.archive_detector import load_detector
        model, ckpt = load_detector(model_path)
        _CACHED_DETECTOR = model
        _CACHED_CHECKPOINT = ckpt
        return _CACHED_DETECTOR, _CACHED_CHECKPOINT
    except Exception as e:
        print(f"[!] Không thể nạp mô hình archive_detector: {e}")
        return None, None


def generate_overlapping_tiles(
    img_h: int,
    img_w: int,
    tile_size: int = 800,
    overlap: int = 200
) -> List[Tuple[int, int, int, int]]:
    """
    Sinh danh sách các ô con chồng lấn (Overlapping Tiles) để quét biển số nhỏ ở xa.
    Trả về danh sách (x, y, w, h).
    """
    tiles = []
    step = tile_size - overlap

    for y in range(0, img_h, step):
        for x in range(0, img_w, step):
            tw = min(tile_size, img_w - x)
            th = min(tile_size, img_h - y)
            if tw >= 200 and th >= 200:
                tiles.append((x, y, tw, th))
            if x + tw >= img_w:
                break
        if y + th >= img_h:
            break

    return tiles


def archive_detector_proposals(
    image_rgb: np.ndarray,
    scale_factor: float = 1.0,
    conf_thresh: float = 0.20,
    use_tiling_for_large_images: bool = True
) -> List[Dict[str, Any]]:
    """
    Sinh các ứng viên Bounding Box từ mô hình CNN đã huấn luyện (archive detector).

    Args:
        image_rgb: Mảng numpy ảnh gốc RGB.
        scale_factor: Hệ số tỉ lệ giữa ảnh resize DIP và ảnh gốc (bbox trả về ở hệ tọa độ resize DIP).
        conf_thresh: Ngưỡng tin cậy của detector.
        use_tiling_for_large_images: Tự động chia ô chồng lấn cho ảnh lớn.

    Returns:
        Danh sách các dictionary ứng viên tương thích với pipeline.
    """
    model, ckpt = get_detector_model()
    if model is None:
        return []

    try:
        from learning.archive_detector import predict
        H, W = image_rgb.shape[:2]

        all_boxes = []

        # 1. Dự đoán trên toàn bộ ảnh gốc
        preds_full = predict(model, image_rgb, score=conf_thresh)
        if len(preds_full) > 0:
            for p in preds_full:
                all_boxes.append((float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))

        # 2. Xử lý Overlapping Tiling cho các biển ở xa trong ảnh độ phân giải cao
        if use_tiling_for_large_images and (W >= 1200 or H >= 1200):
            tiles = generate_overlapping_tiles(H, W, tile_size=900, overlap=250)
            for tx, ty, tw, th in tiles:
                tile_img = image_rgb[ty:ty+th, tx:tx+tw]
                tile_preds = predict(model, tile_img, score=conf_thresh)
                for tp in tile_preds:
                    gx0 = float(tp[0] + tx)
                    gy0 = float(tp[1] + ty)
                    gx1 = float(tp[2] + tx)
                    gy1 = float(tp[3] + ty)
                    all_boxes.append((gx0, gy0, gx1, gy1, float(tp[4])))

        if not all_boxes:
            return []

        # 3. NMS để loại bỏ các box trùng lặp giữa các tile và toàn ảnh
        from image_processing.candidate_filter import compute_iou_and_ios
        all_boxes.sort(key=lambda b: b[4], reverse=True)

        kept_boxes = []
        for b in all_boxes:
            bx0, by0, bx1, by1, score = b
            bw = max(1.0, bx1 - bx0)
            bh = max(1.0, by1 - by0)
            current_bbox = (int(round(bx0)), int(round(by0)), int(round(bw)), int(round(bh)))

            is_dup = False
            for kb in kept_boxes:
                iou, ios = compute_iou_and_ios(current_bbox, kb['orig_bbox'])
                if iou > 0.40 or ios > 0.70:
                    is_dup = True
                    break

            if not is_dup:
                kept_boxes.append({
                    'orig_bbox': current_bbox,
                    'score': score
                })

        # 4. Đóng gói thành các ứng viên chuẩn hóa theo tỉ lệ `scale_factor`
        candidates = []
        for idx, item in enumerate(kept_boxes):
            ox, oy, ow, oh = item['orig_bbox']
            # Chuyển đổi sang hệ tọa độ resize (dùng cho quy trình nội bộ)
            dx = int(round(ox * scale_factor))
            dy = int(round(oy * scale_factor))
            dw = max(10, int(round(ow * scale_factor)))
            dh = max(8, int(round(oh * scale_factor)))

            ar = float(dw) / float(dh) if dh > 0 else 1.0
            if ar < 0.45 or ar > 6.5:
                # Tỷ lệ cực đoan không thể là biển số xe (kể cả biển nghiêng)
                continue

            # Kiểm tra màu sắc phụ trợ trên ảnh gốc (đèn hậu, gói hàng bao bì đỏ)
            crop_rgb = image_rgb[oy:oy+oh, ox:ox+ow]
            red_ratio = 0.0
            if crop_rgb.size > 0:
                r_ch = crop_rgb[:, :, 0].astype(float)
                g_ch = crop_rgb[:, :, 1].astype(float)
                b_ch = crop_rgb[:, :, 2].astype(float)
                red_mask = (r_ch > 125) & (r_ch > 1.35 * g_ch) & (r_ch > 1.35 * b_ch)
                red_ratio = float(np.mean(red_mask))

            # Nếu hộp thẳng đứng hẹp mà lại có màu đỏ đậm chiếm đa số -> túi hàng / đèn hậu
            if ar < 0.75 and red_ratio > 0.25:
                continue

            plate_type = 'single_row' if ar >= 2.20 else 'double_row'

            cand = {
                'id': -100 - idx,
                'bbox': (dx, dy, dw, dh),
                'quad_corners': None,
                'angle': 0.0,
                'area': int(dw * dh * 0.45),
                'bbox_area': dw * dh,
                'density': 0.45,
                'aspect_ratio': ar,
                'score': float(item['score']),  # Không cộng khống +0.15
                'source': 'archive_detector',
                'detector_confidence': float(item['score']),
                'plate_type': plate_type,
                'is_detector_proposal': True,
                'red_ratio': red_ratio,
            }
            candidates.append(cand)

        return candidates
    except Exception as e:
        print(f"[!] Lỗi khi chạy archive_detector_proposals: {e}")
        return []

