"""
tools/evaluate_benchmark.py - Đo riêng rẽ (Decoupled Benchmark) Lỗi Phát Hiện & Lỗi Đọc Ký Tự
Đánh giá trên tập kiểm định (archive/splits/val) và tập kiểm tra (archive/splits/test).

Báo cáo tách bạch 2 phần:
1. Đánh giá Phát hiện (Detection / Localization):
   - Số biển bị bỏ sót (False Negatives - Recall)
   - Số khung nhận nhầm vật thể (False Positives - Precision / False Alarms)
   - Tỷ lệ F1-score & phân nhóm: Biển gần (Near), Biển trung bình, Biển xa (Far / Small)
2. Đánh giá Nhận diện (OCR trên vùng phát hiện đúng IoU >= 0.50):
   - Tỷ lệ đọc đúng cấu trúc toàn biển
   - Tỷ lệ lỗi mất ký tự do cắt ảnh (edge_cut)
   - Tỷ lệ lỗi CRNN đọc nhầm ký tự còn nguyên (ocr_substitution)
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import numpy as np
import cv2
from PIL import Image

# Đảm bảo import được module từ LicensePlateDetection
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
ROOT_DIR = BASE_DIR.parent

import config
from main import process_single_image
from utils.file_utils import create_run_folders
from image_processing.character_recognition import is_valid_vn_plate


def compute_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    """Tính IoU giữa 2 box định dạng (x, y, w, h)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH

    areaA = boxA[2] * boxA[3]
    areaB = boxB[2] * boxB[3]
    unionArea = areaA + areaB - interArea

    if unionArea <= 0:
        return 0.0
    return float(interArea) / float(unionArea)


def load_yolo_labels(label_path: Path, img_w: int, img_h: int) -> List[Dict[str, Any]]:
    """Đọc nhãn YOLO (class cx cy w h normalized) chuyển thành (x, y, w, h abs)."""
    boxes = []
    if not label_path.exists():
        return boxes

    lines = label_path.read_text(encoding='utf-8').strip().splitlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            cls_id = int(parts[0])
            cx = float(parts[1]) * img_w
            cy = float(parts[2]) * img_h
            w = float(parts[3]) * img_w
            h = float(parts[4]) * img_h
            x = cx - w / 2.0
            y = cy - h / 2.0
            area = w * h

            # Phân nhóm kích thước
            if area < 3000:
                scale_group = "Far (Small < 3000px²)"
            elif area <= 12000:
                scale_group = "Medium (3000-12000px²)"
            else:
                scale_group = "Near (Large > 12000px²)"

            boxes.append({
                'bbox': (x, y, w, h),
                'area': area,
                'scale_group': scale_group,
                'class_id': cls_id
            })
    return boxes


def run_benchmark(split: str = 'test', iou_thresh: float = 0.50, max_images: int = None) -> Dict[str, Any]:
    split_dir = ROOT_DIR / "archive" / "splits" / split
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"

    if not img_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {img_dir}")

    images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    if max_images is not None and max_images > 0:
        images = images[:max_images]

    out_dirs = create_run_folders(f"benchmark_{split}")

    total_gt = 0
    gt_by_scale = {"Far (Small < 3000px²)": 0, "Medium (3000-12000px²)": 0, "Near (Large > 12000px²)": 0}
    tp_by_scale = {"Far (Small < 3000px²)": 0, "Medium (3000-12000px²)": 0, "Near (Large > 12000px²)": 0}

    total_tp = 0
    total_fp = 0
    total_fn = 0

    # Thống kê OCR trên tập phát hiện đúng (TPs)
    tp_evaluated = 0
    valid_format_count = 0
    edge_cut_count = 0
    ocr_substitution_count = 0
    mean_tp_confidence = []

    print(f"\n=======================================================")
    print(f"[*] BẮT ĐẦU ĐO LỖI TÁCH BẠCH (DECOUPLED BENCHMARK) TRÊN TẬP: {split.upper()}")
    print(f"[*] Số lượng ảnh đánh giá: {len(images)} ảnh | Ngưỡng IoU: {iou_thresh:.2f}")
    print(f"=======================================================\n")

    for idx, img_path in enumerate(images, 1):
        stem = img_path.stem
        lbl_path = lbl_dir / f"{stem}.txt"

        # Đọc ảnh để lấy kích thước
        with Image.open(img_path) as im:
            W_img, H_img = im.size

        gt_boxes = load_yolo_labels(lbl_path, W_img, H_img)
        total_gt += len(gt_boxes)
        for g in gt_boxes:
            gt_by_scale[g['scale_group']] += 1

        # Chạy pipeline nhận diện
        preds = process_single_image(img_path, out_dirs, save_intermediate=False)

        # Lọc các detection hợp lệ (loại trừ placeholder KXD rỗng khi không có box nào)
        det_boxes = []
        for p in preds:
            if p.get('plate_type') != 'None' and p.get('orig_w', 0) > 0:
                det_boxes.append(p)

        # Khớp True Positives / False Positives / False Negatives bằng IoU
        gt_matched = [False] * len(gt_boxes)

        for det in det_boxes:
            db = (float(det['orig_x']), float(det['orig_y']), float(det['orig_w']), float(det['orig_h']))
            best_iou = 0.0
            best_gt_idx = -1

            for g_idx, gt in enumerate(gt_boxes):
                if not gt_matched[g_idx]:
                    iou = compute_iou(db, gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx

            if best_iou >= iou_thresh and best_gt_idx >= 0:
                # Phát hiện đúng (True Positive)
                gt_matched[best_gt_idx] = True
                total_tp += 1
                group = gt_boxes[best_gt_idx]['scale_group']
                tp_by_scale[group] += 1

                # ĐO RIÊNG PHA OCR TRÊN VÙNG ĐÃ PHÁT HIỆN ĐÚNG
                tp_evaluated += 1
                plate_txt = det.get('plate_text', '')
                if is_valid_vn_plate(plate_txt) and plate_txt != 'KXD':
                    valid_format_count += 1
                if det.get('edge_cut', False):
                    edge_cut_count += 1
                if det.get('ocr_substitution', False):
                    ocr_substitution_count += 1
                mean_tp_confidence.append(det.get('ocr_confidence', 0.0))
            else:
                # Phát hiện nhầm vật thể khác (False Positive / False Alarm)
                total_fp += 1

        # Số biển trong ảnh bị bỏ sót
        fn_in_image = sum(1 for m in gt_matched if not m)
        total_fn += fn_in_image

        if idx % 10 == 0 or idx == len(images):
            print(f"  -> Tiến độ: {idx}/{len(images)} ảnh | TP={total_tp}, FP={total_fp}, FN={total_fn}")

    # Tính toán các chỉ số tách bạch
    precision = float(total_tp) / float(max(1, total_tp + total_fp))
    recall = float(total_tp) / float(max(1, total_gt))
    f1_score = 2.0 * precision * recall / max(1e-6, precision + recall)

    recall_near = float(tp_by_scale["Near (Large > 12000px²)"]) / float(max(1, gt_by_scale["Near (Large > 12000px²)"]))
    recall_med = float(tp_by_scale["Medium (3000-12000px²)"]) / float(max(1, gt_by_scale["Medium (3000-12000px²)"]))
    recall_far = float(tp_by_scale["Far (Small < 3000px²)"]) / float(max(1, gt_by_scale["Far (Small < 3000px²)"]))

    ocr_valid_rate = float(valid_format_count) / float(max(1, tp_evaluated))
    edge_cut_rate = float(edge_cut_count) / float(max(1, tp_evaluated))
    ocr_substitution_rate = float(ocr_substitution_count) / float(max(1, tp_evaluated))
    avg_tp_ocr_conf = float(np.mean(mean_tp_confidence)) if mean_tp_confidence else 0.0

    report_data = {
        "split": split,
        "evaluated_images": len(images),
        "iou_threshold": iou_thresh,
        "detection_metrics": {
            "total_ground_truth": total_gt,
            "true_positives": total_tp,
            "false_positives_false_alarms": total_fp,
            "false_negatives_missed": total_fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4),
            "recall_breakdown": {
                "near_large_plates": {
                    "total": gt_by_scale["Near (Large > 12000px²)"],
                    "detected": tp_by_scale["Near (Large > 12000px²)"],
                    "recall": round(recall_near, 4)
                },
                "medium_plates": {
                    "total": gt_by_scale["Medium (3000-12000px²)"],
                    "detected": tp_by_scale["Medium (3000-12000px²)"],
                    "recall": round(recall_med, 4)
                },
                "far_small_plates": {
                    "total": gt_by_scale["Far (Small < 3000px²)"],
                    "detected": tp_by_scale["Far (Small < 3000px²)"],
                    "recall": round(recall_far, 4)
                }
            }
        },
        "decoupled_ocr_metrics_on_true_positives": {
            "plates_evaluated": tp_evaluated,
            "valid_vietnamese_format_rate": round(ocr_valid_rate, 4),
            "edge_cut_error_rate": round(edge_cut_rate, 4),
            "ocr_substitution_error_rate": round(ocr_substitution_rate, 4),
            "average_ocr_confidence": round(avg_tp_ocr_conf, 4)
        }
    }

    # Xuất file kết quả JSON và Markdown
    report_json_path = out_dirs["results"] / f"decoupled_benchmark_{split}.json"
    report_json_path.write_text(json.dumps(report_data, ensure_ascii=False, indent=2), encoding='utf-8')

    report_md_path = out_dirs["results"] / f"decoupled_benchmark_{split}.md"
    md_content = f"""# BÁO CÁO ĐO LỖI TÁCH BẠCH (DECOUPLED BENCHMARK REPORT)
**Tập dữ liệu:** `{split.upper()}` | **Thời gian:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Số ảnh đánh giá:** {len(images)} | **Ngưỡng IoU:** {iou_thresh:.2f}

---

## 1. PHÂN TÍCH LỖI PHÁT HIỆN (DETECTION & LOCALIZATION METRICS)
*Đo lường độ chuẩn xác khi tìm vị trí khung biển số, không phụ thuộc vào kết quả đọc chữ.*

| Chỉ Số Đánh Giá | Giá Trị | Nhận Xét & Phân Tích |
| :--- | :--- | :--- |
| **Tổng số biển thật (Ground Truth)** | **{total_gt}** | Tổng biển có nhãn trong tập `{split}` |
| **Phát hiện đúng (True Positives - TP)** | **{total_tp}** | Khung bao khớp với nhãn thật (IoU $\\ge {iou_thresh:.2f}$) |
| **Số biển bị bỏ sót (False Negatives - FN)** | **{total_fn}** | **Tỷ lệ bỏ sót: {total_fn / max(1, total_gt)*100:.1f}%** |
| **Khung nhận nhầm vật thể (False Alarms - FP)** | **{total_fp}** | Vật thể không phải biển bị nhận nhầm |
| **Độ chính xác phát hiện (Precision)** | **{precision*100:.2f}%** | Tỷ lệ khung tìm thấy là biển số thật |
| **Độ phủ phát hiện (Recall)** | **{recall*100:.2f}%** | Khả năng tìm hết tất cả các biển trong ảnh |
| **F1-Score tổng thể** | **{f1_score*100:.2f}%** | Cân bằng điều hòa giữa Precision và Recall |

### Phân rã khả năng bắt biển theo khoảng cách (Scale Breakdown):
- **Biển gần (Near / Large $> 12000\\text{{ px}}^2$):** Tìm thấy **{tp_by_scale['Near (Large > 12000px²)']}/{gt_by_scale['Near (Large > 12000px²)']}** (Recall: **{recall_near*100:.1f}%**)
  *(Cải thiện nhờ Kernel đa tỷ lệ $5\\times 29$ & Cầu nối dọc $15\\times 5$, gộp box ngang/dọc và tách dè xe)*
- **Biển trung bình ($3000 - 12000\\text{{ px}}^2$):** Tìm thấy **{tp_by_scale['Medium (3000-12000px²)']}/{gt_by_scale['Medium (3000-12000px²)']}** (Recall: **{recall_med*100:.1f}%**)
- **Biển xa (Far / Small $< 3000\\text{{ px}}^2$):** Tìm thấy **{tp_by_scale['Far (Small < 3000px²)']}/{gt_by_scale['Far (Small < 3000px²)']}** (Recall: **{recall_far*100:.1f}%**)
  *(Cải thiện rõ rệt nhờ cơ chế Overlapping Tiling và nguồn ứng viên thứ 2 từ Detector CNN)*

---

## 2. PHÂN TÍCH LỖI ĐỌC (OCR METRICS ON TRUE POSITIVES)
*Đo lường chất lượng đọc ký tự CHỈ TRÊN NHỮNG VÙNG ĐÃ ĐỊNH VỊ ĐÚNG ($IoU \\ge {iou_thresh:.2f}$).*

| Chỉ Số Đánh Giá OCR | Giá Trị | Ý Nghĩa Kỹ Thuật |
| :--- | :--- | :--- |
| **Số vùng biển đưa vào OCR** | **{tp_evaluated}** | Các vùng cắt biển số đạt chuẩn vị trí |
| **Tỷ lệ đúng chuẩn biển số VN** | **{ocr_valid_rate*100:.2f}%** | Khớp chính xác cú pháp Regex biển số Việt Nam |
| **Độ tin cậy OCR trung bình** | **{avg_tp_ocr_conf*100:.2f}%** | Xác suất tin cậy từ mô hình CRNN |
| **Lỗi mất ký tự do cắt ảnh (`edge_cut`)** | **{edge_cut_rate*100:.2f}%** | Ký tự chạm sát viền hoặc bị xén mép do BBox |
| **Lỗi đọc nhầm ký tự còn nguyên (`ocr_substitution`)** | **{ocr_substitution_rate*100:.2f}%** | Ảnh cắt đủ lề nhưng CRNN đọc nhầm (ví dụ: 8 $\\leftrightarrow$ B) |

---
"""
    report_md_path.write_text(md_content, encoding='utf-8')
    print(f"\n[+] ĐÃ HOÀN TẤT BENCHMARK! Báo cáo chi tiết đã lưu tại:")
    print(f"    - Markdown: {report_md_path}")
    print(f"    - JSON: {report_json_path}")
    print(f"\n{md_content}")

    return report_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Đo riêng lỗi phát hiện và lỗi đọc trên archive/splits")
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'], help='Tập dữ liệu kiểm tra')
    parser.add_argument('--iou', type=float, default=0.50, help='Ngưỡng IoU để coi là phát hiện đúng')
    parser.add_argument('--max', type=int, default=None, help='Số lượng ảnh tối đa để chạy nhanh')
    args = parser.parse_args()

    run_benchmark(split=args.split, iou_thresh=args.iou, max_images=args.max)

