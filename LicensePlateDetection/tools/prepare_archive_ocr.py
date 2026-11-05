"""Extract high-quality plate crops from archive, apply consensus pseudo-labelling via CRNN,
and import approved plates into DatasetStore for training."""
from pathlib import Path
from collections import Counter, defaultdict
import json, re, sys
import cv2
import numpy as np
from PIL import Image

base = Path(__file__).resolve().parents[1]
root_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(base))

from learning.line_model import recognize, split_rows
from learning.dataset import DatasetStore, clean_label, image_digest
from image_processing.character_recognition import is_valid_vn_plate


def parse_yolo_labels(lbl_path, img_w, img_h, min_w=70, min_h=25):
    boxes = []
    if not lbl_path.exists():
        return boxes
    for line in lbl_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            cls, cx, cy, bw, bh = map(float, parts)
            cw = bw * img_w
            ch = bh * img_h
            if cw >= min_w and ch >= min_h:
                x1 = max(0, int((cx - bw / 2.0) * img_w))
                y1 = max(0, int((cy - bh / 2.0) * img_h))
                x2 = min(img_w, int((cx + bw / 2.0) * img_w))
                y2 = min(img_h, int((cy + bh / 2.0) * img_h))
                boxes.append((x1, y1, x2, y2, cw, ch))
    return boxes


def run_archive_pseudo_labeling(max_base_samples=None, min_confidence=0.85):
    print("=== BẮT ĐẦU TRÍCH XUẤT & GÁN NHÃN ĐỒNG THUẬN (CONSENSUS OCR) TỪ ARCHIVE ===")
    train_images_dir = root_dir / 'archive/splits/train/images'
    train_labels_dir = root_dir / 'archive/splits/train/labels'
    
    # Gom nhóm các ảnh theo base_id (ví dụ: 109_jpg.rf.* -> base_id = '109')
    all_imgs = sorted(list(train_images_dir.glob('*.*')))
    groups = defaultdict(list)
    for p in all_imgs:
        stem = p.stem
        if '_jpg.rf.' in p.name:
            base_id = p.name.split('_jpg.rf.')[0]
        elif '_png.rf.' in p.name:
            base_id = p.name.split('_png.rf.')[0]
        else:
            base_id = stem
        groups[base_id].append(p)
        
    print(f"Tổng số ảnh trong tập train: {len(all_imgs)} ({len(groups)} nhóm xe/ảnh gốc độc lập)")
    
    store = DatasetStore()
    existing_plate_ids = {row['plate_id'] for row in store.list_examples()}
    print(f"Số biển đã có sẵn trong cơ sở dữ liệu huấn luyện: {len(existing_plate_ids)}")
    
    saved_count = 0
    skipped_syntax = 0
    skipped_conf = 0
    skipped_consensus = 0
    skipped_exists = 0
    
    group_items = list(groups.items())
    if max_base_samples:
        group_items = group_items[:max_base_samples]
        
    approved_records = []
    
    for idx, (base_id, file_list) in enumerate(group_items, 1):
        if idx % 50 == 0:
            print(f"  -> Đang xử lý nhóm {idx}/{len(group_items)}... (Đã lưu {saved_count} biển)")
            
        # Tìm các kết quả đọc qua các biến thể tăng cường của nhóm
        variant_predictions = []
        variant_crops = []
        
        for fpath in file_list:
            lbl_path = train_labels_dir / (fpath.stem + '.txt')
            im_bgr = cv2.imread(str(fpath))
            if im_bgr is None:
                continue
            h, w = im_bgr.shape[:2]
            boxes = parse_yolo_labels(lbl_path, w, h)
            if not boxes:
                continue
                
            # Với mỗi bounding box trong ảnh
            for (x1, y1, x2, y2, cw, ch) in boxes:
                crop = im_bgr[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop_pil = Image.fromarray(crop_rgb)
                
                aspect = cw / ch
                ptype = 'double_row' if aspect < 2.3 else 'single_row'
                
                try:
                    res = recognize(crop_pil, plate_type=ptype)
                    text = res.get('plate_text', '')
                    conf = float(res.get('avg_confidence', 0.0))
                    chars = res.get('characters', [])
                    rows_pred = res.get('line_predictions', [])
                    
                    variant_predictions.append({
                        'text': text,
                        'conf': conf,
                        'ptype': ptype,
                        'res': res,
                        'crop_pil': crop_pil,
                        'file': fpath.name
                    })
                except Exception:
                    continue
                    
        if not variant_predictions:
            continue
            
        # 1. Tìm bản đọc phổ biến nhất (Majority vote)
        text_counts = Counter(v['text'] for v in variant_predictions)
        best_text, best_count = text_counts.most_common(1)[0]
        
        # 2. Kiểm tra tính hợp lệ cú pháp biển số Việt Nam
        if not is_valid_vn_plate(best_text):
            skipped_syntax += 1
            continue
            
        # 3. Kiểm tra độ tin cậy và sự đồng thuận
        matching_variants = [v for v in variant_predictions if v['text'] == best_text]
        mean_conf = np.mean([v['conf'] for v in matching_variants])
        
        if mean_conf < min_confidence:
            skipped_conf += 1
            continue
            
        # Nếu có từ 2 biến thể trở lên, cần ít nhất 50% số biến thể đồng thuận
        if len(variant_predictions) >= 2 and (best_count / len(variant_predictions)) < 0.5:
            skipped_consensus += 1
            continue
            
        # 4. Kiểm tra xem biển này đã từng được lưu hay chưa
        if best_text in existing_plate_ids:
            skipped_exists += 1
            continue
            
        # 5. Chọn crop có chất lượng tốt nhất (độ phân giải cao hoặc không phải ảnh tăng cường nặng)
        best_variant = max(matching_variants, key=lambda v: (v['crop_pil'].size[0] * v['crop_pil'].size[1], v['conf']))
        crop_pil = best_variant['crop_pil']
        ptype = best_variant['ptype']
        
        # 6. Tách dòng
        try:
            if ptype == 'double_row':
                # Tìm phân tách dòng từ predictions
                rows_pred = best_variant['res'].get('line_predictions', [])
                if len(rows_pred) == 2:
                    row1 = clean_label(rows_pred[0]['text'])
                    row2 = clean_label(rows_pred[1]['text'])
                else:
                    # Ước tính từ text
                    prefix_match = re.match(r'^(\d{2}[A-Z]{1,2}\d?)(.*)$', best_text)
                    if prefix_match:
                        row1, row2 = prefix_match.group(1), prefix_match.group(2)
                    else:
                        row1 = best_text[:4]
                        row2 = best_text[4:]
                split_ratio = float(best_variant['res'].get('row_split_ratio', 0.5) or 0.5)
            else:
                row1 = clean_label(best_text)
                row2 = ''
                split_ratio = None
                
            ident = store.save(
                image=crop_pil,
                row1=row1,
                row2=row2,
                plate_type=ptype,
                split='train',
                split_ratio=split_ratio,
                source_group=base_id,
                provenance=f'Archive consensus (conf={mean_conf:.3f}, votes={best_count}/{len(variant_predictions)})'
            )
            existing_plate_ids.add(best_text)
            saved_count += 1
            approved_records.append({
                'id': ident,
                'plate_text': best_text,
                'row1': row1,
                'row2': row2,
                'ptype': ptype,
                'conf': round(float(mean_conf), 3),
                'votes': f"{best_count}/{len(variant_predictions)}"
            })
        except Exception as exc:
            continue
            
    print("\n=== TỔNG KẾT GÁN NHÃN ĐỒNG THUẬN TỪ ARCHIVE ===")
    print(f"• Số biển số chất lượng cao được DUYỆT & LƯU vào Train: {saved_count}")
    print(f"• Bỏ qua do cú pháp không khớp biển VN:               {skipped_syntax}")
    print(f"• Bỏ qua do độ tin cậy thấp (< {min_confidence}):          {skipped_conf}")
    print(f"• Bỏ qua do không đạt đồng thuận giữa các biến thể:  {skipped_consensus}")
    print(f"• Bỏ qua do biển đã có sẵn trong tập dữ liệu:        {skipped_exists}")
    
    # Thống kê tổng số mẫu trong DatasetStore hiện tại
    stats = {s: len(store.list_examples(s)) for s in ('train', 'validation', 'test')}
    print(f"\n📊 Tổng số mẫu hiện có trong DatasetStore: Train={stats['train']}, Val={stats['validation']}, Test={stats['test']}")
    return approved_records


if __name__ == '__main__':
    run_archive_pseudo_labeling()

