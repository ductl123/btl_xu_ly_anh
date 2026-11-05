"""Train the plate verifier using ground truth plate bounding boxes and hard negatives from archive."""
from pathlib import Path
import json, sys, random
import cv2
import numpy as np

base = Path(__file__).resolve().parents[1]
root_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(base))
from image_processing.object_verifier import descriptor


def compute_iou(boxA, boxB):
    # box: (x1, y1, x2, y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea
    return interArea / max(1, unionArea)


def extract_features_from_split(split_name, rng, max_images=None):
    images_dir = root_dir / 'archive/splits' / split_name / 'images'
    labels_dir = root_dir / 'archive/splits' / split_name / 'labels'
    
    img_files = sorted(list(images_dir.glob('*.*')))
    if max_images:
        img_files = img_files[:max_images]
        
    features = []
    labels = []
    
    pos_count = 0
    neg_count = 0
    
    for img_path in img_files:
        lbl_path = labels_dir / (img_path.stem + '.txt')
        if not lbl_path.exists():
            continue
            
        im_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if im_gray is None:
            continue
        h, w = im_gray.shape[:2]
        if h < 20 or w < 20:
            continue
            
        gt_boxes = []
        for line in lbl_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                _, cx, cy, bw, bh = map(float, parts)
                x1 = max(0, int((cx - bw / 2.0) * w))
                y1 = max(0, int((cy - bh / 2.0) * h))
                x2 = min(w, int((cx + bw / 2.0) * w))
                y2 = min(h, int((cy + bh / 2.0) * h))
                if (x2 - x1) >= 20 and (y2 - y1) >= 12:
                    gt_boxes.append((x1, y1, x2, y2))
                    
        # 1. Trích xuất Positive patches
        for box in gt_boxes:
            x1, y1, x2, y2 = box
            patch = im_gray[y1:y2, x1:x2]
            if patch.size > 0:
                features.append(descriptor(patch))
                labels.append(1.0)
                pos_count += 1
                
                # Data augmentation nhẹ cho positive
                if split_name == 'train':
                    # Biến thể chỉnh sáng nhẹ
                    p_bright = np.clip(patch.astype(float) * rng.uniform(0.85, 1.15), 0, 255).astype(np.uint8)
                    features.append(descriptor(p_bright))
                    labels.append(1.0)
                    pos_count += 1
                    
        # 2. Hard Negative Mining: Vùng có Sobel-X mạnh nhưng IoU < 0.05 với GT box
        # Tính Sobel-X để tìm các vùng dễ bị nhận nhầm thành biển số (nan hoa, lưới tản nhiệt, đèn...)
        gx = np.abs(cv2.Sobel(im_gray, cv2.CV_32F, 1, 0))
        gx_integral = cv2.integral(gx)
        
        # Thử lấy các ô ngẫu nhiên
        sample_attempts = 0
        found_negs = 0
        target_negs = len(gt_boxes) * 2  # 2 negative per positive
        
        while sample_attempts < 20 and found_negs < target_negs:
            sample_attempts += 1
            # Random aspect ratio: 1.2 to 4.5
            ar = rng.uniform(1.2, 4.5)
            nh = rng.integers(int(max(15, h * 0.08)), int(max(16, h * 0.35)))
            nw = int(nh * ar)
            if nw >= w or nh >= h:
                continue
            nx1 = rng.integers(0, w - nw)
            ny1 = rng.integers(0, h - nh)
            nx2 = nx1 + nw
            ny2 = ny1 + nh
            neg_box = (nx1, ny1, nx2, ny2)
            
            # Kiểm tra IoU với mọi GT box
            if any(compute_iou(neg_box, gt) > 0.04 for gt in gt_boxes):
                continue
                
            neg_patch = im_gray[ny1:ny2, nx1:nx2]
            if neg_patch.size > 0:
                features.append(descriptor(neg_patch))
                labels.append(-1.0)
                neg_count += 1
                found_negs += 1
                
    return np.array(features, np.float32), np.array(labels, np.float32), pos_count, neg_count


def main():
    print("=== BẮT ĐẦU HUẤN LUYỆN PLATE VERIFIER TỪ ARCHIVE ===")
    rng = np.random.default_rng(20261105)
    
    # 1. Trích xuất đặc trưng từ train split
    print("Trích xuất đặc trưng tập train (archive/splits/train)...")
    X_train, y_train, train_pos, train_neg = extract_features_from_split('train', rng)
    print(f"  -> Tập train: {len(y_train)} mẫu ({train_pos} positive, {train_neg} negative)")
    
    # 2. Thêm dữ liệu gốc hiện có từ data/plate_verifier/manifest.json
    orig_data_dir = base / 'data/plate_verifier'
    if (orig_data_dir / 'manifest.json').exists():
        orig_records = json.loads((orig_data_dir / 'manifest.json').read_text(encoding='utf-8'))
        orig_feats, orig_lbls = [], []
        for r in orig_records:
            fpath = orig_data_dir / r['file']
            if fpath.exists():
                im = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
                if im is not None:
                    orig_feats.append(descriptor(im))
                    orig_lbls.append(float(r['label']))
        if orig_feats:
            X_train = np.vstack([X_train, np.array(orig_feats, np.float32)])
            y_train = np.concatenate([y_train, np.array(orig_lbls, np.float32)])
            print(f"  -> Bổ sung {len(orig_lbls)} mẫu kiểm định ban đầu vào train")
            
    # 3. Trích xuất đặc trưng tập val
    print("Trích xuất đặc trưng tập val (archive/splits/val)...")
    X_val, y_val, val_pos, val_neg = extract_features_from_split('val', rng)
    print(f"  -> Tập validation: {len(y_val)} mẫu ({val_pos} positive, {val_neg} negative)")
    
    # 4. Huấn luyện Ridge Regression Linear Classifier
    print("\nHuấn luyện mô hình phân loại Ridge Regression...")
    reg_lambda = 0.08
    XtX = X_train.T @ X_train
    reg_matrix = np.eye(XtX.shape[0], dtype=np.float32) * reg_lambda
    # Không phạt bias (cột cuối cùng)
    reg_matrix[-1, -1] = 0.0
    weights = np.linalg.solve(XtX + reg_matrix, X_train.T @ y_train)
    
    # 5. Đánh giá chất lượng mô hình
    train_preds = np.where(X_train @ weights > 0, 1.0, -1.0)
    train_acc = np.mean(train_preds == y_train)
    
    val_preds = np.where(X_val @ weights > 0, 1.0, -1.0)
    val_acc = np.mean(val_preds == y_val)
    val_pos_acc = np.mean(val_preds[y_val == 1.0] == 1.0)
    val_neg_acc = np.mean(val_preds[y_val == -1.0] == -1.0)
    
    print("\n=== KẾT QUẢ HUẤN LUYỆN PLATE VERIFIER ===")
    print(f"• Độ chính xác tập Train: {train_acc * 100:.2f}% (trên {len(y_train)} mẫu)")
    print(f"• Độ chính xác tập Val:   {val_acc * 100:.2f}% (trên {len(y_val)} mẫu)")
    print(f"  + Nhận đúng biển thật (Sensitivity / Recall): {val_pos_acc * 100:.2f}%")
    print(f"  + Lọc sạch nền giả (Specificity):             {val_neg_acc * 100:.2f}%")
    
    # 6. Lưu mô hình và metadata
    out_model = orig_data_dir / 'object_model.npz'
    np.savez(out_model, weights=weights)
    
    metadata = {
        "training_examples": int(len(y_train)),
        "training_positives": int(train_pos),
        "training_negatives": int(train_neg),
        "train_accuracy": float(train_acc),
        "val_examples": int(len(y_val)),
        "val_accuracy": float(val_acc),
        "val_recall_positive": float(val_pos_acc),
        "val_specificity_negative": float(val_neg_acc),
        "source": "Tập huấn luyện archive/splits + calibration manifest",
        "feature_dim": int(len(weights))
    }
    (orig_data_dir / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\n✅ Đã lưu trọng số mới vào: {out_model}")
    print(f"✅ Đã cập nhật metadata tại: {orig_data_dir / 'metadata.json'}")


if __name__ == '__main__':
    main()

