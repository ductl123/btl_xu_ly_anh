"""
main.py - Chương trình chính phát hiện biển số xe từ hình ảnh giao thông
Đề tài: License Plate Detection from Traffic Images Using Edge Detection and Morphological Operations

Chức năng:
1. Xử lý một ảnh đơn hoặc xử lý hàng loạt toàn bộ ảnh trong thư mục input.
2. Thực hiện quy trình xử lý ảnh số với Python, NumPy và OpenCV:
   - Bước 1: Tiền xử lý (RGB -> Grayscale, Resize, Gaussian Blur)
   - Bước 2: Bắt cạnh dọc (Sobel-X)
   - Bước 3: Nhị phân hóa (Otsu Thresholding)
   - Bước 4: Xử lý hình thái học (Closing hình chữ nhật & Opening khử nhiễu)
   - Bước 5: Gán nhãn thành phần liên thông (BFS Connected Component Labeling)
   - Bước 6: Lọc ứng viên biển số (Candidate Filtering: tỷ lệ w/h, diện tích, mật độ điểm trắng, độ sáng nền, xác thực ký tự)
3. Vẽ Bounding Box, phân loại biển 1 dòng / 2 dòng, cắt riêng từng biển số.
4. Lưu tất cả ảnh trung gian, ảnh kết quả, ảnh cắt biển số và xuất bảng thống kê CSV/Excel.
5. Tích hợp chạy thực nghiệm so sánh (Filters, Sobel vs Canny, SE sizes, Thresholds).
"""

import os
import sys
import re
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from PIL import Image, ImageOps

# Import cấu hình tập trung
import config

# Import các thuật toán tự cài đặt
from image_processing.preprocessing import (
    rgb_to_grayscale,
    resize_image,
    gaussian_blur,
    enhance_contrast_dynamic,
)
from image_processing.edge_detection import sobel_x, combined_sobel
from image_processing.thresholding import otsu_threshold
from image_processing.morphology import (
    create_structuring_element,
    binary_closing,
    binary_opening,
)
from image_processing.connected_components import (
    find_connected_components,
    create_colored_label_image,
)
from image_processing.candidate_filter import filter_license_plate_candidates, refine_quad_corners
from image_processing.character_recognition import recognize_license_plate, is_valid_vn_plate

# Import tiện ích lưu trữ và đồ họa
from utils.image_utils import (
    load_image,
    save_image,
    draw_bounding_boxes,
    crop_license_plate,
    order_quad_points,
    create_pipeline_summary_grid,
)
from utils.file_utils import (
    get_image_paths,
    export_results_to_csv,
    export_results_to_excel,
    create_run_folders,
)
from compare import run_all_comparisons


# Định dạng Regex chuẩn biển số xe Việt Nam: 2 chữ số mã tỉnh + 1-2 ký tự series + dãy 3-5 số (kèm dấu chấm nếu biển 5 số)
VN_PLATE_REGEX = re.compile(r'^\d{2}[A-Z0-9]{1,2}[- ]?\d{3,5}(\.\d{2})?$')


def is_valid_vietnamese_plate_format(plate_text: str) -> bool:
    """Kiểm tra chuỗi ký tự nhận diện có khớp với định dạng chuẩn biển số xe Việt Nam.
    Ủy quyền cho bộ kiểm tra regex chuẩn is_valid_vn_plate.
    """
    from image_processing.character_recognition import is_valid_vn_plate
    return is_valid_vn_plate(plate_text)


def process_single_image(
    image_path: Path,
    output_dirs: Dict[str, Path],
    save_intermediate: bool = False,
    show_plot: bool = False
) -> List[Dict[str, Any]]:
    """
    Quy trình xử lý hoàn chỉnh cho 1 bức ảnh:
    Thực hiện 6 bước xử lý ảnh số, lưu ảnh trung gian và xuất kết quả.

    Args:
        image_path: Đường dẫn file ảnh đầu vào.
        output_dirs: Dictionary chứa các đường dẫn thư mục lưu kết quả.
        save_intermediate: Có lưu ảnh của từng bước trung gian hay không.
        show_plot: Có hiển thị cửa sổ đồ thị kết quả hay không.

    Returns:
        Danh sách thông tin các biển số tìm thấy để ghi vào CSV.
    """
    image_name = image_path.name
    stem = image_path.stem
    print(f"\n=======================================================")
    print(f"[*] Đang xử lý ảnh: {image_name}")

    # Tạo thư mục trung gian riêng cho từng ảnh để tránh lẫn lộn
    inter_img_dir = output_dirs["intermediate"] / stem
    main_inter_img_dir = output_dirs["main_intermediate"] / stem
    if save_intermediate:
        inter_img_dir.mkdir(parents=True, exist_ok=True)
        main_inter_img_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # BƯỚC 1: TIỀN XỬ LÝ (PREPROCESSING)
    # -------------------------------------------------------------
    try:
        orig_pil, orig_np = load_image(image_path)
    except Exception as e:
        print(f"[!] Lỗi đọc file ảnh {image_path}: {e}")
        return []

    # 1.1 Chuẩn hóa kích thước nếu ảnh quá lớn
    resized_pil, scale_factor = resize_image(orig_pil, target_max_dim=config.TARGET_MAX_DIM)
    resized_np = np.array(resized_pil)

    # 1.2 Chuyển RGB sang Grayscale và kéo giãn tương phản động (chống chụp tối / chụp lóa)
    raw_gray_np = rgb_to_grayscale(resized_np)
    gray_np = enhance_contrast_dynamic(raw_gray_np, p_low=2.0, p_high=98.0)

    # Nâng cao độ tương phản thích nghi cục bộ CLAHE nếu được bật trong cấu hình
    if getattr(config, 'USE_CLAHE_ENHANCEMENT', True):
        from image_processing.preprocessing import apply_clahe_contrast
        gray_np = apply_clahe_contrast(
            gray_np,
            clip_limit=getattr(config, 'CLAHE_CLIP_LIMIT', 2.5),
            tile_grid_size=getattr(config, 'CLAHE_TILE_GRID_SIZE', (8, 8))
        )

    # Optional, explicit restoration experiment before the edge branch.
    restoration_applied = bool(getattr(config, 'USE_WIENER_RESTORATION', False))
    detection_gray = gray_np
    if restoration_applied:
        from learning.restoration import wiener_restore
        detection_gray = wiener_restore(gray_np)

    # 1.3 Khử nhiễu bằng bộ lọc Gaussian
    blurred_np = gaussian_blur(
        detection_gray,
        kernel_size=config.GAUSSIAN_KERNEL_SIZE,
        sigma=config.GAUSSIAN_SIGMA
    )

    def _save_inter(img, name):
        if save_intermediate:
            save_image(img, inter_img_dir / name)
            save_image(img, main_inter_img_dir / name)

    _save_inter(gray_np, "01_grayscale.png")
    if restoration_applied:
        _save_inter(detection_gray, "01b_wiener_restored.png")
    _save_inter(blurred_np, "02_gaussian_blurred.png")
    print(f"  [1/6] Tiền xử lý hoàn tất: Kích thước xử lý {resized_np.shape[1]}x{resized_np.shape[0]} (Scale={scale_factor:.3f})")

    # -------------------------------------------------------------
    # BƯỚC 2: PHÁT HIỆN CẠNH DỌC (SOBEL-X CHUẨN)
    # -------------------------------------------------------------
    sobel_x_np = sobel_x(blurred_np)
    _save_inter(sobel_x_np, "03_sobel_x.png")
    print(f"  [2/6] Phát hiện cạnh dọc Sobel-X hoàn tất")

    # -------------------------------------------------------------
    # BƯỚC 3: NHỊ PHÂN HÓA (OTSU THRESHOLDING)
    # -------------------------------------------------------------
    binary_np, otsu_t = otsu_threshold(sobel_x_np)
    _save_inter(binary_np, "04_binary_otsu.png")
    print(f"  [3/6] Nhị phân hóa Otsu thành công với ngưỡng tối ưu T* = {otsu_t}")

    # -------------------------------------------------------------
    # BƯỚC 4: XỬ LÝ HÌNH THÁI HỌC (MORPHOLOGY)
    # -------------------------------------------------------------
    # Phép Closing với SE hình chữ nhật nằm ngang để nối các nét ký tự
    se_close = create_structuring_element(
        height=config.CLOSING_KERNEL_HEIGHT,
        width=config.CLOSING_KERNEL_WIDTH,
        shape='rect'
    )
    closed_np = binary_closing(binary_np, se_close)

    # Phép Opening với SE nhỏ để khử nhiễu đốm nhỏ
    se_open = create_structuring_element(
        height=config.OPENING_KERNEL_HEIGHT,
        width=config.OPENING_KERNEL_WIDTH,
        shape='rect'
    )
    opened_np = binary_opening(closed_np, se_open)

    _save_inter(closed_np, "05_morph_closing.png")
    _save_inter(opened_np, "06_morph_opening.png")
    print(f"  [4/6] Xử lý hình thái học (Closing {config.CLOSING_KERNEL_HEIGHT}x{config.CLOSING_KERNEL_WIDTH} & Opening {config.OPENING_KERNEL_HEIGHT}x{config.OPENING_KERNEL_WIDTH}) hoàn tất")

    # -------------------------------------------------------------
    # BƯỚC 5: GÁN NHÃN VÙNG LIÊN THÔNG (CONNECTED COMPONENT LABELING)
    # -------------------------------------------------------------
    components, label_map = find_connected_components(opened_np, connectivity=config.CONNECTIVITY)

    if save_intermediate:
        ccl_colored = create_colored_label_image(label_map, len(components))
        _save_inter(ccl_colored, "07_connected_components.png")
    print(f"  [5/6] Gán nhãn thành phần liên thông BFS: Tìm thấy {len(components)} vùng liên thông")

    # -------------------------------------------------------------
    # BƯỚC 6: LỌC ỨNG VIÊN BIỂN SỐ (CANDIDATE FILTERING)
    # -------------------------------------------------------------
    config_dict = {
        "SINGLE_ROW_ASPECT_RATIO_MIN": config.SINGLE_ROW_ASPECT_RATIO_MIN,
        "SINGLE_ROW_ASPECT_RATIO_MAX": config.SINGLE_ROW_ASPECT_RATIO_MAX,
        "DOUBLE_ROW_ASPECT_RATIO_MIN": config.DOUBLE_ROW_ASPECT_RATIO_MIN,
        "DOUBLE_ROW_ASPECT_RATIO_MAX": config.DOUBLE_ROW_ASPECT_RATIO_MAX,
        "MIN_AREA_RATIO": config.MIN_AREA_RATIO,
        "MAX_AREA_RATIO": config.MAX_AREA_RATIO,
        "MIN_WIDTH_RATIO": config.MIN_WIDTH_RATIO,
        "MAX_WIDTH_RATIO": config.MAX_WIDTH_RATIO,
        "MIN_HEIGHT_RATIO": config.MIN_HEIGHT_RATIO,
        "MAX_HEIGHT_RATIO": config.MAX_HEIGHT_RATIO,
        "MIN_WHITE_DENSITY": config.MIN_WHITE_DENSITY,
        "MAX_WHITE_DENSITY": config.MAX_WHITE_DENSITY,
        "NMS_IOU_THRESHOLD": config.NMS_IOU_THRESHOLD,
    }
    candidates = filter_license_plate_candidates(
        components,
        opened_np.shape,
        config_dict,
        sobel_np=sobel_x_np,
        orig_img_np=orig_np,
        scale_factor=scale_factor
    )

    # BỔ SUNG THÊM: Nhánh phụ Combined Sobel dành riêng cho biển số xoay nghiêng/dọc lớn (|angle| >= 20°)
    try:
        from image_processing.candidate_filter import compute_iou_and_ios
        sob_comb = combined_sobel(blurred_np, weight_x=0.8, weight_y=0.4)
        bin_comb, _ = otsu_threshold(sob_comb)
        closed_comb = binary_closing(bin_comb, se_close)
        opened_comb = binary_opening(closed_comb, se_open)
        comps_comb, _ = find_connected_components(opened_comb, connectivity=config.CONNECTIVITY)
        cands_comb = filter_license_plate_candidates(
            comps_comb,
            opened_comb.shape,
            config_dict,
            sobel_np=sob_comb,
            orig_img_np=orig_np,
            scale_factor=scale_factor
        )
        for c_tilted in cands_comb:
            ang = c_tilted.get('angle', 0.0)
            if abs(ang) >= 20.0 or abs(ang + 90.0) <= 25.0:
                is_dup = False
                for ex in candidates:
                    iou, _ = compute_iou_and_ios(ex['bbox'], c_tilted['bbox'])
                    if iou > 0.40:
                        is_dup = True
                        break
                if not is_dup:
                    candidates.append(c_tilted)
    except Exception as e:
        print(f"[!] Nhánh tìm biển nghiêng không hoàn tất: {e}")

    # BỔ SUNG 1: Nhánh Morphology đa tỷ lệ (Multi-scale kernels) dành riêng cho biển cận cảnh (close-up)
    if getattr(config, 'LARGE_CLOSING_KERNEL_HEIGHT', None):
        try:
            from image_processing.candidate_filter import compute_iou_and_ios
            se_large = create_structuring_element(
                height=config.LARGE_CLOSING_KERNEL_HEIGHT,
                width=config.LARGE_CLOSING_KERNEL_WIDTH,
                shape='rect'
            )
            se_bridge = create_structuring_element(
                height=getattr(config, 'VERTICAL_BRIDGE_HEIGHT', 15),
                width=getattr(config, 'VERTICAL_BRIDGE_WIDTH', 5),
                shape='rect'
            )
            closed_large = binary_closing(binary_np, se_large)
            closed_bridge = binary_closing(closed_large, se_bridge)
            opened_large = binary_opening(closed_bridge, se_open)
            comps_large, _ = find_connected_components(opened_large, connectivity=config.CONNECTIVITY)
            cands_large = filter_license_plate_candidates(
                comps_large,
                opened_large.shape,
                config_dict,
                sobel_np=sobel_x_np,
                orig_img_np=orig_np,
                scale_factor=scale_factor
            )
            for cl in cands_large:
                is_dup = False
                for ex in candidates:
                    iou, _ = compute_iou_and_ios(ex['bbox'], cl['bbox'])
                    if iou > 0.40:
                        is_dup = True
                        break
                if not is_dup:
                    cl['source'] = 'morph_multiscale'
                    candidates.append(cl)
        except Exception as e:
            print(f"[!] Nhánh morphology đa tỉ lệ không hoàn tất: {e}")

    # BỔ SUNG 2: Nguồn ứng viên thứ 2 từ Detector CNN đã huấn luyện & Overlapping Tiling cho biển ở xa
    if getattr(config, 'USE_ARCHIVE_DETECTOR_PROPOSALS', True):
        try:
            from image_processing.candidate_filter import compute_iou_and_ios
            from image_processing.archive_detector_adapter import archive_detector_proposals
            det_cands = archive_detector_proposals(
                orig_np,
                scale_factor=scale_factor,
                conf_thresh=getattr(config, 'ARCHIVE_DETECTOR_CONF_THRESH', 0.20),
                use_tiling_for_large_images=getattr(config, 'USE_TILING_FOR_FAR_PLATES', True)
            )
            for dc in det_cands:
                # Không loại bỏ detector candidate khi trùng với DIP ở bước này;
                # giữ cả 2 nhánh cùng đi qua nắn phối cảnh và OCR để chọn phương án tốt nhất.
                is_exact_dup = False
                for ex in candidates:
                    iou, _ = compute_iou_and_ios(ex['bbox'], dc['bbox'])
                    if iou > 0.85:
                        is_exact_dup = True
                        break
                if not is_exact_dup:
                    candidates.append(dc)
        except Exception as e:
            print(f"[!] Nhánh archive_detector_proposals không hoàn tất: {e}")

    from image_processing.plate_verification import boundary_candidates, deduplicate_candidates, plate_evidence, complete_character_layout, recognized_row_evidence
    # A second proposal route works from real plate borders at higher resolution,
    # independent of the old brightness/character-count rejection gates.
    candidates = boundary_candidates(orig_np, scale_factor) + candidates
    print(f"  [6/6] Có {len(candidates)} ứng viên; tiếp tục kiểm tra bố cục chữ và biên biển.")

    # -------------------------------------------------------------
    # BƯỚC 7: CẮT BIỂN SỐ VÀ NHẬN DẠNG KÝ TỰ (OCR)
    # -------------------------------------------------------------
    records = []
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not candidates:
        print(f"  [-] Không phát hiện thấy biển số trong ảnh này (hoặc bị loại do không đủ tiêu chuẩn).")
        records.append({
            "image_name": image_name,
            "plate_index": 0,
            "plate_text": "KXD",
            "plate_type": "None",
            "confidence_score": 0.0,
            "ocr_confidence": 0.0,
            "char_count": 0,
            "orig_x": 0, "orig_y": 0, "orig_w": 0, "orig_h": 0,
            "aspect_ratio": 0.0,
            "white_density": 0.0,
            "pixel_area": 0,
            "cropped_image_path": "N/A",
            "timestamp": timestamp_str
        })
    else:
        import cv2

        # Chuẩn bị quad_corners cho các ứng viên có góc xoay nghiêng:
        orig_np_full = np.array(orig_pil)
        for cand in candidates:
            if cand.get('source') == 'boundary':
                continue
            angle = cand.get('angle', 0.0)
            is_tilted = (abs(angle) >= 10.0 and abs(angle + 90.0) >= 10.0 and abs(angle - 90.0) >= 10.0)
            if cand.get('quad_corners') is None and is_tilted:
                bx, by, bw, bh = cand['bbox']
                rect = ((bx + bw / 2.0, by + bh / 2.0), (bw, bh), angle)
                box = cv2.boxPoints(rect)
                cand['quad_corners'] = order_quad_points(box)

            # A merged vehicle box can be axis-aligned while the plate is tilted.
            refined_quad, refined_ang = refine_quad_corners(
                orig_np_full,
                cand['bbox'],
                scale_factor=scale_factor,
                plate_type=cand['plate_type']
            )
            if refined_quad is not None:
                cand['quad_corners'] = refined_quad
                cand['angle'] = refined_ang
                cand['boundary_refined'] = True
                lo = np.floor(refined_quad.min(axis=0)).astype(int)
                hi = np.ceil(refined_quad.max(axis=0)).astype(int)
                cand['bbox'] = (int(lo[0]), int(lo[1]), int(hi[0]-lo[0]), int(hi[1]-lo[1]))
                q = refined_quad
                qw = max(np.linalg.norm(q[1]-q[0]), np.linalg.norm(q[2]-q[3]))
                qh = max(np.linalg.norm(q[3]-q[0]), np.linalg.norm(q[2]-q[1]))
                cand['aspect_ratio'] = float(qw / max(qh, 1.0))
        # Gom ứng viên trước OCR:
        # 1. Loại bỏ các mảnh rác quá nhỏ không thể chứa ký tự biển số (kể cả biển xa nhất)
        valid_size_cands = []
        for c in candidates:
            bx, by, bw, bh = c['bbox']
            if bw >= 28 and bh >= 16 and (bw * bh) >= 600:
                valid_size_cands.append(c)
        candidates = valid_size_cands

        # 2. Không cộng điểm khống +0.5 cho boundary; dùng cùng tiêu chí đánh giá cho cả 3 nhánh
        top_k = getattr(config, 'TOP_CANDIDATES_COUNT', 10)
        sorted_cands = sorted(candidates, key=lambda c: c.get('score', 0.0), reverse=True)
        deduped_pre_ocr = []
        for cand in sorted_cands:
            is_dup = False
            c_area = cand['bbox'][2] * cand['bbox'][3]
            for prev in deduped_pre_ocr:
                p_area = prev['bbox'][2] * prev['bbox'][3]
                iou, ios = compute_iou_and_ios(cand['bbox'], prev['bbox'])
                same_src = (cand.get('source') == prev.get('source'))
                thresh = 0.45 if same_src else 0.80
                # Chỉ coi là trùng nếu IoU cao hoặc nếu 2 hộp có kích thước tương đồng (tỷ lệ diện tích >= 0.5)
                area_ratio = min(c_area, p_area) / max(c_area, p_area, 1.0)
                if iou > thresh or (ios > 0.85 and area_ratio >= 0.50):
                    is_dup = True
                    break
            if not is_dup:
                deduped_pre_ocr.append(cand)
            if len(deduped_pre_ocr) >= max(top_k, 20):
                break
        candidates = deduped_pre_ocr

        # Đánh giá OCR đa hướng (Dual Orientation Evaluation):
        # Với mỗi ứng viên, thử nghiệm cả crop chuẩn Bounding Box nằm ngang và crop nắn phẳng (nếu có góc nghiêng).
        # Không loại cứng trước OCR; để toàn bộ ứng viên có cơ hội nắn phẳng và nhận dạng ký tự.
        evaluated_cands = []
        for cand in candidates:
            # Phương án A: Crop ngang chuẩn Bounding Box
            crop_std = crop_license_plate(
                orig_pil,
                cand['bbox'],
                scale_factor=scale_factor,
                margin_ratio=max(config.CROP_MARGIN_RATIO, 0.12)
            )
            ocr_std = {} if cand.get('source') == 'boundary' else recognize_license_plate(crop_std, plate_type=cand['plate_type'])

            # Phương án B: Crop nắn phẳng theo phối cảnh Quad Corners (nếu có nghiêng)
            has_tilt = (cand.get('quad_corners') is not None) or (abs(cand.get('angle', 0.0)) >= 10.0 and abs(cand.get('angle', 0.0) + 90.0) >= 10.0)
            if has_tilt:
                crop_rot = crop_license_plate(
                    orig_pil,
                    cand['bbox'],
                    scale_factor=scale_factor,
                    margin_ratio=max(config.CROP_MARGIN_RATIO, 0.16),
                    quad_corners=cand.get('quad_corners'),
                    ordered_quad=cand.get('ordered_quad',False),
                    target_aspect_ratio=cand.get('target_aspect_ratio')
                )
                ocr_rot = recognize_license_plate(crop_rot, plate_type=cand['plate_type'], geometry_rectified=cand.get('boundary_refined', False))

                std_valid = is_valid_vietnamese_plate_format(ocr_std.get('plate_text', ''))
                rot_valid = is_valid_vietnamese_plate_format(ocr_rot.get('plate_text', ''))

                if cand.get('boundary_refined', False):
                    # A validated physical boundary determines geometry; OCR must
                    # not replace it with an oversized vehicle crop.
                    cropped_pil = crop_rot
                    ocr_res = ocr_rot
                elif rot_valid and not std_valid:
                    cropped_pil = crop_rot
                    ocr_res = ocr_rot
                elif std_valid and not rot_valid:
                    cropped_pil = crop_std
                    ocr_res = ocr_std
                else:
                    score_std = (3.0 if std_valid else 0.0) + ocr_std.get('char_count', 0) * 0.15 + ocr_std.get('avg_confidence', 0.0)
                    score_rot = (3.0 if rot_valid else 0.0) + ocr_rot.get('char_count', 0) * 0.15 + ocr_rot.get('avg_confidence', 0.0)
                    if score_rot > score_std:
                        cropped_pil = crop_rot
                        ocr_res = ocr_rot
                    else:
                        cropped_pil = crop_std
                        ocr_res = ocr_std
            else:
                cropped_pil = crop_std
                ocr_res = ocr_std

            plate_text = ocr_res.get("plate_text", "KXD")
            char_count = ocr_res.get("char_count", 0)
            avg_conf = ocr_res.get("avg_confidence", 0.0)

            # Cập nhật ảnh phẳng đã nắn thẳng
            if ocr_res.get("deskewed_image") is not None:
                cropped_pil = ocr_res["deskewed_image"]

            cand['plate_text'] = plate_text
            cand['ocr_res'] = ocr_res
            cand['cropped_pil'] = cropped_pil

            # Điểm ưu tiên: CHỈ cộng +2.0 điểm khi chuỗi ký tự khớp chính xác với Regex biển số VN
            # Không thưởng điểm cho chuỗi ngẫu nhiên hoặc sai cấu trúc để tránh BBox rác đè mất biển thật
            sort_score = cand['score']
            if is_valid_vietnamese_plate_format(plate_text):
                sort_score += 2.0 + avg_conf

            cand['sort_score'] = sort_score
            evaluated_cands.append(cand)

        # Sắp xếp giảm dần theo điểm ưu tiên thực tế
        evaluated_cands.sort(key=lambda c: c['sort_score'], reverse=True)

        # Formatting a random patch as a plate is not object verification.
        kept_candidates = []
        rejected = []
        for candidate in evaluated_cands:
            result = candidate['ocr_res']
            evidence = plate_evidence(candidate['cropped_pil'], candidate['plate_type'])
            if evidence is None:
                allow_repaired=True
                if result.get('recognizer','').startswith('crnn'):
                    from image_processing.object_verifier import object_margin
                    margin=object_margin(np.asarray(candidate['cropped_pil'].convert('L')))
                    original_layout=candidate.get('layout_evidence',{})
                    verified_before_crop=(candidate.get('source')=='boundary'
                        and original_layout.get('score',0)>=.82
                        and sum(original_layout.get('counts',[]))>=7)
                    allow_repaired=(margin is not None and margin>.2) or verified_before_crop
                if allow_repaired:
                    evidence = recognized_row_evidence(result,candidate['cropped_pil'],candidate['plate_type'])
                elif result.get('recognizer','').startswith('crnn'):
                    # A small, unfamiliar plate can be rejected by the object
                    # verifier. Recover only with an independent visual reader
                    # agreeing exactly, plus its measured character-row evidence.
                    from image_processing.character_recognition import recognize_license_plate_template
                    traditional=recognize_license_plate_template(candidate['cropped_pil'],candidate['plate_type'],True)
                    if traditional.get('plate_text')==candidate['plate_text']:
                        evidence=recognized_row_evidence(traditional,candidate['cropped_pil'],candidate['plate_type'])
                        if evidence is not None:evidence['source']='independent_template_confirmation'
            if evidence is None:
                # Cứu ứng viên nếu OCR đọc ra định dạng biển số Việt Nam hợp lệ hoặc có viền vật lý 4 góc
                plate_txt = candidate.get('plate_text', '')
                is_valid_format = is_valid_vietnamese_plate_format(plate_txt)
                conf = result.get('avg_confidence', 0.0)
                is_mudguard = candidate.get('mudguard_separated', False)
                has_physical_quad = candidate.get('boundary_refined', False)

                # Tuyệt đối KHÔNG tự động cấp 0.86 cho candidate chỉ vì là 'archive_detector'.
                # Chỉ cứu khi:
                # 1) OCR đọc đúng chuẩn cú pháp biển số Việt Nam (conf >= 0.55), HOẶC
                # 2) Đã tách dè thành công và đúng format, HOẶC
                # 3) Có viền vật lý 4 góc (boundary_refined) và nhận dạng ký tự tin cậy (conf >= 0.70)
                if (is_valid_format and conf >= 0.55) or (is_mudguard and is_valid_format) or (has_physical_quad and conf >= 0.70):
                    evidence = {
                        'score': 0.86,
                        'counts': [len(r.get('text', '')) for r in result.get('line_predictions', [])] or [4, 5],
                        'alignment': 0.08,
                        'foreground': 0.30,
                        'source': 'ocr_detector_confirmation'
                    }
                else:
                    rejected.append({'bbox':candidate['bbox'],'reason':'no_character_row_layout'})
                    continue
            lines = [row['text'] for row in result.get('line_predictions', [])]
            counts = evidence.get('counts', [])
            if (result.get('recognizer', '').startswith('crnn') and len(lines) == 2
                    and counts == [len(lines[0]) + 1, len(lines[1])]
                    and len(lines[0]) == 4):
                # Retry a missing prefix character only when the visible layout
                # supports a longer row and both photometric views agree.
                from learning.line_model import recognize as read_crnn
                crop = candidate['cropped_pil'].convert('RGB')
                gamma = [round(255 * (value / 255) ** .65) for value in range(256)]
                enhanced = (ImageOps.autocontrast(crop, cutoff=1), crop.point(gamma * 3))
                readings = [read_crnn(view, candidate['plate_type']) for view in enhanced]
                first, second = readings
                new_lines = [row['text'] for row in first['line_predictions']]
                if (len(new_lines) == 2 and new_lines[1] == lines[1]
                        and len(new_lines[0]) == counts[0]
                        and any(new_lines[0][:i] + new_lines[0][i+1:] == lines[0]
                                for i in range(len(new_lines[0])))
                        and first['plate_text'] == second['plate_text']
                        and is_valid_vn_plate(first['plate_text'])
                        and not first['needs_review'] and not second['needs_review']):
                    first['deskewed_image'] = crop
                    result = candidate['ocr_res'] = first
                    candidate['plate_text'] = first['plate_text']
                    lines = new_lines
            incomplete_serial = (result.get('recognizer', '').startswith('crnn')
                                 and len(lines) == 2 and len(counts) == 2
                                 and counts[1] == 5 and len(lines[1]) == 4
                                 and result.get('needs_review'))
            if incomplete_serial:
                result.setdefault('uncertainty_reasons', []).append(
                    'Dòng số có dấu hiệu thiếu ký tự, cần đối chiếu ảnh')
            complete = complete_character_layout(result, candidate['plate_type'])
            confidence = result.get('avg_confidence', 0.)
            is_trustworthy = (candidate.get('boundary_refined', False)
                              or candidate.get('source') == 'boundary'
                              or candidate.get('mudguard_separated', False)
                              or is_valid_vietnamese_plate_format(candidate['plate_text']))
            if not is_trustworthy and (not complete or confidence < .60):
                rejected.append({'bbox':candidate['bbox'],'reason':'unverified_legacy_patch'})
                continue
            char_details=result.get('characters',[])
            uncertain_fraction=sum(c.get('match_details',{}).get('needs_review',True) for c in char_details)/max(1,len(char_details))
            syntax_fraction=sum(c.get('match_details',{}).get('syntax_conflict',False) for c in char_details)/max(1,len(char_details))
            if candidate['plate_type']=='double_row' and not re.match(r'^\d{2}',candidate['plate_text']):
                # Nonstandard prefixes are reported verbatim for review. The
                # province/series slot assumptions do not apply to their top row.
                body=[c for c in char_details if c.get('row')==2]
                syntax_fraction=sum(c.get('match_details',{}).get('syntax_conflict',False) for c in body)/max(1,len(body))
            unreliable = incomplete_serial or confidence < config.OCR_MIN_MEAN_SCORE or syntax_fraction>.35
            # Preserve unusual but clearly separated glyphs (e.g. T61), while
            # withholding strings with unresolved type conflicts at several slots.
            if syntax_fraction > .15 and result.get('needs_review', True):
                unreliable = True
            if re.match(r'^\d{2}', candidate['plate_text']) and not is_valid_vn_plate(candidate['plate_text']):
                unreliable = True
            if result.get('recognizer','').startswith('crnn') and not is_valid_vn_plate(candidate['plate_text']) and confidence<.90:
                unreliable=True
            if evidence.get('localization_only') and (confidence<.78 or uncertain_fraction>.45):
                unreliable = True
            if not complete or confidence < .60 or unreliable:
                # Tiêu chuẩn giữ KXD đồng nhất cho cả DIP và Detector:
                # Chỉ giữ KXD khi có bằng chứng vật lý chắc chắn là biển số (viền biển 4 góc hoặc nguồn boundary đã xác thực):
                # - Có viền thực tế được xác thực (boundary_refined hoặc source=='boundary' hoặc mudguard_separated)
                # - VÀ evidence score >= 0.82
                # - VÀ tỷ lệ khung hình sau nắn hợp lý (0.85 <= ar <= 4.5)
                # - VÀ không có màu đỏ chiếm ưu thế (không phải đèn hậu/gói hàng đỏ)
                ar = candidate.get('aspect_ratio', 1.0)
                red_ratio = candidate.get('red_ratio', 0.0)
                has_solid_border = (candidate.get('boundary_refined', False)
                                    or candidate.get('source') == 'boundary'
                                    or candidate.get('mudguard_separated', False))

                if not has_solid_border or evidence.get('score', 0.0) < 0.82 or ar < 0.85 or ar > 4.5 or red_ratio > 0.30:
                    rejected.append({'bbox': candidate['bbox'], 'reason': 'insufficient_evidence_for_kxd'})
                    continue

                result['raw_text'] = candidate['plate_text']
                candidate['plate_text'] = 'KXD'
                result['needs_review'] = True
                result['recognition_status'] = 'unreadable'
            else:
                result['recognition_status'] = 'read' if not result.get('needs_review', True) else 'review'
            candidate['layout_evidence'] = evidence
            kept_candidates.append(candidate)

        # Deduplication dựa trên sort_score (ưu tiên phương án đọc đúng biển số VN và điểm cao nhất)
        from image_processing.candidate_filter import compute_iou_and_ios
        kept_candidates = sorted(kept_candidates, key=lambda c: c.get('sort_score', c.get('score', 0.0)), reverse=True)
        deduped_final = []
        for cand in kept_candidates:
            if any((lambda ov: ov[0] > 0.40 or ov[1] > 0.75)(compute_iou_and_ios(cand['bbox'], prev['bbox'])) for prev in deduped_final):
                continue
            deduped_final.append(cand)
            if len(deduped_final) >= 16:
                break
        kept_candidates = deduped_final
        if save_intermediate:
            (inter_img_dir/'candidate_rejections.json').write_text(
                json.dumps(rejected,ensure_ascii=False,indent=2),encoding='utf-8')

        candidates = kept_candidates

        if not candidates:
            print(f"  [-] Không phát hiện thấy biển số trong ảnh này (hoặc bị loại do không đủ tiêu chuẩn).")
            records.append({
                "image_name": image_name,
                "plate_index": 0,
                "plate_text": "KXD",
                "plate_type": "None",
                "confidence_score": 0.0,
                "ocr_confidence": 0.0,
                "char_count": 0,
                "orig_x": 0, "orig_y": 0, "orig_w": 0, "orig_h": 0,
                "aspect_ratio": 0.0,
                "white_density": 0.0,
                "pixel_area": 0,
                "cropped_image_path": "N/A",
                "timestamp": timestamp_str
            })
            # Continue to save the current image even when every proposal was
            # rejected. The GUI must never retain another image's old preview.

        for idx, cand in enumerate(candidates, start=1):
            cand['id'] = idx
            x, y, w, h = cand['bbox']
            orig_x = int(round(x / scale_factor))
            orig_y = int(round(y / scale_factor))
            orig_w = int(round(w / scale_factor))
            orig_h = int(round(h / scale_factor))

            cropped_pil = cand['cropped_pil']
            ocr_res = cand['ocr_res']
            plate_text = cand['plate_text']
            from learning.confirmed_crop import confirmed_crop_rows
            confirmed = confirmed_crop_rows(cropped_pil, cand['plate_type'])
            if confirmed and ''.join(c for c in plate_text if c.isalnum()) != ''.join(confirmed):
                from image_processing.character_recognition import format_plate_display
                raw_text = ocr_res.get('raw_text') or plate_text
                plate_text = format_plate_display(''.join(confirmed), cand['plate_type'])
                cand['plate_text'] = plate_text
                ocr_res = cand['ocr_res'] = {
                    **ocr_res, 'raw_text': raw_text, 'plate_text': plate_text,
                    'recognizer': 'confirmed-crop', 'recognition_status': 'confirmed',
                    'needs_review': False, 'uncertain_positions': [], 'uncertainty_reasons': [],
                    'char_count': sum(map(len, confirmed)),
                }

            from image_processing.mudguard_filter import check_edge_cut_and_substitution
            edge_cut, ocr_sub = check_edge_cut_and_substitution(cand, ocr_res, cropped_pil.width, cropped_pil.height)
            cand['edge_cut'] = edge_cut
            cand['ocr_substitution'] = ocr_sub
            ocr_res['edge_cut'] = edge_cut
            ocr_res['ocr_substitution'] = ocr_sub

            crop_filename = f"{stem}_plate_{idx}_{cand['plate_type']}.png"
            crop_path = output_dirs["cropped"] / crop_filename
            save_image(cropped_pil, crop_path)
            save_image(cropped_pil, output_dirs["main_cropped"] / crop_filename)

            # Save the exact OCR input and per-character ranking for image review.
            if save_intermediate:
                from image_processing.character_recognition import binarize_cropped_plate
                debug_dir = inter_img_dir / f"ocr_plate_{idx}"
                debug_dir.mkdir(parents=True, exist_ok=True)
                mask = binarize_cropped_plate(cropped_pil)
                save_image(mask.astype(np.uint8) * 255, debug_dir / "otsu_mask.png")
                debug_data = {k: v for k, v in ocr_res.items() if k != "deskewed_image"}
                (debug_dir / "recognition.json").write_text(
                    json.dumps(debug_data, ensure_ascii=False, indent=2), encoding="utf-8")
                if ocr_res.get('recognizer','').startswith('crnn'):
                    from learning.line_model import split_rows
                    rows,_=split_rows(cropped_pil,cand['plate_type'],ocr_res.get('row_split_ratio'))
                    for row_idx,(line,_) in enumerate(rows,1):
                        save_image(line,debug_dir/f'line_{row_idx}.png')
                else:
                    for char_idx, char in enumerate(ocr_res.get("characters", [])):
                        cx, cy, cw, ch = char["bbox"]
                        patch = cropped_pil.crop((cx, cy, cx + cw, cy + ch))
                        save_image(patch, debug_dir / f"char_{char_idx:02d}.png")

            print(f"      + Biển #{idx}: Biển số=[{plate_text}], Kiểu={cand['plate_type']}, Box=[x={orig_x}, y={orig_y}, w={orig_w}, h={orig_h}], AR={cand['aspect_ratio']:.2f}, Score={cand['score']:.2f}")

            records.append({
                "image_name": image_name,
                "plate_index": idx,
                "plate_text": plate_text,
                "plate_type": cand['plate_type'],
                "confidence_score": cand['score'],
                "ocr_confidence": ocr_res.get("avg_confidence", 0.0),
                "needs_review": ocr_res.get("needs_review", True),
                "uncertain_positions": ocr_res.get("uncertain_positions", []),
                "recognizer": ocr_res.get("recognizer", "unknown"),
                "uncertainty_reasons": ocr_res.get("uncertainty_reasons", []),
                "restoration_applied": restoration_applied,
                "recognition_status": ocr_res.get("recognition_status", "review"),
                "raw_ocr_text": ocr_res.get("raw_text", ""),
                "layout_evidence": cand.get("layout_evidence", {}),
                "char_count": ocr_res.get("char_count", 0),
                "edge_cut": edge_cut,
                "ocr_substitution": ocr_sub,
                "orig_x": orig_x,
                "orig_y": orig_y,
                "orig_w": orig_w,
                "orig_h": orig_h,
                "aspect_ratio": round(cand['aspect_ratio'], 3),
                "white_density": round(cand['density'], 3),
                "pixel_area": cand['area'],
                "cropped_image_path": str(crop_path),
                "timestamp": timestamp_str
            })

    # Vẽ Bounding Box lên ảnh gốc độ phân giải cao (kèm biển số nhận diện)
    detected_pil = draw_bounding_boxes(orig_pil, candidates, scale_factor=scale_factor)
    save_image(detected_pil, output_dirs["detected"] / f"{stem}_detected.png")
    save_image(detected_pil, output_dirs["main_detected"] / f"{stem}_detected.png")

    # Vẽ bảng tổng kết 8 bước bằng Matplotlib
    if save_intermediate:
        steps_dict = {
            'original_resized': resized_np,
            'grayscale': gray_np,
            'blurred': blurred_np,
            'gaussian_k': config.GAUSSIAN_KERNEL_SIZE,
            'sobel_x': sobel_x_np,
            'binary': binary_np,
            'otsu_t': otsu_t,
            'closing': closed_np,
            'closing_se': f"{config.CLOSING_KERNEL_HEIGHT}x{config.CLOSING_KERNEL_WIDTH}",
            'opening': opened_np,
            'opening_se': f"{config.OPENING_KERNEL_HEIGHT}x{config.OPENING_KERNEL_WIDTH}",
        }
        summary_grid_path = inter_img_dir / "08_pipeline_summary_grid.png"
        create_pipeline_summary_grid(steps_dict, detected_pil, candidates, summary_grid_path, show_plot=show_plot)
        create_pipeline_summary_grid(steps_dict, detected_pil, candidates, main_inter_img_dir / "08_pipeline_summary_grid.png", show_plot=False)

    return records


def process_batch(
    image_paths: List[Path],
    output_dirs: Dict[str, Path],
    save_intermediate: bool = False
) -> None:
    """
    Xử lý hàng loạt toàn bộ danh sách ảnh trong thư mục đầu vào.
    Tổng hợp kết quả vào file CSV.
    """
    total = len(image_paths)
    if total == 0:
        print("[!] Không tìm thấy ảnh nào trong thư mục input!")
        return

    print(f"\n=======================================================")
    print(f"[*] Bắt đầu xử lý hàng loạt {total} bức ảnh...")
    print(f"[*] Thư mục lưu kết quả phiên chạy: {output_dirs['run_dir']}")
    print(f"=======================================================")

    all_records: List[Dict[str, Any]] = []

    for i, img_p in enumerate(image_paths, start=1):
        print(f"\n>>> Tiến độ: [{i}/{total}]")
        records = process_single_image(img_p, output_dirs, save_intermediate=save_intermediate, show_plot=False)
        all_records.extend(records)

    # Xuất báo cáo CSV & Excel (.xlsx)
    csv_run_path = output_dirs["results"] / "detection_results.csv"
    csv_main_path = output_dirs["main_results"] / "detection_results.csv"
    xlsx_run_path = output_dirs["results"] / "detection_results.xlsx"
    xlsx_main_path = output_dirs["main_results"] / "detection_results.xlsx"

    export_results_to_csv(all_records, csv_run_path)
    export_results_to_csv(all_records, csv_main_path)
    export_results_to_excel(all_records, xlsx_run_path)
    export_results_to_excel(all_records, xlsx_main_path)

    # Thống kê tổng kết
    plates_found = sum(1 for r in all_records if r["plate_index"] > 0)
    images_with_plate = len(set(r["image_name"] for r in all_records if r["plate_index"] > 0))

    print(f"\n=======================================================")
    print(f"[*] TỔNG KẾT HOÀN THÀNH XỬ LÝ HÀNG LOẠT:")
    print(f"  - Tổng số ảnh đã xử lý: {total}")
    print(f"  - Số ảnh phát hiện được biển số: {images_with_plate}/{total} ({images_with_plate/total*100:.1f}%)")
    print(f"  - Tổng số biển số đã phát hiện & cắt: {plates_found}")
    print(f"  - File CSV thống kê: {csv_main_path}")
    print(f"  - File Excel (.xlsx) chuyên nghiệp: {xlsx_main_path}")
    print(f"=======================================================\n")


def interactive_menu() -> None:
    """
    Giao diện dòng lệnh tương tác trực tiếp khi người dùng khởi động chương trình.
    """
    output_dirs = create_run_folders(config.OUTPUT_DIR)

    while True:
        print("\n" + "="*65)
        print(" CHƯƠNG TRÌNH PHÁT HIỆN BIỂN SỐ XE TỪ ẢNH GIAO THÔNG")
        print(" (Edge Detection & Morphological Operations - Pure Python/NumPy)")
        print("="*65)
        print(" 1. Chọn 1 ảnh để xử lý (từ input/images hoặc đường dẫn tùy chọn)")
        print(" 2. Xử lý toàn bộ ảnh trong thư mục input/images/")
        print(" 3. Chạy chuỗi thực nghiệm so sánh (Filters, Sobel vs Canny, SE, Otsu)")
        print(" 4. Mở thư mục kết quả output")
        print(" 5. Mở giao diện đồ họa trực quan (Desktop GUI)")
        print(" 0. Thoát chương trình")
        print("-" * 65)

        choice = input("Nhập lựa chọn của bạn [0-5]: ").strip()

        if choice == '1':
            images = get_image_paths(config.INPUT_DIR)
            if not images:
                custom_path = input("Thư mục input/images trống. Nhập đường dẫn file ảnh: ").strip()
                if custom_path and Path(custom_path).is_file():
                    target = Path(custom_path)
                else:
                    print("[!] Đường dẫn không hợp lệ.")
                    continue
            else:
                print("\nDanh sách ảnh có sẵn trong input/images/:")
                for idx, p in enumerate(images[:10], start=1):
                    print(f"  {idx}. {p.name}")
                if len(images) > 10:
                    print(f"  ... và còn {len(images) - 10} ảnh khác.")
                sel = input(f"Chọn số thứ tự [1-{min(10, len(images))}] hoặc gõ tên/đường dẫn ảnh (Enter để chọn ảnh #1): ").strip()

                if not sel:
                    target = images[0]
                elif sel.isdigit() and 1 <= int(sel) <= len(images):
                    target = images[int(sel) - 1]
                elif Path(sel).is_file():
                    target = Path(sel)
                else:
                    print("[!] Lựa chọn không hợp lệ.")
                    continue

            recs = process_single_image(target, output_dirs, save_intermediate=True, show_plot=True)
            csv_path = output_dirs["main_results"] / "single_run_results.csv"
            xlsx_path = output_dirs["main_results"] / "single_run_results.xlsx"
            export_results_to_csv(recs, csv_path)
            export_results_to_excel(recs, xlsx_path)
            print(f"[+] Kết quả lưu tại: {output_dirs['main_detected']}")

        elif choice == '2':
            images = get_image_paths(config.INPUT_DIR)
            if not images:
                print("[!] Thư mục input/images đang trống! Hãy copy ảnh vào trước.")
            else:
                process_batch(images, output_dirs, save_intermediate=True)

        elif choice == '3':
            images = get_image_paths(config.INPUT_DIR)
            if not images:
                sample_path = input("Nhập đường dẫn file ảnh để chạy thực nghiệm: ").strip()
                target = Path(sample_path) if Path(sample_path).is_file() else None
            else:
                target = images[0]

            if target and target.is_file():
                run_all_comparisons(target, output_dirs["main_results"])
            else:
                print("[!] Không tìm thấy file ảnh hợp lệ để chạy thực nghiệm.")

        elif choice == '4':
            print(f"Thư mục output: {config.OUTPUT_DIR.resolve()}")
            if sys.platform == 'win32':
                os.system(f'explorer "{config.OUTPUT_DIR.resolve()}"')

        elif choice == '5':
            from gui import launch_gui
            launch_gui()

        elif choice == '0':
            print("Tạm biệt!")
            break
        else:
            print("[!] Lựa chọn không hợp lệ, vui lòng chọn lại.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="License Plate Detection from Traffic Images Using Edge Detection and Morphology"
    )
    parser.add_argument("--image", type=str, help="Đường dẫn đến 1 file ảnh cần xử lý")
    parser.add_argument("--batch", action="store_true", help="Xử lý toàn bộ ảnh trong thư mục input/images")
    parser.add_argument("--compare", action="store_true", help="Chạy các bài thực nghiệm so sánh thuật toán")
    parser.add_argument("--gui", action="store_true", help="Mở giao diện người dùng đồ họa (Desktop GUI)")
    parser.add_argument("--no-intermediate", action="store_true", help="Không lưu các ảnh bước trung gian để tiết kiệm dung lượng")
    parser.add_argument("--show", action="store_true", help="Hiển thị popup matplotlib khi xử lý xong ảnh")

    args = parser.parse_args()

    # Mở giao diện đồ họa GUI nếu có cờ --gui
    if args.gui:
        from gui import launch_gui
        launch_gui()
        return

    # Khởi tạo thư mục phiên chạy
    output_dirs = create_run_folders(config.OUTPUT_DIR)

    # 1. Chạy so sánh thực nghiệm nếu có cờ --compare
    if args.compare:
        images = get_image_paths(config.INPUT_DIR)
        target = Path(args.image) if args.image else (images[0] if images else None)
        if target and target.is_file():
            run_all_comparisons(target, output_dirs["main_results"])
        else:
            print("[!] Không tìm thấy ảnh để chạy so sánh thực nghiệm.")
        return

    # 2. Xử lý ảnh đơn qua command line
    if args.image:
        img_p = Path(args.image)
        if not img_p.is_file():
            print(f"[!] File ảnh không tồn tại: {args.image}")
            sys.exit(1)
        recs = process_single_image(
            img_p,
            output_dirs,
            save_intermediate=not args.no_intermediate,
            show_plot=args.show
        )
        csv_path = output_dirs["main_results"] / "detection_results.csv"
        xlsx_path = output_dirs["main_results"] / "detection_results.xlsx"
        export_results_to_csv(recs, csv_path)
        export_results_to_excel(recs, xlsx_path)
        return

    # 3. Xử lý hàng loạt qua command line
    if args.batch:
        images = get_image_paths(config.INPUT_DIR)
        process_batch(images, output_dirs, save_intermediate=not args.no_intermediate)
        return

    # 4. Nếu không có đối số dòng lệnh, mở menu tương tác thân thiện
    interactive_menu()


if __name__ == "__main__":
    main()

