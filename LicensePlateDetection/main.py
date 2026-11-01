"""
main.py - Chương trình chính phát hiện biển số xe từ hình ảnh giao thông
Đề tài: License Plate Detection from Traffic Images Using Edge Detection and Morphological Operations

Chức năng:
1. Xử lý một ảnh đơn hoặc xử lý hàng loạt toàn bộ ảnh trong thư mục input.
2. Thực hiện quy trình 6 bước xử lý ảnh số tự cài đặt thuần Python & NumPy:
   - Bước 1: Tiền xử lý (RGB -> Grayscale, Resize, Gaussian Blur)
   - Bước 2: Bắt cạnh dọc (Sobel-X)
   - Bước 3: Nhị phân hóa (Otsu Thresholding)
   - Bước 4: Xử lý hình thái học (Closing hình chữ nhật & Opening khử nhiễu)
   - Bước 5: Gán nhãn thành phần liên thông (BFS Connected Component Labeling)
   - Bước 6: Lọc ứng viên biển số (Candidate Filtering: tỷ lệ w/h, diện tích, mật độ điểm trắng)
3. Vẽ Bounding Box, phân loại biển 1 dòng / 2 dòng, cắt riêng từng biển số.
4. Lưu tất cả ảnh trung gian, ảnh kết quả, ảnh cắt biển số và xuất bảng thống kê CSV.
5. Tích hợp chạy thực nghiệm so sánh (Filters, Sobel vs Canny, SE sizes, Thresholds).
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from PIL import Image

# Import cấu hình tập trung
import config

# Import các thuật toán tự cài đặt
from image_processing.preprocessing import (
    rgb_to_grayscale,
    resize_image,
    gaussian_blur,
    enhance_contrast_dynamic,
)
from image_processing.edge_detection import sobel_x
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
from image_processing.candidate_filter import filter_license_plate_candidates
from image_processing.character_recognition import recognize_license_plate

# Import tiện ích lưu trữ và đồ họa
from utils.image_utils import (
    load_image,
    save_image,
    draw_bounding_boxes,
    crop_license_plate,
    create_pipeline_summary_grid,
)
from utils.file_utils import (
    get_image_paths,
    export_results_to_csv,
    export_results_to_excel,
    create_run_folders,
)
from compare import run_all_comparisons


def process_single_image(
    image_path: Path,
    output_dirs: Dict[str, Path],
    save_intermediate: bool = True,
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

    # 1.3 Khử nhiễu bằng bộ lọc Gaussian
    blurred_np = gaussian_blur(
        gray_np,
        kernel_size=config.GAUSSIAN_KERNEL_SIZE,
        sigma=config.GAUSSIAN_SIGMA
    )

    def _save_inter(img, name):
        if save_intermediate:
            save_image(img, inter_img_dir / name)
            save_image(img, main_inter_img_dir / name)

    _save_inter(gray_np, "01_grayscale.png")
    _save_inter(blurred_np, "02_gaussian_blurred.png")
    print(f"  [1/6] Tiền xử lý hoàn tất: Kích thước xử lý {resized_np.shape[1]}x{resized_np.shape[0]} (Scale={scale_factor:.3f})")

    # -------------------------------------------------------------
    # BƯỚC 2: PHÁT HIỆN CẠNH DỌC (SOBEL-X)
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
        sobel_np=sobel_x_np
    )
    print(f"  [6/6] Bộ lọc ứng viên: Đã xác định {len(candidates)} biển số hợp lệ!")

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
        # Tự động mở rộng hộp bao nếu bắt được dòng 1 của biển 2 dòng (chụp nghiêng/chụp từ trên cao)
        for cand in candidates:
            if cand['plate_type'] == 'double_row' and cand['aspect_ratio'] > 1.35:
                cx, cy, cw, ch = cand['bbox']
                new_h = int(cw * 0.95)
                new_w = int(cw * 1.15)
                new_x = max(0, cx - int(cw * 0.08))
                new_y = max(0, cy - int(ch * 0.05))
                if new_y + new_h <= opened_np.shape[0] and new_x + new_w <= opened_np.shape[1]:
                    cand['bbox'] = (new_x, new_y, new_w, new_h)
                    cand['aspect_ratio'] = round(new_w / float(new_h), 3)

        # Chạy OCR sơ bộ trên tất cả ứng viên và xếp hạng lại:
        evaluated_cands = []
        for cand in candidates:
            cropped_pil = crop_license_plate(
                orig_pil,
                cand['bbox'],
                scale_factor=scale_factor,
                margin_ratio=config.CROP_MARGIN_RATIO
            )
            ocr_res = recognize_license_plate(cropped_pil, plate_type=cand['plate_type'])
            plate_text = ocr_res.get("plate_text", "KXD")
            char_count = ocr_res.get("char_count", 0)
            avg_conf = ocr_res.get("avg_confidence", 0.0)

            cand['plate_text'] = plate_text
            cand['ocr_res'] = ocr_res
            cand['cropped_pil'] = cropped_pil

            # Điểm ưu tiên: ứng viên có ký tự rõ ràng (> 3 ký tự) được đưa ngay lên Top 1
            sort_score = cand['score']
            if char_count >= 4 and plate_text != "KXD":
                sort_score += 1.5 + avg_conf
            elif char_count >= 2:
                sort_score += 0.5 + avg_conf

            cand['sort_score'] = sort_score
            evaluated_cands.append(cand)

        # Sắp xếp giảm dần theo điểm ưu tiên thực tế
        evaluated_cands.sort(key=lambda c: c['sort_score'], reverse=True)
        candidates = evaluated_cands

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

            crop_filename = f"{stem}_plate_{idx}_{cand['plate_type']}.png"
            crop_path = output_dirs["cropped"] / crop_filename
            save_image(cropped_pil, crop_path)
            save_image(cropped_pil, output_dirs["main_cropped"] / crop_filename)

            print(f"      + Biển #{idx}: Biển số=[{plate_text}], Kiểu={cand['plate_type']}, Box=[x={orig_x}, y={orig_y}, w={orig_w}, h={orig_h}], AR={cand['aspect_ratio']:.2f}, Score={cand['score']:.2f}")

            records.append({
                "image_name": image_name,
                "plate_index": idx,
                "plate_text": plate_text,
                "plate_type": cand['plate_type'],
                "confidence_score": cand['score'],
                "ocr_confidence": ocr_res.get("avg_confidence", 0.0),
                "char_count": ocr_res.get("char_count", 0),
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
    save_intermediate: bool = True
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
