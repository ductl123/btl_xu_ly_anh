"""
utils/file_utils.py - Quản lý file, quét thư mục và xuất báo cáo CSV & Excel (.xlsx)
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Union


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def get_image_paths(input_dir: Union[str, Path]) -> List[Path]:
    """
    Quét và lấy toàn bộ danh sách đường dẫn các file ảnh hợp lệ trong thư mục đầu vào.
    Sắp xếp theo thứ tự bảng chữ cái để xử lý tuần tự nhất quán.
    """
    directory = Path(input_dir)
    if not directory.is_dir():
        return []

    images = [
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS
    ]
    images.sort(key=lambda x: x.name)
    return images


def export_results_to_csv(records: List[Dict[str, Any]], output_path: Union[str, Path]) -> None:
    """
    Xuất bảng thống kê kết quả phát hiện biển số ra file CSV.
    Sử dụng mã hóa 'utf-8-sig' để hỗ trợ hiển thị tiếng Việt hoàn hảo khi mở bằng Microsoft Excel.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_name",
        "plate_index",
        "plate_text",            # Số của biển số xe nhận dạng được
        "plate_type",
        "confidence_score",
        "ocr_confidence",
        "char_count",
        "orig_x",
        "orig_y",
        "orig_w",
        "orig_h",
        "aspect_ratio",
        "white_density",
        "pixel_area",
        "cropped_image_path",
        "timestamp"
    ]

    with open(path, mode="w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)


def export_results_to_excel(records: List[Dict[str, Any]], output_path: Union[str, Path]) -> None:
    """
    Xuất bảng thống kê kết quả phát hiện và nhận diện biển số xe ra file Microsoft Excel (.xlsx).
    Tự động căn chỉnh độ rộng cột, định dạng tiêu đề màu xanh chuyên nghiệp và làm nổi bật cột Biển số xe.
    """
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    except ImportError:
        print("[!] Thư viện openpyxl chưa được cài đặt. Bỏ qua xuất file .xlsx.")
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Ket_Qua_Nhan_Dien"

    # Định nghĩa cấu trúc các cột cho file Excel
    columns = [
        ("Tên File Ảnh", "image_name", 28),
        ("Biển Số Nhận Diện", "plate_text", 22),
        ("Loại Biển", "plate_type", 14),
        ("Thứ Tự", "plate_index", 10),
        ("Điểm Tin Cậy BBox", "confidence_score", 18),
        ("Độ Tin Cậy Ký Tự", "ocr_confidence", 18),
        ("Số Ký Tự", "char_count", 12),
        ("Tọa Độ X", "orig_x", 12),
        ("Tọa Độ Y", "orig_y", 12),
        ("Chiều Rộng W", "orig_w", 14),
        ("Chiều Cao H", "orig_h", 14),
        ("Tỷ Lệ w/h", "aspect_ratio", 14),
        ("Mật Độ Trắng", "white_density", 14),
        ("Số Pixel Cạnh", "pixel_area", 14),
        ("Đường Dẫn Ảnh Cắt", "cropped_image_path", 35),
        ("Thời Gian Xử Lý", "timestamp", 20),
    ]

    # Kiểu dáng (Styles)
    header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
    header_font = Font(name="Segoe UI", size=11, bold=True, color="FFFFFF")
    plate_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
    plate_font = Font(name="Segoe UI", size=11, bold=True, color="002060")
    regular_font = Font(name="Segoe UI", size=10)
    center_align = Alignment(horizontal="center", vertical="center")
    left_align = Alignment(horizontal="left", vertical="center")

    thin_border = Border(
        left=Side(style="thin", color="D3D3D3"),
        right=Side(style="thin", color="D3D3D3"),
        top=Side(style="thin", color="D3D3D3"),
        bottom=Side(style="thin", color="D3D3D3")
    )

    # Ghi dòng tiêu đề
    for col_idx, (col_name, _, col_width) in enumerate(columns, start=1):
        cell = ws.cell(row=1, column=col_idx, value=col_name)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center_align
        col_letter = openpyxl.utils.get_column_letter(col_idx)
        ws.column_dimensions[col_letter].width = col_width

    # Ghi dữ liệu
    for row_idx, rec in enumerate(records, start=2):
        for col_idx, (_, field_key, _) in enumerate(columns, start=1):
            val = rec.get(field_key, "")
            cell = ws.cell(row=row_idx, column=col_idx, value=val)
            cell.font = regular_font
            cell.border = thin_border
            cell.alignment = center_align

            # Căn lề trái cho tên file và đường dẫn
            if field_key in ["image_name", "cropped_image_path"]:
                cell.alignment = left_align

            # Làm nổi bật cột Biển số xe
            if field_key == "plate_text":
                cell.fill = plate_fill
                cell.font = plate_font

    ws.row_dimensions[1].height = 28
    for r in range(2, len(records) + 2):
        ws.row_dimensions[r].height = 20

    wb.save(path)
    print(f"  [+] Đã xuất file Excel kết quả chi tiết tại: {path.name}")


def create_run_folders(base_output_dir: Union[str, Path], run_name: str = None) -> Dict[str, Path]:
    """
    Tạo cấu trúc thư mục đầu ra cho phiên làm việc hiện tại, kèm timestamp để tránh ghi đè.
    Đồng thời duy trì đường dẫn tương thích tới các thư mục chính.
    """
    base = Path(base_output_dir)
    if run_name is None:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    run_dir = base / "runs" / run_name
    intermediate_dir = run_dir / "intermediate"
    detected_dir = run_dir / "detected"
    cropped_dir = run_dir / "cropped"
    results_dir = run_dir / "results"

    for d in [intermediate_dir, detected_dir, cropped_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return {
        "run_dir": run_dir,
        "intermediate": intermediate_dir,
        "detected": detected_dir,
        "cropped": cropped_dir,
        "results": results_dir,
        "main_intermediate": base / "intermediate",
        "main_detected": base / "detected",
        "main_cropped": base / "cropped",
        "main_results": base / "results",
    }
