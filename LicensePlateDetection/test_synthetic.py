"""
test_synthetic.py - Kiểm thử tự động với ảnh mô phỏng (Synthetic Test)
Tạo ảnh giả lập chứa phương tiện và biển số xe để kiểm tra tính toàn vẹn của 6 bước xử lý ảnh.
"""

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from main import process_single_image
from utils.file_utils import create_run_folders
import config


def generate_synthetic_plate_image(output_path: Path) -> Path:
    """
    Sinh ảnh giả lập xe với biển số trắng viền đen, chứa các nét ký tự thẳng đứng màu đen.
    Kích thước ảnh 800x600.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (800, 600), color=(140, 150, 160)) # Nền đường xám
    draw = ImageDraw.Draw(img)

    # Thân xe ô tô màu xanh sẫm
    draw.rectangle([150, 200, 650, 480], fill=(40, 50, 90))
    # Kính xe
    draw.rectangle([220, 230, 580, 310], fill=(70, 90, 120))
    # Cản sau xe
    draw.rectangle([170, 400, 630, 460], fill=(30, 35, 45))

    # Biển số xe (trắng, viền đen) tại vị trí trung tâm cản sau
    # Tỷ lệ w=180, h=40 -> aspect ratio = 4.5 (biển 1 dòng)
    plate_x1, plate_y1 = 310, 410
    plate_x2, plate_y2 = 490, 450
    draw.rectangle([plate_x1, plate_y1, plate_x2, plate_y2], fill=(250, 250, 250), outline=(0, 0, 0), width=2)

    # Mô phỏng các nét ký tự thẳng đứng (tạo cạnh dọc Sobel)
    # Ví dụ ký tự '30F - 123.45' gồm các nét thẳng đứng dày 3-4px
    char_positions = [325, 340, 355, 375, 395, 410, 425, 445, 460, 475]
    for cx in char_positions:
        draw.rectangle([cx, plate_y1 + 8, cx + 4, plate_y2 - 8], fill=(10, 10, 10))

    img.save(output_path)
    print(f"[+] Đã tạo ảnh kiểm thử giả lập tại: {output_path}")
    return output_path


def run_unit_tests():
    print("\n" + "="*60)
    print(" CHẠY KIỂM THỬ ĐƠN VỊ VÀ ẢNH GIẢ LẬP (UNIT & INTEGRATION TEST)")
    print("="*60)

    test_dir = config.BASE_DIR / "tests"
    test_dir.mkdir(parents=True, exist_ok=True)
    synth_img_path = test_dir / "synthetic_car_plate.png"
    generate_synthetic_plate_image(synth_img_path)

    output_dirs = create_run_folders(config.OUTPUT_DIR, run_name="test_synthetic_run")

    records = process_single_image(synth_img_path, output_dirs, save_intermediate=True, show_plot=False)

    print(f"\n[*] Kết quả kiểm thử trên ảnh giả lập:")
    print(f"  - Số biển số phát hiện được: {len(records)}")
    assert len(records) > 0, "LỖI: Không phát hiện được biển số trên ảnh giả lập!"

    found = False
    for r in records:
        print(f"  + Ứng viên: Type={r['plate_type']}, Box=[x={r['orig_x']}, y={r['orig_y']}, w={r['orig_w']}, h={r['orig_h']}], Score={r['confidence_score']}")
        # Vị trí gốc là [310, 410, 180, 40]
        if abs(r['orig_x'] - 310) < 50 and abs(r['orig_y'] - 410) < 50:
            found = True
            print("  --> [XÁC NHẬN] Đã phát hiện chính xác vị trí biển số xe mô phỏng!")

    assert found, "LỖI: Tọa độ biển số phát hiện không khớp với vị trí mô phỏng!"
    print("\n[OK] TOÀN BỘ KIỂM THỬ THÀNH CÔNG VÀ CHÍNH XÁC 100%!\n")


if __name__ == "__main__":
    run_unit_tests()

