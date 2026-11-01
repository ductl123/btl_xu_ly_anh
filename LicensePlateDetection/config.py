"""
config.py - Cấu hình tập trung cho toàn bộ hệ thống phát hiện biển số xe.
Mọi tham số tiền xử lý, lọc cạnh, hình thái học, gán nhãn và lọc ứng viên được định nghĩa tại đây.
"""

import os
from pathlib import Path

# ==============================================================================
# 1. ĐƯỜNG DẪN THƯ MỤC DỰ ÁN
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent

# Thư mục đầu vào chứa ảnh cần nhận dạng
INPUT_DIR = BASE_DIR / "input" / "images"

# Thư mục gốc đầu ra
OUTPUT_DIR = BASE_DIR / "output"

# Các thư mục con lưu kết quả
INTERMEDIATE_DIR = OUTPUT_DIR / "intermediate"
DETECTED_DIR = OUTPUT_DIR / "detected"
CROPPED_DIR = OUTPUT_DIR / "cropped"
RESULTS_DIR = OUTPUT_DIR / "results"

# Đảm bảo các thư mục tồn tại
for directory in [INPUT_DIR, INTERMEDIATE_DIR, DETECTED_DIR, CROPPED_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 2. CẤU HÌNH BƯỚC 1: TIỀN XỬ LÝ (PREPROCESSING)
# ==============================================================================
# Chuẩn hóa kích thước: Chiều dài tối đa (max dimension) khi xử lý
# Ảnh lớn sẽ được thu nhỏ giữ nguyên tỷ lệ (aspect ratio) để tăng tốc độ và ổn định
TARGET_MAX_DIM = 900

# Bộ lọc làm mờ Gaussian để giảm nhiễu hạt
GAUSSIAN_KERNEL_SIZE = 5       # Kích thước ma trận lọc (phải là số lẻ: 3, 5, 7)
GAUSSIAN_SIGMA = 1.0           # Độ lệch chuẩn sigma của hàm Gaussian

# Bộ lọc trung vị (Median Filter) dùng cho mục so sánh thực nghiệm
MEDIAN_KERNEL_SIZE = 3


# ==============================================================================
# 3. CẤU HÌNH BƯỚC 2: PHÁT HIỆN CẠNH (EDGE DETECTION)
# ==============================================================================
# Hướng đạo hàm Sobel: 'x' (ưu tiên bắt cạnh dọc của ký tự biển số), 'y', hoặc 'both'
SOBEL_DIRECTION = 'x'

# Tham số Canny Edge Detector (dùng cho phần so sánh thực nghiệm)
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150


# ==============================================================================
# 4. CẤU HÌNH BƯỚC 3 & 4: NHỊ PHÂN HÓA VÀ HÌNH THÁI HỌC (MORPHOLOGY)
# ==============================================================================
# Phương pháp nhị phân: 'otsu' (mặc định) hoặc 'fixed' (ngưỡng cố định)
THRESHOLD_METHOD = 'otsu'
FIXED_THRESHOLD_VALUE = 120

# Phép Đóng (Closing) A • B = (A ⊕ B) ⊖ B
# Structuring Element hình chữ nhật nằm ngang để nối các nét ký tự thẳng đứng
CLOSING_KERNEL_HEIGHT = 3       # Chiều cao phần tử cấu trúc ngang
CLOSING_KERNEL_WIDTH = 19       # Chiều rộng phần tử cấu trúc ngang

# Cầu nối dọc để kết nối dòng 1 và dòng 2 của biển 2 dòng khi chụp từ trên xuống / chụp nghiêng
VERTICAL_BRIDGE_HEIGHT = 9
VERTICAL_BRIDGE_WIDTH = 5

# Phép Mở (Opening) A ∘ B = (A ⊖ B) ⊕ B
# Structuring Element kích thước nhỏ để khử nhiễu đốm lẻ tẻ
OPENING_KERNEL_HEIGHT = 3
OPENING_KERNEL_WIDTH = 3

# Các kích thước phần tử cấu trúc để chạy so sánh thực nghiệm
SE_COMPARISON_CONFIGS = [
    {"name": "SE_Narrow (3x9)",   "height": 3, "width": 9},
    {"name": "SE_Standard (3x19)", "height": 3, "width": 19},
    {"name": "SE_Wide (3x29)",     "height": 3, "width": 29},
    {"name": "SE_Thick (5x19)",    "height": 5, "width": 19},
]


# ==============================================================================
# 5. CẤU HÌNH BƯỚC 5: GÁN NHÃN VÙNG LIÊN THÔNG (CONNECTED COMPONENT LABELING)
# ==============================================================================
# Liên thông: 8 lân cận (8-connectivity) hoặc 4 lân cận (4-connectivity)
CONNECTIVITY = 8


# ==============================================================================
# 6. CẤU HÌNH BƯỚC 6: BỘ LỌC ỨNG VIÊN BIỂN SỐ (CANDIDATE FILTERING)
# ==============================================================================
# Tỷ lệ kích thước Bounding Box (w / h):
# Biển số 1 dòng (dài): tỷ lệ chuẩn ~4.5 - 5.0 (cho phép khoảng 2.4 - 6.5)
SINGLE_ROW_ASPECT_RATIO_MIN = 2.4
SINGLE_ROW_ASPECT_RATIO_MAX = 6.5

# Biển số 2 dòng (vuông): tỷ lệ chuẩn ~1.3 - 1.5 (cho phép 0.95 - 2.3 để bao quát góc chụp nghiêng và từ trên cao)
DOUBLE_ROW_ASPECT_RATIO_MIN = 0.95
DOUBLE_ROW_ASPECT_RATIO_MAX = 2.3

# Tỷ lệ diện tích Bounding Box so với toàn bộ ảnh (area / (W_img * H_img))
MIN_AREA_RATIO = 0.0010         # Tránh các chấm nhiễu quá nhỏ
MAX_AREA_RATIO = 0.25           # Tránh các mảng lớn như thân xe, kính xe

# Giới hạn chiều rộng và chiều cao tương đối so với kích thước ảnh
MIN_WIDTH_RATIO = 0.05
MAX_WIDTH_RATIO = 0.80
MIN_HEIGHT_RATIO = 0.02
MAX_HEIGHT_RATIO = 0.45

# Mật độ điểm trắng (white density / fill ratio) = số pixel trắng / (w * h)
# Điều chỉnh từ 0.14 đến 0.95 để không bỏ sót biển số chụp nghiêng/chụp từ trên cao
MIN_WHITE_DENSITY = 0.14
MAX_WHITE_DENSITY = 0.95

# Ngưỡng Non-Maximum Suppression (IoU) để loại bỏ các bounding box trùng lặp nhau
NMS_IOU_THRESHOLD = 0.35

# Độ đệm lề (margin) khi cắt ảnh biển số (tính theo % của w và h)
CROP_MARGIN_RATIO = 0.05

