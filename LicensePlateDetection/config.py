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

# Tùy chọn lưu ảnh trung gian: Đặt False để không sinh file ảnh rác (Sobel, Otsu, Closing...) ra đĩa
SAVE_INTERMEDIATE_STEPS = False


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
VERTICAL_BRIDGE_HEIGHT = 15
VERTICAL_BRIDGE_WIDTH = 5

# Structuring element mở rộng cho biển số cận cảnh (close-up)
LARGE_CLOSING_KERNEL_HEIGHT = 5
LARGE_CLOSING_KERNEL_WIDTH = 29

# Tùy chọn tiền xử lý cân bằng sáng cục bộ thích nghi CLAHE
USE_CLAHE_ENHANCEMENT = True
CLAHE_CLIP_LIMIT = 2.5
CLAHE_TILE_GRID_SIZE = (8, 8)

# Tùy chọn nguồn đề xuất ứng viên thứ 2 từ Detector học sâu (CNN archive detector)
USE_ARCHIVE_DETECTOR_PROPOSALS = True
ARCHIVE_DETECTOR_CONF_THRESH = 0.20
USE_TILING_FOR_FAR_PLATES = True
TILING_SIZE = 900
TILING_OVERLAP = 250

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
# Tỷ lệ kích thước Bounding Box (w / h) chuẩn biển số xe Việt Nam:
# Biển số 1 dòng (ô tô chữ nhật dài): tỷ lệ chuẩn ~3.0 - 5.5 (cho phép 2.45 - 6.0 khi chụp xiên/nghiêng)
SINGLE_ROW_ASPECT_RATIO_MIN = 2.45
SINGLE_ROW_ASPECT_RATIO_MAX = 6.0

# Biển số 2 dòng (xe máy, ô tô vuông): tỷ lệ chuẩn 1.00 - 2.25 (hỗ trợ biển chụp nghiêng/từ dưới lên, loại bỏ khung đứng w/h < 1.0)
DOUBLE_ROW_ASPECT_RATIO_MIN = 1.00
DOUBLE_ROW_ASPECT_RATIO_MAX = 2.25

# Tỷ lệ diện tích Bounding Box động so với toàn bộ ảnh (area / (W_img * H_img))
# Đặt ngưỡng tối thiểu 0.10% (0.0010) để loại bỏ nhiễu nan hoa/trục bánh xe nhưng giữ biển số thật
MIN_AREA_RATIO = 0.0010
MAX_AREA_RATIO = 0.22

# Giới hạn chiều rộng và chiều cao tương đối so với kích thước ảnh
MIN_WIDTH_RATIO = 0.040
MAX_WIDTH_RATIO = 0.85
MIN_HEIGHT_RATIO = 0.015
MAX_HEIGHT_RATIO = 0.45

# Mật độ điểm trắng (white density / fill ratio) = số pixel trắng / (w * h)
MIN_WHITE_DENSITY = 0.12
MAX_WHITE_DENSITY = 0.95

# Ngưỡng Non-Maximum Suppression (IoU) để loại bỏ các bounding box trùng lặp nhau
NMS_IOU_THRESHOLD = 0.35

# Số lượng ứng viên hình học tiềm năng tối đa giữ lại chuyển sang bước nhận diện OCR
# Tăng lên 16 để không bỏ sót các biển nhỏ phía sau trong ảnh có nhiều xe (Multi-vehicle retention)
TOP_CANDIDATES_COUNT = 16

# Tỷ lệ cắt tỉa lề đáy để loại bỏ chữ phấn hoặc bùn bẩn dè xe
TRIM_BOTTOM_RATIO = 0.12

# Độ đệm lề (margin) khi cắt ảnh biển số (tính theo % của w và h)
CROP_MARGIN_RATIO = 0.05

# Similarity score floor; not a calibrated probability. Low scores keep only localization.
OCR_MIN_MEAN_SCORE = 0.75

# Line OCR learns weights from explicitly labelled plate rows.
OCR_BACKEND = os.environ.get('PLATE_OCR_BACKEND', 'crnn')
# Optional restoration assumes a mild Gaussian blur; compare before enabling.
USE_WIENER_RESTORATION = False
