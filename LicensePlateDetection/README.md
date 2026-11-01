# License Plate Detection from Traffic Images Using Edge Detection and Morphological Operations

> **Phát hiện biển số xe từ hình ảnh giao thông sử dụng phương pháp phát hiện cạnh và các thao tác hình thái học**  
> Dự án môn học Xử lý ảnh số (Digital Image Processing)  
> Cài đặt thuần túy bằng **Python** và **NumPy** (Không sử dụng YOLO, OpenCV, PyTorch, Deep Learning).

---

## 1. Giới thiệu dự án

Hệ thống nhận dạng biển số tự động (Automatic Number Plate Recognition - ANPR) là thành phần cốt lõi trong các ứng dụng giao thông thông minh, trạm thu phí không dừng và bãi đỗ xe tự động.

Đặc trưng mấu chốt của vùng biển số xe là: **chứa mật độ cao các cạnh thẳng đứng nằm sát nhau** do các chữ số và chữ cái in trên nền tương phản tạo ra, trong khi thân xe, mặt đường hay kính xe thường có bề mặt trơn nhẵn với mật độ cạnh thấp.

Hệ thống áp dụng quy trình 6 bước xử lý ảnh kinh điển:
1. **Tiền xử lý (Preprocessing):** Chuyển ảnh RGB sang Grayscale, chuẩn hóa kích thước và khử nhiễu bằng bộ lọc Gaussian Blur 2D.
2. **Phát hiện cạnh dọc (Vertical Edge Detection):** Áp dụng toán tử đạo hàm Sobel theo phương X ($Sobel-X$) để làm nổi bật các cạnh thẳng đứng của ký tự, chuẩn hóa trị tuyệt đối về dải $[0, 255]$.
3. **Nhị phân hóa (Binarization):** Tự động xác định ngưỡng tối ưu bằng thuật toán Otsu ($Otsu's\ Thresholding$) cực đại hóa phương sai liên lớp $\sigma_B^2(t)$.
4. **Xử lý hình thái học (Morphological Operations):** 
   - Phép Đóng ($Closing = (A \oplus B) \ominus B$) với phần tử cấu trúc chữ nhật nằm ngang (SE $3 \times 19$) để gộp các nét ký tự thẳng đứng thành một khối đặc liền nét.
   - Phép Mở ($Opening = (A \ominus B) \oplus B$) với phần tử cấu trúc nhỏ ($3 \times 3$) để triệt tiêu các đốm nhiễu li ti.
5. **Gán nhãn thành phần liên thông (Connected Component Labeling):** Thuật toán loang BFS (8-connectivity) phát hiện các khối trắng, tính hộp bao Bounding Box $(x, y, w, h)$, diện tích điểm trắng và mật độ lấp đầy (white density).
6. **Lọc ứng viên (Candidate Filtering):** Lọc theo tỷ lệ rộng/cao $w/h$ (biển 1 dòng $\approx 4.5 - 5.0$, biển 2 dòng $\approx 1.3 - 1.5$), tỷ lệ diện tích và mật độ điểm trắng, áp dụng Non-Maximum Suppression (NMS) để loại bỏ trùng lặp.

---

## 2. Cấu trúc thư mục dự án

```text
LicensePlateDetection/
│
├── main.py                         # Điểm vào chính: Chế độ 1 ảnh, hàng loạt (batch) hoặc menu CLI
├── config.py                       # Quản lý tập trung toàn bộ tham số thuật toán
├── compare.py                      # Module thực nghiệm so sánh (Filters, Sobel vs Canny, SE, Otsu)
├── requirements.txt                # Danh sách thư viện phụ thuộc
├── README.md                       # Tài liệu hướng dẫn sử dụng chi tiết
│
├── image_processing/               # Gói xử lý ảnh số tự cài đặt thuần Python & NumPy
│   ├── __init__.py
│   ├── preprocessing.py            # RGB->Gray, Resize, Separable Gaussian Blur, Median Blur
│   ├── edge_detection.py           # Sobel-X, Sobel-Y, Canny Edge Detection
│   ├── thresholding.py             # Thuật toán Otsu tự động & Fixed Thresholding
│   ├── morphology.py               # Dilation, Erosion, Closing, Opening tự cài đặt
│   ├── connected_components.py     # BFS Connected Component Labeling, tính BBox, Density
│   └── candidate_filter.py         # Lọc tỷ lệ hình học, kích thước, mật độ, NMS
│
├── utils/                          # Gói tiện ích phụ trợ
│   ├── __init__.py
│   ├── image_utils.py              # Đọc/lưu ảnh (Pillow), vẽ BBox, cắt biển số, vẽ biểu đồ Matplotlib
│   └── file_utils.py               # Quét file ảnh, xuất báo cáo CSV (hỗ trợ tiếng Việt Excel)
│
├── input/
│   └── images/                     # Thư mục chứa các ảnh đầu vào cần xử lý (JPG, PNG)
│
└── output/                         # Thư mục lưu kết quả đầu ra
    ├── intermediate/               # Ảnh trung gian từng bước (Grayscale, Sobel, Otsu, Closing, ...)
    ├── detected/                   # Ảnh gốc có vẽ Bounding Box bao quanh biển số
    ├── cropped/                    # Ảnh cắt riêng từng biển số phát hiện được
    ├── results/                    # File bảng tính CSV và các đồ thị so sánh thực nghiệm
    └── runs/                       # Lưu trữ chi tiết theo từng phiên chạy (tránh ghi đè)
```

---

## 3. Yêu cầu công nghệ & Cài đặt môi trường

### Yêu cầu
- Python 3.8 trở lên (Khuyến nghị Python 3.10 - 3.14).
- Thư viện: `numpy`, `pillow`, `matplotlib`.

### Các bước cài đặt

1. **Mở terminal/cmd tại thư mục dự án `LicensePlateDetection`:**
   ```bash
   cd LicensePlateDetection
   ```

2. **Cài đặt thư viện bằng pip:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Hoặc nếu dùng Python Launcher trên Windows: `py -m pip install -r requirements.txt`)*

---

## 4. Hướng dẫn chạy chương trình

Hệ thống hỗ trợ 3 cách chạy linh hoạt:

### Cách 1: Khởi chạy Giao diện đồ họa trực quan (Desktop GUI) - Khuyến nghị
```bash
python gui.py
# hoặc
python main.py --gui
```
Giao diện cửa sổ Desktop trực quan:
- Nút **📁 Chọn Ảnh**: Chọn ảnh bất kỳ để xử lý và hiển thị ngay kết quả.
- Tab **🎯 Kết Quả Phát Hiện (BBox)**: Hiển thị ảnh gốc kèm Bounding Box màu viền xanh lá / vàng.
- Tab **🧩 Lưới 8 Bước Xử Lý**: Xem ảnh tổng hợp 8 bước trung gian.
- Tab **📐 Cạnh Dọc Sobel-X**: Xem chi tiết cạnh dọc do toán tử Sobel-X trích xuất.
- Tab **⬛ Nhị Phân & Closing**: Xem ảnh nhị phân và ảnh sau phép đóng hình thái học.
- Bảng danh sách biển số bên phải: Click vào từng biển số để xem ảnh cắt riêng (Cropped) và chi tiết thông số tọa độ, tỷ lệ w/h, độ tin cậy.
- Nút **⚡ Xử Lý Hàng Loạt** & **📊 So Sánh Thực Nghiệm**.

### Cách 2: Chạy Menu tương tác thân thiện qua Terminal
Chỉ cần chạy lệnh không tham số:
```bash
python main.py
```
*(Trên Windows có thể dùng: `py main.py`)*

Màn hình hiển thị menu chọn lựa:
- **Phím 1:** Chọn 1 ảnh từ thư mục `input/images/` để xử lý và hiển thị đồ thị 8 bước chi tiết.
- **Phím 2:** Xử lý hàng loạt toàn bộ ảnh có trong thư mục `input/images/`.
- **Phím 3:** Chạy 4 bài thực nghiệm so sánh thuật toán.
- **Phím 4:** Mở nhanh thư mục kết quả `output/` trên File Explorer.
- **Phím 5:** Khởi động giao diện đồ họa Desktop GUI.
- **Phím 0:** Thoát chương trình.

### Cách 2: Xử lý 1 bức ảnh cụ thể qua dòng lệnh (CLI)
```bash
python main.py --image "input/images/ten_anh.jpg"
```
*Tùy chọn bổ sung:*
- `--show`: Mở popup cửa sổ Matplotlib hiển thị ngay sơ đồ 8 bước.
- `--no-intermediate`: Bỏ qua lưu ảnh trung gian để tăng tốc độ.

### Cách 3: Xử lý hàng loạt toàn bộ thư mục input
```bash
python main.py --batch
```

### Cách 4: Chạy chuỗi thực nghiệm so sánh (Experimental Comparisons)
```bash
python main.py --compare
```
Sẽ tự động xuất 4 ảnh phân tích so sánh chất lượng cao vào `output/results/`:
1. `so_sanh_1_bo_loc_gaussian_vs_median.png`: So sánh làm mờ khử nhiễu.
2. `so_sanh_2_sobel_vs_canny.png`: So sánh Sobel-X, Sobel-Y, Magnitude và Canny.
3. `so_sanh_3_kich_thuoc_structuring_element.png`: So sánh các kích thước phần tử cấu trúc SE trong phép Closing.
4. `so_sanh_4_phan_nguong_otsu_vs_co_dinh.png`: So sánh ngưỡng Otsu tự thích nghi và ngưỡng cố định.

---

## 5. Vị trí lưu kết quả đầu ra

Sau khi chạy xong, kết quả được phân loại rõ ràng:
- **`output/detected/`**: Chứa ảnh gốc kèm Bounding Box màu xanh lục (biển 1 dòng) hoặc vàng cam (biển 2 dòng), mã số biển và điểm tin cậy (Confidence Score).
- **`output/cropped/`**: Chứa ảnh cắt riêng từng biển số theo tên file dạng: `{tên_ảnh}_plate_{thứ_tự}_{loại_biển}.png`.
- **`output/intermediate/{tên_ảnh}/`**: Chứa toàn bộ 8 ảnh minh chứng từng bước:
  - `01_grayscale.png`: Ảnh mức xám.
  - `02_gaussian_blurred.png`: Ảnh sau lọc Gaussian.
  - `03_sobel_x.png`: Ảnh gradient cạnh dọc Sobel-X.
  - `04_binary_otsu.png`: Ảnh nhị phân Otsu.
  - `05_morph_closing.png`: Ảnh sau phép Closing.
  - `06_morph_opening.png`: Ảnh sau phép Opening.
  - `07_connected_components.png`: Bản đồ các vùng liên thông (màu ngẫu nhiên).
  - `08_pipeline_summary_grid.png`: Lưới tổng hợp trực quan 8 giai đoạn.
- **`output/results/detection_results.csv`**: File CSV thống kê chi tiết (mã hóa `utf-8-sig` mở bằng Excel không lỗi font): Tên ảnh, thứ tự biển, loại biển, điểm số, tọa độ $(x, y, w, h)$, tỷ lệ $w/h$, mật độ điểm trắng, đường dẫn ảnh cắt.

---

## 6. Cách tùy chỉnh tham số trong `config.py`

Mọi tham số đều được quản lý tập trung và có chú thích tiếng Việt trong `config.py`:
- `TARGET_MAX_DIM` (mặc định `900`): Kích thước cạnh dài tối đa khi xử lý (tự động quy đổi tọa độ ngược lại kích thước gốc khi vẽ và cắt).
- `GAUSSIAN_KERNEL_SIZE` & `GAUSSIAN_SIGMA` (mặc định `5` và `1.0`): Độ mờ khử nhiễu.
- `CLOSING_KERNEL_HEIGHT` & `CLOSING_KERNEL_WIDTH` (mặc định `3` và `19`): Kích thước SE hình chữ nhật nằm ngang. Nếu biển số trong ảnh chụp quá nhỏ, có thể giảm width xuống `13 - 15`; nếu biển số to, có thể tăng lên `21 - 25`.
- `SINGLE_ROW_ASPECT_RATIO_MIN/MAX` (mặc định `2.6` đến `6.2`): Dải tỷ lệ rộng/cao cho biển 1 dòng.
- `DOUBLE_ROW_ASPECT_RATIO_MIN/MAX` (mặc định `1.05` đến `2.2`): Dải tỷ lệ rộng/cao cho biển 2 dòng.
- `MIN_WHITE_DENSITY` & `MAX_WHITE_DENSITY` (mặc định `0.25` và `0.95`): Mật độ điểm trắng của khối biển số.

---

## 7. Giải thích chức năng từng module

| Module | Chức năng chính |
| :--- | :--- |
| `image_processing/preprocessing.py` | Chuyển RGB sang Grayscale theo trọng số quang học, chuẩn hóa kích thước, lọc Gaussian 2D phân rã (Separable Convolution), lọc trung vị (Median Filter). |
| `image_processing/edge_detection.py` | Tích chập toán tử đạo hàm Sobel-X để tập trung bắt cạnh dọc của ký tự, chuẩn hóa trị tuyệt đối về $[0, 255]$; cài đặt thuật toán Canny (Gaussian -> Gradient -> NMS -> Hysteresis) phục vụ so sánh. |
| `image_processing/thresholding.py` | Tính toán histogram 256 mức xám, xác suất tích lũy, cực đại hóa phương sai giữa hai lớp để tự động chọn ngưỡng Otsu $T^*$; phân ngưỡng cố định. |
| `image_processing/morphology.py` | Tự cài đặt các phép toán hình thái học (Dilation, Erosion, Closing, Opening) bằng NumPy vectorization, tối ưu tốc độ xử lý nhanh gấp hàng trăm lần vòng lặp lồng. |
| `image_processing/connected_components.py` | Triển khai thuật toán BFS (8-connectivity) duyệt tìm các vùng liên thông, tính Bounding Box $(x, y, w, h)$, diện tích điểm trắng và mật độ lấp đầy. |
| `image_processing/candidate_filter.py` | Lọc các vùng ứng viên dựa trên tỷ lệ hình học (biển 1 dòng / 2 dòng), kích thước tương đối và mật độ; tính điểm tin cậy (Confidence Score); áp dụng NMS loại bỏ khung trùng lặp. |
| `utils/image_utils.py` | Đọc ảnh, lưu ảnh đa định dạng, vẽ Bounding Box kèm nhãn lên ảnh gốc (quy đổi scale factor), cắt vùng biển số có thêm lề an toàn (margin), tạo biểu đồ tổng quan 8 bước bằng Matplotlib. |
| `utils/file_utils.py` | Quét file ảnh định dạng JPG/PNG/BMP, tạo thư mục phiên chạy (runs) tránh ghi đè kết quả, ghi xuất file CSV chuẩn UTF-8-BOM. |
| `compare.py` | Tự động sinh 4 biểu đồ so sánh thực nghiệm đối chiếu các kỹ thuật xử lý ảnh phục vụ mục báo cáo đề tài. |
| `main.py` | Điều phối toàn bộ pipeline, hỗ trợ CLI và Menu tương tác cho người dùng. |

