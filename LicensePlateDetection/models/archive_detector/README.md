# Huấn luyện bộ phát hiện biển số từ archive

Đã hoàn tất **16 vòng huấn luyện trên CPU**. Bộ phát hiện lưu phiên bản tốt nhất ở vòng 9 theo điểm kiểm định. Có chương trình riêng để chọn ảnh, xem khung và lưu ảnh biển cắt.

## Dữ liệu đã dùng

| Tập | Số ảnh | Số khung biển | Vai trò |
|---|---:|---:|---|
| Train | 2.036 | 2.713 | Cập nhật trọng số: 509 tệp không có hậu tố tăng cường và **1.527 tệp `_aug`**. |
| Kiểm định | 146 | 191 | Chọn trọng số và ngưỡng điểm phát hiện. |
| Kiểm tra cuối | 73 | 94 | Chỉ đánh giá sau khi chọn trọng số. |

Nguồn là `D:\hoc_tap\hoc_tap\nam_4\xu_li_anh\btl\archive\splits`, có một lớp YOLO `Bien-so`. Đã kiểm tra 2.255 ảnh đọc được và 2.998 khung hợp lệ. Không thấy nhóm tên ảnh gốc hoặc tệp giống hệt vượt qua ranh giới các tập; do nguồn chưa có chuỗi ký tự, chưa thể xác nhận mọi ảnh chụp cùng một biển ở các thời điểm khác nhau đã được gom nhóm.

Mô hình CNN nhỏ nhận ảnh 320×320, giữ tỷ lệ ảnh và thêm lề, dự đoán tâm và kích thước khung biển. Không tải thêm thư viện hoặc trọng số bên ngoài.

## Kết quả phát hiện

| Chỉ số | Kiểm định | Kiểm tra cuối |
|---|---:|---:|
| AP50 | 79,58% | **78,84%** |
| Khung đúng / tổng số biển | 142/191 | **71/94** |
| Khung dự đoán sai | 16 | 6 |
| Biển bỏ sót | 49 | 23 |
| Precision | 89,87% | 92,21% |
| Recall | 74,35% | 75,53% |

Một khung được tính đúng khi giao với khung nhãn đạt IoU từ 0,50 và chưa có khung khác ghép với biển đó. Ngưỡng điểm 0,528 được chọn trên tập kiểm định rồi giữ nguyên trên tập kiểm tra. AP50 đo chất lượng phát hiện vị trí biển trên dải điểm; đây chưa phải kết quả đọc chữ. AP50–95 trên kiểm tra cuối là 41,67%.

Bản lưu đã được nạp lại và chạy thử thành công trên một ảnh kiểm tra. `ket_qua_phat_hien_archive.jpg` minh họa cả các trường hợp phát hiện được và bỏ sót.

## Mở để chạy thử

Trong `D:\hoc_tap\hoc_tap\nam_4\xu_li_anh\btl\LicensePlateDetection`, mở **`Run_Archive_Detector.bat`** rồi chọn một ảnh. Công cụ mở thư mục kết quả gồm ảnh vẽ khung, thư mục `crops` và `detections.json`.

Trọng số nằm ở `D:\hoc_tap\hoc_tap\nam_4\xu_li_anh\btl\LicensePlateDetection\models\archive_detector\best.pth`. Báo cáo theo vòng, bảng kê ảnh và dự đoán kiểm tra nằm cùng thư mục.

Đây là công cụ phát hiện riêng để thử trọng số mới. Bộ phát hiện mặc định trong GUI cũ và trọng số CRNN đang dùng không bị thay thế.

Để huấn luyện lại từ chính ảnh và nhãn đã kiểm tra, mở PowerShell ở thư mục chương trình và chạy:

```powershell
& ..\.venv\Scripts\python.exe tools\train_archive_detector.py --manifest models\archive_detector\dataset_manifest.json --output models\archive_detector_trial --epochs 16
```

## Đã chuẩn bị ảnh để gán nhãn OCR

Đã cắt **777 vùng biển** theo khung có sẵn từ các tệp không có hậu tố `_aug`: 556 train, 148 kiểm định, 73 kiểm tra. Bỏ 187 khung quá nhỏ khỏi hàng chờ gán chữ; các ảnh chứa chúng vẫn được dùng cho phát hiện theo tập tương ứng. Việc ưu tiên tệp không có hậu tố giúp gán nhãn một lần mỗi ảnh nguồn, thay vì nhập lại cho từng biến thể.

Kho ảnh nằm tại `D:\hoc_tap\hoc_tap\nam_4\xu_li_anh\btl\LicensePlateDetection\data\archive_ocr_queue`. ZIP bàn giao chứa ảnh, bảng kê và ảnh xem mẫu. **Đợt này chỉ huấn luyện bộ phát hiện**, vì nguồn chưa có nhãn chữ/số để huấn luyện OCR có giám sát.

1. Mở ảnh trong `crops` và điền `row1`, `row2` trong `cho_gan_nhan_ocr.csv`. Biển một dòng chỉ điền `row1`. Đọc bằng mắt, bỏ qua ảnh bị cắt thiếu hoặc quá mờ. Nhãn có thể nhập có dấu chấm/gạch ngang; khi lưu chương trình sẽ bỏ dấu phân cách.
2. Giữ cột `split` và `source_group`. Nếu thấy cùng biển xuất hiện ở hai tập, xử lý trùng trước khi đưa vào học. Đừng điền bừa các ký tự không nhìn rõ.
3. Giữ riêng các dòng đã điền vào một bản CSV mới **cùng thư mục** với bảng gốc để đường dẫn ảnh còn đúng. Có thể nhập CSV đã xác nhận bằng công cụ `tools/import_training_labels.py`, hoặc mở ảnh biển cắt trong cửa sổ **Gán nhãn & Huấn luyện** để xem gợi ý rồi xác nhận tay.
4. Sau khi có nhãn gốc, muốn dùng lại cho bản `_aug` thì kiểm tra đúng vùng biển, thứ tự dòng và chiều chữ của từng bản trước khi kế thừa nhãn. Biến thể chỉ được ở cùng tập với ảnh nguồn.

Các ảnh crop mới chưa được ghi vào kho OCR chính. File CSV để trống các cột chữ/số và không lấy dự đoán OCR tự động làm nhãn.
