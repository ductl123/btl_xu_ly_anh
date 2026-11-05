# OCR có trọng số và huấn luyện bổ sung

Chọn ảnh để nhận dạng không làm thay đổi trọng số. Chỉ ảnh có nhãn được lưu vào
tập **Huấn luyện**, sau đó chạy **Huấn luyện bổ sung**, mới được dùng để cập nhật
mô hình. Ảnh Kiểm định và Kiểm tra cuối không được đưa vào bước cập nhật gradient.

Mô hình là VGG–BiLSTM–CTC, khởi tạo bằng trọng số `english_g2` của EasyOCR.
Các lớp gốc và trọng số có ghi nguồn/giấy phép tại `learning/vendor/` và
`models/line_ocr/pretrained_metadata.json`. Nhãn sử dụng 30 ký tự của dự án.
Mạng nhận cả dòng ảnh; không cần gán hộp cho từng ký tự. Các dấu chấm/gạch ngang
được coi là khoảng ngắt CTC, kết quả hiển thị chỉ gồm chữ và số.

## Dùng trên giao diện

1. Mở `Run_With_Learning.bat`, chọn **Học sâu CRNN**.
2. Chọn ảnh giao thông và chọn một biển trong bảng kết quả.
3. Mở **Gán nhãn & Huấn luyện** → **Lấy biển đang chọn**.
4. Nhìn ảnh, nhập chữ dòng trên và số dòng dưới. Điều chỉnh thanh tách dòng nếu
   đường cắt chưa nằm giữa hai dòng. Biển một dòng chỉ dùng ô nhãn thứ nhất.
   Nếu ảnh đang ngược, dùng **Xoay 90°** cho đến khi chữ đúng chiều rồi nhập nhãn.
5. Chọn tập dữ liệu và bấm **Lưu nhãn đã xác nhận**.
6. Bấm **Huấn luyện bổ sung**. Mô hình bắt đầu từ phiên bản đang dùng, học lại
   toàn bộ ảnh train cũ và mới; vì thế không chỉ ghi nhớ ảnh mới nhất.
7. Khi hoàn tất, chọn lại ảnh ở màn hình chính để nạp phiên bản đã được chấp nhận.

Có thể chọn nhiều ảnh biển đã cắt trong một lần mở. Sau mỗi lần lưu nhãn, ảnh
tiếp theo sẽ xuất hiện. Nếu chương trình chưa tìm được vùng biển từ ảnh giao thông,
cần cung cấp ảnh cắt đúng vùng biển cho bước gán nhãn; huấn luyện bộ đọc này không
tự huấn luyện thêm một bộ phát hiện vật thể.

## Chia dữ liệu

- **Huấn luyện:** dùng để học trọng số.
- **Kiểm định:** so sánh và chọn phiên bản sau mỗi vòng học.
- **Kiểm tra cuối:** đánh giá riêng; không dùng để học hoặc chọn phiên bản.

Cùng một biển số, cùng ảnh gốc hoặc cùng ảnh cắt không được nằm ở nhiều tập.
Đổi ánh sáng/xoay một ảnh không tạo thành một ảnh kiểm tra độc lập. Ảnh trùng
không làm tăng số mẫu. Bỏ mẫu trong bảng chỉ vô hiệu hóa nhãn, không xóa ảnh gốc.

Mỗi lần train lưu `models/line_ocr/runs/<version>/last.pth`, báo cáo và lịch sử.
`best.pth` chỉ được chọn khi lỗi kiểm định không xấu hơn và có cải thiện theo
tiêu chí lựa chọn trong `trainer.py` (kiểm định được ưu tiên, tập train phá hòa).
`active.json` trỏ đến phiên bản đang dùng; cập nhật bằng thay file nguyên tử sau
khi train hoàn tất. Dừng/lỗi giữa chừng không thay thế mô hình đang dùng.

## Dòng lệnh

```text
python tools/train_line_ocr.py --epochs 20 --batch-size 8 --device auto
python tools/evaluate_line_ocr.py --split test
python tools/import_training_labels.py labels.csv
```

CSV nhập nhãn có cột `image,row1,row2,split,split_ratio`; đường dẫn ảnh tương đối
với file CSV. Để trống `row2` cho biển một dòng, `split_ratio` có thể để trống để
ước lượng. Với nhiều ảnh cắt từ cùng ảnh gốc, thêm cùng một `source_group`.

## Đủ năm nhóm xử lý ảnh của đề tài

| Nhóm | Trong chương trình |
|---|---|
| Nâng cao chất lượng | Kéo giãn tương phản; CLAHE ở nhánh tìm vùng/kiểm tra ảnh |
| Lọc và khôi phục | Gaussian; tùy chọn Wiener theo giả định mờ Gaussian |
| Biên | Sobel-X, Canny |
| Phân đoạn | Thành phần liên thông/contour, tách vùng biển, tách dòng |
| Hình thái học | Closing, opening và các phép giãn/co hỗ trợ |

Wiener là một tùy chọn thí nghiệm cho nhánh tìm biển. Thông số mờ giả định có
thể không đúng với ảnh thực; không bật mặc định cho mọi ảnh. Ảnh dòng đưa vào
CRNN giữ thông tin mức xám, không bị ép nhị phân. Bộ đọc Truyền thống vẫn được
giữ để đối chứng trên cùng ảnh.

## Giới hạn cần ghi trong báo cáo

Bộ khởi tạo có 34 ảnh biển đã gán nhãn: 28 train, 2 kiểm định, 4 kiểm tra cuối.
Đây là dữ liệu phát triển từ ảnh của người dùng. Một số ảnh đã dùng để phát triển
bộ phát hiện cũ; chúng chỉ được giữ riêng đối với việc fine-tune bộ đọc hiện tại.
Tập kiểm tra rất nhỏ, không phải một đánh giá tổng quát trên ảnh giao thông mới.
Nhiều ảnh có nhãn đúng và đa dạng có thể giúp mô hình tiến bộ, không bảo đảm
thêm ảnh là mọi chỉ số sẽ tăng. Điểm softmax cũng không bảo đảm nội dung đúng.

Lần cài hiện tại chạy được bằng PyTorch CPU. Chương trình tự dùng CUDA nếu môi
trường PyTorch có hỗ trợ; việc máy có card NVIDIA chưa đồng nghĩa thư viện đã
được cài bản CUDA. Thêm dữ liệu cho các chữ ít xuất hiện và tăng số biển độc lập
ở tập kiểm định/kiểm tra trước khi báo cáo độ chính xác thực tế.
