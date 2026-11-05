# Kết quả huấn luyện bổ sung từ bộ 224 ảnh tăng cường

Ngày: 05/11/2026.

**Đã kích hoạt trọng số mới trong chương trình.**

## Đã huấn luyện gì?

- Tiếp tục từ mô hình đang dùng, chạy **12 vòng**, batch 8, learning rate 0,00005 trên CPU; seed 20261105.
- Dùng **196 ảnh biển cắt tăng cường + 28 ảnh biển gốc = 224 mẫu biển để học**. Đây là số ảnh biển cắt, không phải tất cả 224 ảnh toàn cảnh mới đều tham gia học.
- 196 mẫu tăng cường được cắt từ 154 ảnh toàn cảnh mới, thuộc 22 ảnh nguồn của tập học. Một ảnh toàn cảnh có thể chứa nhiều biển.
- 35 ảnh toàn cảnh mới thuộc các nguồn kiểm định/kiểm tra được giữ riêng. 35 ảnh toàn cảnh còn lại thuộc năm ảnh gốc chưa có nhãn chắc chắn nên chưa được dùng.
- Kế thừa nhãn đã có từ đúng ảnh gốc qua phép đối chiếu hình học; không lấy kết quả OCR làm nhãn huấn luyện. 34/34 vùng biển gốc đã đối chiếu được với nguồn.
- Có đủ bảy điều kiện trong tập học, mỗi điều kiện 28 mẫu biển cắt. Không dùng bộ 20.000 ảnh chữ giả lập ở đợt này.
- Giữ cố định CNN, cập nhật các lớp đọc chuỗi và đầu ra CTC. Độ thay đổi trọng số đầu ra L2: 0.540188.

## Kết quả trước và sau

| Tập ảnh biển cắt | Số ảnh | Đúng trước học | Đúng sau học | CER trước → sau |
|---|---:|---:|---:|---:|
| Ảnh gốc thuộc tập học | 28 | 27 | 28 | 0.40% → 0.00% |
| Biến thể thuộc tập học | 196 | 128 | 182 | 11.88% → 2.70% |
| Ảnh gốc kiểm định | 2 | 2 | 2 | 0.00% → 0.00% |
| Biến thể kiểm định | 14 | 9 | 10 | 10.08% → 6.72% |
| Ảnh gốc kiểm tra cuối | 4 | 4 | 4 | 0.00% → 0.00% |
| Biến thể kiểm tra cuối | 28 | 23 | 23 | 4.20% → 3.78% |

CER là số lỗi ký tự chia số ký tự chuẩn; thấp hơn là tốt hơn. Điểm tập học chỉ cho thấy mức khớp dữ liệu đã học, không phải độ chính xác trên ảnh mới ngoài thực tế.

Các biến thể kiểm định xuất phát từ hai biển gốc, còn biến thể kiểm tra xuất phát từ bốn biển thuộc ba ảnh gốc. Chúng không tương đương 14 hoặc 28 cảnh chụp độc lập. Tập dữ liệu còn nhỏ để kết luận khả năng tổng quát.

## Cách chọn phiên bản

Trong 12 vòng, chương trình ưu tiên tập kiểm định gốc và sau đó lỗi tập học. Phiên bản tốt nhất được lưu ở vòng **11**. Trước khi kích hoạt, kiểm tra thêm rằng số biển đúng và CER trên biến thể kiểm định không kém mốc ban đầu, đồng thời CER trên ảnh gốc thuộc tập học không tăng. **Không dùng tập kiểm tra cuối để chọn phiên bản.**

Lý do không kích hoạt, nếu có: Không có; đã qua các điều kiện kiểm định.

## Phạm vi áp dụng

Đây là huấn luyện **bộ đọc ký tự CRNN sau khi đã cắt và chỉnh thẳng biển**. Các mẫu xoay 45–90° được đưa về chiều đọc bằng ma trận biến đổi đã biết trước khi học OCR. Đợt này không huấn luyện bộ phát hiện vị trí biển hoặc bộ ước lượng góc; không thể dùng kết quả này để khẳng định pipeline tự tìm và xoay đúng mọi biển nghiêng.

Một số biến thể bị mờ hoặc loá quá mạnh vẫn có thể đọc sai. Các dự đoán sai được giữ trong báo cáo JSON để đối chiếu, không bị sửa bằng nhãn chuẩn khi tính điểm.

## Vị trí trọng số và dữ liệu

- Phiên bản đang dùng: `20261105T034509Z_521574`.
- Thư mục lần học: `D:\hoc_tap\hoc_tap\nam_4\xu_li_anh\btl\LicensePlateDetection\models\line_ocr\runs\20261105T034509Z_521574`.
- Bản sao lưu trước cập nhật: `D:\hoc_tap\hoc_tap\nam_4\xu_li_anh\btl\LicensePlateDetection\backups\augmented_training_20261105T034509Z_521574`.
- Kho nhãn chính: đã thêm 196 mẫu thuộc tập học; tổng cộng 230 mẫu đang hoạt động. Giữ nguyên nhãn cũ và các tập gốc.
- Kho kiểm tra đầy đủ, gồm cả biến thể giữ riêng: `D:\hoc_tap\hoc_tap\nam_4\xu_li_anh\btl\LicensePlateDetection\data\ocr_training_augmented_20261105T034509Z_521574`.

Đóng rồi mở lại chương trình, hoặc chọn lại ảnh để bộ đọc nạp phiên bản đang dùng. Khi học thêm sau này, tiếp tục giữ ảnh gốc và mọi biến thể trong cùng tập dữ liệu.

Tệp `so_sanh_train_anh_tang_cuong.csv` đi kèm phân tách kết quả theo bảy điều kiện và theo train/validation/test. Tệp `ket_qua_train_anh_tang_cuong.json` giữ dự đoán từng mẫu.

Sau khi cập nhật, đã xác nhận chương trình nạp đúng phiên bản mới, 34 nhãn cũ giữ nguyên và kiểm tra trực tiếp pipeline trên một ảnh gốc cùng biến thể mờ của nó: cả hai đọc ra 18MD207473. Đây là kiểm tra tích hợp, không phải phép đánh giá đầy đủ trên 224 ảnh toàn cảnh.
