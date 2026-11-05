# Sinh dữ liệu giả lập và bổ sung ảnh thật cho OCR biển số

**Đã tạo sẵn 20.000 ảnh** trong dự án ngày 05/11/2026: 8.000 dòng mã đầu, 8.000 dòng số và 4.000 chuỗi một dòng, đủ 30 ký tự. Lô dữ liệu: `data/synthetic_ocr/batches/c28a72079fe521c06cb5/`. Bạn không cần sinh lại lô này trước khi thử huấn luyện.

## Những gì đã triển khai

- Bộ sinh ảnh dòng chữ bằng Pillow, dùng Bahnschrift, Arial Bold và Arial Narrow Bold có sẵn trong Windows. Có thể đưa thêm font `.ttf` của bạn. Đây là các font thay thế; không coi Bahnschrift là bản sao hoàn toàn của font biển số Việt Nam.
- Ảnh gồm dòng mã đầu, dòng số phía dưới và chuỗi một dòng. Nhãn chỉ chứa **30 ký tự `0123456789ABCDEFGHKLMNPSTUVXYZ`**; dấu `.` và `-` có thể xuất hiện trong ảnh nhưng bị loại khỏi nhãn và kết quả OCR.
- Biến đổi ngẫu nhiên: nghiêng tối đa ±15°, thay đổi độ đậm, sáng/tối, tương phản, bóng sáng, nhiễu, làm mờ, vết xước mảnh và nén JPEG. Có nền trắng, vàng và một phần nền xanh. Chừa khoảng cách giữa ký tự để hạn chế nét dính do font quá đậm.
- GUI tự gợi ý nhãn từ CRNN hiện tại, cho bạn sửa và bấm **Lưu nhãn đã xác nhận**. Gợi ý không tự trở thành nhãn huấn luyện.
- Có thể trộn ảnh giả và ảnh thật khi huấn luyện bổ sung; tiếp tục từ trọng số đang dùng, không học lại từ đầu mỗi lần.

Sinh nhiều ảnh không bảo đảm độ chính xác 90–95%. Nhãn được tạo trực tiếp từ chuỗi đã vẽ nên không cần người gõ lại, nhưng ảnh biến đổi vẫn có thể khó đọc. Chất lượng phải được đo trên ảnh thật chưa tham gia học. Bộ sinh tạo biến thể để học hình dáng ký tự, không phải công cụ xác nhận mọi mã biển sinh ra đều là biển đăng ký hợp lệ.

## 1. Dùng ngay trong giao diện

Thư mục chương trình:

`D:\hoc_tap\hoc_tap\nam_4\xu_li_anh\btl\LicensePlateDetection`

1. Mở `Run_With_Learning.bat`. Nếu chương trình đang mở từ trước, đóng rồi mở lại để nạp phần cập nhật.
2. Chọn **Gán nhãn & Huấn luyện**.
   Nếu màn hình nhỏ, cuộn cửa sổ xuống để thấy phần sinh dữ liệu và huấn luyện.
3. Nhập số ảnh ở ô **Ảnh giả cần sinh**, chẳng hạn `20000` hoặc `50000`, rồi bấm **Sinh ảnh giả**.
4. Chờ thông báo hoàn tất. Bấm **Xem ảnh mẫu** để kiểm tra hình ảnh và nhãn. Mỗi lần bấm sinh là một lô mới với hạt ngẫu nhiên mới.
5. Tích **Trộn ảnh giả khi train** nếu muốn dùng dữ liệu giả trong lần học tiếp theo.

Sinh ảnh không tự bắt đầu huấn luyện và không đổi trọng số đang dùng. Có thể dừng sinh; lô chưa hoàn tất sẽ không được đưa vào học.

## 2. Thu ảnh thật và xác nhận nhãn

1. Ở màn hình chính, chọn ảnh xe để pipeline phát hiện, chỉnh góc và cắt biển.
2. Chọn biển trong bảng kết quả, mở **Gán nhãn & Huấn luyện**, bấm **Lấy biển đang chọn ở màn hình chính**. Hoặc chọn **Mở nhiều ảnh biển đã cắt**.
3. Chọn đúng **1 dòng / 2 dòng**. Nếu ảnh nằm ngang/ngược, bấm **Xoay 90°** đến đúng chiều. Điều chỉnh thanh tách để hai ảnh nhỏ chứa trọn từng dòng và không lẫn dòng bên cạnh.
4. CRNN tự điền gợi ý vào ô còn trống. Có thể bấm **Gợi ý OCR** để yêu cầu lại. Gợi ý không ghi đè nội dung bạn đã nhập. Nếu đổi ảnh hoặc vị trí tách, kết quả OCR cũ không được áp dụng sang ảnh mới.
5. Nhìn ảnh, sửa từng ký tự. Ví dụ ảnh ghi `18-C1` / `277.11`, nhãn lưu là `18C1` / `27711`.
6. Chọn tập dữ liệu, rồi bấm **Lưu nhãn đã xác nhận**. Với nhiều ảnh, chương trình chuyển sang ảnh tiếp theo sau khi lưu.

Nếu cắt sai hoặc ảnh không thể đọc chắc chắn, sửa phần cắt/chọn ảnh khác trước khi lưu. Không xác nhận theo dự đoán của máy khi không nhìn được ký tự. Dữ liệu thật nằm trong `data/ocr_training`, tách riêng dữ liệu giả. Lịch sử lưu giữ cả gợi ý OCR và hành động xác nhận để đối chiếu.

Chia theo **xe/biển và ảnh gốc**, không chia các bản chụp của cùng một biển vào cả train và kiểm tra. Chương trình chặn việc đưa cùng biển hoặc cùng nguồn ảnh gốc sang nhiều tập. Với ảnh tự cắt bên ngoài, hãy giữ các ảnh cùng nguồn trong cùng một tập vì chương trình không thể tự biết mọi quan hệ giữa các tệp độc lập.

- **Huấn luyện:** ảnh dùng cập nhật trọng số.
- **Kiểm định:** ảnh thật riêng dùng so sánh và chọn phiên bản.
- **Kiểm tra cuối:** ảnh thật riêng để đánh giá; không dùng học hoặc chọn mô hình.

## 3. Huấn luyện sau khi thêm ảnh

Chương trình yêu cầu có ảnh thật trong cả tập Huấn luyện và Kiểm định. Tích **Trộn ảnh giả khi train**, chọn số vòng rồi bấm **Huấn luyện bổ sung**.

Ô **Tối đa/vòng** có ý nghĩa:

| Giá trị | Cách sử dụng |
|---|---|
| `256` | Dùng tối đa 256 ảnh giả mỗi vòng; phù hợp chạy thử trên CPU. Chuyển dần qua các ảnh khác ở những vòng tiếp theo. |
| `1000` | Dùng tối đa 1.000 ảnh giả mỗi vòng; lâu hơn. |
| `0` | Dùng toàn bộ ảnh giả hoàn tất mỗi vòng; chạy 20.000–50.000 ảnh có thể rất lâu trên CPU. |

Ví dụ 20 vòng × 256 ảnh chỉ sử dụng tối đa 5.120 ảnh giả, không phải cả 20.000 ảnh. Hệ thống phát lại ảnh thật để số lượt dòng thật và dòng giả xấp xỉ cân bằng; việc này hạn chế dữ liệu giả lấn át tập thật ít ảnh nhưng cũng tăng thời gian học. Sau giai đoạn có ảnh giả, có thể bỏ tích **Trộn ảnh giả khi train** và học tiếp bằng toàn bộ ảnh thật đã xác nhận.

Hiện tại môi trường cài PyTorch bản CPU. Phần huấn luyện giữ cố định CNN trích xuất đặc trưng đã học sẵn, cập nhật các lớp đọc chuỗi BiLSTM và đầu ra CTC. Đây là huấn luyện thích nghi từ mô hình có sẵn, không phải huấn luyện toàn bộ CRNN ngẫu nhiên từ đầu.

Mỗi lần học đều so sánh với mô hình hiện tại. Chỉ kích hoạt phiên bản đáp ứng điều kiện chọn của chương trình: ưu tiên lỗi ký tự trên kiểm định thật; khi bằng nhau mới xét train thật rồi mẫu train giả. Các điểm đo trên ảnh giả là **điểm tập học**, không được báo thành độ chính xác trên ảnh thật. Chương trình lưu cả báo cáo và trọng số thử nghiệm; nếu không đạt điều kiện hoặc bị dừng thì giữ mô hình đang dùng.

Tập kiểm định thật hiện còn nhỏ: cần tiếp tục bổ sung xe/biển mới, ảnh ô tô và xe máy, sáng/tối, nghiêng, khoảng cách khác nhau trước khi kết luận khả năng tổng quát.

## 4. Sinh bằng lệnh để tái lập dữ liệu

Mở PowerShell tại thư mục chương trình:

```powershell
& ..\.venv\Scripts\python.exe tools\generate_synthetic_ocr.py --count 20000 --seed 20261105
```

Sinh thêm 50.000 ảnh với hạt khác:

```powershell
& ..\.venv\Scripts\python.exe tools\generate_synthetic_ocr.py --count 50000 --seed 20261106
```

Font tự chọn:

```powershell
& ..\.venv\Scripts\python.exe tools\generate_synthetic_ocr.py --count 20000 --seed 20261107 --font "C:\Windows\Fonts\bahnschrift.ttf" --font "C:\Windows\Fonts\arialbd.ttf"
```

Cùng số ảnh, seed, font và tập biển cần loại trừ sẽ dùng lại lô đã hoàn tất, không nhân đôi dữ liệu. Thay font hoặc tham số sẽ sinh lô riêng. Chương trình loại chuỗi biển trùng với các biển kiểm định/kiểm tra thật đã biết; khi huấn luyện cũng lọc lại theo tập thật hiện tại.

Huấn luyện thử với 256 ảnh giả/vòng:

```powershell
& ..\.venv\Scripts\python.exe tools\train_line_ocr.py --epochs 20 --synthetic --synthetic-per-epoch 256
```

Dùng toàn bộ ảnh giả/vòng: đổi `256` thành `0`. Lệnh mặc định tiếp tục trọng số đang dùng. Tùy chọn `--from-pretrained` bắt đầu thử từ CRNN gốc có sẵn, vẫn giữ cơ chế so sánh với mô hình đang dùng trước khi kích hoạt. Với `--start-checkpoint`, có thể chỉ định một trọng số thử nghiệm để học tiếp.

## 5. Dữ liệu được lưu thế nào?

Mỗi lô trong `data/synthetic_ocr/batches/<mã_lô>/` có:

| Tệp | Nội dung |
|---|---|
| `images/` | Ảnh JPEG dòng chữ, cao 64 px, rộng thay đổi theo nội dung. |
| `labels.tsv` | Mỗi dòng: `đường_dẫn_ảnh<TAB>nhãn_không_dấu_phân_cách`. |
| `manifest.jsonl` | Nhãn, chuỗi thực sự được vẽ, font, biến đổi, mã băm ảnh và nguồn synthetic/train. |
| `dataset.json` | Tham số sinh, seed, số lượng, thống kê 30 ký tự và trạng thái hoàn tất. |
| `preview.jpg` | Bảng 48 ảnh đầu kèm nhãn để kiểm tra nhanh. |

Ví dụ ảnh `18-C1` có `rendered_text = "18-C1"`, `text = "18C1"`; ảnh `277.11` có nhãn `27711`. Bộ đọc dữ liệu kiểm tra nhãn, mã băm và trạng thái lô trước khi dùng. Không tự đưa ảnh giả vào tập kiểm định hoặc kiểm tra thật.

## 6. Phạm vi thay đổi

Các bước nâng cao chất lượng, lọc/khôi phục, phát hiện biên, phân đoạn và hình thái học vẫn thuộc pipeline OpenCV hiện có. Bộ dữ liệu này huấn luyện **bộ đọc dòng chữ sau khi cắt biển**. Nó không tự dạy bộ phát hiện tìm ra biển bị bỏ sót hoặc sửa mọi trường hợp cắt nhầm vật thể; các lỗi phát hiện cần dữ liệu vị trí biển và đánh giá riêng.

## 7. Kết quả kiểm tra khi bàn giao

- Đọc và kiểm tra đủ **20.000/20.000 ảnh**, khoảng 73,6 MB riêng tệp ảnh; mã băm đều khớp, nhãn khớp chuỗi đã vẽ, đủ chiều dài đầu vào cho CTC.
- Đủ 30 ký tự; mỗi chữ cái có 871–939 lượt xuất hiện. Phân bố này không cân bằng hoàn toàn giữa chữ và số vì dòng số chiếm nhiều ảnh.
- Không có chuỗi biển đầy đủ trùng các biển kiểm định/kiểm tra thật đã biết.
- **37 kiểm tra tự động đạt**, có kiểm tra cập nhật trọng số bằng phép học thực trên bộ dữ liệu thử tách riêng. Sau khi bổ sung thanh cuộn, chạy lại 17 kiểm tra liên quan đến luồng học/gán nhãn và đều đạt.
- Gợi ý OCR chạy được trên GUI với ảnh thật; kiểm tra không ghi đè sửa tay và không tự lưu nhãn. Cuộn được đến các nút huấn luyện trên cửa sổ cao 700 px.
- 34 nhãn thật và trọng số đang dùng được giữ nguyên. **Chưa chạy một đợt huấn luyện đầy đủ trên bộ 20.000 ảnh**; bạn bắt đầu bằng nút Huấn luyện bổ sung khi muốn dùng dữ liệu mới.

Bản kiểm tra chi tiết: `kiem_tra_du_lieu_gia_lap.json` đi kèm tài liệu trong thư mục bàn giao. Điểm kiểm tra tệp và kiểm tra chương trình không phải thước đo độ chính xác OCR ngoài thực tế.
