# Các điểm sai, còn thiếu và hướng sửa tiền xử lý biển số xe máy, ô tô

Ngày rà soát: 04/11/2026. Dự án: `D:/hoc_tap/hoc_tap/nam_4/xu_li_anh/btl/LicensePlateDetection`.

## 1. Kết luận và phạm vi

Code đã có nền tảng phù hợp: chuyển xám, thu nhỏ giữ tỷ lệ, kéo giãn tương phản, Gaussian, hai nhánh Sobel, hình thái học, lọc ứng viên, crop từ ảnh gốc, làm nét crop và OCR phân biệt một/hai dòng. Không cần thay toàn bộ pipeline; cần sửa các điểm cụ thể và bổ sung khả năng thích ứng.

Ưu tiên cao nhất là **Otsu, hình học crop, xử lý một/hai dòng và chuẩn hóa ảnh nhỏ**. Tiếp theo là ánh sáng không đều, cực tính chữ/nền và bộ lọc độ sáng. Không nên bật đồng thời mọi phép làm nét, cân bằng sáng và morphology.

Trong báo cáo này:

- **Lỗi đã kiểm chứng:** có ví dụ tái hiện bằng hàm của dự án.
- **Thiếu sót xác định từ code:** code thiếu một bước hoặc có giới hạn rõ ràng; ảnh hưởng thực tế cần đo trên bộ ảnh.
- **Rủi ro cần thực nghiệm:** tham số hoặc giả định có thể gây sai nhưng chưa có tỷ lệ lỗi đo được.

Đường dẫn bên dưới tương đối với thư mục dự án. Báo cáo tập trung vào xử lý ảnh đầu vào và crop biển trước nhận dạng; một số vấn đề phân đoạn/OCR được đưa vào vì quyết định ảnh tiền xử lý có dùng được hay không. Chưa sửa code nguồn và chưa chạy đánh giá toàn bộ bộ ảnh có đáp án.

## 2. Danh sách cần sửa, theo ưu tiên

| Mã | Vấn đề | Bằng chứng | Ưu tiên |
|---|---|---|---|
| P01 | Quy ước phân lớp Otsu sai tại pixel bằng ngưỡng | Đã tái hiện ảnh toàn đen và ảnh hai mức thành toàn trắng | Sửa trước |
| P02 | Sắp góc theo Y không giữ đúng cạnh biển; chỉnh crop phải đồng bộ | Đã tái hiện hình chữ nhật xoay 30° bị chọn cạnh ngắn làm chiều rộng | Sửa trước |
| P03 | “Nắn phối cảnh” hiện dựa trên hình chữ nhật xoay, chưa biểu diễn đúng méo hình thang | `refine_quad_corners` dùng `minAreaRect`/`boxPoints` | Cao |
| P04 | Xoay thử crop có thể cắt mất chữ và sinh vùng đệm tối | `rotate` không `expand=True`, không chỉ định màu đệm | Cao |
| P05 | Chọn một/hai dòng còn phụ thuộc hình học hộp bao, chưa đủ chắc cho biển nghiêng | Phân loại theo bbox trước rồi mới xét OBB | Cao |
| P06 | Kernel cố định chưa thích ứng tỷ lệ biển và bố cục hai dòng | Closing 3×19, opening 3×3 | Cao |
| P07 | Chưa chuẩn hóa kích thước crop OCR; ngưỡng pixel dễ loại nét mảnh | Upscale chưa tích hợp; lọc `bw<3`, `area<25` | Cao |
| P08 | Nhị phân hóa crop chưa có phương án cục bộ cho bóng/lóa không đều | Có Sauvola nhưng chưa gọi | Vừa–cao |
| P09 | Cực tính chữ/nền chưa được xử lý xuyên suốt | Xác thực ứng viên và OCR đều ưu tiên nét tối | Cao nếu hỗ trợ nền tối |
| P10 | Bộ lọc màu/độ sáng có thể loại biển thật trước OCR | Loại đỏ và độ sáng là điều kiện cứng | Vừa–cao |
| P11 | Gaussian và unsharp cố định, chưa theo chất lượng ảnh | Luôn blur ở phát hiện; luôn sharpen ở OCR | Vừa |
| P12 | Chưa có nhánh sửa nét đứt/bùn bẩn được kiểm soát | Black-hat/bridge chưa được gọi | Vừa |
| P13 | Chuẩn hóa ký tự làm mất tỷ lệ; sửa phải đồng bộ template | Patch và template đều kéo về khung cố định | Vừa, thuộc giáp ranh OCR |
| P14 | Chưa áp dụng hướng EXIF trước xử lý | Đọc ảnh rồi convert RGB | Nên bổ sung |
| P15 | Lỗi bị nuốt và thiếu thông tin lý do loại ứng viên | Nhiều `except` im lặng/cho qua | Cao để chẩn đoán |

## 3. Phân tích từng vấn đề và hướng giải quyết

### P01 — Sửa Otsu trước khi chỉnh tham số khác

**Vị trí:** `image_processing/thresholding.py:42–64`.

Code tính lớp thấp của histogram gồm `pixel <= t`, nhưng lại đưa `pixel >= t` vào lớp trắng. Khi ngưỡng bằng 0, nền 0 cũng thành trắng.

**Kiểm tra đã chạy bằng chính hàm nguồn:**

| Ảnh tổng hợp | Ngưỡng Otsu | Trắng với code hiện tại | Trắng khi dùng `>` |
|---|---:|---:|---:|
| Toàn 0 | 0 | 100% | 0% |
| Hai mức 0 và 200, mỗi mức 50% | 0 | 100% | 50% |

**Hướng sửa:**

```python
# Phù hợp với cách chia lớp của histogram hiện tại:
binary_img = np.where(img_uint8 > t_star, 255, 0).astype(np.uint8)
# Nếu cần lấy lớp tối trực tiếp theo ngưỡng Otsu:
dark_mask = img_uint8 <= t_star
```

Kiểm tra riêng ảnh rỗng, ảnh hằng và ảnh rất ít tương phản. Với ảnh hằng, nên đánh dấu không đủ thông tin thay vì tiếp tục xem kết quả phân lớp là có ý nghĩa. Rà soát các nơi dùng ngưỡng trả về; không thay máy móc dấu so sánh của `t_opt` trong OCR vì đó là ngưỡng đã pha trộn, khác ngưỡng Otsu thuần.

**Áp dụng:** cả xe máy và ô tô; ảnh gradient có nhiều pixel 0 đặc biệt cần kiểm tra.

### P02 — Chuẩn hóa bốn góc và ánh xạ crop thành một khối

**Vị trí:** `utils/image_utils.py:16–46`, `209–256`.

Sắp hai điểm Y nhỏ nhất làm cạnh trên không luôn đúng với cạnh vật lý của biển. Kiểm tra hình chữ nhật 200×40 xoay 30° cho kết quả chiều rộng/cao theo thứ tự điểm là 40/200. Nhánh `np.roll` hiện có vai trò đổi cạnh lại, nên không được xóa nó riêng lẻ rồi chỉ swap kích thước.

**Hướng sửa:**

1. Sắp điểm theo thứ tự chu vi; kiểm tra bốn đỉnh phân biệt, lồi, không tự cắt, diện tích đủ lớn.
2. Chọn cặp cạnh chiều rộng theo hình học biển; tính kích thước từ chính các cạnh đang được ánh xạ.
3. Giữ nguyên tỷ lệ hình học khi warp, rồi xử lý chữ xuôi/ngược riêng.
4. Chỉ thử thêm hướng OCR khi còn mơ hồ; hiện đã có cơ chế thử 180° trong `character_recognition.py:898`.

Không thay ngay bằng cách tổng–hiệu tọa độ trong tài liệu cũ: trên hình thoi đối xứng, cách đó đã tái hiện chọn trùng đỉnh, chỉ còn ba điểm phân biệt. `np.roll` theo chu vi cũng không tự tạo phản chiếu; lỗi phải được đánh giá trên toàn chuỗi sắp góc → chọn cạnh → warp → định hướng.

**Nghiệm thu:** ảnh có nhãn ở bốn góc, các góc 0°, ±15°, ±30°, ±45°, ±60°, 90°, 180°, thứ tự điểm đầu vào bị xáo; không đảo gương, không méo, không mất chữ.

### P03 — Chưa nắn đúng phối cảnh hình thang

**Vị trí:** `image_processing/candidate_filter.py:697–734`.

`refine_quad_corners` lấy `cv2.minAreaRect` rồi `cv2.boxPoints`. Bốn điểm này thuộc một hình chữ nhật xoay, không phải bốn góc thật của biển bị chụp xiên. Dùng `warpPerspective` ở sau không đủ để khôi phục phối cảnh nếu điểm nguồn đã chỉ là hình chữ nhật bao.

**Hướng sửa:** tìm contour biên biển, thử xấp xỉ đa giác bốn đỉnh, kiểm tra lồi/diện tích/cạnh và mức phù hợp với vùng chữ. Nếu đường viền không rõ, có thể ghép giao điểm các đường biên. Giữ `minAreaRect` làm phương án dự phòng để xoay thẳng khi không tìm được tứ giác đáng tin cậy. Không ép mọi contour thành tứ giác.

**Khác biệt:** biển ô tô dài bị chụp lệch ngang dễ làm chữ đầu/cuối chênh kích thước; biển xe máy chụp từ trên xuống dễ thành hình thang và lệch khoảng cách hai dòng. Cần ảnh kiểm tra cho cả hai.

### P04 — Xoay crop để deskew có nguy cơ cắt mép

**Vị trí:** `image_processing/character_recognition.py:676–679`.

`cropped_pil.rotate(ang, resample=...)` giữ khung cũ theo mặc định. Chữ sát mép có thể bị cắt; vùng mới phát sinh mặc định tối có thể bị nhị phân hóa thành foreground.

**Hướng sửa:** dùng canvas mở rộng hoặc thêm lề trước xoay; chọn màu đệm theo nền đã ước lượng hoặc dùng border replicate. Sau xoay xác định lại ROI hợp lệ, chuẩn hóa kích thước và bảo đảm tọa độ ký tự gắn với đúng crop. Không chỉ thêm `expand=True` mà giữ nguyên các giả định chiều cao/lề trước đó.

**Áp dụng:** ô tô một dòng có ký tự ngoài cùng dễ chạm mép; xe máy cần giữ đủ cả dòng trên và dưới. Đây là rủi ro xác định từ cách gọi hàm, chưa đo mức cắt trên ảnh thực tế.

### P05 — Phân loại một/hai dòng cần kiểm tra lại sau crop

**Vị trí:** `candidate_filter.py:540–557`; `character_recognition.py:681–696`, `725–737`.

Code ưu tiên phân loại theo tỷ lệ bbox ngang; chỉ xét OBB khi chưa phân loại được. Biển dài khi xoay có thể có bbox nhìn gần vuông và bị gán hai dòng trước khi xét tỷ lệ OBB. Khi loại bố cục sai, các ngưỡng chiều cao ký tự và cách chia dòng tiếp tục sai theo.

Code đã có tách dòng theo khoảng hở tâm Y; không phải hoàn toàn thiếu tách dòng. Tuy nhiên bước chọn góc và bước tách dòng cuối dùng tiêu chí khác nhau, và tỷ lệ hộp chưa đủ để quyết định chắc chắn bố cục.

**Hướng sửa:**

- Dùng hình học sau nắn thẳng kết hợp phân bố vùng chữ để xác nhận một/hai dòng.
- Khi chưa rõ, giữ hai giả thuyết bố cục và so sánh chất lượng phân đoạn.
- Dùng profile ngang hoặc phân cụm tâm Y có chuẩn hóa theo chiều cao chữ; kiểm tra hai hàng có đủ chữ và khoảng cách hợp lý.
- Không suy ra “xe máy = hai dòng, ô tô = một dòng”. Ô tô cũng có biển hai dòng; ưu tiên bố cục quan sát được trên ảnh.

### P06 — Kernel cần theo tỷ lệ và bố cục

**Vị trí:** `config.py:69–79`, `main.py:177–190`; bước gộp vùng tại `candidate_filter.py:503`.

Closing 3×19 và opening 3×3 áp dụng cho mọi ảnh. Ảnh nhỏ không được phóng ở `resize_image`, nên một kernel có thể vừa với biển này nhưng quá lớn/nhỏ với biển khác. Opening cũng có thể xóa nét mảnh.

**Hướng sửa:** thử số ít cấu hình kernel/tỷ lệ, dùng 3×19 hiện tại làm bản chuẩn để so sánh. Có thể bắt đầu bằng kernel theo kích thước xử lý, nhưng về sau nên liên hệ với kích thước vùng biển hoặc nét chữ ước lượng.

- **Một dòng:** nối khoảng cách giữa chữ theo phương ngang, tránh nhập với lưới tản nhiệt/cản xe.
- **Hai dòng:** cho phép hai dải chữ tạo thành ứng viên chung dựa trên chồng lấp ngang, khoảng cách dọc và độ rộng tương thích. Đánh giá cơ chế `merge_vertical_split_boxes` đã có trước khi thêm nối dọc mới.
- Giữ morphology tìm vùng biển tách biệt morphology sửa nét OCR. Không dùng kernel nối cả biển lên ảnh chữ.

### P07 — Chuẩn hóa crop nhỏ trước khi phân đoạn

**Vị trí:** `preprocessing.py:290–318`; `main.py:324–342`; `character_recognition.py:600–618`.

`upscale_low_resolution` chưa được tích hợp vào luồng OCR cuối. Trong khi đó có ngưỡng tuyệt đối `bw < 3`, diện tích bbox `<25`; nét mảnh trong crop nhỏ có thể bị loại. Bước xác thực ứng viên đã phóng ROI lên cao 60 px (`candidate_filter.py:194–198`), nhưng không thay thế chuẩn hóa crop OCR.

**Hướng sửa:** thêm hàm chuẩn bị crop dùng chung cho crop ngang và crop nắn; phóng giữ tỷ lệ theo độ cao ký tự hoặc cấu hình riêng một/hai dòng; giới hạn mức phóng để kiểm soát thời gian. Thử các mức đích trên tập chỉnh tham số, không coi 120 px luôn tối ưu.

Nếu truyền RGB vào `upscale_low_resolution`, hàm hiện chỉ nội suy; làm nét trong hàm chỉ áp dụng cho mảng xám. OCR sau đó đã unsharp, vì vậy cần kiểm soát để không làm nét hai lần khi thay luồng.

Nếu biển mất ngay sau khi toàn cảnh bị thu nhỏ xuống 900 px, phóng crop sau phát hiện không cứu được. Cần thử nhánh phát hiện ở độ phân giải cao hơn hoặc chia vùng có chồng lấp cho ảnh toàn cảnh lớn; kiểm soát tọa độ và gộp hộp trùng.

### P08 — Bổ sung nhị phân hóa cục bộ cho ánh sáng không đều

**Vị trí:** `character_recognition.py:534–558`; `thresholding.py:79–143`.

Hiện crop được sharpen, kéo giãn tương phản, rồi chọn một ngưỡng toàn cục đã hiệu chỉnh. Bóng che nửa biển hoặc phản sáng cục bộ có thể cần ngưỡng theo vùng; Sauvola có sẵn nhưng chưa gọi.

**Hướng sửa:** tạo các phương án độc lập: Otsu hiện tại, Sauvola và tùy chọn CLAHE nhẹ rồi phân ngưỡng. Chọn theo bố cục chữ, số thành phần hợp lý, độ đồng đều kích thước và kết quả nhận dạng. Block size phải phù hợp độ cao chữ sau chuẩn hóa.

Không OR mặc định tất cả mặt nạ vì dễ nối chữ và tăng nhiễu. Không nối tiếp mọi phép tăng tương phản. Phần đã cháy trắng hoàn toàn không còn nét để các phép này khôi phục.

### P09 — Cực tính chữ sáng/nền tối phải xuyên suốt pipeline

**Vị trí:** `candidate_filter.py:231`; `character_recognition.py:550–558`.

OCR có nhánh đổi ngưỡng cho ảnh tối nhưng vẫn dùng `enhanced < t_opt`; đây chưa phải đảo cực tính. Bước xác thực ứng viên cũng dùng inverse Otsu.

**Hướng sửa:** nếu tập mục tiêu có chữ sáng/nền tối, xét hai mặt nạ cực tính tại cả bước xác thực lẫn OCR, giữ quy ước nội bộ thống nhất: `True`/255 là nét chữ. Chọn phương án theo cấu trúc chữ và bố cục; tỷ lệ foreground chỉ là một tín hiệu phụ.

Ngưỡng “foreground >60% thì đảo” có thể sai với ảnh bẩn, crop lẫn nền hoặc chữ bị dính. Không dùng nó làm quyết định duy nhất.

### P10 — Điều kiện màu/độ sáng quá cứng có thể chặn biển trước OCR

**Vị trí:** `candidate_filter.py:55–58`, `123–127`, `496`, `587–595`.

Độ sáng tối thiểu phụ thuộc mean toàn ảnh, nên cảnh nền sáng và biển nằm trong bóng có thể bất lợi. Tuy nhiên code đã có điều kiện cứu bằng p75; không được chỉ nhìn mean ROI rồi kết luận bị loại.

Vùng đỏ bị loại theo trung bình RGB. Nếu phạm vi gồm biển nền đỏ, bộ lọc này cần xét lại; nếu phạm vi loại trừ nhóm đó thì ghi rõ giới hạn. Không bỏ bộ lọc đỏ mà không kiểm tra báo nhầm đèn hậu.

**Hướng sửa:** log các thống kê và lý do loại trước; cân nhắc dùng màu/sáng như điểm phạt, hoặc giữ ngoại lệ khi có cấu trúc chữ mạnh. Kết hợp tương phản cục bộ, mật độ nét và bố cục. Không coi vùng tương phản cao là chắc chắn biển vì bánh xe/lưới tản nhiệt cũng có đặc điểm đó.

### P11 — Khử nhiễu/làm nét cần có lựa chọn

**Vị trí:** `main.py:143–148`; `character_recognition.py:534`.

Gaussian 5×5 luôn chạy ở phát hiện; unsharp luôn chạy ở crop OCR. Có thể blur mất nét nhỏ hoặc sharpen mạnh nhiễu/viền.

**Hướng sửa:** giữ nhánh chuẩn và thử Gaussian nhẹ/bỏ blur/làm nét nhẹ theo nhóm ảnh. Nếu dùng Laplacian variance, đo ở kích thước chuẩn và ưu tiên ROI; ảnh toàn cảnh có nền nét không chứng minh biển nét. Hiệu chỉnh ngưỡng bằng dữ liệu, không lấy 300 làm hằng số chung.

Unsharp đã được sử dụng; vấn đề là cách lựa chọn cường độ và thời điểm, không phải thiếu hoàn toàn hàm làm nét. Nhòe chuyển động nặng không được giải quyết chắc chắn bằng tăng sharpen.

### P12 — Bổ sung sửa nét đứt có điều kiện

**Vị trí:** `morphology.py:222–256`; import bridge tại `character_recognition.py:36` nhưng chưa gọi.

**Hướng sửa:** thử black-hat cho chữ tối, top-hat cho chữ sáng; chọn kernel theo độ dày nét. Dùng bridge/closing nhỏ khi có dấu hiệu chữ bị đứt; giữ thêm phương án không sửa để so sánh. Kiểm tra không nhập hai chữ, không bít lỗ 0/8/B và không biến bùn thành chữ.

Không phải mọi biển bẩn đều cần bridge hoặc black-hat. Chữ bị vật cản che hoàn toàn cần được đánh dấu không chắc chắn.

### P13 — Giữ tỷ lệ ký tự phải đồng bộ template

**Vị trí:** `character_recognition.py:64–72`, `426`, `434`, `442`.

Patch bị kéo về 20×32, làm mất tỷ lệ gốc. Tuy vậy template cũng được kéo tương tự nên đây là lựa chọn biểu diễn đồng nhất, chưa đủ bằng chứng gọi là nguyên nhân mọi lỗi OCR.

**Hướng sửa:** thử resize giữ tỷ lệ + padding cho cả patch và template chuẩn/đậm/mảnh, dùng chung một hàm. Đánh giá lại luật mật độ vùng và lỗ ký tự vì padding thay đổi chúng. So sánh riêng nhóm 1/7/T/L, 0/O/D, 8/B, 6/9.

Đây là bước chuẩn bị ký tự cho nhận dạng; không phải tiền xử lý ảnh toàn cảnh. Không đổi riêng `normalize_char_patch` rồi giữ nguyên template.

### P14 — Chuẩn hóa hướng ảnh điện thoại

**Vị trí:** `utils/image_utils.py:63–65`.

Ảnh được convert RGB mà chưa áp dụng EXIF orientation. Trình xem có thể tự xoay ảnh, còn mảng pixel xử lý vẫn khác hướng.

**Hướng sửa:** dùng `ImageOps.exif_transpose(img)` trước convert RGB; toàn bộ hiển thị, phát hiện, tọa độ và crop dùng cùng ảnh đã chuẩn hóa. Kiểm tra ảnh không EXIF và ảnh có hướng xoay/phản chiếu.

### P15 — Thêm thông tin chẩn đoán và xử lý lỗi rõ ràng

**Vị trí:** `main.py:260–261`, `image_utils.py:265–266`, `candidate_filter.py:271–272`.

Nhánh Combined Sobel/warp có thể lỗi rồi bị bỏ qua im lặng. Xác thực ký tự có thể trả `True` khi exception, khiến “không kiểm tra được” bị xem như “đã đạt”.

**Hướng sửa:** ghi tên ảnh, bước, tham số và nguyên nhân lỗi; phân biệt ba trạng thái đạt/không đạt/không kiểm tra được. Lưu các ảnh trung gian của từng nhánh khi bật chẩn đoán; ghi lý do loại ở từng cổng hình học, sáng, màu, ký tự và NMS.

## 4. Điều chỉnh riêng theo bố cục biển

| Trường hợp | Điều cần bảo toàn | Cách xử lý nên ưu tiên |
|---|---|---|
| Xe máy, biển hai dòng | Đủ hai dòng; không nhập dòng, không mất dòng trên do crop/viền | Nắn hình thang, kiểm tra gộp vùng hai dòng, chia hàng theo phân bố chữ sau deskew |
| Ô tô, biển một dòng | Chữ đầu/cuối, khoảng cách dãy dài, tỷ lệ ngang | Crop đủ lề, xoay không cắt mép, chuẩn hóa theo chiều cao chữ, kernel ngang theo tỷ lệ |
| Ô tô, biển hai dòng | Bố cục hai dòng như quan sát thực tế | Chọn nhánh theo bố cục ảnh, không ép một dòng theo loại phương tiện |
| Biển nhỏ ở ảnh xa | Nét mảnh trước khi phân đoạn | Phát hiện đủ độ phân giải, crop ảnh gốc, phóng giữ tỷ lệ rồi chọn threshold |
| Biển bẩn/vàng/bóng râm | Tương phản ký tự, không loại sớm theo sáng/màu | Thống kê cục bộ, ngưỡng thích nghi, kiểm tra cấu trúc chữ và ngoại lệ lọc |

## 5. Pipeline đề xuất

### A. Phát hiện vùng biển trên toàn cảnh

Đọc ảnh → áp dụng hướng EXIF → giữ bản gốc → tạo ảnh xử lý theo tỷ lệ → chuyển xám/tăng tương phản vừa phải → chọn khử nhiễu nhẹ → Sobel-X và Combined Sobel hiện có → Otsu đã sửa → morphology theo cấu hình/tỷ lệ → CCL/gộp vùng → chấm ứng viên bằng hình học, cạnh và bố cục chữ → gộp hộp trùng → giữ một số ứng viên tốt.

Nhánh Combined Sobel đã tồn tại; điều cần thử là điều kiện gộp ứng viên tại `main.py:251` và tiêu chí chất lượng. Chỉ chuyển thứ tự chạy lên đầu không tự cải thiện kết quả.

### B. Chuẩn bị crop để OCR

Crop từ bản gốc → tìm tứ giác thật nếu có → warp/deskew giữ tỷ lệ và không cắt chữ → xác nhận một/hai dòng → chuẩn hóa độ cao chữ → thử một số phương án tương phản/ngưỡng/cực tính cần thiết → loại khung/ốc vít có kiểm soát → sửa nét khi có bằng chứng → phân đoạn theo dòng → chuẩn hóa patch đồng bộ template → nhận dạng.

Nên có hàm chuẩn bị crop chung để crop ngang và crop nắn được đánh giá công bằng. Mỗi phương án giữ metadata về tỷ lệ, hướng và phép biến đổi để đưa tọa độ ký tự về đúng ảnh nếu cần.

Không nên dùng số chữ tối đa hoặc regex hợp lệ làm tiêu chí duy nhất chọn mặt nạ: nhiễu cũng có thể tạo nhiều thành phần và chuỗi sai vẫn có thể hợp định dạng.

## 6. Thứ tự sửa thực tế và tiêu chí nghiệm thu

1. **Sửa Otsu:** kiểm tra ảnh hằng, hai mức, gradient; sau đó chạy lại tập ảnh chuẩn vì tác động lên cả hai nhánh cạnh.
2. **Sửa hình học crop:** kiểm tra điểm góc, phối cảnh, mép chữ và các hướng xoay; giữ fallback khi không đủ bằng chứng hình học.
3. **Chuẩn hóa crop và bố cục:** dùng cùng quy trình cho cả crop; thử biển xe máy hai dòng, ô tô một và hai dòng.
4. **Thêm log cổng lọc:** biết ứng viên mất ở đâu trước khi hạ ngưỡng sáng hoặc nới kích thước.
5. **Thử từng cải tiến:** kernel đa tỷ lệ, Sauvola/CLAHE, cực tính, khử nhiễu/làm nét, bridge/black-hat. Mỗi lần chỉ đổi một yếu tố so với bản chuẩn.
6. **Cuối cùng mới đổi biểu diễn template:** đánh giá riêng vì có thể thay đổi nhiều luật nhận dạng.

Chuẩn bị tập có hộp biển và chuỗi đúng, chia nhóm phương tiện/bố cục/chất lượng ảnh. Tách tập chỉnh tham số và tập kiểm tra; các biến thể từ cùng ảnh đặt trong cùng tập.

Đo recall phát hiện, số vùng báo nhầm, tỷ lệ đọc đúng toàn biển, OCR trên crop đúng và thời gian. Báo cáo riêng xe máy hai dòng, ô tô một dòng, ô tô hai dòng; kiểm tra trường hợp bản cũ đúng nhưng bản mới sai. Chưa có những số đo này thì không thể khẳng định một cấu hình là tốt nhất.

## 7. Ngoài phạm vi tiền xử lý nhưng cần chú ý khi đánh giá

`character_recognition.py:849–852` ép vị trí thứ ba ở biển một dòng là chữ và mọi vị trí còn lại là số; `862` còn cắt phần thân còn năm ký tự. Nếu tập mục tiêu có định dạng khác, OCR có thể sai dù tiền xử lý tốt. Cần khai báo phạm vi định dạng và kiểm tra luật nhận dạng riêng, tránh tiếp tục tăng morphology để chữa lỗi do luật OCR.

Các nhận xét trong báo cáo dựa trên code và các kiểm tra tổng hợp đã nêu. Tác động lên ảnh thực tế, giá trị tham số tối ưu và mức tăng độ chính xác còn cần thực nghiệm; không có thay đổi code nguồn nào được thực hiện trong lần rà soát này.
