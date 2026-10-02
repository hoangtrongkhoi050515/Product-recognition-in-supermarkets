# LOG TRÌNH BÀY – CHECKPOINT 1 (Bài 1) – NHÓM 14
Đề tài: Ứng dụng Machine Learning nhận diện sản phẩm trong siêu thị (hướng Self-Checkout)
File báo cáo đi kèm: `NHOM_14_BAO_CAO_DU_AN_HOC_MAY_26001_v3.docx`

---

## 1. Mở đầu bằng 1 câu (15 giây)
"Nhóm em xây dựng mô hình nhận diện 12 sản phẩm thật từ ảnh để làm nền cho quầy thanh toán tự động. Hôm nay nhóm báo cáo: phân công, cơ sở lý thuyết, định hướng phương pháp và kết quả thu thập dữ liệu đợt 1."

## 2. Kịch bản nói theo từng mục (khoảng 5–7 phút)

| Mục | Nói gì | Chỉ vào đâu trong báo cáo |
|---|---|---|
| Phân công | 8 nhiệm vụ theo tiến độ, mỗi người 3 nhiệm vụ: Khôi (Demo/Code, Kết quả thực nghiệm, Phối hợp hoàn thiện), Phúc (Định hướng phương pháp, Chạy chương trình & huấn luyện, Phương pháp/mô hình & hoàn thiện), Thiện (Cấu trúc báo cáo, Cơ sở lý thuyết, Thu thập & tiền xử lý) | Bảng phân công |
| Chương 1 | Phát hiện đối tượng khác phân loại ảnh: một ảnh có nhiều sản phẩm → cần YOLO. Các thuật toán KNN/SVM/Random Forest/CNN/Transfer Learning để so sánh trên ảnh cắt. Chỉ số: Precision, Recall, F1, mAP, IoU | Mục 1.1, 1.4–1.7 |
| Định hướng phương pháp | Nhánh chính: YOLO (PyTorch + Ultralytics) phát hiện + đếm + tính tiền. Nhánh so sánh: cắt từng khung → HOG + histogram màu → KNN/SVM/RF; CNN; Transfer Learning. Huấn luyện trên Google Colab (GPU T4) vì máy cá nhân không có GPU NVIDIA; giao diện Streamlit | Mục 2.1.4, 2.3, 2.4 |
| Dữ liệu | 12 sản phẩm lấy từ hóa đơn GO! Thái Nguyên (24/09/2026). Tự chụp, gán nhãn trên Roboflow, xuất YOLOv8. Quy tắc: khung ôm sát, mỗi sản phẩm một khung, TH true MILK mỗi hộp một khung | Mục 2.1.3, 2.2 |
| Kết quả đợt 1 | 227 ảnh, 932 khung, 195 ảnh (86%) nhiều sản phẩm, 0 lỗi nhãn. Chưa cân bằng: TH 356 khung, Lay's Stax chỉ 2 | Mục 3.1, Bảng 3.1, Hình 3.1 |
| Kế hoạch | Chụp bổ sung 14/10 và 18/10, kiểm tra chéo 19–21/10, rồi chia tập, huấn luyện | Phần "Bước tiếp theo" dưới đây |

## 3. Nhật ký tiến độ (đã làm)
- Chốt danh mục 12 sản phẩm (thay Red Bull bằng Pepsi lon 320 ml; Vinamilk dạng bịch, chỉ giữ loại có đường; thêm TH true MILK hộp lẻ).
- Dựng khung mã nguồn: cấu hình `configs/` (config.yaml, products.csv), pipeline dữ liệu (kiểm tra nhãn, thống kê, chia tập, cắt ảnh), test tự động. Chạy được trên 3 máy của nhóm.
- Chụp đợt 1: Khôi 27 ảnh, Phúc 100 ảnh, Thiện 100 ảnh.
- Gán nhãn trên Roboflow; thử Auto Label nhưng chỉ nhận ra 1/27 ảnh nên chuyển sang gán thủ công.
- Viết `import_roboflow.py` để: đổi mã lớp theo products.csv (Roboflow xếp tên lớp theo chữ cái), chuyển nhãn đa giác thành khung, thay ảnh cũ khi ảnh được sửa lại và xuất lại.
- Sửa các khung rất nhỏ (vẽ nhầm) rồi xuất lại bản v2.

## 4. Con số cần thuộc
- 12 lớp; 227 ảnh; 932 khung; trung bình 4,11 khung/ảnh; 195 ảnh nhiều sản phẩm (86%); 0 lỗi.
- Theo lớp (khung): TH 356 · Pepsi 81 · Aquafina 79 · Chocopie 72 · Handy Hảo Hảo 63 · Omachi bắp bò 63 · Oreo 57 · Omachi tô tôm 57 · KitKat 44 · Handy Tomyum 30 · Vinamilk 28 · Lay's Stax 2.
- Mục tiêu: 600–800 ảnh, tối thiểu 200 khung/lớp. Hiện chỉ TH đạt; 11 lớp còn lại cần thêm khoảng 1.624 khung.
- Đơn giá: TH 9.100 đ/hộp (= 36.400 / 4), Pepsi 10.300, Aquafina 4.000, Lay's Stax 30.200, Chocopie 29.900.
- Chia tập mặc định 70/15/15 theo ảnh gốc, cắt ảnh sau khi chia (tránh rò rỉ), bỏ khung dưới 32 pixel.

## 5. Rủi ro và cách trả lời nếu thầy/cô hỏi
- **Dữ liệu còn ít, mất cân bằng?** Đúng, mới là đợt 1 (227/600–800 ảnh). Lay's Stax (2 khung) và 3 lớp dưới 50 khung sẽ được ưu tiên chụp ở 14/10 và 18/10; khi huấn luyện dùng thêm tăng cường dữ liệu và cân nhắc trọng số lớp.
- **Vì sao tách từng hộp TH mà không tính cả lốc?** Để cùng một cách đếm với các hàng lẻ khác và khớp giá từng hộp; mỗi hộp một khung kể cả trong lốc.
- **Vì sao chọn YOLO mà còn so sánh KNN/SVM?** Yêu cầu học phần cần so sánh thuật toán; nhưng phân loại chỉ xử lý một sản phẩm/ảnh, còn quầy thanh toán có nhiều sản phẩm nên YOLO là mô hình chính.
- **Chia train/valid/test có rò rỉ không?** Chia theo ảnh gốc, phân tầng, cố định random seed; cắt ảnh sau khi chia.
- **Nhãn có đúng không?** Có script kiểm tra tự động (0 lỗi) và đã sửa khung nhỏ; kiểm tra chéo giữa các thành viên dự kiến 19–21/10.
- **Không có GPU?** Huấn luyện trên Google Colab (T4); máy cá nhân chỉ chuẩn bị dữ liệu và chạy giao diện.
- **Chưa có kết quả mô hình?** Đúng, Bài 1 là giai đoạn cơ sở lý thuyết + dữ liệu; huấn luyện bắt đầu sau khi đủ dữ liệu (sau 21/10).

## 6. Bước tiếp theo (lịch dự kiến)
- 08/10: chụp Bộ 1 (ưu tiên Lay's Stax, Vinamilk, Handy Tomyum, KitKat).
- 14/10: chụp Bộ 2. · 18/10: chụp Bộ 3.
- 19–21/10: bổ sung, kiểm tra chéo nhãn → mốc M2 (21/10).
- Sau đó: chia tập, cắt ảnh, huấn luyện YOLO và các mô hình phân loại trên Colab.

## 7. Việc cần nhóm chốt (đã tô vàng trong báo cáo)
1. Dạng đầu vào hỗ trợ: ảnh tĩnh / video / camera (mục 2.1.2).
2. Thiết bị chụp, độ phân giải ảnh gốc, địa điểm chụp (mục 2.2.1, 3.1.1).
3. Ngưỡng che khuất khi gán nhãn (mục 2.2.2).
4. Phiên bản YOLO: YOLOv8 / YOLO11 / YOLO26 (mục 2.3.6) và backbone Transfer Learning: MobileNetV2 / ResNet50 (mục 2.3.5).
5. Xác nhận tỉ lệ chia 70/15/15 (mục 2.2.6); góc xoay tăng cường (mục 2.2.5).
6. Tên viết tắt của Pepsi trên hóa đơn (Bảng 2.1 đang ghi "chưa có").
7. Sau khi mở file: bấm Ctrl+A rồi F9 để cập nhật mục lục và số trang.

## 8. Việc dọn dữ liệu trước khi trình bày
Thư mục `data/raw` còn 2 cặp ảnh cũ (bản trước khi sửa) nên thống kê chạy ra 229 ảnh/944 khung thay vì 227/932. Xoá 4 file rồi chạy lại `python -m src.data.dataset_stats` thì số khớp với báo cáo:
- `images` và `labels`: `phuc_0110_0063_jpg.rf.219f607a6cbea572c51c940b20cadcfd` (.jpg và .txt)
- `images` và `labels`: `thien_0110_0009_jpg.rf.01eab0491a0a8f0ff95814adda4bdc7a` (.jpg và .txt)
