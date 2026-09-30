# KẾ HOẠCH DỰ ÁN – NHẬN DIỆN SẢN PHẨM SIÊU THỊ (NHÓM 14)

> Hướng ứng dụng: Quầy thanh toán tự động (Self-Checkout)
> Thời gian dự kiến: **10 tuần, 01/10/2026 → 09/12/2026** (còn dư thời gian dự phòng trước hạn nộp)
> Trạng thái: 🟩 xong · 🟨 đang làm · ⬜ chưa làm · ❓ chờ nhóm quyết định

---

## 1. Các quyết định đã chốt

| Hạng mục | Quyết định |
|---|---|
| Bài toán chính | Phát hiện đối tượng (Object Detection) – YOLO, PyTorch + Ultralytics |
| Bài toán so sánh (yêu cầu học phần) | Phân loại ảnh cắt từ khung bao: KNN, SVM, Random Forest, CNN, Transfer Learning |
| Danh mục | 19 sản phẩm theo hóa đơn GO! Thái Nguyên 24/09/2026 (`configs/products.csv`) |
| Đầu ra | Nhận diện + đếm + tính tiền + xuất hóa đơn + giao diện |
| Giao diện | Streamlit |

## 2. Các quyết định còn chờ (❓)

| # | Hạng mục | Đề xuất của Claude | Cần chốt trước |
|---|---|---|---|
| Q1 | Nơi huấn luyện | Viết/chạy thử code trên máy cá nhân, huấn luyện YOLO và CNN trên **Google Colab (GPU T4)**. Nếu máy MSI có GPU NVIDIA ≥ 4 GB VRAM thì có thể huấn luyện tại máy | Tuần 4 |
| Q2 | Dạng đầu vào | **Bắt buộc:** ảnh tải lên + chụp 1 khung hình từ webcam. **Mở rộng (tuần 9, nếu kịp):** camera liên tục | Tuần 7 |
| Q3 | Phiên bản YOLO | So sánh **YOLO11n** (ổn định, nhiều tài liệu) với **YOLO26n** (mới nhất, 01/2026, không cần NMS) | Tuần 5 |
| Q4 | Công cụ gán nhãn | **Roboflow** (làm việc nhóm online, xuất sẵn định dạng YOLO) hoặc LabelImg (offline) | Tuần 2 |
| Q5 | Tên sản phẩm & đơn vị | Đối chiếu bao bì thật; sữa TH: nhận diện theo **lốc** hay **hộp** | Tuần 2 |
| Q6 | Tỉ lệ chia tập | 70 / 15 / 15 (đang để mặc định trong `configs/config.yaml`) | Tuần 4 |

## 3. Tiến độ theo giai đoạn

| GĐ | Thời gian | Nội dung | Phụ trách chính | Kết quả bàn giao |
|---|---|---|---|---|
| 0 | T1 (01–07/10) | Chốt yêu cầu, dựng khung mã nguồn, cài môi trường | Khôi | Khung thư mục, `configs/`, `src/`, README 🟨 |
| 1 | T2–T3 (08–21/10) | Chụp ảnh thử 20 ảnh/lớp → kiểm tra → chụp đủ; gán nhãn khung bao | Thiện (cả nhóm cùng chụp) | `data/raw/` đầy đủ ảnh + nhãn |
| 2 | T4 (22–28/10) | Kiểm tra nhãn, thống kê, chia tập, cắt ảnh cho nhánh phân loại | Thiện | `data/yolo/`, `data/crops/`, báo cáo thống kê (mục 3.1) |
| 3 | T5–T6 (29/10–11/11) | Nhánh A: đặc trưng HOG + màu, KNN/SVM/RF; CNN; Transfer Learning | Phúc | Mô hình + bảng kết quả (mục 3.2, 3.3.1) |
| 4 | T6–T7 (05–18/11) | Nhánh B: huấn luyện, tinh chỉnh YOLO; đánh giá mAP, tốc độ | Phúc + Khôi | `best.pt` + kết quả (mục 3.2.6, 3.3.2–3.3.5) |
| 5 | T8–T9 (19/11–02/12) | Hệ thống: module nhận diện, tính tiền, hóa đơn, giao diện Streamlit | Khôi | Ứng dụng demo chạy được |
| 6 | T9–T10 (26/11–09/12) | Thử nghiệm thực tế, viết Chương 3, Mở đầu, Kết luận, slide, tổng duyệt | Cả nhóm | Báo cáo hoàn chỉnh + demo |

### Mốc kiểm tra (milestone)
- **M1 – 07/10:** môi trường chạy được `python -m src.data.dataset_stats` trên máy cả 3 thành viên.
- **M2 – 21/10:** ≥ 200 đối tượng/lớp đã gán nhãn *(đề xuất, xem mục 4)*.
- **M3 – 28/10:** dữ liệu đã chia tập, không còn lỗi nhãn.
- **M4 – 18/11:** YOLO đạt mAP@0.5 trên tập Validation ≥ *(nhóm đặt mục tiêu sau lần huấn luyện đầu)*.
- **M5 – 02/12:** demo nhận diện → tính tiền → hóa đơn chạy trọn vẹn.
- **M6 – 09/12:** nộp báo cáo.

## 4. Chỉ tiêu dữ liệu (đề xuất)
- Tổng khoảng **600–800 ảnh**, trong đó ≥ 60% là ảnh có **nhiều sản phẩm** (2–6 sản phẩm/ảnh) để giống khay thanh toán thật.
- Mỗi lớp **≥ 200 đối tượng** (khung bao); các cặp dễ nhầm (Hảo Hảo/Tomyum, Vinamilk ít đường/có đường, các loại Omachi, 2 loại KitKat) chụp nhiều hơn.
- Khoảng 5% ảnh nền không có sản phẩm (giảm phát hiện nhầm).
- Chi tiết: `docs/HUONG_DAN_CHUP_ANH.md`, `docs/HUONG_DAN_GAN_NHAN.md`.

## 5. Rủi ro và cách xử lý

| Rủi ro | Cách xử lý |
|---|---|
| Dữ liệu ít hoặc lệch lớp | Kiểm tra thống kê sau mỗi đợt chụp (`dataset_stats`), chụp bổ sung lớp thiếu |
| Nhãn sai hoặc không thống nhất | Quy tắc gán nhãn chung + kiểm tra chéo + công cụ tự phát hiện lỗi nhãn |
| Nhầm giữa các sản phẩm giống nhau | Chụp nhiều góc có chữ phân biệt; phân tích ma trận nhầm lẫn |
| Thiếu GPU | Dùng Colab; YOLO bản nano; giảm kích thước ảnh |
| Chậm tiến độ viết báo cáo | Viết Chương 3 song song ngay khi có kết quả từng giai đoạn |

## 6. Quy ước làm việc
- Toàn bộ tham số nằm trong `configs/` – **không ghi cứng** trong code.
- Mỗi chức năng một module trong `src/`, chạy bằng `python -m src.<module>`.
- Kết quả huấn luyện, thống kê lưu vào `Logs/`; mô hình đã huấn luyện lưu vào `models/`.
- Khuyến nghị dùng Git/GitHub để quản lý mã nguồn chung (không đưa `data/` lên Git).
