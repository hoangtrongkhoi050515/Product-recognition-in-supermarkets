# Hướng dẫn đưa ứng dụng lên Streamlit Community Cloud

Ứng dụng quầy thanh toán (`app/streamlit_app.py`) chạy được trên Streamlit Community Cloud (miễn phí) trực tiếp từ repo GitHub công khai này.

## Các tệp dành cho bản cloud

| Tệp | Vai trò |
|---|---|
| `app/requirements.txt` | Thư viện cần cho ứng dụng (PyTorch bản CPU). Streamlit Cloud dùng tệp này thay cho `requirements.txt` ở gốc. |
| `packages.txt` | Thư viện hệ thống Linux mà OpenCV cần (`libgl1`, `libglib2.0-0`). |
| `.streamlit/config.toml` | Màu giao diện, dùng chung cho máy cá nhân và cloud. |
| `models/yolo26n/weights/best.pt` | Trọng số YOLO26n, phải có trong repo thì bản cloud mới nhận diện được. |

## Các bước

1. Đẩy code mới nhất lên GitHub (nhánh `main`), kiểm tra trên GitHub có tệp `models/yolo26n/weights/best.pt`.
2. Vào https://share.streamlit.io, đăng nhập bằng tài khoản GitHub.
3. Chọn **Create app** → **Deploy a public app from GitHub**, điền:
   - Repository: `hoangtrongkhoi050515/Product-recognition-in-supermarkets`
   - Branch: `main`
   - Main file path: `app/streamlit_app.py`
   - App URL: tự đặt tên, ví dụ `quay-thanh-toan-nhom14`
4. Mở **Advanced settings**, chọn Python **3.12**, rồi bấm **Deploy**. Lần đầu cài thư viện mất khoảng 5–10 phút.
5. Mỗi lần đẩy code mới lên nhánh `main`, ứng dụng trên cloud tự cập nhật.

## Lưu ý

- Ổ đĩa trên cloud không lưu lâu dài: lịch sử hóa đơn, tệp PDF trong `Logs/receipts` và ngưỡng chỉnh ở trang Quản trị sẽ trở về mặc định mỗi khi ứng dụng khởi động lại.
- Ứng dụng tự "ngủ" khi không có người dùng vài ngày; lần mở sau bấm nút đánh thức và chờ khoảng 1 phút.
- Mã PIN trang Quản trị nằm trong `configs/config.yaml` của repo công khai, nên ai đọc repo cũng biết PIN.
- Mô hình chạy trên CPU của máy chủ miễn phí, mỗi ảnh mất khoảng vài trăm mili giây, chậm hơn máy cá nhân.
