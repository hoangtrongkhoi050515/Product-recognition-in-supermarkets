# Nhận diện sản phẩm siêu thị – Quầy thanh toán tự động (Nhóm 14)

Học phần Học máy – Trường ĐH CNTT&TT, ĐH Thái Nguyên. GVHD: TS. Ngô Hữu Huy.


## Cấu trúc thư mục

```
NHOM_14_DU_AN_HOC_MAY/
├── configs/
│   ├── config.yaml        # MỌI tham số: đường dẫn, tỉ lệ chia, ngưỡng...
│   └── products.csv       # 12 lớp + bảng giá (nguồn dữ liệu duy nhất về lớp)
├── data/
│   ├── roboflow_export_v1/ # bản xuất YOLOv8 từ Roboflow (giải nén, không đưa lên Git)
│   ├── raw/images/        # ảnh gốc (nhóm chụp)
│   ├── raw/labels/        # nhãn YOLO .txt xuất từ công cụ gán nhãn
│   ├── yolo/              # [tự sinh] train/val/test + data.yaml
│   └── crops/             # [tự sinh] ảnh cắt cho nhánh phân loại
├── app/                   # giao diện Streamlit: streamlit_app.py (điều hướng), kiosk.py (quầy), admin.py (quản trị), common.py
├── docs/                  # hướng dẫn chụp ảnh, gán nhãn
│   └── figures/           # hình minh họa báo cáo + script vẽ lại
├── Documents/             # báo cáo Word
├── Logs/                  # [tự sinh] thống kê, nhật ký huấn luyện
├── models/                # [tự sinh] trọng số mô hình
├── notebooks/             # notebook huấn luyện trên Colab
├── src/
│   ├── config.py          # đọc configs/config.yaml
│   ├── catalog.py         # danh mục sản phẩm + bảng giá
│   ├── data/              # kiểm tra nhãn, thống kê, chia tập, cắt ảnh
│   ├── classic/           # [khung rỗng] KNN / SVM / Random Forest
│   ├── deep/              # [khung rỗng] CNN, Transfer Learning
│   ├── detection/         # YOLO: train.py, evaluate.py
│   └── billing/           # tính tiền (billing.py), hóa đơn PDF (receipt.py)
└── tests/                 # kiểm thử tự động
```

Các thư mục đánh dấu `[khung rỗng]` chỉ có `__init__.py`, mã nguồn sẽ bổ sung theo tiến độ trong `KE_HOACH_DU_AN.md`.

## Cài đặt
#Clone repo
git clone https://github.com/hoangtrongkhoi050515/Product-recognition-in-supermarkets.git
#Giải nén và mở Terminal tại thư mục
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```

## Quy trình dữ liệu

Luôn chạy lệnh tại thư mục gốc dự án.

| Bước | Lệnh | Kết quả |
|---|---|---|
| 1. Chụp ảnh, gán nhãn | xem `docs/` | bản xuất YOLOv8 từ Roboflow |
| 1b. Nhập bản xuất Roboflow | `python -m src.data.import_roboflow` | `data/raw/images`, `data/raw/labels` |
| 2. Kiểm tra nhãn + thống kê | `python -m src.data.dataset_stats` | `Logs/data_check/raw/` |
| 3. Chia Train/Val/Test | `python -m src.data.split_dataset --clean` | `data/yolo/` + `data.yaml` |
| 4. Tạo ảnh cắt (phân loại) | `python -m src.data.make_crops --clean` | `data/crops/` |
| (Khi chuyển máy/Colab) | `python -m src.data.split_dataset --yaml-only` | sinh lại `data.yaml` |

Kiểm thử: `python -m pytest -q`

## Huấn luyện YOLO (Colab)

Mở `notebooks/colab_train_yolo.ipynb` trên Google Colab (GPU T4), hoặc chạy tay:

| Bước | Lệnh | Kết quả |
|---|---|---|
| Huấn luyện | `python -m src.detection.train --model yolo11n` (hoặc `yolo26n`) | `models/<tên mô hình>/weights/best.pt` |
| Đánh giá | `python -m src.detection.evaluate --model yolo11n --split test` | `Logs/detection/<tên>_test/` (metrics.json, per_class.csv, ma trận nhầm lẫn) |

Tham số (kích thước ảnh, epoch, batch...) nằm ở mục `detection` của `configs/config.yaml`.

## Chạy chương trình quầy thanh toán (Streamlit)

1. Chép trọng số YOLO26n (`best.pt`, lấy từ Drive: `NHOM_14_ket_qua_yolo/yolo26n/weights/`) vào `models/yolo26n/weights/best.pt`.
2. Cài thư viện: `pip install -r requirements.txt`
3. Tại thư mục gốc dự án: `streamlit run app/streamlit_app.py`

Chương trình có hai trang, chọn ở thanh trên cùng:

- **Quầy thanh toán** (dành cho khách): chọn ảnh, hệ thống nhận diện và lập giỏ hàng; khách sửa số lượng, đổi hoặc xóa dòng nhận sai, thêm sản phẩm bị sót, rồi bấm Thanh toán để nhận hóa đơn PDF (lưu vào `Logs/receipts/`) và mã QR thanh toán mô phỏng.
- **Quản trị** (mã PIN ở `app.admin_pin` trong `configs/config.yaml`, mặc định `1414`): chỉnh ngưỡng độ tin cậy và thử trên ảnh, xem chỉ số mô hình, lịch sử hóa đơn (`Logs/receipts/lich_su.csv`), thống kê bán hàng và bảng giá. Cài đặt lưu ở `Logs/cai_dat.json` và có hiệu lực ngay ở trang Quầy thanh toán.

Tham số mặc định (mô hình, ngưỡng tin cậy, tên cửa hàng) nằm ở mục `app` của `configs/config.yaml`. Sau khi cập nhật mã nguồn, hãy tắt hẳn Streamlit rồi chạy lại để nạp các tệp mới.
