# HƯỚNG DẪN GÁN NHÃN KHUNG BAO (Giai đoạn 1)

Công cụ: **Roboflow** (nhóm đã chốt 01/10/2026). **Xuất theo định dạng YOLO (.txt)**.

## 1. Danh sách lớp – BẮT BUỘC đúng thứ tự
Tạo lớp trong công cụ theo đúng thứ tự `class_id` của `configs/products.csv` (0 → 11), dùng cột `code`:

```
0 th_milk_180ml        4 handy_haohao         8 oreo_blueberry
1 aquafina_500ml       5 handy_tomyum         9 omachi_beef
2 kitkat_17g           6 lays_stax_lobster   10 chocopie_orion
3 pepsi_320ml          7 vinamilk_sugar      11 omachi_bowl_shrimp
```
Sai thứ tự lớp = nhãn sai toàn bộ. Roboflow thường sắp xếp lớp theo bảng chữ cái khi xuất, nên `class_id` trong file `.txt` có thể KHÁC thứ tự trên.
Dự án sẽ có bước chuyển đổi `class_id` theo tên lớp (`code`) – vì vậy **tên lớp trong Roboflow phải đúng từng ký tự với cột `code`**.

## 2. Quy tắc vẽ khung
1. Khung **ôm sát** phần nhìn thấy của sản phẩm, không chừa nền thừa, không cắt mất mép.
2. **Mỗi sản phẩm một khung**, kể cả hai sản phẩm cùng loại đặt cạnh nhau.
3. Sản phẩm bị che khuất: *(nhóm chốt ngưỡng – đề xuất: vẫn gán nếu còn thấy ≥ 30% và còn nhận ra được)*.
4. Sản phẩm bị cắt ở mép ảnh: vẫn gán nếu còn nhận ra được.
5. Đồ vật không thuộc danh mục: **không** gán nhãn.
6. Ảnh nền (không có sản phẩm): để file nhãn rỗng hoặc không có file nhãn.
7. Không chắc là lớp nào → **không đoán**, ghi tên file vào danh sách để cả nhóm xem.

**Riêng sữa TH true MILK:** mỗi **hộp** một khung, kể cả khi các hộp còn dính thành lốc (lốc 4 hộp = 4 khung).

## 3. Lưu ý kỹ thuật
- Ảnh chụp bằng điện thoại có thông tin xoay (EXIF). Nếu dùng Roboflow, bật **Auto-Orient**;
  nếu dùng LabelImg, kiểm tra ảnh hiển thị đúng chiều trước khi vẽ.
- **Không** bật resize/augmentation trong Roboflow khi xuất – việc này do code của dự án xử lý.
- Xuất xong: ảnh vào `data/raw/images/`, file `.txt` vào `data/raw/labels/` (cùng tên với ảnh).

## 4. Kiểm tra sau khi gán
```bash
python -m src.data.dataset_stats
```
- Mở `Logs/data_check/raw/issues.csv`: sửa hết dòng **ERROR**, xem lại dòng **WARN**.
- Mở `class_distribution.png`: lớp nào thấp hơn các lớp khác nhiều → chụp bổ sung.
- **Kiểm tra chéo**: mỗi thành viên xem ngẫu nhiên 10% ảnh do người khác gán nhãn.

## 5. Thiết lập Roboflow (làm một lần, do 1 người tạo rồi mời 2 người còn lại)
1. Tạo một **Project** loại *Object Detection*, đặt tên `nhom14-sanpham-sieuthi`; mời 2 thành viên với quyền Editor.
2. Tạo đủ 12 lớp với tên đúng bằng cột `code` (mục 1).
3. Tải ảnh lên theo từng đợt chụp; dùng chức năng **Assign** để chia mỗi người một phần ảnh, tránh gán trùng.
4. Gán nhãn xong, tạo **Version** với: Auto-Orient bật; **không** Resize, **không** Augmentation.
5. **Export** → định dạng *YOLOv8* (tương thích Ultralytics) → tải file `.zip` về máy, lưu bản sao ngoài Git (thư mục `data/` không đưa lên Git).
