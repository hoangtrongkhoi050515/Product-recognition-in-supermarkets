# HƯỚNG DẪN GÁN NHÃN KHUNG BAO (Giai đoạn 1)

Công cụ: *(chờ nhóm chốt – đề xuất Roboflow hoặc LabelImg)*. Dù dùng công cụ nào, **xuất theo định dạng YOLO (.txt)**.

## 1. Danh sách lớp – BẮT BUỘC đúng thứ tự
Tạo lớp trong công cụ theo đúng thứ tự `class_id` của `configs/products.csv` (0 → 18), dùng cột `code`:

```
0 th_milk_180ml        5 chacheer_sunflower   10 vinamilk_less_sugar  15 omachi_beef
1 aquafina_500ml       6 handy_haohao         11 vinamilk_sugar       16 chocopie_orion
2 kitkat_17g           7 handy_tomyum         12 ong_tho_condensed    17 qua_than_chili
3 kitkat_chunky        8 lays_stax_lobster    13 oreo_blueberry       18 omachi_bowl_shrimp
4 redbull_250ml        9 corn_snack_choco     14 omachi_seafood
```
Sai thứ tự lớp = nhãn sai toàn bộ. Nếu công cụ tự sắp xếp theo bảng chữ cái (Roboflow hay làm vậy),
báo lại để viết thêm bước chuyển đổi.

## 2. Quy tắc vẽ khung
1. Khung **ôm sát** phần nhìn thấy của sản phẩm, không chừa nền thừa, không cắt mất mép.
2. **Mỗi sản phẩm một khung**, kể cả hai sản phẩm cùng loại đặt cạnh nhau.
3. Sản phẩm bị che khuất: *(nhóm chốt ngưỡng – đề xuất: vẫn gán nếu còn thấy ≥ 30% và còn nhận ra được)*.
4. Sản phẩm bị cắt ở mép ảnh: vẫn gán nếu còn nhận ra được.
5. Đồ vật không thuộc danh mục: **không** gán nhãn.
6. Ảnh nền (không có sản phẩm): để file nhãn rỗng hoặc không có file nhãn.
7. Không chắc là lớp nào → **không đoán**, ghi tên file vào danh sách để cả nhóm xem.

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
