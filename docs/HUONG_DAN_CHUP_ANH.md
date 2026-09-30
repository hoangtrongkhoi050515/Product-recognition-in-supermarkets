# HƯỚNG DẪN CHỤP ẢNH DỮ LIỆU (Giai đoạn 1: 08–21/10)

Mục tiêu: bộ ảnh giống điều kiện **quầy thanh toán thật** – nhìn từ trên xuống khay, nhiều sản phẩm cùng lúc.

## 1. Chuẩn bị
- Mua/mượn đủ 19 sản phẩm trong `configs/products.csv` (nên có 2–3 đơn vị mỗi loại để chụp nhiều món cùng lúc).
- **Cố định một "khay thanh toán"**: mặt bàn/tấm nền, điện thoại hoặc webcam đặt ở độ cao ~40–60 cm, nhìn từ trên xuống hoặc chéo nhẹ.
- Thiết bị: *(nhóm ghi lại: tên điện thoại/webcam, độ phân giải)* – ghi vào báo cáo mục 2.2.1.
- Tắt chế độ làm đẹp, HDR quá mức; ảnh JPG, độ phân giải ≥ 1280 px cạnh dài là đủ.

## 2. Đặt tên file
`<người chụp>_<ngày>_<stt>.jpg` – ví dụ `thien_1010_0001.jpg`. Không dấu, không khoảng trắng.
Mỗi ảnh là một file riêng; **không** chỉnh sửa, cắt ảnh sau khi chụp.

## 3. Phân bổ ảnh (đề xuất, tổng 600–800 ảnh)

| Loại ảnh | Tỉ lệ | Mô tả |
|---|---|---|
| Đơn lẻ | ~30% | 1 sản phẩm, xoay đủ các mặt: trước, sau, cạnh, nằm, đứng |
| Nhiều sản phẩm | ~60% | 2–6 sản phẩm ngẫu nhiên; có ảnh đặt sát nhau, che khuất 10–30% |
| Ảnh nền | ~5% | khay trống / đồ vật không thuộc danh mục (ví, điện thoại, túi nilon) |
| Khó | ~5% | ánh sáng yếu, ngược sáng, bị lóa, hơi mờ |

## 4. Nguyên tắc để mô hình học tốt
1. **Đa dạng**: đổi góc, khoảng cách, vị trí trên khay, hướng sản phẩm sau mỗi ảnh.
2. **Ánh sáng**: chụp ở ít nhất 3 điều kiện (đèn trắng, đèn vàng, ánh sáng tự nhiên).
3. **Nền**: ít nhất 2–3 loại nền (khay thực tế, bàn gỗ, nền sáng).
4. **Cặp dễ nhầm** – chụp **nhiều hơn** và đặt chung trong cùng ảnh:
   - Mì ly Handy Hảo Hảo ↔ Handy Tomyum
   - Sữa Vinamilk ít đường ↔ có đường
   - Mì Omachi hải sản ↔ bắp bò ↔ mì tô Omachi tôm
   - KitKat 17 g ↔ KitKat Chunky
5. **Cân bằng**: mỗi lớp ≥ 200 lần xuất hiện. Sau mỗi buổi chụp + gán nhãn, chạy
   `python -m src.data.dataset_stats` để xem lớp nào còn thiếu.

## 5. Quy trình làm theo đợt
1. **Đợt thử (08–10/10):** ~20 ảnh/lớp → gán nhãn → chạy thống kê → cả nhóm xem lại quy tắc.
2. **Đợt chính (11–18/10):** chụp đủ chỉ tiêu, chia đều cho 3 thành viên.
3. **Bổ sung (19–21/10):** chụp thêm lớp thiếu, cặp dễ nhầm.

## 6. Lưu trữ
Ảnh gốc chép vào `data/raw/images/`. Nên sao lưu thêm lên Google Drive chung của nhóm.
