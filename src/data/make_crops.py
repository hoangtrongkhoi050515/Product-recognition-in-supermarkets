"""Cắt từng sản phẩm theo khung bao để tạo bộ dữ liệu PHÂN LOẠI (nhánh A).

Đầu vào : data/yolo/{images,labels}/{train,val,test}   (đã chia tập)
Đầu ra  : data/crops/{train,val,test}/<code>/<tên_ảnh>_<stt>.jpg
Cắt SAU khi chia tập nên ảnh cắt từ cùng một ảnh gốc luôn nằm cùng một tập
(tránh rò rỉ dữ liệu giữa Train và Test).

Cách chạy:
    python -m src.data.make_crops           # thêm --clean để xóa data/crops cũ
"""
from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter

from PIL import Image, ImageOps

from src.catalog import load_catalog
from src.config import load_config
from src.data.dataset_stats import scan_dataset
from src.data.yolo_labels import Box

SPLITS = ("train", "val", "test")


def pad_box(box: Box, padding: float) -> Box:
    """Nới rộng khung theo tỉ lệ padding mỗi phía (vẫn ở tọa độ chuẩn hóa)."""
    return Box(box.class_id, box.xc, box.yc, box.w * (1 + 2 * padding), box.h * (1 + 2 * padding))


def main() -> None:
    parser = argparse.ArgumentParser(description="Tạo bộ dữ liệu phân loại từ khung bao")
    parser.add_argument("--clean", action="store_true", help="xóa data/crops cũ trước khi cắt")
    args = parser.parse_args()

    cfg, catalog = load_config(), load_catalog()
    params = cfg.section("crops")
    padding, min_size = float(params.get("padding", 0.0)), int(params.get("min_size", 0))
    yolo_dir, crops_dir = cfg.path("yolo_dir"), cfg.path("crops_dir")

    if crops_dir.exists() and any(crops_dir.iterdir()):
        if not args.clean:
            raise SystemExit(f"{crops_dir} đã có dữ liệu. Thêm --clean để xóa và cắt lại.")
        shutil.rmtree(crops_dir)

    stats = {s: Counter() for s in SPLITS}
    skipped = Counter()
    for split in SPLITS:
        scan = scan_dataset(yolo_dir / "images" / split, yolo_dir / "labels" / split, catalog)
        if not scan.items:
            raise SystemExit(f"Chưa có dữ liệu tập {split}. Chạy `python -m src.data.split_dataset` trước.")
        for p in catalog:  # tạo đủ 19 thư mục lớp, kể cả lớp chưa có ảnh
            (crops_dir / split / p.code).mkdir(parents=True, exist_ok=True)
        for img_path, boxes in scan.items:
            if not boxes:
                continue
            # exif_transpose: xoay ảnh điện thoại về đúng chiều trước khi cắt
            with Image.open(img_path) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                for k, box in enumerate(boxes):
                    x1, y1, x2, y2 = pad_box(box, padding).to_xyxy(*im.size)
                    if min(x2 - x1, y2 - y1) < min_size:
                        skipped[split] += 1
                        continue
                    code = catalog.by_id(box.class_id).code
                    im.crop((x1, y1, x2, y2)).save(crops_dir / split / code / f"{img_path.stem}_{k}.jpg", quality=95)
                    stats[split][box.class_id] += 1

    log_dir = cfg.path("logs_dir") / "data_check"
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "crops_report.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "code"] + list(SPLITS))
        for p in catalog:
            w.writerow([p.class_id, p.code] + [stats[s].get(p.class_id, 0) for s in SPLITS])
    for s in SPLITS:
        print(f"{s:5s}: {sum(stats[s].values()):6d} ảnh cắt (bỏ {skipped[s]} khung quá nhỏ)")
    print(f"Đã lưu tại {crops_dir}\nBáo cáo: {log_dir / 'crops_report.csv'}")


if __name__ == "__main__":
    main()
