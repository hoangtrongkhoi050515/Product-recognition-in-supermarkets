"""Chia bộ dữ liệu gốc thành Train/Validation/Test theo định dạng Ultralytics YOLO.

Hướng tiếp cận:
  - Chia theo ẢNH GỐC (không theo khung bao) để một ảnh không nằm ở hai tập.
  - Phân tầng tham lam (greedy stratified): xét ảnh chứa lớp hiếm trước, mỗi ảnh được
    đưa vào tập đang "thiếu" các lớp của ảnh đó nhiều nhất so với tỉ lệ mục tiêu,
    nhờ vậy mọi lớp đều có mặt ở cả 3 tập với tỉ lệ gần đúng cấu hình.
  - Sinh file data.yaml với danh sách lớp lấy từ configs/products.csv.

Cách chạy:
    python -m src.data.split_dataset            # chia và sao chép sang data/yolo
    python -m src.data.split_dataset --clean    # xóa data/yolo cũ trước khi chia
    python -m src.data.split_dataset --yaml-only  # chỉ sinh lại data.yaml (vd khi chuyển máy/Colab)
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import Counter
from pathlib import Path

import yaml

from src.catalog import Catalog, load_catalog
from src.config import load_config
from src.data.dataset_stats import scan_dataset
from src.data.yolo_labels import Box, write_label_file

SPLITS = ("train", "val", "test")


def stratified_split(items: list[tuple[Path, list[Box]]], ratios: dict[str, float], seed: int
                     ) -> dict[str, list[tuple[Path, list[Box]]]]:
    """Phân tầng tham lam theo số đối tượng của từng lớp."""
    rng = random.Random(seed)
    items = items[:]
    rng.shuffle(items)  # xáo trộn trước để thứ tự chụp không ảnh hưởng

    total = Counter(b.class_id for _, boxes in items for b in boxes)
    n_img = len(items)
    n_box = max(sum(total.values()), 1)
    target = {s: {c: ratios[s] * n for c, n in total.items()} for s in SPLITS}
    target_img = {s: ratios[s] * n_img for s in SPLITS}
    cur = {s: Counter() for s in SPLITS}
    cur_img = {s: 0 for s in SPLITS}

    def rarity(item):  # ảnh chứa lớp càng hiếm càng được xếp trước; ảnh nền xếp cuối
        classes = {b.class_id for b in item[1]}
        return min((total[c] for c in classes), default=float("inf"))

    result = {s: [] for s in SPLITS}
    for item in sorted(items, key=rarity):
        counts = Counter(b.class_id for b in item[1])
        # Điểm của mỗi tập = tổng "độ thiếu tương đối" (target - hiện có) / target của các lớp
        # trong ảnh; lớp hiếm được nhân trọng số lớn hơn để luôn phủ đủ 3 tập.
        # Hòa điểm thì chọn tập còn thiếu nhiều ảnh hơn.
        def score(s: str) -> tuple[float, float]:
            need = sum(n * (target[s][c] - cur[s][c]) / target[s][c] * (n_box / total[c])
                       for c, n in counts.items())
            return need, (target_img[s] - cur_img[s]) / target_img[s]

        best = max(SPLITS, key=score)
        result[best].append(item)
        cur[best].update(counts)
        cur_img[best] += 1
    return result


def write_data_yaml(yolo_dir: Path, catalog: Catalog) -> Path:
    """Sinh data.yaml cho Ultralytics (đường dẫn tuyệt đối theo máy đang chạy)."""
    data = {
        "path": str(yolo_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(catalog),
        "names": {p.class_id: p.code for p in catalog},
    }
    out = yolo_dir / "data.yaml"
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Chia bộ dữ liệu YOLO thành train/val/test")
    parser.add_argument("--clean", action="store_true", help="xóa thư mục data/yolo cũ trước khi chia")
    parser.add_argument("--force", action="store_true", help="vẫn chia dù còn lỗi nhãn (bỏ qua dòng lỗi)")
    parser.add_argument("--yaml-only", action="store_true", help="chỉ sinh lại data.yaml")
    args = parser.parse_args()

    cfg, catalog = load_config(), load_catalog()
    yolo_dir = cfg.path("yolo_dir")

    if args.yaml_only:
        print(f"Đã ghi {write_data_yaml(yolo_dir, catalog)}")
        return

    ratios = cfg.section("split")
    if abs(sum(ratios[s] for s in SPLITS) - 1.0) > 1e-6:
        raise SystemExit(f"Tổng tỉ lệ chia phải bằng 1.0, hiện tại: {ratios}")

    scan = scan_dataset(cfg.path("raw_images"), cfg.path("raw_labels"), catalog)
    if not scan.items:
        raise SystemExit(f"Không tìm thấy ảnh trong {cfg.path('raw_images')}")
    if scan.n_errors and not args.force:
        raise SystemExit(f"Còn {scan.n_errors} lỗi nhãn. Chạy `python -m src.data.dataset_stats` để xem, "
                         "sửa xong rồi chia lại (hoặc thêm --force).")

    if yolo_dir.exists() and any(yolo_dir.iterdir()):
        if not args.clean:
            raise SystemExit(f"{yolo_dir} đã có dữ liệu. Thêm --clean để xóa và chia lại.")
        shutil.rmtree(yolo_dir)

    parts = stratified_split(scan.items, ratios, cfg.seed)

    report = []
    for split, items in parts.items():
        img_out, lbl_out = yolo_dir / "images" / split, yolo_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img, boxes in items:
            shutil.copy2(img, img_out / img.name)
            write_label_file(lbl_out / f"{img.stem}.txt", boxes)  # ghi lại nhãn đã làm sạch
        counts = Counter(b.class_id for _, boxes in items for b in boxes)
        report.append((split, len(items), counts))

    yaml_path = write_data_yaml(yolo_dir, catalog)

    # Báo cáo phân chia: số ảnh + số đối tượng từng lớp ở mỗi tập
    log_dir = cfg.path("logs_dir") / "data_check"
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "split_report.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "code"] + [f"{s}_doi_tuong" for s in SPLITS])
        for p in catalog:
            w.writerow([p.class_id, p.code] + [c.get(p.class_id, 0) for _, _, c in report])
        w.writerow(["", "TONG_SO_ANH"] + [n for _, n, _ in report])

    for split, n, counts in report:
        print(f"{split:5s}: {n:5d} ảnh, {sum(counts.values()):6d} đối tượng")
    print(f"Đã ghi {yaml_path}\nBáo cáo: {log_dir / 'split_report.csv'}")


if __name__ == "__main__":
    main()
