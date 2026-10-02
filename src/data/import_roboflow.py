"""Nhập bộ dữ liệu xuất từ Roboflow (định dạng YOLOv8) vào data/raw/.

Bản xuất của Roboflow lệch với quy ước của dự án ở hai điểm:
  1. class_id được xếp theo bảng chữ cái, khác thứ tự trong configs/products.csv.
  2. Nhãn vẽ bằng công cụ đa giác (Polygon / Smart Polygon / Auto Label) được xuất
     thành nhiều cặp tọa độ, không phải 4 giá trị của khung bao.

Script này đọc tên lớp trong data.yaml, đổi class_id theo products.csv, đổi đa giác
về khung bao nhỏ nhất, rồi gộp train/valid/test vào chung data/raw/ (việc chia tập
sẽ do split_dataset.py làm lại theo cấu hình của dự án).

Cách chạy:
    python -m src.data.import_roboflow               # đọc từ paths.roboflow_export
    python -m src.data.import_roboflow --dry-run     # chỉ kiểm tra, không ghi file
    python -m src.data.import_roboflow --force       # ghi đè ảnh/nhãn đã có trong data/raw
"""
from __future__ import annotations

import argparse
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.catalog import Catalog, load_catalog
from src.config import load_config
from src.data.yolo_labels import Box, pair_images_labels, write_label_file

SPLITS = ("train", "valid", "test")  # tên thư mục tập của Roboflow


@dataclass
class ImportReport:
    """Kết quả một lần nhập, dùng để in thống kê và kiểm thử."""
    images: Counter = field(default_factory=Counter)   # số ảnh đã nhập theo tập
    boxes: Counter = field(default_factory=Counter)    # số khung theo class_id (đã đổi số)
    kinds: Counter = field(default_factory=Counter)    # "khung" / "đa giác"
    existed: list[str] = field(default_factory=list)   # ảnh đã có sẵn trong data/raw -> bỏ qua
    no_label: list[str] = field(default_factory=list)  # ảnh không có file nhãn -> bỏ qua
    errors: list[str] = field(default_factory=list)    # ảnh có dòng nhãn lỗi -> bỏ qua


def load_class_names(yaml_path: Path) -> list[str]:
    """Đọc danh sách tên lớp (theo thứ tự class_id của Roboflow) từ data.yaml."""
    with open(yaml_path, encoding="utf-8") as f:
        names = yaml.safe_load(f)["names"]
    if isinstance(names, dict):  # một số bản xuất ghi dạng {id: tên}
        names = [names[k] for k in sorted(names)]
    return [str(n).strip() for n in names]


def build_id_map(names: list[str], catalog: Catalog) -> dict[int, int]:
    """Bảng đổi class_id Roboflow -> class_id của products.csv, khớp theo tên (code)."""
    extra = sorted(set(names) - set(catalog.codes))
    if extra:
        raise SystemExit(
            f"Roboflow có lớp không nằm trong products.csv: {extra}\n"
            "Kiểm tra lại tên lớp trên Roboflow (phải đúng từng ký tự với cột `code`)."
        )
    return {i: catalog.by_code(n).class_id for i, n in enumerate(names)}


def parse_line(line: str, id_map: dict[int, int]) -> tuple[Box, str]:
    """Đổi một dòng nhãn Roboflow thành Box (đã đổi class_id). Lỗi -> ValueError."""
    parts = line.split()
    try:
        old_id = int(float(parts[0]))
        nums = [float(v) for v in parts[1:]]
    except ValueError:
        raise ValueError("giá trị không phải số") from None
    if old_id not in id_map:
        raise ValueError(f"class_id {old_id} không có trong data.yaml")

    if len(nums) == 4:                                  # đã là khung bao
        xc, yc, w, h = nums
        kind = "khung"
    elif len(nums) >= 6 and len(nums) % 2 == 0:         # đa giác: x1 y1 x2 y2 ...
        xs, ys = nums[0::2], nums[1::2]
        x1, x2 = max(0.0, min(xs)), min(1.0, max(xs))   # khung nhỏ nhất ôm đa giác
        y1, y2 = max(0.0, min(ys)), min(1.0, max(ys))
        xc, yc, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        kind = "đa giác"
    else:
        raise ValueError(f"cần 4 giá trị (khung) hoặc số chẵn >= 6 (đa giác), có {len(nums)}")

    if w <= 0 or h <= 0 or not all(0.0 <= v <= 1.0 for v in (xc, yc, w, h)):
        raise ValueError("tọa độ ngoài [0, 1] hoặc kích thước <= 0")
    return Box(id_map[old_id], xc, yc, w, h), kind


def convert_label(path: Path, id_map: dict[int, int]) -> tuple[list[Box], Counter, list[str]]:
    """Đọc một file nhãn Roboflow. Trả về (các khung, đếm theo loại dòng, danh sách lỗi)."""
    boxes: list[Box] = []
    kinds: Counter = Counter()
    errors: list[str] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            box, kind = parse_line(line, id_map)
        except ValueError as e:
            errors.append(f"{path.name} dòng {i}: {e}")
            continue
        boxes.append(box)
        kinds[kind] += 1
    return boxes, kinds, errors


def run_import(src: Path, raw_images: Path, raw_labels: Path, catalog: Catalog,
               image_exts: list[str], force: bool = False, dry_run: bool = False) -> ImportReport:
    """Nhập toàn bộ train/valid/test của bản xuất Roboflow vào raw_images / raw_labels."""
    id_map = build_id_map(load_class_names(src / "data.yaml"), catalog)
    report = ImportReport()
    seen: set[str] = set()  # tên ảnh đã nhập, phát hiện trùng giữa các tập
    if not dry_run:
        raw_images.mkdir(parents=True, exist_ok=True)
        raw_labels.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        pairs, _ = pair_images_labels(src / split / "images", src / split / "labels", image_exts)
        for img, lbl in pairs:
            if lbl is None:
                report.no_label.append(img.name)
                continue
            if img.name in seen:
                report.errors.append(f"{img.name}: trùng tên giữa các tập")
                continue
            seen.add(img.name)
            dest_img = raw_images / img.name
            if dest_img.exists() and not force:
                report.existed.append(img.name)
                continue
            boxes, kinds, errors = convert_label(lbl, id_map)
            if errors:  # không nhập một nửa file: bỏ cả ảnh để người dùng sửa trên Roboflow
                report.errors.extend(errors)
                continue
            if not dry_run:
                shutil.copy2(img, dest_img)
                write_label_file(raw_labels / f"{img.stem}.txt", boxes)
            report.images[split] += 1
            report.kinds.update(kinds)
            report.boxes.update(b.class_id for b in boxes)
    return report


def print_report(report: ImportReport, catalog: Catalog, dry_run: bool) -> None:
    """In thống kê dễ đọc sau khi nhập."""
    print("=== KẾT QUẢ " + ("(CHẠY THỬ – chưa ghi file) " if dry_run else "") + "===")
    total = sum(report.images.values())
    print(f"Ảnh nhập: {total}  " + "  ".join(f"{s}={report.images[s]}" for s in SPLITS))
    print(f"Dòng nhãn: {report.kinds['khung']} khung, {report.kinds['đa giác']} đa giác (đã đổi thành khung)")
    print("\nSố khung theo lớp:")
    for p in catalog:
        print(f"  {p.class_id:>2}  {p.code:<22} {report.boxes[p.class_id]:>5}")
    if report.existed:
        print(f"\nBỏ qua {len(report.existed)} ảnh đã có trong data/raw (dùng --force để ghi đè).")
    if report.no_label:
        print(f"\nCẢNH BÁO: {len(report.no_label)} ảnh không có file nhãn, đã bỏ qua: {report.no_label[:5]}")
    if report.errors:
        print(f"\nLỖI: {len(report.errors)} vấn đề, các ảnh liên quan đã bỏ qua. 20 lỗi đầu:")
        for e in report.errors[:20]:
            print("  -", e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Nhập bản xuất YOLOv8 của Roboflow vào data/raw")
    parser.add_argument("--src", type=Path, help="thư mục giải nén bản xuất (mặc định lấy từ config)")
    parser.add_argument("--force", action="store_true", help="ghi đè ảnh/nhãn đã có trong data/raw")
    parser.add_argument("--dry-run", action="store_true", help="chỉ kiểm tra và thống kê, không ghi file")
    args = parser.parse_args()

    cfg, catalog = load_config(), load_catalog()
    src = args.src or cfg.path("roboflow_export")
    if not (src / "data.yaml").exists():
        raise SystemExit(f"Không thấy {src / 'data.yaml'}. Kiểm tra lại đường dẫn bản xuất Roboflow.")

    report = run_import(
        src, cfg.path("raw_images"), cfg.path("raw_labels"), catalog,
        cfg.section("data_check").get("image_exts", [".jpg", ".jpeg", ".png"]),
        force=args.force, dry_run=args.dry_run,
    )
    print_report(report, catalog, args.dry_run)
    if report.errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
