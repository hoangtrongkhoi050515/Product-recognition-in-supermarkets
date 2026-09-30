"""Kiểm tra chất lượng nhãn và thống kê bộ dữ liệu.

Cách chạy (tại thư mục gốc dự án):
    python -m src.data.dataset_stats                 # kiểm tra data/raw
    python -m src.data.dataset_stats --split train   # kiểm tra data/yolo/.../train

Kết quả lưu tại Logs/data_check/<tên>/:
    summary.txt, class_counts.csv, issues.csv, class_distribution.png
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from src.catalog import Catalog, load_catalog
from src.config import load_config
from src.data.yolo_labels import Box, LabelIssue, pair_images_labels, read_label_file


@dataclass
class DatasetScan:
    """Kết quả quét một cặp thư mục ảnh/nhãn."""
    items: list[tuple[Path, list[Box]]] = field(default_factory=list)  # (ảnh, các khung)
    issues: list[LabelIssue] = field(default_factory=list)

    @property
    def n_errors(self) -> int:
        return sum(1 for i in self.issues if i.level == "ERROR")

    def class_counts(self) -> Counter:
        return Counter(b.class_id for _, boxes in self.items for b in boxes)

    def image_counts(self) -> Counter:
        """Số ảnh có chứa mỗi lớp."""
        return Counter(c for _, boxes in self.items for c in {b.class_id for b in boxes})


def scan_dataset(img_dir: Path, lbl_dir: Path, catalog: Catalog) -> DatasetScan:
    """Quét toàn bộ ảnh + nhãn, ghi nhận lỗi. Dùng chung cho các script khác."""
    cfg = load_config().section("data_check")
    pairs, orphan_labels = pair_images_labels(img_dir, lbl_dir, cfg["image_exts"])
    scan = DatasetScan()
    for img, lbl in pairs:
        if lbl is None:
            scan.issues.append(LabelIssue(img.name, 0, "WARN", "ảnh chưa có file nhãn (coi là ảnh nền nếu cố ý)"))
            scan.items.append((img, []))
            continue
        boxes, issues = read_label_file(lbl, len(catalog), cfg.get("min_box_size", 0.0))
        scan.issues.extend(issues)
        scan.items.append((img, boxes))
    for lbl in orphan_labels:
        scan.issues.append(LabelIssue(lbl.name, 0, "ERROR", "file nhãn không có ảnh tương ứng"))
    return scan


def save_report(scan: DatasetScan, catalog: Catalog, out_dir: Path, title: str) -> str:
    """Ghi các file thống kê và trả về đoạn tóm tắt dạng văn bản."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts, img_counts = scan.class_counts(), scan.image_counts()
    n_img = len(scan.items)
    n_multi = sum(1 for _, b in scan.items if len(b) >= 2)
    n_bg = sum(1 for _, b in scan.items if not b)
    n_box = sum(counts.values())

    # Bảng số lượng theo lớp
    with open(out_dir / "class_counts.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "code", "name", "so_doi_tuong", "so_anh_chua_lop"])
        for p in catalog:
            w.writerow([p.class_id, p.code, p.name, counts.get(p.class_id, 0), img_counts.get(p.class_id, 0)])

    # Danh sách lỗi
    with open(out_dir / "issues.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["muc_do", "file", "dong", "noi_dung"])
        for i in sorted(scan.issues, key=lambda x: (x.level, x.file, x.line)):
            w.writerow([i.level, i.file, i.line, i.message])

    # Biểu đồ phân bố lớp
    _plot_distribution(catalog, counts, out_dir / "class_distribution.png", title)

    vals = [counts.get(p.class_id, 0) for p in catalog]
    lines = [
        f"=== THỐNG KÊ: {title} ===",
        f"Số ảnh                : {n_img}",
        f"Ảnh nhiều sản phẩm    : {n_multi} ({n_multi / max(n_img, 1):.0%})",
        f"Ảnh nền (không nhãn)  : {n_bg}",
        f"Tổng số đối tượng     : {n_box}  (trung bình {n_box / max(n_img, 1):.2f}/ảnh)",
        f"Đối tượng/lớp         : ít nhất {min(vals)} – nhiều nhất {max(vals)}",
        f"Lỗi (ERROR)           : {scan.n_errors}",
        f"Cảnh báo (WARN)       : {len(scan.issues) - scan.n_errors}",
    ]
    missing = [p.code for p in catalog if counts.get(p.class_id, 0) == 0]
    if missing:
        lines.append(f"Lớp CHƯA có dữ liệu    : {', '.join(missing)}")
    summary = "\n".join(lines)
    (out_dir / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    return summary


def _plot_distribution(catalog: Catalog, counts: Counter, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")  # không cần màn hình
    import matplotlib.pyplot as plt

    names = [p.code for p in catalog]
    vals = [counts.get(p.class_id, 0) for p in catalog]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(names[::-1], vals[::-1], color="#2563eb")
    for y, v in enumerate(vals[::-1]):
        ax.text(v, y, f" {v}", va="center", fontsize=8)
    ax.set_xlabel("Số đối tượng (khung bao)")
    ax.set_title(f"Phân bố số đối tượng theo lớp – {title}")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kiểm tra nhãn và thống kê bộ dữ liệu YOLO")
    parser.add_argument("--split", choices=["train", "val", "test"],
                        help="kiểm tra một tập đã chia trong data/yolo (mặc định: kiểm tra data/raw)")
    args = parser.parse_args()

    cfg, catalog = load_config(), load_catalog()
    if args.split:
        root = cfg.path("yolo_dir")
        img_dir, lbl_dir, name = root / "images" / args.split, root / "labels" / args.split, args.split
    else:
        img_dir, lbl_dir, name = cfg.path("raw_images"), cfg.path("raw_labels"), "raw"

    scan = scan_dataset(img_dir, lbl_dir, catalog)
    out_dir = cfg.path("logs_dir") / "data_check" / name
    print(save_report(scan, catalog, out_dir, name))
    print(f"\nChi tiết lưu tại: {out_dir}")
    if scan.n_errors:
        print("CẢNH BÁO: còn lỗi nhãn – xem issues.csv và sửa trước khi chia tập.")


if __name__ == "__main__":
    main()
