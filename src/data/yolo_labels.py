"""Hàm dùng chung để đọc, kiểm tra và chuyển đổi nhãn định dạng YOLO.

Định dạng mỗi dòng trong file .txt:  class_id x_center y_center width height
(4 giá trị tọa độ đã chuẩn hóa về đoạn [0, 1]).
Các script dataset_stats / split_dataset / make_crops đều dùng module này
để không lặp lại logic đọc nhãn.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Box:
    class_id: int
    xc: float
    yc: float
    w: float
    h: float

    def to_xyxy(self, img_w: int, img_h: int) -> tuple[int, int, int, int]:
        """Đổi sang tọa độ pixel (x1, y1, x2, y2), đã kẹp trong biên ảnh."""
        x1 = (self.xc - self.w / 2) * img_w
        y1 = (self.yc - self.h / 2) * img_h
        x2 = (self.xc + self.w / 2) * img_w
        y2 = (self.yc + self.h / 2) * img_h
        return (
            max(0, int(round(x1))), max(0, int(round(y1))),
            min(img_w, int(round(x2))), min(img_h, int(round(y2))),
        )


@dataclass
class LabelIssue:
    """Một lỗi/cảnh báo phát hiện được trong file nhãn."""
    file: str
    line: int          # số dòng (1-based), 0 nếu lỗi ở cấp file
    level: str         # "ERROR" (phải sửa) hoặc "WARN" (nên xem lại)
    message: str


def read_label_file(path: Path, num_classes: int, min_box_size: float = 0.0
                    ) -> tuple[list[Box], list[LabelIssue]]:
    """Đọc một file nhãn, trả về danh sách khung hợp lệ và danh sách lỗi."""
    boxes: list[Box] = []
    issues: list[LabelIssue] = []
    text = path.read_text(encoding="utf-8").strip()
    for i, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            issues.append(LabelIssue(path.name, i, "ERROR", f"cần 5 giá trị, có {len(parts)} (nhãn segmentation?)"))
            continue
        try:
            cid = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:])
        except ValueError:
            issues.append(LabelIssue(path.name, i, "ERROR", "giá trị không phải số"))
            continue
        if not 0 <= cid < num_classes:
            issues.append(LabelIssue(path.name, i, "ERROR", f"class_id {cid} ngoài khoảng 0..{num_classes - 1}"))
            continue
        if not all(0.0 <= v <= 1.0 for v in (xc, yc, w, h)) or w <= 0 or h <= 0:
            issues.append(LabelIssue(path.name, i, "ERROR", "tọa độ chưa chuẩn hóa về [0, 1] hoặc kích thước <= 0"))
            continue
        if min(w, h) < min_box_size:
            issues.append(LabelIssue(path.name, i, "WARN", f"khung rất nhỏ (w={w:.3f}, h={h:.3f})"))
        boxes.append(Box(cid, xc, yc, w, h))
    return boxes, issues


def write_label_file(path: Path, boxes: Iterable[Box]) -> None:
    """Ghi danh sách khung ra file .txt theo định dạng YOLO."""
    lines = [f"{b.class_id} {b.xc:.6f} {b.yc:.6f} {b.w:.6f} {b.h:.6f}" for b in boxes]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def list_images(folder: Path, exts: Iterable[str]) -> list[Path]:
    """Liệt kê ảnh (không phân biệt hoa thường phần mở rộng), sắp xếp ổn định."""
    exts = {e.lower() for e in exts}
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)


def pair_images_labels(img_dir: Path, lbl_dir: Path, exts: Iterable[str]
                       ) -> tuple[list[tuple[Path, Path | None]], list[Path]]:
    """Ghép ảnh với file nhãn cùng tên.

    Trả về: (danh sách (ảnh, nhãn hoặc None), danh sách file nhãn không có ảnh).
    """
    images = list_images(img_dir, exts)
    labels = {p.stem: p for p in lbl_dir.glob("*.txt")} if lbl_dir.exists() else {}
    labels.pop("classes", None)  # file danh sách lớp do LabelImg/Roboflow sinh ra
    pairs = [(img, labels.pop(img.stem, None)) for img in images]
    return pairs, sorted(labels.values())
