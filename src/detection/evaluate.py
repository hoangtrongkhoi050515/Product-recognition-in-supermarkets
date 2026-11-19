"""Đánh giá một lượt huấn luyện YOLO trên tập Validation hoặc Test.

Xuất ra Logs/detection/<tên lượt>_<tập>/:
  - metrics.json     : mAP@0.5, mAP@0.5:0.95, Precision, Recall, tốc độ (ms/ảnh)
  - per_class.csv    : Precision, Recall, AP50, AP50-95 của từng sản phẩm
  - confusion_matrix.png (do Ultralytics vẽ)

Cách chạy:
    python -m src.detection.evaluate --run yolo26n --split test
    python -m src.detection.evaluate --run yolo26n_ft832 --split val --imgsz 832
"""
from __future__ import annotations

import argparse
import csv
import json

from src.catalog import load_catalog
from src.detection.common import best_weights, data_yaml_path, get_detection_cfg, logs_dir


def main() -> None:
    cfg = get_detection_cfg()
    parser = argparse.ArgumentParser(description="Đánh giá YOLO")
    parser.add_argument("--run", required=True, help="tên thư mục trong models/, vd yolo26n, yolo26n_ft640")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--imgsz", type=int, default=cfg["imgsz"],
                        help="kích thước ảnh khi đánh giá (nên bằng imgsz lúc huấn luyện lượt đó)")
    args = parser.parse_args()

    from ultralytics import YOLO

    catalog = load_catalog()
    out_dir = logs_dir(args.run, args.split)
    out_dir.mkdir(parents=True, exist_ok=True)

    res = YOLO(str(best_weights(args.run))).val(
        data=str(data_yaml_path()),
        split=args.split,
        imgsz=args.imgsz,
        batch=cfg["batch"],
        iou=cfg["iou"],
        device=cfg["device"],
        workers=cfg["workers"],
        plots=True,
        project=str(out_dir.parent.resolve()),
        name=out_dir.name,
        exist_ok=True,
    )

    # Số liệu tổng hợp; res.speed tính bằng mili giây trên mỗi ảnh
    summary = {
        "run": args.run,
        "split": args.split,
        "imgsz": args.imgsz,
        "precision": float(res.box.mp),
        "recall": float(res.box.mr),
        "map50": float(res.box.map50),
        "map50_95": float(res.box.map),
        "ms_per_image": {k: round(float(v), 2) for k, v in res.speed.items()},
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Số liệu theo từng lớp (res.box.ap_class_index là các lớp thực sự có mặt trong tập)
    with open(out_dir / "per_class.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "code", "name", "precision", "recall", "ap50", "ap50_95"])
        for i, class_id in enumerate(res.box.ap_class_index):
            p = catalog.by_id(int(class_id))
            w.writerow([p.class_id, p.code, p.name,
                        round(float(res.box.p[i]), 4), round(float(res.box.r[i]), 4),
                        round(float(res.box.ap50[i]), 4), round(float(res.box.ap[i]), 4)])

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Đã lưu kết quả tại {out_dir}")


if __name__ == "__main__":
    main()
