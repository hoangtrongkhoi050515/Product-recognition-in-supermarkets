"""So sánh mô hình cũ và các lượt huấn luyện thêm trên CÙNG một tập (đọc kết quả của evaluate.py).

Quy tắc chọn mô hình: chọn theo tập Validation (mAP50-95), tập Test chỉ để báo cáo,
tránh "chọn theo Test" làm số liệu Test bị lạc quan.

Cách chạy:
    python -m src.detection.compare --runs yolo26n yolo26n_ft640 yolo26n_ft832
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from src.detection.common import logs_dir

METRICS = ["precision", "recall", "map50", "map50_95"]


def load_summary(runs: list[str], split: str) -> pd.DataFrame:
    """Bảng số liệu tổng hợp, mỗi dòng một lượt; cột Δ so với lượt đầu tiên (mô hình cũ)."""
    rows = []
    for run in runs:
        m = json.loads((logs_dir(run, split) / "metrics.json").read_text(encoding="utf-8"))
        rows.append({"run": run, "imgsz": m.get("imgsz"), **{k: m[k] for k in METRICS},
                     "ms_inference": m["ms_per_image"]["inference"]})
    df = pd.DataFrame(rows).set_index("run")
    for k in ("map50", "map50_95"):
        df[f"Δ{k}"] = df[k] - df[k].iloc[0]
    return df.round(4)


def load_per_class(runs: list[str], split: str, metric: str = "ap50_95") -> pd.DataFrame:
    """Bảng `metric` theo từng sản phẩm, mỗi cột một lượt, sắp lớp yếu nhất (mô hình cũ) lên đầu."""
    cols = {run: pd.read_csv(logs_dir(run, split) / "per_class.csv", encoding="utf-8-sig")
            .set_index("name")[metric] for run in runs}
    return pd.DataFrame(cols).sort_values(runs[0]).round(4)


def main() -> None:
    parser = argparse.ArgumentParser(description="So sánh các lượt huấn luyện YOLO")
    parser.add_argument("--runs", nargs="+", required=True, help="lượt đầu tiên là mô hình gốc để tính Δ")
    args = parser.parse_args()

    for split in ("val", "test"):
        print(f"\n=== Tập {split.upper()} – tổng hợp ===")
        print(load_summary(args.runs, split).to_string())
        print(f"\n=== Tập {split.upper()} – AP50-95 theo sản phẩm ===")
        print(load_per_class(args.runs, split).to_string())

    best = load_summary(args.runs, "val")["map50_95"].idxmax()
    print(f"\nLượt có mAP50-95 tập Validation cao nhất: {best}")


if __name__ == "__main__":
    main()
