"""Huấn luyện YOLO (Ultralytics) trên bộ dữ liệu 12 sản phẩm.

Hướng tiếp cận:
  - Fine-tune từ trọng số pretrained COCO (dữ liệu chỉ ~680 ảnh train nên không học từ đầu).
  - Mọi tham số lấy từ mục `detection` của configs/config.yaml; chạy YOLO11n hay YOLO26n
    chỉ khác tham số --model, các cấu hình còn lại giữ nguyên để so sánh công bằng.
  - Cùng seed với toàn dự án để kết quả tái lập được.
  - Kết quả (biểu đồ loss, ma trận nhầm lẫn, best.pt) lưu ở models/<tên mô hình>/.

Cách chạy:
    python -m src.detection.train --model yolo11n
    python -m src.detection.train --model yolo26n
    python -m src.detection.train --model yolo11n --epochs 1 --fraction 0.05   # chạy thử nhanh
"""
from __future__ import annotations

import argparse

from src.config import load_config
from src.detection.common import data_yaml_path, get_detection_cfg


def main() -> None:
    cfg = get_detection_cfg()
    parser = argparse.ArgumentParser(description="Huấn luyện YOLO")
    parser.add_argument("--model", required=True, choices=sorted(cfg["models"]),
                        help="tên mô hình khai báo trong detection.models")
    parser.add_argument("--epochs", type=int, default=cfg["epochs"], help="ghi đè số epoch")
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="tỉ lệ ảnh train được dùng (<1 để chạy thử nhanh)")
    args = parser.parse_args()

    from ultralytics import YOLO  # import trễ để --help chạy được khi chưa cài ultralytics

    project_cfg = load_config()
    model = YOLO(cfg["models"][args.model])  # tự tải trọng số pretrained lần đầu
    model.train(
        data=str(data_yaml_path()),
        imgsz=cfg["imgsz"],
        epochs=args.epochs,
        patience=cfg["patience"],
        batch=cfg["batch"],
        device=cfg["device"],
        workers=cfg["workers"],
        seed=project_cfg.seed,
        fraction=args.fraction,
        project=str(project_cfg.path("models_dir").resolve()),
        name=args.model,
        exist_ok=True,  # chạy lại thì ghi đè đúng thư mục của mô hình đó
        plots=True,     # biểu đồ loss, PR curve, ma trận nhầm lẫn
    )


if __name__ == "__main__":
    main()
