"""Huấn luyện YOLO26n trên bộ dữ liệu 12 sản phẩm.

Hai chế độ:
  1. Huấn luyện lần đầu (đã chạy 08/10/2026): fine-tune từ trọng số pretrained COCO,
     tham số ở mục `detection` của configs/config.yaml, kết quả ở models/yolo26n/.
  2. Huấn luyện thêm (--finetune <lượt>): xuất phát từ models/yolo26n/weights/best.pt,
     dùng lr nhỏ + tăng cường dữ liệu mạnh hơn (mục `detection.finetune`).
     Mỗi lượt lưu riêng ở models/yolo26n_<lượt>/ nên mô hình cũ không bị ghi đè.

Cách chạy:
    python -m src.detection.train --model yolo26n                      # lần đầu
    python -m src.detection.train --finetune ft640                     # huấn luyện thêm, ảnh 640
    python -m src.detection.train --finetune ft832                     # huấn luyện thêm, ảnh 832
    python -m src.detection.train --finetune ft640 --epochs 1 --fraction 0.05   # chạy thử nhanh
"""
from __future__ import annotations

import argparse

from src.config import load_config
from src.detection.common import best_weights, data_yaml_path, finetune_run_name, get_detection_cfg


def build_train_args(cfg: dict, finetune: str | None) -> tuple[str, str, dict]:
    """Trả về (trọng số xuất phát, tên lượt, tham số model.train) – tách riêng để kiểm thử được."""
    common = {"imgsz": cfg["imgsz"], "epochs": cfg["epochs"], "patience": cfg["patience"],
              "batch": cfg["batch"], "device": cfg["device"], "workers": cfg["workers"]}
    if finetune is None:
        return cfg["models"]["yolo26n"], "yolo26n", common

    ft = cfg["finetune"]
    if finetune not in ft["runs"]:
        raise SystemExit(f"Không có lượt '{finetune}'. Các lượt khai báo: {sorted(ft['runs'])}")
    args = {**common,
            "epochs": ft["epochs"], "patience": ft["patience"], "optimizer": ft["optimizer"],
            "lr0": ft["lr0"], "warmup_epochs": ft["warmup_epochs"], "cos_lr": ft["cos_lr"],
            **ft["augment"],
            **ft["runs"][finetune]}  # tham số riêng của lượt (vd imgsz) ghi đè sau cùng
    return str(best_weights(ft["base"])), finetune_run_name(finetune), args


def main() -> None:
    cfg = get_detection_cfg()
    parser = argparse.ArgumentParser(description="Huấn luyện YOLO26n")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model", choices=sorted(cfg["models"]), help="huấn luyện lần đầu từ pretrained COCO")
    mode.add_argument("--finetune", choices=sorted(cfg["finetune"]["runs"]),
                      help="huấn luyện thêm từ best.pt hiện có")
    parser.add_argument("--epochs", type=int, help="ghi đè số epoch")
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="tỉ lệ ảnh train được dùng (<1 để chạy thử nhanh)")
    args = parser.parse_args()

    from ultralytics import YOLO  # import trễ để --help chạy được khi chưa cài ultralytics

    weights, run_name, train_args = build_train_args(cfg, args.finetune)
    if args.epochs:
        train_args["epochs"] = args.epochs
    project_cfg = load_config()
    print(f"Xuất phát từ {weights} -> models/{run_name}/  tham số: {train_args}")
    YOLO(weights).train(
        data=str(data_yaml_path()),
        seed=project_cfg.seed,
        fraction=args.fraction,
        project=str(project_cfg.path("models_dir").resolve()),
        name=run_name,
        exist_ok=True,  # chạy lại thì ghi đè đúng thư mục của lượt đó
        plots=True,     # biểu đồ loss, PR curve, ma trận nhầm lẫn
        **train_args,
    )


if __name__ == "__main__":
    main()
