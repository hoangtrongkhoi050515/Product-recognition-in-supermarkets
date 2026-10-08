"""Hàm dùng chung cho huấn luyện, đánh giá và dự đoán YOLO."""
from __future__ import annotations

from pathlib import Path

from src.config import load_config


def get_detection_cfg() -> dict:
    """Mục `detection` trong configs/config.yaml."""
    return load_config().section("detection")


def data_yaml_path() -> Path:
    """Đường dẫn data.yaml của bộ dữ liệu đã chia (data/yolo/data.yaml)."""
    path = load_config().path("yolo_dir") / "data.yaml"
    if not path.exists():
        raise SystemExit(f"Chưa có {path}. Chạy `python -m src.data.split_dataset` "
                         "(hoặc thêm --yaml-only khi đã chép sẵn data/yolo, vd trên Colab).")
    return path


def run_dir(model_name: str) -> Path:
    """Thư mục lưu kết quả của một mô hình: models/<tên mô hình>/."""
    return load_config().path("models_dir") / model_name


def best_weights(model_name: str) -> Path:
    """Trọng số tốt nhất sau huấn luyện: models/<tên>/weights/best.pt."""
    path = run_dir(model_name) / "weights" / "best.pt"
    if not path.exists():
        raise SystemExit(f"Chưa có {path}. Hãy huấn luyện trước: "
                         f"python -m src.detection.train --model {model_name}")
    return path
