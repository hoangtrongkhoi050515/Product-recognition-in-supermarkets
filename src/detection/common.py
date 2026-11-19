"""Hàm dùng chung cho huấn luyện, đánh giá và so sánh YOLO."""
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


def run_dir(run_name: str) -> Path:
    """Thư mục kết quả của một lượt huấn luyện: models/<tên lượt>/ (vd yolo26n, yolo26n_ft640)."""
    return load_config().path("models_dir") / run_name


def finetune_run_name(run: str) -> str:
    """Tên thư mục của lượt huấn luyện thêm, vd 'ft640' -> 'yolo26n_ft640'."""
    return f"{get_detection_cfg()['finetune']['base']}_{run}"


def best_weights(run_name: str) -> Path:
    """Trọng số tốt nhất của một lượt: models/<tên lượt>/weights/best.pt."""
    path = run_dir(run_name) / "weights" / "best.pt"
    if not path.exists():
        raise SystemExit(f"Chưa có {path}. Hãy huấn luyện lượt '{run_name}' trước "
                         "(python -m src.detection.train ...).")
    return path


def logs_dir(run_name: str, split: str) -> Path:
    """Nơi lưu kết quả đánh giá: Logs/detection/<tên lượt>_<tập>/."""
    return load_config().path("logs_dir") / "detection" / f"{run_name}_{split}"
