"""Kiểm thử cấu hình huấn luyện YOLO (không cần cài ultralytics hay GPU)."""
from src.config import load_config
from src.detection.common import get_detection_cfg


def test_cau_hinh_yolo_day_du():
    cfg = get_detection_cfg()
    assert set(cfg["models"]) == {"yolo11n", "yolo26n"}
    assert cfg["imgsz"] == 640 and cfg["epochs"] == 100 and cfg["batch"] == 16
    assert cfg["patience"] <= cfg["epochs"]


def test_duong_dan_models_tu_config():
    assert load_config().path("models_dir").name == "models"
