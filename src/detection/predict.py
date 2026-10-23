"""Nhận diện sản phẩm trên một ảnh bằng mô hình YOLO đã huấn luyện (dùng cho ứng dụng)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config import load_config


@dataclass(frozen=True)
class Detection:
    class_id: int
    confidence: float
    xyxy: tuple[float, float, float, float]


def weights_path(model_name: str | None = None) -> Path:
    cfg = load_config()
    name = model_name or cfg.section("app")["model"]
    path = cfg.path("models_dir") / name / "weights" / "best.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Chưa có {path}. Chép best.pt từ Drive (NHOM_14_ket_qua_yolo/{name}/weights/) vào đúng đường dẫn này.")
    return path


class Detector:
    """Nạp mô hình một lần, dùng lại cho mọi lượt quét."""

    def __init__(self, model_name: str | None = None):
        from ultralytics import YOLO
        self.model = YOLO(str(weights_path(model_name)))
        self.imgsz = int(load_config().section("app").get("imgsz", 640))

    def detect(self, image: np.ndarray, conf: float) -> list[Detection]:
        """`image`: mảng RGB (H, W, 3). Trả về danh sách phát hiện, sắp theo độ tin cậy giảm dần."""
        res = self.model.predict(image[..., ::-1], imgsz=self.imgsz, conf=conf, verbose=False)[0]
        out = [Detection(int(c), float(p), tuple(map(float, b)))
               for c, p, b in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist(), res.boxes.xyxy.tolist())]
        return sorted(out, key=lambda d: -d.confidence)
