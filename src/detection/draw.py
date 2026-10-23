"""Vẽ khung bao và nhãn sản phẩm lên ảnh kết quả."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.catalog import Catalog
from src.detection.predict import Detection

_PALETTE = ["#ef4444", "#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899",
            "#14b8a6", "#f97316", "#6366f1", "#84cc16", "#06b6d4", "#a855f7"]
_FONTS = ["C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
          "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", "/Library/Fonts/Arial.ttf"]


def _font(size: int):
    for p in _FONTS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def short_name(name: str) -> str:
    """Bỏ phần ghi chú trong ngoặc: 'Sữa TH true MILK 180 ml (hộp lẻ)' -> 'Sữa TH true MILK 180 ml'."""
    return name.split(" (")[0]


def draw_detections(image: np.ndarray, dets: list[Detection], catalog: Catalog,
                    faded: list[Detection] | None = None) -> Image.Image:
    """Vẽ khung màu cho `dets`; `faded` (dưới ngưỡng tin cậy) vẽ nét mảnh màu xám để so sánh ngưỡng."""
    img = Image.fromarray(image).convert("RGB")
    w, h = img.size
    scale = max(w, h) / 1000
    font, thick = _font(max(12, int(16 * scale))), max(2, int(3 * scale))
    d = ImageDraw.Draw(img)
    for det in faded or []:
        d.rectangle(list(det.xyxy), outline="#9ca3af", width=max(1, thick // 2))
        d.text((det.xyxy[0] + 3, det.xyxy[1] + 2), f"{det.confidence:.2f}", fill="#6b7280", font=font)
    for det in dets:
        color = _PALETTE[det.class_id % len(_PALETTE)]
        x1, y1, x2, y2 = det.xyxy
        d.rectangle([x1, y1, x2, y2], outline=color, width=thick)
        label = f"{short_name(catalog.by_id(det.class_id).name)} {det.confidence:.2f}"
        tw = d.textlength(label, font=font)
        th = font.size + 4
        ty = y1 - th if y1 - th > 0 else y1
        d.rectangle([x1, ty, x1 + tw + 6, ty + th], fill=color)
        d.text((x1 + 3, ty + 1), label, fill="white", font=font)
    return img
