"""Cài đặt vận hành do trang Quản trị chỉnh (ngưỡng tin cậy, hiện khung bị loại), lưu ra tệp JSON.

Trang Quầy thanh toán đọc lại mỗi lần chạy, nên thay đổi ở trang Quản trị có hiệu lực ngay ở mọi trình duyệt.
"""
from __future__ import annotations

import json
from pathlib import Path

DEFAULTS = {"conf": 0.45, "show_faded": False}


def load_settings(path: Path, defaults: dict | None = None) -> dict:
    out = {**DEFAULTS, **(defaults or {})}
    try:
        out.update({k: v for k, v in json.loads(Path(path).read_text(encoding="utf-8")).items() if k in out})
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    out["conf"] = min(0.95, max(0.05, float(out["conf"])))
    out["show_faded"] = bool(out["show_faded"])
    return out


def save_settings(path: Path, settings: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
