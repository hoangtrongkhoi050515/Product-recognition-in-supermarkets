"""Đọc cấu hình chung của dự án.

Mọi module khác chỉ lấy tham số và đường dẫn qua `load_config()`,
nhờ vậy khi đổi tham số chỉ cần sửa `configs/config.yaml`.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Thư mục gốc dự án = thư mục cha của thư mục src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


@dataclass(frozen=True)
class Config:
    """Bọc dict cấu hình, cung cấp hàm lấy đường dẫn tuyệt đối."""

    raw: dict[str, Any]
    root: Path

    def path(self, key: str) -> Path:
        """Trả về đường dẫn tuyệt đối của khóa trong mục `paths`."""
        value = self.raw["paths"][key]
        p = Path(value)
        return p if p.is_absolute() else (self.root / p)

    def section(self, name: str) -> dict[str, Any]:
        """Lấy một mục cấu hình (vd: 'split', 'crops')."""
        return self.raw.get(name, {})

    @property
    def seed(self) -> int:
        return int(self.raw.get("project", {}).get("seed", 42))


@lru_cache(maxsize=None)
def load_config(config_path: str | Path = DEFAULT_CONFIG) -> Config:
    """Đọc file YAML một lần và lưu đệm cho các lần gọi sau."""
    config_path = Path(config_path)
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(raw=raw, root=PROJECT_ROOT)
