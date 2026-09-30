"""Danh mục sản phẩm (lớp nhận diện) và bảng giá.

`configs/products.csv` là nguồn dữ liệu DUY NHẤT về lớp: thứ tự class_id dùng cho
nhãn YOLO, tên thư mục ảnh cắt (code), tên hiển thị và đơn giá khi tính tiền
đều lấy từ đây để tránh lệch thông tin giữa các module.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from src.config import load_config


@dataclass(frozen=True)
class Product:
    class_id: int      # chỉ số lớp trong nhãn YOLO (bắt đầu từ 0)
    code: str          # mã ASCII, dùng làm tên thư mục/tên lớp trong mô hình
    name: str          # tên tiếng Việt hiển thị trên giao diện, hóa đơn
    receipt_name: str  # tên viết tắt trên hóa đơn gốc (để đối chiếu)
    unit: str          # đơn vị tính
    price: int         # đơn giá (VNĐ, đã gồm VAT)


class Catalog:
    """Tra cứu sản phẩm theo class_id hoặc code."""

    def __init__(self, products: list[Product]):
        self.products = sorted(products, key=lambda p: p.class_id)
        self._by_id = {p.class_id: p for p in self.products}
        self._by_code = {p.code: p for p in self.products}
        self._validate()

    def _validate(self) -> None:
        ids = [p.class_id for p in self.products]
        if ids != list(range(len(ids))):
            raise ValueError(f"class_id phải liên tục từ 0..{len(ids) - 1}, hiện tại: {ids}")
        if len(self._by_code) != len(self.products):
            raise ValueError("Có mã sản phẩm (code) bị trùng trong products.csv")

    def __len__(self) -> int:
        return len(self.products)

    def __iter__(self):
        return iter(self.products)

    def by_id(self, class_id: int) -> Product:
        return self._by_id[class_id]

    def by_code(self, code: str) -> Product:
        return self._by_code[code]

    @property
    def codes(self) -> list[str]:
        """Danh sách code theo đúng thứ tự class_id (dùng cho data.yaml)."""
        return [p.code for p in self.products]


def load_catalog(csv_path: str | Path | None = None) -> Catalog:
    """Đọc products.csv (UTF-8, có hoặc không có BOM)."""
    csv_path = Path(csv_path) if csv_path else load_config().path("products")
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    products = [
        Product(
            class_id=int(r["class_id"]),
            code=r["code"].strip(),
            name=r["name"].strip(),
            receipt_name=r["receipt_name"].strip(),
            unit=r["unit"].strip(),
            price=int(r["price"]),
        )
        for r in rows
    ]
    return Catalog(products)
