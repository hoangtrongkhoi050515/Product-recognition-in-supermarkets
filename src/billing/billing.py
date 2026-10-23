"""Tính tiền từ kết quả nhận diện: đếm theo lớp, tra đơn giá, tính thành tiền và tổng tiền."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from src.catalog import Catalog


@dataclass
class BillLine:
    class_id: int
    name: str
    unit: str
    unit_price: int
    quantity: int

    @property
    def amount(self) -> int:
        return self.unit_price * self.quantity


@dataclass
class Bill:
    lines: list[BillLine] = field(default_factory=list)

    @property
    def total_quantity(self) -> int:
        return sum(l.quantity for l in self.lines)

    @property
    def total(self) -> int:
        return sum(l.amount for l in self.lines)


def count_classes(class_ids: Iterable[int]) -> dict[int, int]:
    """Đếm số khung bao theo mã lớp (mỗi khung = một sản phẩm)."""
    return dict(Counter(int(c) for c in class_ids))


def build_bill(counts: dict[int, int], catalog: Catalog) -> Bill:
    """Tạo hóa đơn từ {class_id: số lượng}; bỏ qua dòng có số lượng <= 0, sắp theo class_id."""
    lines = []
    for cid in sorted(counts):
        qty = int(counts[cid])
        if qty <= 0:
            continue
        p = catalog.by_id(cid)
        lines.append(BillLine(cid, p.name, p.unit, p.price, qty))
    return Bill(lines)


def fmt_money(value: int, currency: str = "đ") -> str:
    """12345 -> '12.345 đ' (dấu chấm phân cách hàng nghìn kiểu Việt Nam)."""
    return f"{int(value):,}".replace(",", ".") + f" {currency}"
