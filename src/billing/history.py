"""Lịch sử hóa đơn và thống kê bán hàng (lưu trong tệp CSV, mỗi dòng một hóa đơn)."""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.billing.billing import Bill
from src.catalog import Catalog

COLUMNS = ["invoice_no", "time", "total_quantity", "total", "items"]


def append_invoice(history_path: Path, bill: Bill, invoice_no: str, when: datetime) -> None:
    """Ghi thêm một hóa đơn; `items` là JSON {class_id: số lượng}. Cùng số hóa đơn thì ghi đè dòng cũ."""
    history_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r for r in _read(history_path) if r["invoice_no"] != invoice_no]
    rows.append({"invoice_no": invoice_no, "time": when.strftime("%Y-%m-%d %H:%M:%S"),
                 "total_quantity": bill.total_quantity, "total": bill.total,
                 "items": json.dumps({str(l.class_id): l.quantity for l in bill.lines})})
    with open(history_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def _read(history_path: Path) -> list[dict]:
    if not history_path.exists():
        return []
    with open(history_path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_history(history_path: Path) -> pd.DataFrame:
    """Bảng hóa đơn mới nhất trước; cột total, total_quantity là số nguyên."""
    df = pd.DataFrame(_read(history_path), columns=COLUMNS)
    if df.empty:
        return df
    df["total"] = df["total"].astype(int)
    df["total_quantity"] = df["total_quantity"].astype(int)
    return df.sort_values("time", ascending=False).reset_index(drop=True)


def product_stats(df: pd.DataFrame, catalog: Catalog) -> pd.DataFrame:
    """Số lượng bán và doanh thu theo từng sản phẩm (đủ 12 sản phẩm, kể cả chưa bán)."""
    qty = {p.class_id: 0 for p in catalog}
    for raw in df["items"]:
        for cid, q in json.loads(raw).items():
            qty[int(cid)] += int(q)
    return pd.DataFrame([{"Sản phẩm": p.name, "Số lượng bán": qty[p.class_id],
                          "Doanh thu": qty[p.class_id] * p.price} for p in catalog])
