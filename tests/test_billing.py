"""Kiểm thử tính tiền và hóa đơn (không cần mô hình)."""
from src.billing.billing import build_bill, count_classes, fmt_money
from src.billing.receipt import make_receipt_pdf
from src.catalog import load_catalog


def test_dem_va_tinh_tien():
    cat = load_catalog()
    counts = count_classes([0, 0, 1, 11])
    assert counts == {0: 2, 1: 1, 11: 1}
    bill = build_bill(counts, cat)
    assert bill.total_quantity == 4
    assert bill.total == 2 * 9100 + 4000 + 13900


def test_bo_dong_so_luong_khong():
    bill = build_bill({0: 0, 3: 2}, load_catalog())
    assert [l.class_id for l in bill.lines] == [3] and bill.total == 20600


def test_dinh_dang_tien():
    assert fmt_money(1234567) == "1.234.567 đ"


def test_xuat_hoa_don_pdf():
    bill = build_bill({0: 2, 6: 1}, load_catalog())
    pdf = make_receipt_pdf(bill, "Nhóm 14", "HD0001")
    assert pdf[:4] == b"%PDF" and len(pdf) > 1000


def test_lich_su_va_thong_ke(tmp_path):
    from datetime import datetime
    from src.billing.history import append_invoice, load_history, product_stats
    cat = load_catalog()
    f = tmp_path / "lich_su.csv"
    assert load_history(f).empty
    append_invoice(f, build_bill({0: 2, 3: 1}, cat), "HD1", datetime(2026, 10, 22, 9, 0, 0))
    append_invoice(f, build_bill({0: 1}, cat), "HD2", datetime(2026, 10, 22, 10, 0, 0))
    append_invoice(f, build_bill({0: 1}, cat), "HD2", datetime(2026, 10, 22, 10, 0, 0))   # ghi đè cùng số hóa đơn
    df = load_history(f)
    assert list(df["invoice_no"]) == ["HD2", "HD1"] and df["total"].sum() == 2 * 9100 + 10300 + 9100
    st_ = product_stats(df, cat)
    assert st_.loc[0, "Số lượng bán"] == 3 and st_.loc[3, "Doanh thu"] == 10300 and len(st_) == 12
