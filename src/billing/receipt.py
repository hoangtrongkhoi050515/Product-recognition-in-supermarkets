"""Xuất hóa đơn PDF theo mẫu phiếu tính tiền của siêu thị."""
from __future__ import annotations

import io
import unicodedata
from datetime import datetime
from pathlib import Path

from reportlab.lib.pagesizes import A6
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from src.billing.billing import Bill, fmt_money

_BUNDLED = Path(__file__).resolve().parent.parent.parent / "app" / "assets" / "fonts"

# Phông có đủ dấu tiếng Việt: ưu tiên phông đóng gói trong repo (chạy được trên cloud), sau đó Windows, Linux, macOS
_FONT_CANDIDATES = [
    (str(_BUNDLED / "DejaVuSans.ttf"), str(_BUNDLED / "DejaVuSans-Bold.ttf")),
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
    ("/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial Bold.ttf"),
]


def _register_fonts() -> tuple[str, str, bool]:
    """Trả về (phông thường, phông đậm, có_dấu). Không có phông TTF thì dùng Helvetica và bỏ dấu để không làm sập ứng dụng."""
    for regular, bold in _FONT_CANDIDATES:
        if Path(regular).exists() and Path(bold).exists():
            try:
                pdfmetrics.registerFont(TTFont("Receipt", regular))
                pdfmetrics.registerFont(TTFont("Receipt-Bold", bold))
                return "Receipt", "Receipt-Bold", True
            except Exception:  # tệp phông hỏng: thử ứng viên kế tiếp
                continue
    return "Helvetica", "Helvetica-Bold", False


def _strip_accents(text: str) -> str:
    """Bỏ dấu tiếng Việt (dùng khi chỉ có phông không hỗ trợ dấu)."""
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")


def _wrap(text: str, font: str, size: float, width: float) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if pdfmetrics.stringWidth(trial, font, size) <= width or not cur:
            cur = trial
        else:
            lines.append(cur); cur = w
    return lines + ([cur] if cur else [])


def make_receipt_pdf(bill: Bill, store_name: str, invoice_no: str, when: datetime | None = None,
                     currency: str = "đ") -> bytes:
    """Trả về nội dung file PDF (bytes) của hóa đơn."""
    when = when or datetime.now()
    f, fb, accents = _register_fonts()
    t = (lambda x: x) if accents else _strip_accents
    margin = 14
    lines_h = []
    for l in bill.lines:
        lines_h.append(_wrap(t(l.name), f, 8, 128))
    height = 190 + sum(12 * len(w) + 12 for w in lines_h)
    page_w = A6[0]
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(page_w, max(height, 260)))
    H = max(height, 260)
    y = H - margin - 6
    c.setFont(fb, 10); c.drawCentredString(page_w / 2, y, t(store_name)); y -= 14
    c.setFont(fb, 12); c.drawCentredString(page_w / 2, y, t("HÓA ĐƠN BÁN HÀNG")); y -= 14
    c.setFont(f, 8)
    c.drawString(margin, y, t(f"Số HĐ: {invoice_no}")); y -= 11
    c.drawString(margin, y, t(f"Ngày giờ: {when:%d/%m/%Y %H:%M:%S}")); y -= 8
    c.line(margin, y, page_w - margin, y); y -= 11
    c.setFont(fb, 8)
    c.drawString(margin, y, t("Sản phẩm")); c.drawRightString(page_w - margin - 112, y, t("SL"))
    c.drawRightString(page_w - margin - 62, y, t("Đơn giá")); c.drawRightString(page_w - margin, y, t("Thành tiền")); y -= 4
    c.line(margin, y, page_w - margin, y); y -= 11
    c.setFont(f, 8)
    for l, wrapped in zip(bill.lines, lines_h):
        for k, part in enumerate(wrapped):
            c.drawString(margin, y, t(part))
            if k == len(wrapped) - 1:
                y -= 11
                c.drawRightString(page_w - margin - 112, y, t(f"{l.quantity} {l.unit}"))
                c.drawRightString(page_w - margin - 62, y, t(fmt_money(l.unit_price, "").strip()))
                c.drawRightString(page_w - margin, y, t(fmt_money(l.amount, "").strip()))
            else:
                y -= 11
        y -= 12
    c.line(margin, y + 6, page_w - margin, y + 6); y -= 6
    c.setFont(f, 8); c.drawString(margin, y, t(f"Tổng số lượng: {bill.total_quantity}")); y -= 14
    c.setFont(fb, 11); c.drawString(margin, y, t("TỔNG TIỀN")); c.drawRightString(page_w - margin, y, t(fmt_money(bill.total, currency))); y -= 14
    c.setFont(f, 7); c.drawCentredString(page_w / 2, y, t("Đơn giá đã gồm thuế GTGT. Cảm ơn quý khách!"))
    c.save()
    return buf.getvalue()
