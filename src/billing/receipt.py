"""Xuất hóa đơn PDF theo mẫu phiếu tính tiền của siêu thị."""
from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

from reportlab.lib.pagesizes import A6
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from src.billing.billing import Bill, fmt_money

# Phông có đủ dấu tiếng Việt: thử lần lượt Windows, Linux, macOS
_FONT_CANDIDATES = [
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
    ("/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial Bold.ttf"),
]


def _register_fonts() -> tuple[str, str]:
    for regular, bold in _FONT_CANDIDATES:
        if Path(regular).exists() and Path(bold).exists():
            pdfmetrics.registerFont(TTFont("Receipt", regular))
            pdfmetrics.registerFont(TTFont("Receipt-Bold", bold))
            return "Receipt", "Receipt-Bold"
    raise RuntimeError("Không tìm thấy phông chữ có dấu tiếng Việt (Arial/DejaVu/Liberation) để xuất PDF")


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
    f, fb = _register_fonts()
    margin = 14
    lines_h = []
    for l in bill.lines:
        lines_h.append(_wrap(l.name, f, 8, 128))
    height = 190 + sum(12 * len(w) + 12 for w in lines_h)
    page_w = A6[0]
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(page_w, max(height, 260)))
    H = max(height, 260)
    y = H - margin - 6
    c.setFont(fb, 10); c.drawCentredString(page_w / 2, y, store_name); y -= 14
    c.setFont(fb, 12); c.drawCentredString(page_w / 2, y, "HÓA ĐƠN BÁN HÀNG"); y -= 14
    c.setFont(f, 8)
    c.drawString(margin, y, f"Số HĐ: {invoice_no}"); y -= 11
    c.drawString(margin, y, f"Ngày giờ: {when:%d/%m/%Y %H:%M:%S}"); y -= 8
    c.line(margin, y, page_w - margin, y); y -= 11
    c.setFont(fb, 8)
    c.drawString(margin, y, "Sản phẩm"); c.drawRightString(page_w - margin - 112, y, "SL")
    c.drawRightString(page_w - margin - 62, y, "Đơn giá"); c.drawRightString(page_w - margin, y, "Thành tiền"); y -= 4
    c.line(margin, y, page_w - margin, y); y -= 11
    c.setFont(f, 8)
    for l, wrapped in zip(bill.lines, lines_h):
        for k, part in enumerate(wrapped):
            c.drawString(margin, y, part)
            if k == len(wrapped) - 1:
                y -= 11
                c.drawRightString(page_w - margin - 112, y, f"{l.quantity} {l.unit}")
                c.drawRightString(page_w - margin - 62, y, fmt_money(l.unit_price, "").strip())
                c.drawRightString(page_w - margin, y, fmt_money(l.amount, "").strip())
            else:
                y -= 11
        y -= 12
    c.line(margin, y + 6, page_w - margin, y + 6); y -= 6
    c.setFont(f, 8); c.drawString(margin, y, f"Tổng số lượng: {bill.total_quantity}"); y -= 14
    c.setFont(fb, 11); c.drawString(margin, y, "TỔNG TIỀN"); c.drawRightString(page_w - margin, y, fmt_money(bill.total, currency)); y -= 14
    c.setFont(f, 7); c.drawCentredString(page_w / 2, y, "Đơn giá đã gồm thuế GTGT. Cảm ơn quý khách!")
    c.save()
    return buf.getvalue()
