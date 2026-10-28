"""Trang Quầy thanh toán: màn hình dành cho khách, đi qua ba bước Chọn ảnh, Kiểm tra giỏ hàng, Thanh toán."""
from __future__ import annotations

import io
from datetime import datetime

import streamlit as st
from PIL import UnidentifiedImageError

from common import (CUR, HISTORY, RECEIPT_DIR, app_cfg, banner, catalog, conf_badge, product_icon, reset_cart, scan, settings,
                    ss, thumbnail)
from src.billing.billing import build_bill, count_classes, fmt_money
from src.billing.history import append_invoice
from src.billing.receipt import make_receipt_pdf
from src.detection.draw import draw_detections, short_name

S = settings()
conf = S["conf"]
banner("Quầy thanh toán tự động", "Nhận diện sản phẩm bằng YOLO26n")
has = "raw_dets" in ss
paid = has and "invoice_pdf" in ss
st.markdown(f"<div class='steps'><div class='step {'done' if has else 'on'}'>1. Chọn ảnh</div>"
            f"<div class='step {'done' if paid else ('on' if has else '')}'>2. Kiểm tra giỏ hàng</div>"
            f"<div class='step {'done' if paid else ''}'>3. Thanh toán</div></div>", unsafe_allow_html=True)


def cart_qty(kept) -> tuple[dict, dict]:
    auto = count_classes(d.class_id for d in kept)
    adj = ss.adjust
    return auto, {cid: max(0, auto.get(cid, 0) + adj.get(cid, 0)) for cid in set(auto) | set(adj)}


# ------------------------------------------------------------------ BƯỚC 1: CHỌN ẢNH
if not has:
    st.markdown("""<div class='hero'><h2>Xin chào! Mời bạn thanh toán</h2>
<p>Đặt sản phẩm lên bàn, chụp một ảnh từ trên xuống rồi chọn ảnh bên dưới. Hệ thống tự nhận diện và tính tiền.</p>
<div class='howto'><div><b>1. Chọn ảnh</b>Hệ thống nhận diện ngay, không cần bấm thêm</div>
<div><b>2. Kiểm tra giỏ hàng</b>Sửa số lượng, đổi hoặc xóa dòng nhận sai</div>
<div><b>3. Thanh toán</b>Nhận hóa đơn PDF và mã QR thanh toán</div></div></div>""", unsafe_allow_html=True)
    up = st.file_uploader("Chọn ảnh khu vực thanh toán", type=["jpg", "jpeg", "png", "webp", "bmp", "heic", "heif"], key=f"up_{ss.get('up_n', 0)}",
                          help="Mẹo: nền bàn có màu, mỗi sản phẩm lộ ít nhất một nửa.")
    if up is not None:
        try:
            with st.spinner("Đang nhận diện sản phẩm..."):
                scan(up)
            st.rerun()
        except UnidentifiedImageError:
            st.error("Không đọc được tệp ảnh này. Hãy chọn ảnh JPG, PNG hoặc HEIC.")
    cards = "".join(f"<div class='pcard'><div class='ic'>{product_icon(p, 84)}</div><div class='nm'>{short_name(p.name)}</div>"
                    f"<div class='pr'>{fmt_money(p.price, CUR)}</div></div>" for p in catalog)
    st.markdown("**Sản phẩm có bán tại quầy**")
    st.markdown(f"<div class='pgrid'>{cards}</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------ BƯỚC 2: KIỂM TRA GIỎ HÀNG
elif not paid:
    kept = [d for d in ss.raw_dets if d.confidence >= conf]
    faded = [d for d in ss.raw_dets if 0.10 <= d.confidence < conf] if S["show_faded"] else []
    auto, qty = cart_qty(kept)
    adj = ss.adjust
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        if not kept:
            st.error("Không tìm thấy sản phẩm nào trong ảnh. Hãy chụp lại rõ hơn, đặt sản phẩm vào giữa khung hình, "
                     "hoặc thêm sản phẩm thủ công ở bên phải.")
        st.image(draw_detections(ss.image, kept, catalog, faded), use_container_width=True,
                 caption=f"{ss.img_name}: nhận diện {len(kept)} sản phẩm trong {ss.infer_ms / 1000:.1f} giây")
        if st.button("📷 Chọn ảnh khác", use_container_width=True):
            reset_cart()
            st.rerun()
    with right:
        bill = build_bill(qty, catalog)
        st.markdown(f"<div class='total-card'><div class='lbl'>TỔNG TIỀN</div><div class='row'><div class='val'>{fmt_money(bill.total, CUR)}</div>"
                    f"<div class='cnt'>{bill.total_quantity} sản phẩm</div></div></div>", unsafe_allow_html=True)
        if st.button(f"Thanh toán {fmt_money(bill.total, CUR)}", type="primary", icon=":material/payments:",
                     disabled=bill.total_quantity == 0, use_container_width=True):
            now = datetime.now()
            pdf = make_receipt_pdf(bill, app_cfg["store_name"], ss.invoice_no, when=now, currency=CUR)
            try:   # ổ đĩa cloud có thể chỉ đọc: vẫn cho khách tải hóa đơn dù không lưu được lịch sử
                RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
                (RECEIPT_DIR / f"{ss.invoice_no}.pdf").write_bytes(pdf)
                append_invoice(HISTORY, bill, ss.invoice_no, now)
            except OSError:
                st.toast("Không lưu được lịch sử hóa đơn trên máy chủ.", icon="⚠️")
            ss.invoice_pdf, ss.invoice_bill = pdf, bill
            st.rerun()
        low = [c for c in auto if (lambda m: m and sum(m) / len(m) < 0.5)([d.confidence for d in kept if d.class_id == c])]
        if low:
            st.warning(f"Có {len(low)} sản phẩm hệ thống chưa chắc chắn (nhãn đỏ). Bạn kiểm tra giúp, nếu sai thì bấm ⇄ để đổi.")
        st.markdown("**Giỏ hàng** <span class='hint'>· kiểm tra trước khi thanh toán</span>", unsafe_allow_html=True)
        with st.container(border=True):
            rows = sorted(c for c in qty if qty[c] > 0 or c in auto)
            if not rows:
                st.caption("Giỏ hàng trống. Thêm sản phẩm thủ công ở bên dưới.")
            for cid in rows:
                p = catalog.by_id(cid)
                mine = [d for d in kept if d.class_id == cid]
                best = max(mine, key=lambda d: d.confidence) if mine else None
                k = f"{ss.invoice_no}_{cid}_{auto.get(cid, 0)}_{adj.get(cid, 0)}"
                c_img, c_name, c_qty, c_amt, c_swap, c_del = st.columns([0.9, 3.0, 1.3, 1.5, 0.55, 0.55], vertical_alignment="center")
                if best is not None:
                    c_img.image(thumbnail(ss.image, best), width=56)
                else:
                    c_img.markdown(f"<div style='text-align:center'>{product_icon(p, 56)}</div>", unsafe_allow_html=True)
                badge = conf_badge(sum(d.confidence for d in mine) / len(mine)) if mine else "<span class='badge b-mid'>Thêm tay</span>"
                c_name.markdown(f"<div class='cart-name' title='{short_name(p.name)}'>{short_name(p.name)}</div>"
                                f"<div class='cart-sub'>{fmt_money(p.price, CUR)} / {p.unit.lower()}{badge}</div>", unsafe_allow_html=True)
                new = c_qty.number_input("Số lượng", 0, 99, qty[cid], 1, key=f"q_{k}", label_visibility="collapsed")
                c_amt.markdown(f"<div class='cart-amt'>{fmt_money(new * p.price, CUR)}</div>", unsafe_allow_html=True)
                with c_swap.popover("", icon=":material/swap_horiz:", help="Nhận nhầm? Đổi sang sản phẩm đúng"):
                    st.caption(f"Chuyển {new} sản phẩm của dòng này sang:")
                    tgt = st.selectbox("Sản phẩm đúng", [q.class_id for q in catalog if q.class_id != cid],
                                       format_func=lambda i: short_name(catalog.by_id(i).name), key=f"s_{k}")
                    if st.button("Đổi", key=f"sb_{k}", type="primary"):
                        adj[tgt] = adj.get(tgt, 0) + new
                        adj[cid] = -auto.get(cid, 0)
                        st.rerun()
                if c_del.button("", icon=":material/delete:", key=f"d_{k}", help="Xóa dòng này"):
                    adj[cid] = -auto.get(cid, 0)
                    st.rerun()
                if new != qty[cid]:
                    adj[cid] = new - auto.get(cid, 0)
                    st.rerun()
        a1, a2 = st.columns([3, 1.3], vertical_alignment="bottom")
        pick = a1.selectbox("Thêm sản phẩm hệ thống nhận sót", [p.class_id for p in catalog],
                            format_func=lambda i: f"{short_name(catalog.by_id(i).name)} · {fmt_money(catalog.by_id(i).price, CUR)}")
        if a2.button("Thêm", icon=":material/add:", use_container_width=True):
            adj[pick] = adj.get(pick, 0) + 1
            st.rerun()

# ------------------------------------------------------------------ BƯỚC 3: THANH TOÁN
else:
    bill = ss.invoice_bill
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.markdown(f"<div class='paid'><div class='big'>✅</div><h3>Đã tạo hóa đơn {ss.invoice_no}</h3>"
                    f"<div class='hint'>Hóa đơn đã được lưu vào lịch sử của cửa hàng</div></div>", unsafe_allow_html=True)
        lines = "".join(f"<tr><td>{short_name(ln.name)}</td><td class='r'>{ln.quantity} × {fmt_money(ln.unit_price, CUR)}</td>"
                        f"<td class='r'>{fmt_money(ln.amount, CUR)}</td></tr>" for ln in bill.lines)
        st.markdown(f"<div class='receipt' style='margin-top:12px'><table>{lines}"
                    f"<tr class='tot'><td>Tổng cộng ({bill.total_quantity} sản phẩm)</td><td></td><td class='r'>{fmt_money(bill.total, CUR)}</td></tr>"
                    f"</table></div>", unsafe_allow_html=True)
    with right:
        st.markdown(f"<div class='total-card'><div class='lbl'>SỐ TIỀN CẦN THANH TOÁN</div><div class='val'>{fmt_money(bill.total, CUR)}</div></div>",
                    unsafe_allow_html=True)
        try:
            import segno
            buf = io.BytesIO()
            segno.make(f"MO PHONG|{ss.invoice_no}|{bill.total}", error="m").save(buf, kind="png", scale=7, border=2)
            q1, q2 = st.columns([1, 1.3], vertical_alignment="center")
            q1.image(buf.getvalue(), use_container_width=True)
            q2.markdown("**Quét mã để thanh toán**  \n Rất vui được phục vụ bạn! ")
        except ImportError:
            st.caption("Cài thư viện segno (pip install segno) để hiện mã QR thanh toán mô phỏng.")
        st.download_button("Tải hóa đơn (PDF)", ss.invoice_pdf, file_name=f"{ss.invoice_no}.pdf", mime="application/pdf",
                           icon=":material/download:", use_container_width=True)
        if st.button("Lượt khách mới", type="primary", icon=":material/restart_alt:", use_container_width=True):
            reset_cart()
            st.rerun()
        if st.button("Quay lại sửa giỏ hàng", icon=":material/arrow_back:", use_container_width=True,
                     help="Sửa xong bấm Thanh toán lại, hóa đơn cùng số sẽ được cập nhật"):
            ss.pop("invoice_pdf", None)
            st.rerun()
