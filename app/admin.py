"""Trang Quản trị: cài đặt ngưỡng, chỉ số mô hình, lịch sử hóa đơn, thống kê bán hàng và bảng giá."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import streamlit as st

from common import (CUR, HISTORY, METRICS, RAW_CONF, RECEIPT_DIR, SETTINGS, app_cfg, banner, catalog, get_detector, load_image,
                    settings, ss)
from src.billing.billing import fmt_money
from src.billing.history import load_history, product_stats
from src.detection.draw import draw_detections, short_name
from src.settings import save_settings

banner("Quản trị cửa hàng", "Cài đặt nhận diện · Chỉ số mô hình · Lịch sử hóa đơn · Thống kê bán hàng", admin=True)

# ------------------------------------------------------------------ ĐĂNG NHẬP
if not ss.get("admin_ok"):
    _, mid, _ = st.columns([1, 1.1, 1])
    with mid.form("pin"):
        st.markdown("#### 🔒 Đăng nhập quản trị")
        pin = st.text_input("Mã PIN", type="password")
        if st.form_submit_button("Đăng nhập", type="primary", use_container_width=True):
            if pin == str(app_cfg.get("admin_pin", "1414")):
                ss.admin_ok = True
                st.rerun()
            st.error("Sai mã PIN.")
    mid.caption("Vui lòng nhập mã pin.")
    st.stop()

top_l, top_r = st.columns([6, 1])
if top_r.button("Đăng xuất", icon=":material/logout:", use_container_width=True):
    ss.admin_ok = False
    st.rerun()

tab_set, tab_model, tab_hist, tab_stat, tab_price = st.tabs(
    ["Cài đặt nhận diện", "Chỉ số mô hình", "Lịch sử hóa đơn", "Thống kê bán hàng", "Bảng giá"])

# ------------------------------------------------------------------ CÀI ĐẶT NHẬN DIỆN
with tab_set:
    S = settings()
    c1, c2 = st.columns([1, 1.6], gap="large")
    with c1:
        st.markdown("#### Ngưỡng độ tin cậy")
        conf = st.slider("Ngưỡng dùng ở quầy thanh toán", 0.05, 0.95, S["conf"], 0.05,
                         help="Tăng để giảm nhận nhầm, giảm để bớt bỏ sót.")
        faded = st.toggle("Hiện khung bị loại (xám) trên màn hình khách", value=S["show_faded"])
        if (conf, faded) != (S["conf"], S["show_faded"]):
            save_settings(SETTINGS, {"conf": conf, "show_faded": faded})
            st.toast(f"Đã lưu. Quầy thanh toán dùng ngưỡng {conf:.2f}.", icon="✅")
        f1 = METRICS.get("f1_best", {"conf": 0.458, "f1": 0.88})
        st.info(f"Gợi ý: F1 trên tập Test cao nhất ({f1['f1']:.2f}) tại ngưỡng {f1['conf']:.3f}, nên ngưỡng mặc định là 0,45. "
                "Cài đặt được lưu vào tệp Logs/cai_dat.json và có hiệu lực ngay ở trang Quầy thanh toán.")
        st.markdown("#### Thử ngưỡng trên một ảnh")
        up = st.file_uploader("Ảnh thử (không tạo hóa đơn)", type=["jpg", "jpeg", "png", "webp", "bmp"], key="adm_up")
        if up is not None and up.file_id != ss.get("adm_id"):
            with st.spinner("Đang nhận diện..."):
                img = load_image(up)
                t0 = time.perf_counter()
                ss.adm_raw = get_detector().detect(img, RAW_CONF)
                ss.adm_ms = (time.perf_counter() - t0) * 1000
                ss.adm_img, ss.adm_id = img, up.file_id
    with c2:
        raw, img = (ss.get("adm_raw"), ss.get("adm_img")) if up is not None and "adm_raw" in ss else (ss.get("raw_dets"), ss.get("image"))
        if raw is None:
            st.markdown("#### Kết quả thử")
            st.info("Chọn một ảnh thử ở bên trái (hoặc quét một ảnh ở trang Quầy thanh toán trong cùng trình duyệt) "
                    "để xem số nhận diện được giữ và bị loại ở từng ngưỡng.")
        else:
            confs = np.array([d.confidence for d in raw])
            kept = [d for d in raw if d.confidence >= conf]
            st.markdown(f"#### Kết quả thử ở ngưỡng {conf:.2f}")
            a = st.columns(4)
            a[0].metric("Phát hiện thô (≥ 0,05)", len(confs))
            a[1].metric("Giữ lại", len(kept))
            a[2].metric("Bị loại", len(confs) - len(kept))
            a[3].metric("Tin cậy TB (đã giữ)", f"{np.mean([d.confidence for d in kept]):.1%}" if kept else "-")
            st.image(draw_detections(img, kept, catalog, [d for d in raw if 0.10 <= d.confidence < conf]), use_container_width=True,
                     caption="Khung màu: được giữ; khung xám: bị loại do dưới ngưỡng")
            ca, cb = st.columns(2)
            bins = pd.cut(pd.Series(confs), bins=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0], include_lowest=True).value_counts().sort_index()
            bins.index = [f"{max(iv.left, 0):.1f}-{iv.right:.1f}" for iv in bins.index]
            ca.markdown("**Phân bố độ tin cậy**")
            ca.bar_chart(bins.rename("Số nhận diện"), color="#0f766e", height=260)
            rows = []
            for cid in sorted({d.class_id for d in raw}):
                ds = [d.confidence for d in raw if d.class_id == cid]
                rows.append({"Sản phẩm": short_name(catalog.by_id(cid).name), "Giữ": sum(c >= conf for c in ds),
                             "Loại": sum(c < conf for c in ds), "Tin cậy cao nhất": max(ds)})
            cb.markdown("**Theo từng sản phẩm**")
            cb.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, height=260,
                         column_config={"Tin cậy cao nhất": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})

# ------------------------------------------------------------------ CHỈ SỐ MÔ HÌNH
with tab_model:
    t = METRICS["test"]
    st.markdown(f"#### Chất lượng mô hình {METRICS['model']} trên tập Test")
    k = st.columns(5)
    k[0].metric("mAP@0.5", f"{t['map50']:.3f}", help="Độ chính xác trung bình tại IoU 0,5: càng gần 1 càng tốt.")
    k[1].metric("mAP@0.5:0.95", f"{t['map5095']:.3f}", help="Độ chính xác trung bình, tính cả độ khít của khung bao (IoU từ 0,5 đến 0,95).")
    k[2].metric("Precision", f"{t['precision']:.3f}", help="Trong số sản phẩm mô hình báo, bao nhiêu phần là đúng.")
    k[3].metric("Recall", f"{t['recall']:.3f}", help="Trong số sản phẩm thật, mô hình tìm ra được bao nhiêu phần.")
    k[4].metric("Thời gian suy luận", f"{t['ms']} ms/ảnh")

    st.markdown("#### Kết quả theo từng lớp sản phẩm")
    pc = pd.DataFrame(METRICS["per_class"]).rename(columns={"name": "Sản phẩm", "precision": "Precision", "recall": "Recall",
                                                              "map50": "mAP@0.5", "map5095": "mAP@0.5:0.95"})
    pcol = {c: st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f") for c in ["Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"]}
    cc, cd = st.columns([3, 2])
    cc.dataframe(pc[["Sản phẩm", "Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"]], hide_index=True, use_container_width=True,
                 column_config=pcol, height=460)
    cd.markdown("**mAP@0.5:0.95 theo lớp** (thấp là lớp yếu)")
    cd.bar_chart(pc.set_index("Sản phẩm")["mAP@0.5:0.95"], horizontal=True, color="#14b8a6", height=420)
    weak = pc.nsmallest(2, "mAP@0.5:0.95")["Sản phẩm"].tolist()
    st.caption(f"Hai lớp yếu nhất: {weak[0]} và {weak[1]}. Với các lớp này nên kiểm tra kỹ giỏ hàng trước khi xuất hóa đơn.")

    st.markdown("#### So sánh YOLO26n và YOLO11n")
    cmp_ = METRICS["compare"]
    names = {"precision": "Precision (Test)", "recall": "Recall (Test)", "map50": "mAP@0.5 (Test)", "map5095": "mAP@0.5:0.95 (Test)",
             "val_map50": "mAP@0.5 (Val)", "val_map5095": "mAP@0.5:0.95 (Val)"}
    cmp_df = pd.DataFrame({n: {m: cmp_[m][k_] for m in cmp_} for k_, n in names.items()}).T
    ce, cf = st.columns(2)
    ce.dataframe(cmp_df.style.format("{:.3f}"), use_container_width=True)
    cf.bar_chart(cmp_df, color=["#0f766e", "#94a3b8"], height=280)
    cost = pd.DataFrame({m: {"Epoch đã chạy": cmp_[m]["epochs"], "Epoch tốt nhất": cmp_[m]["best_epoch"],
                             "Huấn luyện (giây)": cmp_[m]["train_sec"], "GFLOPs": cmp_[m]["gflops"],
                             "Suy luận (ms/ảnh)": cmp_[m]["ms"]} for m in cmp_})
    st.dataframe(cost, use_container_width=True)

    st.markdown("#### Bộ dữ liệu")
    ds = METRICS["dataset"]
    g = st.columns(4)
    g[0].metric("Số ảnh", f"{ds['images']:,}".replace(",", "."))
    g[1].metric("Số khung bao", f"{ds['boxes']:,}".replace(",", "."))
    g[2].metric("Số lớp", ds["classes"])
    g[3].metric("Train / Val / Test", " / ".join(str(s_["images"]) for s_ in METRICS["split"]))
    st.markdown("**Số khung bao theo lớp**")
    st.bar_chart(pc.set_index("Sản phẩm")["objects"], horizontal=True, color="#0f766e", height=380)

# ------------------------------------------------------------------ LỊCH SỬ
with tab_hist:
    hist = load_history(HISTORY)
    if hist.empty:
        st.info("Chưa có hóa đơn nào. Hóa đơn thanh toán ở trang Quầy thanh toán sẽ được lưu tại đây.")
    else:
        show = hist.assign(total=hist["total"].map(lambda v: fmt_money(v, CUR))).rename(columns={
            "invoice_no": "Số hóa đơn", "time": "Thời gian", "total_quantity": "Số lượng", "total": "Tổng tiền"})
        st.dataframe(show[["Số hóa đơn", "Thời gian", "Số lượng", "Tổng tiền"]], hide_index=True, use_container_width=True)
        h1, h2 = st.columns([3, 2], vertical_alignment="bottom")
        no = h1.selectbox("Mở lại hóa đơn", list(hist["invoice_no"]))
        f = RECEIPT_DIR / f"{no}.pdf"
        if f.exists():
            h2.download_button(f"Tải lại {no} (PDF)", f.read_bytes(), file_name=f.name, mime="application/pdf",
                               icon=":material/download:", use_container_width=True)
        else:
            h2.caption("Không tìm thấy tệp PDF của hóa đơn này (có thể đã bị xóa).")

# ------------------------------------------------------------------ THỐNG KÊ
with tab_stat:
    hist = load_history(HISTORY)
    if hist.empty:
        st.info("Chưa có dữ liệu thống kê. Hãy thanh toán ít nhất một hóa đơn.")
    else:
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Số hóa đơn", len(hist))
        s2.metric("Doanh thu", fmt_money(int(hist["total"].sum()), CUR))
        s3.metric("Trung bình mỗi hóa đơn", fmt_money(int(hist["total"].mean()), CUR))
        s4.metric("Sản phẩm đã bán", int(hist["total_quantity"].sum()))
        stats = product_stats(hist, catalog)
        stats["Sản phẩm"] = stats["Sản phẩm"].map(short_name)
        c_a, c_b = st.columns(2)
        c_a.markdown("**Số lượng bán theo sản phẩm**")
        c_a.bar_chart(stats.set_index("Sản phẩm")["Số lượng bán"], horizontal=True, color="#0f766e")
        c_b.markdown("**Doanh thu theo sản phẩm (đ)**")
        c_b.bar_chart(stats.set_index("Sản phẩm")["Doanh thu"], horizontal=True, color="#14b8a6")

# ------------------------------------------------------------------ BẢNG GIÁ
with tab_price:
    st.dataframe(pd.DataFrame([{"Mã lớp": p.class_id, "Sản phẩm": short_name(p.name), "Đơn vị": p.unit, "Đơn giá": fmt_money(p.price, CUR)}
                               for p in catalog]), hide_index=True, use_container_width=True, height=470)
    #st.caption("Bảng giá đọc từ tệp configs/products.csv; sửa giá trong tệp này rồi khởi động lại chương trình.")
