#"""Phần dùng chung của hai trang: cấu hình, mô hình, giao diện chung và các hàm tiện ích."""
from __future__ import annotations

import base64
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

from src.catalog import load_catalog
from src.config import load_config
from src.detection.predict import Detector
from src.settings import load_settings

cfg = load_config()
app_cfg = cfg.section("app")
catalog = load_catalog()
CUR = app_cfg.get("currency", "đ")
RAW_CONF = 0.05   # phát hiện một lần ở ngưỡng thấp, sau đó lọc theo ngưỡng cài đặt (đổi ngưỡng không cần quét lại)
RECEIPT_DIR = cfg.root / app_cfg.get("receipt_dir", "Logs/receipts")
HISTORY = RECEIPT_DIR / "lich_su.csv"
SETTINGS = cfg.root / "Logs" / "cai_dat.json"
METRICS = json.loads((cfg.root / "configs" / "metrics.json").read_text(encoding="utf-8"))
ICONS = {0: "🥛", 1: "💧", 2: "🍫", 3: "🥤", 4: "🍜", 5: "🍜", 6: "🥔", 7: "🧃", 8: "🍪", 9: "🍜", 10: "🍰", 11: "🍲"}
ss = st.session_state
ASSETS = Path(__file__).resolve().parent / "assets" / "products"


@st.cache_data
def product_img_uri(code: str) -> str:
    """Ảnh minh họa sản phẩm (cắt từ bộ dữ liệu) dưới dạng data URI; rỗng nếu chưa có file."""
    f = ASSETS / f"{code}.jpg"
    return "data:image/jpeg;base64," + base64.b64encode(f.read_bytes()).decode() if f.exists() else ""


def product_icon(p, size: int = 56) -> str:
    """HTML hiển thị ảnh sản phẩm; không có ảnh thì dùng biểu tượng cảm xúc."""
    uri = product_img_uri(p.code)
    if uri:
        return f"<img src='{uri}' style='width:{size}px;height:{size}px;object-fit:contain;border-radius:10px;background:white'>"
    return f"<span style='font-size:{size * .6:.0f}px'>{ICONS.get(p.class_id, '🛒')}</span>"


def settings() -> dict:
    return load_settings(SETTINGS, {"conf": app_cfg.get("conf", 0.45)})


CSS = """
<style>
.block-container {padding-top: 4.6rem; max-width: 1440px;}
section[data-testid="stSidebar"], div[data-testid="stSidebarCollapsedControl"] {display: none;}
.banner {background: linear-gradient(90deg,#0f766e,#14b8a6); color: white; border-radius: 16px;
         padding: 14px 24px; display: flex; align-items: center; gap: 16px; margin-bottom: 14px;}
.banner.admin {background: linear-gradient(90deg,#1e293b,#475569);}
.banner h1 {color: white; font-size: 1.55rem; margin: 0; padding: 0;}
.banner p {margin: 0; opacity: .9; font-size: .9rem;}
.banner .right {margin-left: auto; text-align: right; font-size: .85rem; opacity: .9;}
.steps {display: flex; gap: 10px; margin: 0 0 16px 0;}
.step {flex: 1; border-radius: 10px; padding: 9px 14px; background: #f1f5f9; color: #94a3b8; font-weight: 600; font-size: .95rem;}
.step.on {background: #ccfbf1; color: #0f766e; border: 1px solid #5eead4;}
.step.done {background: #0f766e; color: white;}
div[data-testid="stMetricValue"] {font-size: 1.6rem;}
div[data-testid="stMetric"] {background: #f0fdfa; border: 1px solid #99f6e4; border-radius: 12px; padding: 10px 16px;}
.total-card {background: #0f766e; color: white; border-radius: 16px; padding: 16px 22px; margin: 0 0 12px 0;}
.total-card .lbl {opacity: .85; font-size: .85rem; letter-spacing: .04em;}
.total-card .row {display: flex; justify-content: space-between; align-items: end;}
.total-card .val {font-size: 2.6rem; font-weight: 700; line-height: 1.1;} .total-card .cnt {opacity: .9;}
.cart-name {font-weight: 600; line-height: 1.25; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.cart-sub {color: #64748b; font-size: .82rem; white-space: nowrap;}
.cart-amt {font-weight: 700; color: #0f766e; text-align: right; white-space: nowrap; position: relative; top: -7px;}
.badge {display: inline-block; border-radius: 999px; padding: 1px 9px; font-size: .78rem; font-weight: 600; margin-left: 6px;}
.b-hi {background: #dcfce7; color: #166534;} .b-mid {background: #fef9c3; color: #854d0e;} .b-lo {background: #fee2e2; color: #991b1b;}
.hero {background: linear-gradient(135deg,#f0fdfa,#ccfbf1); border: 1px solid #99f6e4; border-radius: 18px; padding: 26px 30px; margin-bottom: 14px;}
.hero h2 {margin: 0 0 4px 0; color: #0f766e; padding: 0;} .hero p {margin: 0; color: #134e4a;}
.howto {display: grid; grid-template-columns: repeat(3,1fr); gap: 10px; margin-top: 16px;}
.howto div {background: white; border-radius: 12px; padding: 12px 14px; color: #134e4a; font-size: .92rem; border: 1px solid #ccfbf1;}
.howto b {display: block; color: #0f766e; font-size: 1.05rem;}
.pgrid {display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; margin-top: 6px;}
.pcard {background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 10px 8px; text-align: center;}
.pcard .ic {line-height: 1.2; margin-bottom: 4px;} .pcard .ic img {border: 1px solid #f1f5f9;} .pcard .nm {font-size: .78rem; font-weight: 600; color: #0f172a; min-height: 2.3em; line-height: 1.1rem;}
.pcard .pr {color: #0f766e; font-weight: 700; font-size: .85rem;}
.receipt {background: white; border: 1px solid #e2e8f0; border-radius: 16px; padding: 18px 22px;}
.receipt table {width: 100%; border-collapse: collapse; border: none; margin: 0;} .receipt tr {border: none; background: none !important;}
.receipt td {padding: 7px 4px; border: none; border-bottom: 1px dashed #e2e8f0;}
.receipt td.r {text-align: right;} .receipt .tot td {font-weight: 700; font-size: 1.2rem; color: #0f766e; border: none; padding-top: 12px;}
.paid {background: #f0fdf4; border: 1px solid #86efac; border-radius: 16px; padding: 18px 22px; text-align: center;}
.paid .big {font-size: 2.4rem;} .paid h3 {color: #166534; margin: 4px 0; padding: 0;}
.hint {color: #64748b; font-size: .88rem;}
div[data-testid='stImage'] img {max-height: 540px; object-fit: contain;}
section[data-testid='stFileUploaderDropzone'] {border: 2px dashed #14b8a6; background: #f0fdfa; min-height: 110px; border-radius: 14px; padding: 24px;}
button[data-testid='stBaseButton-primary'], button[data-testid='stBaseButton-primaryFormSubmit'] {min-height: 3rem;}
button[data-testid='stBaseButton-primary'] p {font-size: 1.1rem; font-weight: 600;}
</style>
"""


def banner(title: str, sub: str, admin: bool = False) -> None:
    icon = ('<path d="M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 '
            '1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 '
            '.3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 '
            '1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 '
            '1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/>') if admin else \
        '<circle cx="9" cy="20" r="1.4"/><circle cx="18" cy="20" r="1.4"/><path d="M2 3h3l2.6 12.4a1 1 0 0 0 1 .8h8.9a1 1 0 0 0 1-.8L20 7H6"/>'
    st.markdown(f"""<div class="banner {'admin' if admin else ''}">
<svg width="42" height="42" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{icon}</svg>
<div><h1>{title}</h1><p>{sub}</p></div><div class="right">{app_cfg.get('store_name', '')}<br>{datetime.now():%d/%m/%Y}</div></div>""",
                unsafe_allow_html=True)


@st.cache_resource(show_spinner="Đang nạp mô hình...")
def get_detector() -> Detector:
    return Detector()


def reset_cart() -> None:
    for k in ("image", "raw_dets", "adjust", "infer_ms", "invoice_pdf", "invoice_no", "invoice_bill", "scanned_id", "img_name"):
        ss.pop(k, None)
    ss.up_n = ss.get("up_n", 0) + 1   # đổi key của ô chọn ảnh để xóa ảnh cũ


def load_image(file) -> np.ndarray:
    img = ImageOps.exif_transpose(Image.open(file)).convert("RGB")   # xoay đúng chiều ảnh chụp điện thoại
    return np.array(img)


def scan(file) -> None:
    """Nhận diện một lần ở ngưỡng thấp; kết quả được lọc lại theo ngưỡng cài đặt."""
    for k in ("adjust", "invoice_pdf", "invoice_bill"):
        ss.pop(k, None)
    image = load_image(file)
    t0 = time.perf_counter()
    raw = get_detector().detect(image, RAW_CONF)
    ss.infer_ms = (time.perf_counter() - t0) * 1000
    ss.image, ss.raw_dets, ss.adjust = image, raw, {}
    ss.invoice_no = datetime.now().strftime("HD%Y%m%d%H%M%S")
    ss.scanned_id, ss.img_name = file.file_id, file.name


def thumbnail(image: np.ndarray, det, size: int = 72) -> Image.Image:
    x1, y1, x2, y2 = (int(v) for v in det.xyxy)
    crop = Image.fromarray(image[max(y1, 0):y2, max(x1, 0):x2]).convert("RGB")
    crop = ImageOps.contain(crop, (size, size))
    canvas = Image.new("RGB", (size, size), "#f1f5f9")
    canvas.paste(crop, ((size - crop.width) // 2, (size - crop.height) // 2))
    return canvas


def conf_badge(c: float) -> str:
    cls, txt = ("b-hi", "Cao") if c >= 0.8 else (("b-mid", "Vừa") if c >= 0.5 else ("b-lo", "Thấp"))
    return f"<span class='badge {cls}'>{txt} {c:.0%}</span>"
