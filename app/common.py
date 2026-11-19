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
ss = st.session_state
ASSETS = Path(__file__).resolve().parent / "assets" / "products"


@st.cache_data
def product_img_uri(code: str) -> str:
    """Ảnh minh họa sản phẩm (cắt từ bộ dữ liệu) dưới dạng data URI; rỗng nếu chưa có file."""
    f = ASSETS / f"{code}.jpg"
    return "data:image/jpeg;base64," + base64.b64encode(f.read_bytes()).decode() if f.exists() else ""


def product_icon(p, size: int = 56) -> str:
    """HTML hiển thị ảnh sản phẩm; không có ảnh thì dùng ô xám có biểu tượng gói hàng (nét mảnh)."""
    uri = product_img_uri(p.code)
    if uri:
        return f"<img src='{uri}' style='width:{size}px;height:{size}px;object-fit:contain;border-radius:8px;background:white'>"
    ic = size // 2
    return (f"<span class='ph' style='width:{size}px;height:{size}px'><svg width='{ic}' height='{ic}' viewBox='0 0 24 24' fill='none' "
            "stroke='currentColor' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'>"
            "<path d='M21 8 12 3 3 8v8l9 5 9-5z'/><path d='m3 8 9 5 9-5M12 13v8'/></svg></span>")


def settings() -> dict:
    return load_settings(SETTINGS, {"conf": app_cfg.get("conf", 0.45)})


# Giao diện: nền và khung dùng màu xám trung tính, màu teal #0f766e chỉ làm màu nhấn
# (nút chính, tổng tiền, bước đang làm). Khoảng cách theo bội số 4px, chữ đậm tối đa 600.
CSS = """
<style>
:root {--accent: #0f766e; --ink: #0f172a; --ink-2: #475569; --muted: #64748b;
       --raised: #f4f4f5; --line: rgba(15, 23, 42, .09); --num: ui-monospace, "SF Mono", Consolas, monospace;}
.block-container {padding-top: 4.5rem; max-width: 1440px;}
section[data-testid="stSidebar"], div[data-testid="stSidebarCollapsedControl"] {display: none;}
.banner {background: var(--accent); color: white; border-radius: 16px;
         padding: 16px 24px; display: flex; align-items: center; gap: 16px; margin-bottom: 16px;}
.banner.admin {background: #1e293b;}
.banner h1 {color: white; font-size: 1.5rem; font-weight: 600; margin: 0; padding: 0;}
.banner p {margin: 0; opacity: .85; font-size: .9rem;}
.banner .right {margin-left: auto; text-align: right; font-size: .85rem; opacity: .85; font-variant-numeric: tabular-nums;}
.steps {display: flex; gap: 8px; margin: 0 0 16px 0;}
.step {flex: 1; border-radius: 8px; padding: 8px 16px; background: var(--raised); color: var(--muted);
       font-weight: 500; font-size: .95rem; border: 1px solid transparent;}
.step.on {background: white; color: var(--accent); border-color: var(--accent);}
.step.done {background: var(--accent); color: white;}
div[data-testid="stMetricValue"] {font-size: 1.6rem; font-weight: 500; font-variant-numeric: tabular-nums;}
div[data-testid="stMetric"] {background: white; border: 1px solid var(--line); border-radius: 12px; padding: 12px 16px;}
.total-card {background: var(--accent); color: white; border-radius: 16px; padding: 16px 24px; margin: 0 0 12px 0;}
.total-card .lbl {opacity: .85; font-size: .9rem;}
.total-card .row {display: flex; justify-content: space-between; align-items: end;}
.total-card .val {font-size: 2.6rem; font-weight: 600; line-height: 1.1; font-variant-numeric: tabular-nums;}
.total-card .cnt {opacity: .85;}
.cart-name {font-weight: 500; line-height: 1.25; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.cart-sub {color: var(--muted); font-size: .82rem; white-space: nowrap; font-variant-numeric: tabular-nums;}
.cart-amt {font-weight: 600; color: var(--ink); text-align: right; white-space: nowrap; position: relative; top: -7px;
           font-variant-numeric: tabular-nums;}
.badge {display: inline-block; border-radius: 4px; padding: 0 6px; font-size: .75rem; font-weight: 500; margin-left: 8px;}
.b-hi {background: #dcfce7; color: #166534;} .b-mid {background: #fef3c7; color: #854d0e;} .b-lo {background: #fee2e2; color: #991b1b;}
.hero {background: white; border: 1px solid var(--line); border-radius: 16px; padding: 24px 32px; margin-bottom: 16px;}
.hero h2 {margin: 0 0 4px 0; color: var(--ink); font-weight: 600; padding: 0;} .hero p {margin: 0; color: var(--ink-2);}
.howto {display: grid; grid-template-columns: repeat(3,1fr); gap: 12px; margin-top: 16px;}
.howto div {background: var(--raised); border-radius: 12px; padding: 12px 16px; color: var(--ink-2); font-size: .92rem;}
.howto b {display: block; color: var(--ink); font-weight: 500; font-size: 1.05rem;}
.howto b span {color: var(--accent); font-variant-numeric: tabular-nums; margin-right: 4px;}
.pgrid {display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; margin-top: 8px;}
.pcard {background: white; border: 1px solid var(--line); border-radius: 12px; padding: 12px 8px; text-align: center;}
.pcard .ic {line-height: 1.2; margin-bottom: 4px;} .pcard .ic img {border: 1px solid var(--line);}
.pcard .nm {font-size: .8rem; font-weight: 500; color: var(--ink); min-height: 2.3em; line-height: 1.1rem;}
.pcard .pr {color: var(--ink-2); font-size: .85rem; font-variant-numeric: tabular-nums;}
.receipt {background: white; border: 1px solid var(--line); border-radius: 16px; padding: 16px 24px;}
.receipt table {width: 100%; border-collapse: collapse; border: none; margin: 0;} .receipt tr {border: none; background: none !important;}
.receipt td {padding: 8px 4px; border: none; border-bottom: 1px dashed var(--line);}
.receipt td.r {text-align: right; font-variant-numeric: tabular-nums;}
.receipt .tot td {font-weight: 600; font-size: 1.2rem; color: var(--accent); border: none; padding-top: 12px;}
.paid {background: white; border: 1px solid var(--line); border-radius: 16px; padding: 20px 24px; text-align: center;}
.paid .big {display: inline-flex; width: 48px; height: 48px; border-radius: 50%; background: #dcfce7; color: #16a34a;
            align-items: center; justify-content: center;}
.paid h3 {color: var(--ink); font-weight: 600; margin: 8px 0 4px 0; padding: 0;}
.ph {display: inline-flex; align-items: center; justify-content: center; border-radius: 8px; background: var(--raised); color: var(--muted);}
.hint {color: var(--muted); font-size: .88rem;}
div[data-testid='stImage'] img {max-height: 540px; object-fit: contain;}
section[data-testid='stFileUploaderDropzone'] {border: 1px dashed rgba(15, 23, 42, .25); background: var(--raised); min-height: 112px;
                                               border-radius: 12px; padding: 24px;}
button[data-testid='stBaseButton-primary'], button[data-testid='stBaseButton-primaryFormSubmit'] {min-height: 3rem;}
button[data-testid='stBaseButton-primary'] p {font-size: 1.1rem; font-weight: 500;}
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
<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">{icon}</svg>
<div><h1>{title}</h1><p>{sub}</p></div><div class="right">{app_cfg.get('store_name', '')}<br>{datetime.now():%d/%m/%Y}</div></div>""",
                unsafe_allow_html=True)


@st.cache_resource(show_spinner="Đang nạp mô hình...")
def get_detector() -> Detector:
    return Detector()


def reset_cart() -> None:
    for k in ("image", "raw_dets", "adjust", "infer_ms", "invoice_pdf", "invoice_no", "invoice_bill", "scanned_id", "img_name"):
        ss.pop(k, None)
    ss.up_n = ss.get("up_n", 0) + 1   # đổi key của ô chọn ảnh để xóa ảnh cũ


try:   # ảnh HEIC/HEIF của iPhone: cần thư viện pi-heif (có trong app/requirements.txt)
    from pi_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


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
