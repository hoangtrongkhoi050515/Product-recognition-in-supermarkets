"""Chương trình mô phỏng quầy thanh toán tự động (Streamlit).

Chạy tại thư mục gốc dự án:  streamlit run app/streamlit_app.py
Gồm hai trang: Quầy thanh toán (app/kiosk.py, dành cho khách) và Quản trị (app/admin.py, có mã PIN).
Phần dùng chung nằm trong app/common.py.
"""
import streamlit as st

st.set_page_config(page_title="Quầy thanh toán tự động", page_icon="🛒", layout="wide")

from common import CSS  # noqa: E402  (phải gọi set_page_config trước)

st.markdown(CSS, unsafe_allow_html=True)
st.navigation([st.Page("kiosk.py", title="Quầy thanh toán", icon=":material/shopping_cart:", default=True),
               st.Page("admin.py", title="Quản trị", icon=":material/admin_panel_settings:")],
              position="top").run()
