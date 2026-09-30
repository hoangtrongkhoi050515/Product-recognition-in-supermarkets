"""Vẽ lại các hình minh họa dùng trong báo cáo (Hình 1.1, 2.1, 2.2).

Cách chạy (tại thư mục gốc dự án):  python docs/figures/ve_hinh_bao_cao.py
Ảnh PNG được ghi đè vào chính thư mục docs/figures/.
"""
import os
import matplotlib; matplotlib.use('Agg')  # không cần màn hình
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
plt.rcParams['font.family']='DejaVu Sans'
OUT=os.path.dirname(os.path.abspath(__file__))  # lưu ảnh cạnh file script
INK='#1f2937'; BLUE='#dbeafe'; BLUE_E='#1d4ed8'; GREEN='#dcfce7'; GREEN_E='#15803d'; AMB='#fef3c7'; AMB_E='#b45309'; GR='#f3f4f6'

def box(ax,x,y,w,h,txt,fc=BLUE,ec=BLUE_E,fs=10,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.08",fc=fc,ec=ec,lw=1.4))
    ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=fs,color=INK,fontweight='bold' if bold else 'normal',wrap=True)
def arrow(ax,x1,y1,x2,y2):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='-|>',color='#374151',lw=1.4))

# ---- Hình 1.1: minh họa IoU
fig,axs=plt.subplots(1,3,figsize=(9,3.1))
for ax,(title,gt,pr) in zip(axs,[('IoU ≈ 0,14 (kém)',(1,1,4,3),(3.6,2.4,4,3)),('IoU ≈ 0,47 (trung bình)',(1,1,4,3),(2.2,1.6,4,3)),('IoU ≈ 0,82 (tốt)',(1,1,4,3),(1.3,1.2,4,3))]):
    ax.add_patch(Rectangle(gt[:2],gt[2],gt[3],fill=False,ec=GREEN_E,lw=2.2,label='Khung thật (Ground truth)'))
    ax.add_patch(Rectangle(pr[:2],pr[2],pr[3],fill=False,ec='#dc2626',lw=2.2,ls='--',label='Khung dự đoán'))
    x1=max(gt[0],pr[0]);y1=max(gt[1],pr[1]);x2=min(gt[0]+gt[2],pr[0]+pr[2]);y2=min(gt[1]+gt[3],pr[1]+pr[3])
    if x2>x1 and y2>y1: ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fc='#fde68a',alpha=.8))
    inter=max(0,x2-x1)*max(0,y2-y1); u=gt[2]*gt[3]*2-inter
    ax.set_title(f'IoU = {inter/u:.2f}'.replace('.',','),fontsize=11,color=INK)
    ax.set_xlim(0,9);ax.set_ylim(0,6.2);ax.set_aspect('equal');ax.axis('off')
h,l=axs[0].get_legend_handles_labels()
fig.legend(h,l,loc='lower center',ncol=2,frameon=False,fontsize=10)
plt.tight_layout(rect=(0,0.1,1,1)); plt.savefig(os.path.join(OUT,'hinh_1_1_iou.png'),dpi=200); plt.close()

# ---- Hình 2.1: quy trình tổng thể
fig,ax=plt.subplots(figsize=(9,5.2)); ax.set_xlim(0,10); ax.set_ylim(0,6); ax.axis('off')
steps=[('1. Thu thập ảnh\nsản phẩm (19 lớp)',0.2),('2. Gán nhãn\nkhung bao (YOLO)',2.2),('3. Tiền xử lý &\ntăng cường dữ liệu',4.2),('4. Chia tập\nTrain / Val / Test',6.2),('',8.2)]
for t,x in steps[:4]: box(ax,x,4.4,1.7,1.1,t,fs=9.5)
for i in range(3): arrow(ax,steps[i][1]+1.7,4.95,steps[i+1][1],4.95)
# split
ax.plot([7.05,7.05],[4.4,3.75],color='#374151',lw=1.4); ax.plot([2.3,7.8],[3.75,3.75],color='#374151',lw=1.4)
arrow(ax,2.3,3.75,2.3,3.35); arrow(ax,7.8,3.75,7.8,3.35)
box(ax,0.3,1.95,4.0,1.4,'Nhánh A – Phân loại (ảnh cắt theo khung bao)\nKNN · SVM · Random Forest\n(đặc trưng thủ công)  ·  CNN  ·  Transfer Learning',fc=AMB,ec=AMB_E,fs=9)
box(ax,5.7,1.95,4.0,1.4,'Nhánh B – Phát hiện đối tượng (ảnh gốc)\nYOLO (Ultralytics)\nnhiều sản phẩm / ảnh',fc=GREEN,ec=GREEN_E,fs=9)
arrow(ax,2.3,1.95,4.0,1.3); arrow(ax,7.7,1.95,5.8,1.3)
box(ax,2.6,0.5,4.6,0.8,'5. Đánh giá, so sánh & chọn mô hình chính',fc=GR,ec='#4b5563',fs=9.5,bold=True)
box(ax,8.0,0.5,1.9,0.8,'6. Tích hợp\nhệ thống',fc=BLUE,ec=BLUE_E,fs=9.5)
arrow(ax,7.2,0.9,8.0,0.9)
plt.tight_layout(); plt.savefig(os.path.join(OUT,'hinh_2_1_quy_trinh.png'),dpi=200); plt.close()

# ---- Hình 2.2: kiến trúc hệ thống
fig,ax=plt.subplots(figsize=(9,4.6)); ax.set_xlim(0,10); ax.set_ylim(0,5.2); ax.axis('off')
box(ax,0.1,3.4,1.8,1.2,'Đầu vào\nẢnh / Video /\nCamera',fc=GR,ec='#4b5563',fs=9.5)
box(ax,2.4,3.4,2.1,1.2,'Module nhận diện\nMô hình YOLO\nđã huấn luyện',fs=9.5,bold=False)
box(ax,5.0,3.4,2.2,1.2,'Module hậu xử lý\nLọc ngưỡng, NMS,\nđếm theo lớp',fs=9.5)
box(ax,7.7,3.4,2.2,1.2,'Module tính tiền\nTra bảng giá,\ntính thành tiền',fc=AMB,ec=AMB_E,fs=9.5)
for a,b in [(1.9,2.4),(4.5,5.0),(7.2,7.7)]: arrow(ax,a,4.0,b,4.0)
box(ax,7.7,1.5,2.2,1.2,'Module hóa đơn\nTạo & xuất\nhóa đơn',fc=AMB,ec=AMB_E,fs=9.5)
arrow(ax,8.8,3.4,8.8,2.7)
box(ax,2.4,1.5,4.8,1.2,'Giao diện người dùng (GUI)\nHiển thị khung bao, danh sách sản phẩm, tổng tiền\nXác nhận / chỉnh sửa / thanh toán',fc=GREEN,ec=GREEN_E,fs=9.5)
arrow(ax,7.7,2.1,7.2,2.1); arrow(ax,6.1,3.4,6.1,2.7); arrow(ax,3.45,2.7,3.45,3.4)
box(ax,7.7,0.1,2.2,0.9,'Dữ liệu cấu hình\nbảng giá (CSV/JSON)',fc='white',ec='#6b7280',fs=9)
ax.annotate('',xy=(9.6,3.4),xytext=(9.6,1.0),arrowprops=dict(arrowstyle='-|>',color='#6b7280',lw=1.1,ls='--'))
plt.tight_layout(); plt.savefig(os.path.join(OUT,'hinh_2_2_kien_truc.png'),dpi=200); plt.close()
