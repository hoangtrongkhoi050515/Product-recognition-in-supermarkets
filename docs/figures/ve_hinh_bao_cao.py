"""Vẽ lại các hình minh họa dùng trong báo cáo (Hình 1.1, 1.2, 1.3, 2.1, 2.2, 3.1, 3.2).

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

# ---- Hình 2.1: quy trình tổng thể (chỉ dùng YOLO)
fig,ax=plt.subplots(figsize=(9,3.6)); ax.set_xlim(0,10); ax.set_ylim(0,4.2); ax.axis('off')
row1=[('1. Thu thập ảnh\n12 sản phẩm',0.15),('2. Gán nhãn khung bao\n(Roboflow)',2.2),('3. Nhập, kiểm tra\nvà chia 70/15/15',4.25),('4. Huấn luyện\nYOLO11n & YOLO26n',6.3)]
for t,x in row1: box(ax,x,2.6,1.85,1.1,t,fs=9)
for i in range(3): arrow(ax,row1[i][1]+1.85,3.15,row1[i+1][1],3.15)
box(ax,8.35,2.6,1.5,1.1,'5. Đánh giá\ntrên tập Test',fc=GREEN,ec=GREEN_E,fs=9)
arrow(ax,8.15,3.15,8.35,3.15)
box(ax,2.2,0.6,3.0,1.1,'6. So sánh YOLO11n\nvà YOLO26n (mAP, tốc độ)',fc=GR,ec='#4b5563',fs=9,bold=True)
box(ax,6.0,0.6,3.0,1.1,'7. Tích hợp mô hình chính\nvào ứng dụng Streamlit',fc=AMB,ec=AMB_E,fs=9)
ax.plot([9.1,9.1],[2.6,2.15],color='#374151',lw=1.4); ax.plot([3.7,9.1],[2.15,2.15],color='#374151',lw=1.4); arrow(ax,3.7,2.15,3.7,1.7)
arrow(ax,5.2,1.15,6.0,1.15)
plt.tight_layout(); plt.savefig(os.path.join(OUT,'hinh_2_1_quy_trinh.png'),dpi=200); plt.close()

# ---- Hình 1.2: quy trình làm việc trên Roboflow
fig,ax=plt.subplots(figsize=(9,2.6)); ax.set_xlim(0,10); ax.set_ylim(0,2.8); ax.axis('off')
st=[('Tạo dự án\nObject Detection',0.1),('Tải ảnh lên\n(Upload)',2.1),('Vẽ khung bao\nvà gán lớp',4.1),('Tạo phiên bản\n(Version)',6.1),('Xuất dữ liệu\nđịnh dạng YOLOv8',8.1)]
for t,x in st: box(ax,x,0.9,1.8,1.1,t,fs=9)
for i in range(4): arrow(ax,st[i][1]+1.8,1.45,st[i+1][1],1.45)
ax.text(5,0.35,'Kết quả: thư mục train/valid/test + data.yaml → nhập vào dự án bằng import_roboflow.py',ha='center',fontsize=9,color=INK)
plt.tight_layout(); plt.savefig(os.path.join(OUT,'hinh_1_2_roboflow.png'),dpi=200); plt.close()

# ---- Hình 1.3: kiến trúc tổng quát của YOLO
fig,ax=plt.subplots(figsize=(9,3.0)); ax.set_xlim(0,10); ax.set_ylim(0,3.4); ax.axis('off')
box(ax,0.05,1.2,1.4,1.0,'Ảnh đầu vào\n640 × 640',fc=GR,ec='#4b5563',fs=9)
box(ax,2.0,1.2,2.0,1.0,'Backbone\ntrích xuất đặc trưng',fs=9)
box(ax,4.55,1.2,2.0,1.0,'Neck (FPN + PAN)\nkết hợp đa tỉ lệ',fc=AMB,ec=AMB_E,fs=9)
box(ax,7.1,1.2,1.5,1.0,'Head\nkhung + lớp',fc=GREEN,ec=GREEN_E,fs=9)
box(ax,9.0,1.2,0.95,1.0,'Khung\nbao, lớp',fc=GR,ec='#4b5563',fs=8.5)
for a,b in [(1.45,2.0),(4.0,4.55),(6.55,7.1),(8.6,9.0)]: arrow(ax,a,1.7,b,1.7)
for x,t in [(3.0,'P3/P4/P5'),(5.55,'3 mức phân giải')]:
    ax.text(x,0.75,t,ha='center',fontsize=8.5,color='#6b7280')
ax.text(5,2.9,'Kiến trúc tổng quát của mô hình YOLO một giai đoạn',ha='center',fontsize=10,color=INK)
plt.tight_layout(); plt.savefig(os.path.join(OUT,'hinh_1_3_kien_truc_yolo.png'),dpi=200); plt.close()

# ---- Hình 3.1: phân bố số đối tượng theo lớp (984 ảnh) và Hình 3.2: mAP50-95 theo lớp trên tập Test
CLS=['TH true MILK','Aquafina','KitKat','Pepsi','Handy Hảo Hảo','Handy Tomyum',"Lay's Stax",'Vinamilk','Oreo','Omachi bắp bò','Chocopie','Omachi tô tôm']
OBJ=[429,365,357,386,277,351,295,477,352,238,302,273]
fig,ax=plt.subplots(figsize=(9,4.2))
y=range(len(CLS))[::-1]
bars=ax.barh(list(y),OBJ,color='#3b82f6')
ax.set_yticks(list(y)); ax.set_yticklabels(CLS,fontsize=9)
for yy,v in zip(y,OBJ): ax.text(v+4,yy,str(v),va='center',fontsize=8.5,color=INK)
ax.axvline(200,color='#dc2626',ls='--',lw=1); ax.text(203,11.55,'mức tối thiểu 200',color='#dc2626',fontsize=8)
ax.set_xlabel('Số đối tượng (khung bao)'); ax.set_xlim(0,540)
for sp in ('top','right'): ax.spines[sp].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(OUT,'hinh_3_1_phan_bo.png'),dpi=200); plt.close()

M11=[0.540,0.849,0.756,0.746,0.663,0.808,0.812,0.814,0.722,0.838,0.809,0.877]
M26=[0.526,0.896,0.780,0.816,0.734,0.818,0.901,0.933,0.860,0.856,0.809,0.916]
import numpy as np
fig,ax=plt.subplots(figsize=(9,4.4)); x=np.arange(12); w=0.38
ax.bar(x-w/2,M11,w,label='YOLO11n',color='#93c5fd'); ax.bar(x+w/2,M26,w,label='YOLO26n',color='#1d4ed8')
ax.set_xticks(x); ax.set_xticklabels(CLS,rotation=35,ha='right',fontsize=8.5); ax.set_ylim(0,1.0); ax.set_ylabel('mAP@0.5:0.95 (tập Test)')
ax.legend(frameon=False)
for sp in ('top','right'): ax.spines[sp].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(OUT,'hinh_3_2_map_theo_lop.png'),dpi=200); plt.close()

# ---- Hình 2.2: kiến trúc hệ thống
fig,ax=plt.subplots(figsize=(9,4.6)); ax.set_xlim(0,10); ax.set_ylim(0,5.2); ax.axis('off')
box(ax,0.1,3.4,1.8,1.2,'Đầu vào\nẢnh tải lên /\nwebcam',fc=GR,ec='#4b5563',fs=9.5)
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
