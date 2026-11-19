"""Kiểm thử cấu hình huấn luyện YOLO26n (không cần cài ultralytics hay GPU)."""
from src.config import load_config
from src.detection.common import finetune_run_name, get_detection_cfg
from src.detection.train import build_train_args


def test_chi_con_yolo26n():
    cfg = get_detection_cfg()
    assert set(cfg["models"]) == {"yolo26n"}
    assert cfg["patience"] <= cfg["epochs"]


def test_lan_dau_dung_pretrained_coco():
    weights, name, args = build_train_args(get_detection_cfg(), None)
    assert (weights, name) == ("yolo26n.pt", "yolo26n")
    assert args["imgsz"] == 640 and "lr0" not in args


def test_huan_luyen_them_xuat_phat_tu_best_pt_va_khong_ghi_de():
    cfg = get_detection_cfg()
    for run, extra in cfg["finetune"]["runs"].items():
        weights, name, args = build_train_args(cfg, run)
        assert weights.replace("\\", "/").endswith("models/yolo26n/weights/best.pt")
        assert name == finetune_run_name(run) != "yolo26n"   # thư mục riêng, không ghi đè mô hình cũ
        assert args["imgsz"] == extra["imgsz"] and args["imgsz"] % 32 == 0
        assert args["optimizer"] != "auto"                   # auto sẽ bỏ qua lr0
        assert args["lr0"] < 0.01 and args["patience"] <= args["epochs"]


def test_duong_dan_models_tu_config():
    assert load_config().path("models_dir").name == "models"
