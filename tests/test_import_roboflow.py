"""Kiểm thử nhập dữ liệu Roboflow. Chạy: python -m pytest -q"""
from pathlib import Path

import pytest

from src.catalog import load_catalog
from src.data.import_roboflow import build_id_map, convert_label, parse_line, run_import

CODES_ROBOFLOW = sorted(load_catalog().codes)  # Roboflow xếp theo bảng chữ cái
IMG_EXTS = [".jpg"]


def test_id_map_theo_ten_lop():
    cat = load_catalog()
    id_map = build_id_map(CODES_ROBOFLOW, cat)
    # aquafina đứng đầu bảng chữ cái (0) nhưng là lớp 1 trong products.csv
    assert id_map[CODES_ROBOFLOW.index("aquafina_500ml")] == cat.by_code("aquafina_500ml").class_id
    assert sorted(id_map.values()) == list(range(len(cat)))


def test_ten_lop_la_bi_tu_choi():
    with pytest.raises(SystemExit):
        build_id_map(CODES_ROBOFLOW + ["ten_sai"], load_catalog())


def test_da_giac_doi_thanh_khung_nho_nhat():
    box, kind = parse_line("0 0.2 0.3 0.6 0.3 0.4 0.7", {0: 5})
    assert kind == "đa giác" and box.class_id == 5
    assert (box.xc, box.yc) == pytest.approx((0.4, 0.5))
    assert (box.w, box.h) == pytest.approx((0.4, 0.4))


def test_khung_giu_nguyen_va_doi_so_lop():
    box, kind = parse_line("1 0.5 0.5 0.2 0.4", {1: 9})
    assert kind == "khung"
    assert (box.class_id, box.xc, box.w) == (9, 0.5, 0.2)


@pytest.mark.parametrize("line", ["0 0.5 0.5", "0 a b c d", "7 0.5 0.5 0.2 0.2", "0 0.5 0.5 0 0.2"])
def test_dong_loi_bi_phat_hien(line):
    with pytest.raises(ValueError):
        parse_line(line, {0: 0})


def _tao_ban_xuat(tmp_path: Path) -> Path:
    src = tmp_path / "rf"
    names = ", ".join(f"'{c}'" for c in CODES_ROBOFLOW)
    (src).mkdir()
    (src / "data.yaml").write_text(f"nc: {len(CODES_ROBOFLOW)}\nnames: [{names}]\n", encoding="utf-8")
    for split in ("train", "valid", "test"):
        (src / split / "images").mkdir(parents=True)
        (src / split / "labels").mkdir(parents=True)
    return src


def test_nhap_gop_cac_tap_va_doi_so_lop(tmp_path: Path):
    cat = load_catalog()
    src = _tao_ban_xuat(tmp_path)
    pepsi_rf = CODES_ROBOFLOW.index("pepsi_320ml")
    for split, name in (("train", "a"), ("valid", "b"), ("test", "c")):
        (src / split / "images" / f"{name}.jpg").write_bytes(b"x")
        (src / split / "labels" / f"{name}.txt").write_text(
            f"{pepsi_rf} 0.5 0.5 0.2 0.2\n{pepsi_rf} 0.1 0.1 0.3 0.1 0.3 0.3\n", encoding="utf-8")
    (src / "train" / "images" / "khong_nhan.jpg").write_bytes(b"x")

    raw_i, raw_l = tmp_path / "images", tmp_path / "labels"
    rep = run_import(src, raw_i, raw_l, cat, IMG_EXTS)
    assert sum(rep.images.values()) == 3 and rep.no_label == ["khong_nhan.jpg"] and not rep.errors
    assert rep.kinds["khung"] == 3 and rep.kinds["đa giác"] == 3
    pepsi_id = cat.by_code("pepsi_320ml").class_id
    assert rep.boxes[pepsi_id] == 6
    assert (raw_l / "a.txt").read_text().split()[0] == str(pepsi_id)

    # chạy lại: ảnh đã có thì bỏ qua, --force thì ghi đè
    assert len(run_import(src, raw_i, raw_l, cat, IMG_EXTS).existed) == 3
    assert sum(run_import(src, raw_i, raw_l, cat, IMG_EXTS, force=True).images.values()) == 3


def test_dry_run_khong_ghi_file(tmp_path: Path):
    src = _tao_ban_xuat(tmp_path)
    (src / "train" / "images" / "a.jpg").write_bytes(b"x")
    (src / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    raw_i = tmp_path / "images"
    rep = run_import(src, raw_i, tmp_path / "labels", load_catalog(), IMG_EXTS, dry_run=True)
    assert rep.images["train"] == 1 and not raw_i.exists()


def test_convert_label_gom_loi(tmp_path: Path):
    f = tmp_path / "x.txt"
    f.write_text("0 0.5 0.5 0.2 0.2\n0 0.5 0.5\n", encoding="utf-8")
    boxes, _, errors = convert_label(f, {0: 0})
    assert len(boxes) == 1 and len(errors) == 1 and "dòng 2" in errors[0]
