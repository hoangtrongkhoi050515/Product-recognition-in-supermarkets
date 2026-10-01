"""Kiểm thử các hàm xử lý dữ liệu. Chạy: python -m pytest -q"""
from collections import Counter
from pathlib import Path

import pytest

from src.catalog import load_catalog
from src.data.split_dataset import SPLITS, stratified_split
from src.data.yolo_labels import Box, pair_images_labels, read_label_file


def test_catalog_hop_le():
    cat = load_catalog()
    assert len(cat) == 12
    assert cat.by_id(0).code == "th_milk_180ml"
    assert cat.by_code("aquafina_500ml").price == 4000


def test_doc_nhan_phat_hien_loi(tmp_path: Path):
    f = tmp_path / "a.txt"
    f.write_text("0 0.5 0.5 0.2 0.2\n25 0.5 0.5 0.2 0.2\n1 1.5 0.5 0.2 0.2\n2 0.5 0.5\n", encoding="utf-8")
    boxes, issues = read_label_file(f, num_classes=12)
    assert len(boxes) == 1
    assert [i.level for i in issues] == ["ERROR"] * 3


def test_doi_toa_do_pixel():
    assert Box(0, 0.5, 0.5, 0.5, 0.5).to_xyxy(200, 100) == (50, 25, 150, 75)
    assert Box(0, 0.05, 0.5, 0.2, 0.2).to_xyxy(100, 100) == (0, 40, 15, 60)  # kẹp trong biên


def test_ghep_anh_nhan(tmp_path: Path):
    (tmp_path / "img").mkdir(); (tmp_path / "lbl").mkdir()
    (tmp_path / "img" / "a.JPG").write_bytes(b"x")
    (tmp_path / "img" / "b.jpg").write_bytes(b"x")
    (tmp_path / "lbl" / "a.txt").write_text("", encoding="utf-8")
    (tmp_path / "lbl" / "c.txt").write_text("", encoding="utf-8")
    (tmp_path / "lbl" / "classes.txt").write_text("", encoding="utf-8")
    pairs, orphans = pair_images_labels(tmp_path / "img", tmp_path / "lbl", [".jpg"])
    assert [(i.name, l.name if l else None) for i, l in pairs] == [("a.JPG", "a.txt"), ("b.jpg", None)]
    assert [o.name for o in orphans] == ["c.txt"]


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_chia_tap_phan_tang(seed):
    import random
    rng = random.Random(seed)
    items = []
    for k in range(600):  # 600 ảnh giả, 1–5 đối tượng, lớp 11 hiếm
        n = rng.randint(1, 5)
        boxes = [Box(rng.choice(range(11)) if rng.random() > 0.03 else 11, .5, .5, .1, .1) for _ in range(n)]
        items.append((Path(f"{k}.jpg"), boxes))
    ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
    parts = stratified_split(items, ratios, seed)

    # không mất / không trùng ảnh
    names = [p.name for s in SPLITS for p, _ in parts[s]]
    assert sorted(names) == sorted(p.name for p, _ in items)
    # mỗi lớp (kể cả lớp hiếm) có mặt ở cả 3 tập, tỉ lệ lệch không quá 5 điểm %
    total = Counter(b.class_id for _, bs in items for b in bs)
    for s in SPLITS:
        cnt = Counter(b.class_id for _, bs in parts[s] for b in bs)
        for c, n in total.items():
            assert cnt[c] > 0
            assert abs(cnt[c] / n - ratios[s]) < 0.05, (s, c, cnt[c] / n)
