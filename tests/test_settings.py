from src.settings import load_settings, save_settings


def test_defaults_when_missing(tmp_path):
    s = load_settings(tmp_path / "none.json", {"conf": 0.45})
    assert s == {"conf": 0.45, "show_faded": False}


def test_roundtrip_and_clamp(tmp_path):
    p = tmp_path / "s.json"
    save_settings(p, {"conf": 0.6, "show_faded": True})
    assert load_settings(p) == {"conf": 0.6, "show_faded": True}
    save_settings(p, {"conf": 3, "junk": 1})
    assert load_settings(p)["conf"] == 0.95


def test_corrupt_file_falls_back(tmp_path):
    p = tmp_path / "s.json"
    p.write_text("{oops", encoding="utf-8")
    assert load_settings(p, {"conf": 0.3})["conf"] == 0.3
