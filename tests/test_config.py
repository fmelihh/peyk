import json

from peyk.cli import _build_parser
from peyk.config import load_config


def _write(tmp_path, monkeypatch, data):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    (tmp_path / ".peyk.json").write_text(json.dumps(data))


def test_load_config_local(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, {"languages": "tr,en", "use_case": "coding", "bogus": 1})
    cfg = load_config()
    assert cfg["languages"] == "tr,en"
    assert cfg["use_case"] == "coding"
    assert "bogus" not in cfg  # unknown keys filtered


def test_config_missing_returns_empty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
    assert load_config() == {}


def test_config_supplies_recommend_defaults():
    parser = _build_parser({"languages": "tr,en", "context": 32768, "top": 8,
                            "use_case": "coding"})
    args = parser.parse_args(["recommend"])
    assert args.languages == "tr,en"
    assert args.context == 32768
    assert args.top == 8
    assert args.use_case == "coding"


def test_cli_flags_override_config():
    parser = _build_parser({"languages": "tr,en", "context": 32768})
    args = parser.parse_args(["recommend", "--languages", "en", "--context", "4096"])
    assert args.languages == "en"
    assert args.context == 4096


def test_invalid_use_case_in_config_ignored():
    parser = _build_parser({"use_case": "not-real"})
    args = parser.parse_args(["recommend"])
    assert args.use_case is None
