"""Config: defaults, round-trip, tolerance to garbage."""
import json

from vllm_tui.config import Config, load_config, save_config


def test_defaults_are_localhost_and_no_extra_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    cfg = load_config()
    assert cfg.endpoints == ["http://localhost:8000"]
    assert cfg.extra_model_dirs == []


def test_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    save_config(Config(endpoints=["http://localhost:8000", "http://localhost:8001"],
                       extra_model_dirs=["/srv/models"]))
    cfg = load_config()
    assert len(cfg.endpoints) == 2
    assert cfg.extra_model_dirs == ["/srv/models"]


def test_garbage_files_fall_back_to_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    path = tmp_path / "vllm-tui" / "config.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    cfg = load_config()
    assert cfg.endpoints == ["http://localhost:8000"]


def test_unknown_keys_are_ignored_not_fatal(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    path = tmp_path / "vllm-tui" / "config.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"endpoints": ["http://h:1"], "future_thing": 42}))
    assert load_config().endpoints == ["http://h:1"]
