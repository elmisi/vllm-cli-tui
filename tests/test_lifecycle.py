"""Tests for the serve-plan layer: endpoint parsing, config reading,
nvidia-smi parsing, port probing, and the plan builder."""
from __future__ import annotations

import json
import socket

from vllm_tui.lifecycle import (
    Check,
    build_serve_plan,
    parse_endpoint,
    parse_free_vram,
    port_free,
    read_max_model_len,
)
from vllm_tui.local_models import LocalModel


def _model(**kw) -> LocalModel:
    base = dict(name="m", size_bytes=0, source="hf-cache",
                serveable=True, detail="LlamaForCausalLM")
    base.update(kw)
    return LocalModel(**base)


# --- parse_endpoint ---------------------------------------------------------

def test_parse_endpoint_full():
    assert parse_endpoint("http://localhost:8000") == ("localhost", 8000)


def test_parse_endpoint_https_custom_port():
    assert parse_endpoint("https://10.0.0.5:9001") == ("10.0.0.5", 9001)


def test_parse_endpoint_no_scheme():
    assert parse_endpoint("10.0.0.5:9001") == ("10.0.0.5", 9001)


def test_parse_endpoint_bare_host():
    assert parse_endpoint("10.0.0.5") == ("10.0.0.5", 8000)


def test_parse_endpoint_trailing_slash():
    assert parse_endpoint("http://localhost:8000/") == ("localhost", 8000)


# --- read_max_model_len -----------------------------------------------------

def test_read_max_model_len(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"max_position_embeddings": 40960}), encoding="utf-8")
    assert read_max_model_len(tmp_path) == 40960


def test_read_max_model_len_missing(tmp_path):
    assert read_max_model_len(tmp_path) is None


def test_read_max_model_len_corrupt(tmp_path):
    (tmp_path / "config.json").write_text("{not json", encoding="utf-8")
    assert read_max_model_len(tmp_path) is None


def test_read_max_model_len_no_field(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    assert read_max_model_len(tmp_path) is None


# --- parse_free_vram --------------------------------------------------------

def test_parse_free_vram_single():
    assert parse_free_vram("12345\n") == 12345


def test_parse_free_vram_sums_gpus():
    assert parse_free_vram("8192\n16384\n") == 24576


def test_parse_free_vram_empty():
    assert parse_free_vram("") is None


# --- port_free --------------------------------------------------------------

def test_port_free_reflects_binding():
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("", 0))
    port = probe.getsockname()[1]
    try:
        assert port_free(port) is False
    finally:
        probe.close()
    assert port_free(port) is True


# --- build_serve_plan -------------------------------------------------------

def test_command_repo_id_and_flag():
    plan = build_serve_plan(_model(name="org/m"), "http://localhost:8000",
                            vllm_binary="/usr/local/bin/vllm",
                            port_is_free=True, vram_free_mb=24000,
                            max_model_len=32768)
    assert plan.model_arg == "org/m"
    assert plan.command == "/usr/local/bin/vllm serve org/m --port 8000 --max-model-len 32768"
    assert plan.blocking is None


def test_extra_dir_serves_by_path():
    plan = build_serve_plan(
        _model(name="m", source="/data/models", path="/data/models/m"),
        "http://localhost:8000", vllm_binary="vllm", port_is_free=True)
    assert plan.model_arg == "/data/models/m"


def test_fallback_binary_warns():
    plan = build_serve_plan(_model(), "http://localhost:8000",
                            vllm_binary=None, port_is_free=True)
    assert plan.command.startswith("python -m vllm serve")
    assert any(c.level == "warn" and "python -m vllm" in c.text for c in plan.checks)


def test_occupied_port_warns():
    plan = build_serve_plan(_model(), "http://localhost:8000",
                            vllm_binary="vllm", port_is_free=False)
    assert any(c.level == "warn" and "8000" in c.text for c in plan.checks)


def test_unchecked_port_is_info():
    plan = build_serve_plan(_model(), "http://localhost:8000", vllm_binary="vllm")
    assert any(c.level == "info" and "port" in c.text for c in plan.checks)


def test_nonserveable_blocks():
    plan = build_serve_plan(_model(serveable=False, detail="GGUF"),
                            "http://localhost:8000", vllm_binary="vllm",
                            port_is_free=True)
    assert plan.blocking is not None
    assert plan.blocking.level == "fail"
    assert "GGUF" in plan.blocking.text


def test_vram_tight_warns():
    plan = build_serve_plan(_model(size_bytes=20_000_000_000),
                            "http://localhost:8000", vllm_binary="vllm",
                            port_is_free=True, vram_free_mb=16_000)
    assert any(c.level == "warn" and "VRAM" in c.text for c in plan.checks)


def test_vram_ok_is_quiet():
    plan = build_serve_plan(_model(size_bytes=2_000_000_000),
                            "http://localhost:8000", vllm_binary="vllm",
                            port_is_free=True, vram_free_mb=16_000)
    assert not any(c.level == "warn" and "VRAM" in c.text for c in plan.checks)


def test_marks():
    assert Check("ok", "x").mark == "✓"
    assert Check("warn", "x").mark == "⚠"
    assert Check("fail", "x").mark == "✗"
    assert Check("info", "x").mark == "•"