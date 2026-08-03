"""Which local models vLLM can actually load — decided from the files on disk."""
import json

from vllm_tui.local_models import probe_dir_serveability


def _write(path, name, content=b"x"):
    (path / name).write_bytes(content)


def _config(path, architectures):
    (path / "config.json").write_text(json.dumps({"architectures": architectures}))


def test_a_transformers_repo_with_weights_is_serveable(tmp_path):
    _config(tmp_path, ["Qwen2ForCausalLM"])
    _write(tmp_path, "model-00001-of-00002.safetensors")
    result = probe_dir_serveability(tmp_path)
    assert result.serveable is True
    assert result.detail == "Qwen2ForCausalLM"


def test_a_diffusers_repo_is_not(tmp_path):
    _write(tmp_path, "model_index.json")
    _write(tmp_path, "unet.safetensors")
    result = probe_dir_serveability(tmp_path)
    assert result.serveable is False
    assert result.detail == "diffusers"


def test_a_gguf_only_repo_is_not(tmp_path):
    _write(tmp_path, "model-Q4_K_M.gguf")
    result = probe_dir_serveability(tmp_path)
    assert result.serveable is False
    assert result.detail == "GGUF"


def test_an_adapter_only_repo_is_not(tmp_path):
    _write(tmp_path, "adapter_config.json")
    _write(tmp_path, "adapter_model.safetensors")
    result = probe_dir_serveability(tmp_path)
    assert result.serveable is False
    assert result.detail == "adapter-only"


def test_config_without_weights_is_not(tmp_path):
    _config(tmp_path, ["LlamaForCausalLM"])
    result = probe_dir_serveability(tmp_path)
    assert result.serveable is False
    assert result.detail == "no weights"


def test_weights_without_config_are_not(tmp_path):
    _write(tmp_path, "model.safetensors")
    result = probe_dir_serveability(tmp_path)
    assert result.serveable is False
    assert result.detail == "no config.json"


def test_a_broken_config_does_not_crash_the_probe(tmp_path):
    _write(tmp_path, "config.json", b"{not json")
    _write(tmp_path, "model.safetensors")
    result = probe_dir_serveability(tmp_path)
    assert result.serveable is False
    assert result.detail == "no architectures"


def test_adapter_files_do_not_count_as_full_weights(tmp_path):
    _config(tmp_path, ["MistralForCausalLM"])
    _write(tmp_path, "adapter_model.safetensors")
    result = probe_dir_serveability(tmp_path)
    assert result.serveable is False
    assert result.detail == "no weights"


def test_a_missing_directory_is_just_unknown(tmp_path):
    result = probe_dir_serveability(tmp_path / "nope")
    assert result.serveable is False
    assert result.detail == "unreadable"
