"""Red flags a search result can show before any bytes move."""
from vllm_tui.hf_client import ModelFile, detail_flags


def _files(*names):
    return tuple(ModelFile(name=n, size=1) for n in names)


def test_a_normal_repo_has_no_flags():
    files = _files("config.json", "model-00001.safetensors", "tokenizer.json")
    assert detail_flags(files) == ""


def test_gguf_only_is_flagged():
    files = _files("README.md", "model-Q4_K_M.gguf")
    assert "GGUF-only" in detail_flags(files)


def test_gguf_next_to_full_weights_is_not_flagged():
    files = _files("config.json", "model.safetensors", "model-Q4.gguf")
    assert detail_flags(files) == ""


def test_adapter_only_is_flagged():
    files = _files("adapter_config.json", "adapter_model.safetensors", "README.md")
    assert "adapter-only" in detail_flags(files)


def test_missing_config_is_flagged():
    files = _files("model.safetensors")
    assert "no config.json" in detail_flags(files)
