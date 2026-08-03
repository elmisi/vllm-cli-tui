"""HF Hub JSON parsing and quantization inference — no network."""
from vllm_tui.hf_client import (
    ModelHit,
    format_size,
    infer_quantization,
    parse_model_detail,
    parse_search_results,
)

SEARCH_PAYLOAD = [
    {"id": "Qwen/Qwen2.5-7B-Instruct-AWQ", "downloads": 12345, "likes": 67,
     "tags": ["awq", "text-generation"], "gated": False},
    {"id": "meta-llama/Llama-3.1-8B-Instruct", "downloads": 999999, "likes": 4321,
     "tags": ["text-generation"], "gated": "auto"},
    {"id": "someone/broken-entry"},
]

DETAIL_PAYLOAD = {
    "id": "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "gated": False,
    "siblings": [
        {"rfilename": "model-00001-of-00002.safetensors", "size": 4_000_000_000},
        {"rfilename": "model-00002-of-00002.safetensors", "lfs": {"size": 1_500_000_000}},
        {"rfilename": "config.json", "size": 1234},
        {"rfilename": "tokenizer.json"},
    ],
}


def test_search_results_survive_missing_fields():
    hits = parse_search_results(SEARCH_PAYLOAD)
    assert len(hits) == 3
    assert hits[0] == ModelHit(id="Qwen/Qwen2.5-7B-Instruct-AWQ", downloads=12345,
                               likes=67, gated=False, quantization="AWQ")
    assert hits[2].downloads == 0 and hits[2].quantization == ""


def test_gated_is_truthy_for_the_auto_marker():
    hits = parse_search_results(SEARCH_PAYLOAD)
    assert hits[1].gated is True


def test_quantization_is_inferred_from_id_and_tags():
    assert infer_quantization("Org/Model-AWQ", []) == "AWQ"
    assert infer_quantization("Org/model-gptq-int4", []) == "GPTQ"
    assert infer_quantization("Org/Model", ["fp8"]) == "FP8"
    assert infer_quantization("Org/Model-4bit", []) == "INT4"
    assert infer_quantization("Org/Model-GGUF", []) == "GGUF"
    assert infer_quantization("Org/Plain-Model", ["text-generation"]) == ""


def test_detail_sums_sizes_from_both_size_shapes():
    detail = parse_model_detail(DETAIL_PAYLOAD)
    # size can live at the top level or inside lfs; absent sizes count zero
    assert detail.total_bytes == 4_000_000_000 + 1_500_000_000 + 1234
    assert len(detail.files) == 4
    assert detail.files[0].name == "model-00001-of-00002.safetensors"


def test_sizes_format_for_humans():
    assert format_size(0) == "0 B"
    assert format_size(1234) == "1.2 KB"
    assert format_size(5_500_000_000) == "5.5 GB"
