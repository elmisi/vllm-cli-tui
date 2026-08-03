"""The per-row vLLM verdict in search results.

Three states, honestly separated: a "yes" only for architectures known to the
bundled snapshot, a "?" for transformers repos whose architecture the snapshot
does not know (a newer vLLM may well run them), and a certain "no" with its
reason. The expand[] parameter of the Hub API delivers the architecture for
every row in the same single search request.
"""
from vllm_tui.hf_client import ModelHit, hit_verdict, parse_search_results

PAYLOAD = [
    {"id": "Qwen/Qwen2.5-7B-Instruct", "downloads": 1, "likes": 1,
     "library_name": "transformers", "tags": ["text-generation"],
     "config": {"architectures": ["Qwen2ForCausalLM"]}},
    {"id": "brand/new-arch-model", "downloads": 1, "likes": 1,
     "library_name": "transformers", "tags": [],
     "config": {"architectures": ["FrontierNet9ForCausalLM"]}},
    {"id": "bartowski/Some-GGUF", "downloads": 1, "likes": 1,
     "tags": ["gguf"], "config": None},
    {"id": "someone/lora-adapter", "downloads": 1, "likes": 1,
     "library_name": "peft", "tags": [], "config": None},
    {"id": "someone/mystery-repo", "downloads": 1, "likes": 1, "tags": []},
]


def _hits() -> list[ModelHit]:
    return parse_search_results(PAYLOAD)


def test_architecture_and_library_are_parsed_from_the_expanded_payload():
    hits = _hits()
    assert hits[0].architecture == "Qwen2ForCausalLM"
    assert hits[0].library == "transformers"
    assert hits[2].architecture == ""


def test_a_known_architecture_is_a_yes():
    assert hit_verdict(_hits()[0]) == "✓ Qwen2ForCausalLM"


def test_an_unknown_transformers_architecture_is_a_maybe_not_a_no():
    assert hit_verdict(_hits()[1]) == "? FrontierNet9ForCausalLM"


def test_gguf_without_config_is_a_certain_no():
    assert hit_verdict(_hits()[2]) == "no (GGUF)"


def test_a_peft_adapter_is_a_certain_no():
    assert hit_verdict(_hits()[3]) == "no (adapter)"


def test_no_config_at_all_is_a_no_with_its_reason():
    assert hit_verdict(_hits()[4]) == "no (no config)"


def test_the_arch_seen_live_on_a_real_server_is_in_the_snapshot():
    # Verified serving on vLLM 0.21: if this fails, the snapshot regressed.
    hit = ModelHit(id="x", architecture="Qwen3_5ForConditionalGeneration",
                   library="transformers")
    assert hit_verdict(hit).startswith("✓")


def test_mlx_repos_are_a_certain_no_despite_their_valid_architecture():
    """MLX repos carry a correct transformers config, but the weights are in
    Apple's MLX format — found in the field: they showed as ✓."""
    hit = ModelHit(id="mlx-community/Qwen3.5-9B-4bit", library="mlx",
                   architecture="Qwen3_5ForConditionalGeneration")
    assert hit_verdict(hit) == "no (MLX)"
