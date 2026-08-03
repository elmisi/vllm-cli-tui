"""HuggingFace Hub REST client: search and model detail.

The Hub has a real JSON API, so unlike ollama-cli-tui's registry scraper this
is plain urllib over documented endpoints. Parsing is separated from fetching
so the parsers are unit-testable offline.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

HF_API = "https://huggingface.co/api"

_QUANT_PATTERNS: tuple[tuple[str, str], ...] = (
    # Order matters: GPTQ repos often mention int4 too, and GGUF wins over all
    # because vLLM's GGUF support is limited enough to deserve the flag.
    ("GGUF", r"gguf"),
    ("AWQ", r"awq"),
    ("GPTQ", r"gptq"),
    ("FP8", r"fp8"),
    ("INT4", r"int4|4bit|4-bit|w4a16"),
    ("INT8", r"int8|8bit|8-bit|w8a"),
)


@dataclass(frozen=True)
class ModelHit:
    id: str
    downloads: int = 0
    likes: int = 0
    gated: bool = False
    quantization: str = ""
    architecture: str = ""
    library: str = ""


@dataclass(frozen=True)
class ModelFile:
    name: str
    size: int = 0


@dataclass(frozen=True)
class ModelDetail:
    id: str
    gated: bool = False
    files: tuple[ModelFile, ...] = ()
    total_bytes: int = 0


# Architectures known to vLLM's model registry. A snapshot, deliberately:
# the live list changes with every vLLM release and cannot be queried from a
# server. An architecture missing here shows as "?", never as "no" — the
# certain noes come from the repo's format, not from this list.
SUPPORTED_ARCHITECTURES: frozenset[str] = frozenset({
    # text generation
    "ArcticForCausalLM", "BaichuanForCausalLM", "BloomForCausalLM",
    "ChatGLMModel", "ChatGLMForConditionalGeneration",
    "CohereForCausalLM", "Cohere2ForCausalLM", "DbrxForCausalLM",
    "DeciLMForCausalLM", "DeepseekForCausalLM", "DeepseekV2ForCausalLM",
    "DeepseekV3ForCausalLM", "ExaoneForCausalLM", "FalconForCausalLM",
    "GemmaForCausalLM", "Gemma2ForCausalLM", "Gemma3ForCausalLM",
    "GlmForCausalLM", "Glm4ForCausalLM", "GPT2LMHeadModel",
    "GPTBigCodeForCausalLM", "GPTJForCausalLM", "GPTNeoXForCausalLM",
    "GraniteForCausalLM", "GraniteMoeForCausalLM", "InternLM2ForCausalLM",
    "JambaForCausalLM", "LlamaForCausalLM", "MambaForCausalLM",
    "Mamba2ForCausalLM", "MiniCPMForCausalLM", "MiniCPM3ForCausalLM",
    "MistralForCausalLM", "MixtralForCausalLM", "MPTForCausalLM",
    "NemotronForCausalLM", "OlmoForCausalLM", "Olmo2ForCausalLM",
    "OlmoeForCausalLM", "OPTForCausalLM", "PersimmonForCausalLM",
    "PhiForCausalLM", "Phi3ForCausalLM", "PhiMoEForCausalLM",
    "Qwen2ForCausalLM", "Qwen2MoeForCausalLM", "Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM", "Qwen3NextForCausalLM", "SolarForCausalLM",
    "StableLmForCausalLM", "Starcoder2ForCausalLM", "XverseForCausalLM",
    # multimodal generation
    "Gemma3ForConditionalGeneration", "Idefics3ForConditionalGeneration",
    "InternVLChatModel", "LlavaForConditionalGeneration",
    "LlavaNextForConditionalGeneration", "MiniCPMV",
    "Mistral3ForConditionalGeneration", "MllamaForConditionalGeneration",
    "PaliGemmaForConditionalGeneration", "Phi3VForCausalLM",
    "PixtralForConditionalGeneration", "Qwen2VLForConditionalGeneration",
    "Qwen2_5_VLForConditionalGeneration", "Qwen3VLForConditionalGeneration",
    # seen serving live on vLLM 0.21
    "Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM",
})


def hit_verdict(hit: ModelHit) -> str:
    """One cell of truth per search row: ✓ known, ? plausible, no (why)."""
    if hit.architecture:
        marker = "✓" if hit.architecture in SUPPORTED_ARCHITECTURES else "?"
        return f"{marker} {hit.architecture}"
    if hit.library == "peft":
        return "no (adapter)"
    if hit.quantization == "GGUF":
        return "no (GGUF)"
    return "no (no config)"


def infer_quantization(model_id: str, tags: list[str]) -> str:
    hay = (model_id + " " + " ".join(tags)).lower()
    for label, pattern in _QUANT_PATTERNS:
        if re.search(pattern, hay):
            return label
    return ""


def parse_search_results(payload: Any) -> list[ModelHit]:
    hits: list[ModelHit] = []
    if not isinstance(payload, list):
        return hits
    for entry in payload:
        if not isinstance(entry, dict) or not entry.get("id"):
            continue
        tags = [t for t in entry.get("tags") or [] if isinstance(t, str)]
        config = entry.get("config") or {}
        architectures = config.get("architectures") or [] if isinstance(config, dict) else []
        hits.append(
            ModelHit(
                id=str(entry["id"]),
                downloads=int(entry.get("downloads") or 0),
                likes=int(entry.get("likes") or 0),
                # the API returns False, True or "auto"; anything not False is gated
                gated=bool(entry.get("gated")),
                quantization=infer_quantization(str(entry["id"]), tags),
                architecture=str(architectures[0]) if architectures else "",
                library=str(entry.get("library_name") or ""),
            )
        )
    return hits


def parse_model_detail(payload: Any) -> ModelDetail:
    if not isinstance(payload, dict):
        return ModelDetail(id="")
    files: list[ModelFile] = []
    total = 0
    for sibling in payload.get("siblings") or []:
        if not isinstance(sibling, dict) or not sibling.get("rfilename"):
            continue
        size = sibling.get("size")
        if not isinstance(size, int):
            lfs = sibling.get("lfs")
            size = lfs.get("size") if isinstance(lfs, dict) else 0
        size = size if isinstance(size, int) else 0
        files.append(ModelFile(name=str(sibling["rfilename"]), size=size))
        total += size
    return ModelDetail(
        id=str(payload.get("id") or ""),
        gated=bool(payload.get("gated")),
        files=tuple(files),
        total_bytes=total,
    )


def detail_flags(files: tuple[ModelFile, ...]) -> str:
    """Certain noes, visible before any bytes move.

    The Hub cannot tell us whether the installed vLLM release supports an
    architecture, but a repo whose files are GGUF-only, adapter-only or
    config-less is unloadable for sure — worth saying next to the price tag.
    """
    names = [f.name for f in files]
    full = [n for n in names
            if not n.startswith("adapter")
            and (n.endswith(".safetensors") or (n.endswith(".bin") and "model" in n))]
    gguf = [n for n in names if n.endswith(".gguf")]
    flags: list[str] = []
    if gguf and not full:
        flags.append("GGUF-only: vLLM's GGUF support is limited")
    elif "adapter_config.json" in names and not full:
        flags.append("adapter-only: needs its base model, not serveable alone")
    elif "config.json" not in names:
        flags.append("no config.json: not a transformers repo")
    return " • ".join(flags)


def fetch_architectures(model_id: str, *, timeout_s: float = 10.0) -> str:
    """The architectures field of the repo's config.json; "" when unknowable."""
    url = f"https://huggingface.co/{quote(model_id, safe='/')}/raw/main/config.json"
    try:
        request = Request(url, headers={"User-Agent": "vllm-cli-tui"})
        with urlopen(request, timeout=timeout_s) as resp:
            config = json.loads(resp.read().decode("utf-8", errors="replace"))
        architectures = config.get("architectures") or []
        return str(architectures[0]) if architectures else ""
    except Exception:
        return ""


def format_size(size: int) -> str:
    """Decimal units, like the Hub itself displays them."""
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1000 or unit == "TB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1000
    return f"{value:.1f} TB"


def _get_json(url: str, *, timeout_s: float = 10.0) -> Any:
    request = Request(url, headers={"User-Agent": "vllm-cli-tui"})
    with urlopen(request, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def search_models(query: str, *, limit: int = 50, timeout_s: float = 10.0) -> list[ModelHit]:
    params = urlencode({
        "search": query,
        "pipeline_tag": "text-generation",
        "limit": str(limit),
    })
    # expand[] delivers config.architectures for every row in this same
    # request — the per-row verdict costs zero extra round trips. The API
    # rejects expand combined with sort, so results are re-sorted client-side.
    expand = "&".join(f"expand[]={field}" for field in
                      ("config", "library_name", "tags", "downloads", "likes", "gated"))
    hits = parse_search_results(
        _get_json(f"{HF_API}/models?{params}&{expand}", timeout_s=timeout_s)
    )
    hits.sort(key=lambda h: -h.downloads)
    return hits


def fetch_model_detail(model_id: str, *, timeout_s: float = 10.0) -> ModelDetail:
    url = f"{HF_API}/models/{quote(model_id, safe='/')}?blobs=true"
    return parse_model_detail(_get_json(url, timeout_s=timeout_s))
