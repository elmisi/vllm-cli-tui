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
        hits.append(
            ModelHit(
                id=str(entry["id"]),
                downloads=int(entry.get("downloads") or 0),
                likes=int(entry.get("likes") or 0),
                # the API returns False, True or "auto"; anything not False is gated
                gated=bool(entry.get("gated")),
                quantization=infer_quantization(str(entry["id"]), tags),
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
        "sort": "downloads",
        "direction": "-1",
        "limit": str(limit),
    })
    return parse_search_results(_get_json(f"{HF_API}/models?{params}", timeout_s=timeout_s))


def fetch_model_detail(model_id: str, *, timeout_s: float = 10.0) -> ModelDetail:
    url = f"{HF_API}/models/{quote(model_id, safe='/')}?blobs=true"
    return parse_model_detail(_get_json(url, timeout_s=timeout_s))
