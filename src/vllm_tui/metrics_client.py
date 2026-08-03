"""Fetch and parse vLLM's Prometheus metrics.

Pure "data in / data out": the fetch is one urllib call, the parser is a small
regex over the exposition text, and the rates come from diffing two samples.
No Textual imports here.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional
from urllib.request import urlopen

_METRIC_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*$")


@dataclass(frozen=True)
class EndpointSample:
    at: float
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class EndpointStatus:
    url: str
    reachable: bool
    model: str = ""
    max_model_len: Optional[int] = None
    sample: Optional[EndpointSample] = None
    error: str = ""


def parse_prometheus(text: str) -> dict[str, float]:
    """Metric name -> value; same-name series (different labels) are summed.

    Summing is what we want for the one metric vLLM splits by label that we
    read (request_success_total by finished_reason); gauges appear once.
    """
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = _METRIC_LINE.match(line)
        if not m:
            continue
        try:
            value = float(m.group(2))
        except ValueError:
            continue
        out[m.group(1)] = out.get(m.group(1), 0.0) + value
    return out


def derive_rates(before: EndpointSample, after: EndpointSample) -> dict[str, float]:
    """Tokens per second between two samples; a counter reset reads as zero."""
    span = after.at - before.at
    if span <= 0:
        return {}
    out: dict[str, float] = {}
    for metric, key in (
        ("vllm:generation_tokens_total", "generation_tok_s"),
        ("vllm:prompt_tokens_total", "prompt_tok_s"),
    ):
        if metric in after.metrics and metric in before.metrics:
            delta = after.metrics[metric] - before.metrics[metric]
            out[key] = max(0.0, delta / span)
    return out


def probe_endpoint(url: str, *, timeout_s: float = 3.0) -> EndpointStatus:
    """One poll of a vLLM endpoint: served model plus a metrics sample."""
    base = url.rstrip("/")
    model = ""
    max_len: Optional[int] = None
    try:
        with urlopen(f"{base}/v1/models", timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
        entries = payload.get("data") or []
        if entries and isinstance(entries[0], dict):
            model = str(entries[0].get("id") or "")
            raw_len = entries[0].get("max_model_len")
            if isinstance(raw_len, int):
                max_len = raw_len
    except Exception as exc:  # noqa: BLE001 — a down server is data, not a crash
        return EndpointStatus(url=url, reachable=False, error=type(exc).__name__)

    try:
        with urlopen(f"{base}/metrics", timeout=timeout_s) as resp:
            text = resp.read().decode("utf-8", errors="replace")
        sample = EndpointSample(at=time.monotonic(), metrics=parse_prometheus(text))
    except Exception:
        # /v1/models answered, so the server is up; metrics may be disabled.
        sample = None

    return EndpointStatus(url=url, reachable=True, model=model,
                          max_model_len=max_len, sample=sample)
