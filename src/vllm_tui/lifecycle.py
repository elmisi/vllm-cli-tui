"""Serve planning: what `vllm serve` would be, checked before it runs.

Step 1 of the lifecycle work: read-only. This module answers "what is the
exact command to serve this local model, and what would go wrong if I ran
it?" — the plan is displayed, never executed. Process control (systemd /
direct spawn, selected by the `lifecycle` config, default "off") is a later
step and does not live here yet.

Same pure-layer rules as the rest: no Textual imports, IO split from logic
so the logic is unit-testable.
"""
from __future__ import annotations

import json
import re
import shutil
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from .hf_client import format_size
from .local_models import LocalModel


@dataclass(frozen=True)
class Check:
    """One line of the serve plan.

    ok   — verified and fine
    warn — it would start, but something is likely wrong
    fail — it would not start (or would not be usable)
    info — could not be verified from here
    """

    level: str  # "ok" | "warn" | "fail" | "info"
    text: str

    @property
    def mark(self) -> str:
        return {"ok": "✓", "warn": "⚠", "fail": "✗", "info": "•"}.get(self.level, "•")


@dataclass(frozen=True)
class ServePlan:
    model_name: str
    model_arg: str      # what goes on the command line: repo id or absolute path
    command: str        # the full command
    endpoint: str
    port: int
    checks: tuple[Check, ...] = ()

    @property
    def blocking(self) -> Check | None:
        """The first ✗, if any."""
        for check in self.checks:
            if check.level == "fail":
                return check
        return None


def parse_endpoint(url: str) -> tuple[str, int]:
    """(host, port) of a configured endpoint; a missing port defaults to 8000."""
    text = url.strip()
    if "://" not in text:
        text = "http://" + text
    parts = urlsplit(text)
    host = parts.hostname or "localhost"
    port = parts.port or 8000
    return host, port


def read_max_model_len(path: str | Path | None) -> int | None:
    """max_position_embeddings from a local config.json; None when unknowable."""
    if not path:
        return None
    try:
        config = json.loads(Path(path).joinpath("config.json").read_text(encoding="utf-8"))
        value = config.get("max_position_embeddings")
        return int(value) if isinstance(value, int) else None
    except Exception:
        return None


def parse_free_vram(output: str) -> int | None:
    """Total free GPU memory in MB from nvidia-smi csv output; None when no GPU."""
    total = 0
    seen = False
    for line in output.splitlines():
        m = re.fullmatch(r"\s*(\d+)\s*", line)
        if m:
            total += int(m.group(1))
            seen = True
    return total if seen else None


def fetch_free_vram(timeout_s: float = 3.0) -> int | None:
    """Free VRAM across all visible GPUs in MB; None without nvidia-smi or GPUs."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout_s)
        if proc.returncode != 0:
            return None
        return parse_free_vram(proc.stdout)
    except (OSError, subprocess.SubprocessError):
        return None


def port_free(port: int) -> bool:
    """True when the port can be bound (nothing is listening on it)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("", port))
            return True
        except OSError:
            return False


def _model_arg(model: LocalModel) -> str:
    """hf-cache entries serve by repo id (vllm looks in the cache itself);
    extra-dir entries serve by absolute path."""
    if model.source == "hf-cache":
        return model.name
    return model.path or model.name


def build_serve_plan(
    model: LocalModel,
    endpoint: str,
    *,
    vllm_binary: str | None = None,
    port_is_free: bool | None = None,
    vram_free_mb: int | None = None,
    max_model_len: int | None = None,
) -> ServePlan:
    """Assemble the plan. All side-effect inputs are passed in, so this
    function is deterministic and unit-testable."""
    _host, port = parse_endpoint(endpoint)
    binary = vllm_binary or "python -m vllm"
    arg = _model_arg(model)
    command = f"{binary} serve {arg} --port {port}"
    if max_model_len:
        command += f" --max-model-len {max_model_len}"

    checks: list[Check] = []

    if model.serveable:
        checks.append(Check("ok", f"serveable — {model.detail}"))
    else:
        checks.append(Check("fail", f"not serveable ({model.detail})"))

    where = "in the HF cache" if model.source == "hf-cache" else f"from {model.source}"
    checks.append(Check("ok", f"on disk, {format_size(model.size_bytes)} {where}"))

    if vllm_binary:
        checks.append(Check("ok", f"vllm found at {vllm_binary}"))
    else:
        checks.append(Check(
            "warn",
            "vllm not in PATH — command falls back to `python -m vllm` "
            "(needs vllm in the active environment)",
        ))

    if port_is_free is True:
        checks.append(Check("ok", f"port {port} is free"))
    elif port_is_free is False:
        checks.append(Check("warn",
                            f"port {port} is already in use — a server may be running there"))
    else:
        checks.append(Check("info", f"port {port} not checked"))

    if vram_free_mb is None:
        checks.append(Check("info", "VRAM unknown (nvidia-smi unavailable)"))
    elif model.size_bytes > 0:
        # Weights on disk plus headroom for the KV cache and activations.
        estimated_mb = model.size_bytes / 1e6 * 1.25
        if estimated_mb > vram_free_mb:
            checks.append(Check(
                "warn",
                f"VRAM likely tight: model needs ~{format_size(int(estimated_mb * 1e6))} "
                f"with overhead, {vram_free_mb / 1e6:.1f} GB free",
            ))
        else:
            checks.append(Check("ok", f"VRAM OK: {vram_free_mb / 1e6:.1f} GB free"))

    return ServePlan(model_name=model.name, model_arg=arg, command=command,
                     endpoint=endpoint, port=port, checks=tuple(checks))


def plan_for_model(model: LocalModel, endpoint: str) -> ServePlan:
    """build_serve_plan with the real world attached: PATH, port, GPUs, config."""
    max_len = None
    if model.source != "hf-cache" and model.path:
        max_len = read_max_model_len(model.path)
    _host, port = parse_endpoint(endpoint)
    return build_serve_plan(
        model,
        endpoint,
        vllm_binary=shutil.which("vllm"),
        port_is_free=port_free(port),
        vram_free_mb=fetch_free_vram(),
        max_model_len=max_len,
    )