"""Configuration: a small JSON file, tolerant of anything it finds."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_ENDPOINT = "http://localhost:8000"

# Planned modes for process control (start/stop of vllm servers). Step 1
# (the read-only serve plan) does not use it yet; "off" stays the default
# until the later steps land, so the tool keeps its observe-only behaviour.
LIFECYCLE_MODES = ("off", "systemd", "direct")


@dataclass(frozen=True)
class Config:
    endpoints: list[str] = field(default_factory=lambda: [DEFAULT_ENDPOINT])
    extra_model_dirs: list[str] = field(default_factory=list)
    lifecycle: str = "off"


def _config_path() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "vllm-tui" / "config.json"


def load_config() -> Config:
    try:
        data = json.loads(_config_path().read_text(encoding="utf-8"))
    except Exception:
        return Config()
    if not isinstance(data, dict):
        return Config()
    endpoints = [e for e in data.get("endpoints") or [] if isinstance(e, str) and e.strip()]
    dirs = [d for d in data.get("extra_model_dirs") or [] if isinstance(d, str) and d.strip()]
    lifecycle = data.get("lifecycle")
    if not isinstance(lifecycle, str) or lifecycle not in LIFECYCLE_MODES:
        lifecycle = "off"
    return Config(
        endpoints=endpoints or [DEFAULT_ENDPOINT],
        extra_model_dirs=dirs,
        lifecycle=lifecycle,
    )


def save_config(config: Config) -> None:
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"endpoints": config.endpoints, "extra_model_dirs": config.extra_model_dirs,
               "lifecycle": config.lifecycle}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
