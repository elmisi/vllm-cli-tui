"""What is on disk: the HuggingFace cache plus any configured extra dirs.

The cache is read through huggingface_hub's own scanner, so repo ids and
revisions are always interpreted the way the library itself would. Extra dirs
are plain filesystem work: one subdirectory = one model.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Serveability:
    serveable: bool
    detail: str  # the architecture when serveable, the reason when not


@dataclass(frozen=True)
class LocalModel:
    name: str
    size_bytes: int
    source: str          # "hf-cache" or the extra dir it came from
    path: str = ""       # filesystem path when directly deletable
    last_used: str = ""
    serveable: bool = False
    detail: str = ""

    @property
    def deletable(self) -> bool:
        return bool(self.path)


def probe_dir_serveability(path: Path) -> Serveability:
    """Can vLLM load what is in this directory? Decided from the files.

    "Serveable" here means "in a format vLLM loads": a transformers repo with
    a config naming its architecture and full weights. Whether the specific
    architecture is supported by the installed vLLM release cannot be known
    offline — but diffusers trees, GGUF-only and adapter-only repos are
    certain noes, and that is what this filter is for.
    """
    try:
        names = [p.name for p in path.iterdir()]
    except OSError:
        return Serveability(False, "unreadable")

    if "model_index.json" in names:
        return Serveability(False, "diffusers")

    def is_full_weight(name: str) -> bool:
        if name.startswith("adapter"):
            return False
        return name.endswith(".safetensors") or (name.endswith(".bin") and "model" in name)

    has_full = any(is_full_weight(n) for n in names)
    has_gguf = any(n.endswith(".gguf") for n in names)

    if "adapter_config.json" in names and not has_full:
        return Serveability(False, "adapter-only")
    if "config.json" not in names:
        return Serveability(False, "GGUF" if has_gguf and not has_full else "no config.json")

    try:
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        architectures = config.get("architectures") or []
    except Exception:
        architectures = []
    if not architectures:
        return Serveability(False, "no architectures")
    if not has_full:
        return Serveability(False, "no weights")
    return Serveability(True, str(architectures[0]))


def scan_extra_dirs(dirs: list[str]) -> list[LocalModel]:
    models: list[LocalModel] = []
    for raw in dirs:
        root = Path(raw).expanduser()
        try:
            children = sorted(p for p in root.iterdir() if p.is_dir())
        except OSError:
            continue
        for child in children:
            size = 0
            try:
                for f in child.rglob("*"):
                    if f.is_file():
                        size += f.stat().st_size
            except OSError:
                pass
            probe = probe_dir_serveability(child)
            models.append(LocalModel(name=child.name, size_bytes=size,
                                     source=str(root), path=str(child),
                                     serveable=probe.serveable, detail=probe.detail))
    models.sort(key=lambda m: -m.size_bytes)
    return models


def scan_hf_cache() -> list[LocalModel]:
    """Models in the standard HF cache, via the library's own scanner."""
    try:
        from huggingface_hub import scan_cache_dir

        info = scan_cache_dir()
    except Exception:
        return []
    models = []
    for repo in info.repos:
        if repo.repo_type != "model":
            continue
        probe = Serveability(False, "no snapshot")
        revisions = sorted(repo.revisions, key=lambda r: r.last_modified or 0)
        if revisions:
            probe = probe_dir_serveability(Path(revisions[-1].snapshot_path))
        models.append(
            LocalModel(
                name=repo.repo_id,
                size_bytes=repo.size_on_disk,
                source="hf-cache",
                last_used=repo.last_accessed_str or "",
                serveable=probe.serveable,
                detail=probe.detail,
            )
        )
    models.sort(key=lambda m: -m.size_bytes)
    return models


def delete_local_model(model: LocalModel) -> str | None:
    """Delete a model from disk; returns an error string or None.

    Cache entries go through the library's deletion strategy so refs stay
    consistent; extra-dir entries are plain directory trees.
    """
    if model.source == "hf-cache":
        try:
            from huggingface_hub import scan_cache_dir

            info = scan_cache_dir()
            revisions = [
                rev.commit_hash
                for repo in info.repos
                if repo.repo_id == model.name and repo.repo_type == "model"
                for rev in repo.revisions
            ]
            if not revisions:
                return "not found in cache"
            info.delete_revisions(*revisions).execute()
            return None
        except Exception as exc:  # noqa: BLE001
            return type(exc).__name__
    if not model.path:
        return "not deletable"
    try:
        shutil.rmtree(model.path)
        return None
    except OSError as exc:
        return type(exc).__name__
