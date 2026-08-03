"""Model downloads into the HF cache, with observable progress.

`snapshot_download` is used for what it is good at — resume, auth, integrity —
and progress is read from the outside by measuring bytes on disk against the
expected total from the API. That keeps the download loop entirely the
library's and the progress entirely ours.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DownloadState:
    repo_id: str
    expected_bytes: int
    done: bool = False
    error: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def finish(self, error: str = "") -> None:
        with self._lock:
            self.done = True
            self.error = error

    def snapshot(self) -> tuple[bool, str]:
        with self._lock:
            return self.done, self.error


def bytes_on_disk_for(repo_id: str) -> int:
    """Current cache footprint of a repo; 0 when not there yet."""
    try:
        from huggingface_hub import scan_cache_dir

        for repo in scan_cache_dir().repos:
            if repo.repo_id == repo_id and repo.repo_type == "model":
                return repo.size_on_disk
    except Exception:
        pass
    return 0


def start_download(repo_id: str, expected_bytes: int) -> DownloadState:
    """Kick off a snapshot download in a daemon thread and return its state."""
    state = DownloadState(repo_id=repo_id, expected_bytes=expected_bytes)

    def work() -> None:
        try:
            from huggingface_hub import snapshot_download
            from huggingface_hub.utils import disable_progress_bars

            # tqdm writes to stderr, which inside a Textual app scribbles all
            # over the screen; progress is ours, read from bytes on disk.
            disable_progress_bars()
            snapshot_download(repo_id)
            state.finish()
        except Exception as exc:  # noqa: BLE001 — surfaced in the UI, not raised
            state.finish(f"{type(exc).__name__}: {exc}")

    threading.Thread(target=work, name=f"download:{repo_id}", daemon=True).start()
    return state
