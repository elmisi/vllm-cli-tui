"""Download progress: the library downloads, we watch the disk grow."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import ProgressBar, Static

from ..downloads import DownloadState, bytes_on_disk_for, start_download
from ..hf_client import format_size

POLL_S = 2.0


class DownloadProgressScreen(ModalScreen[None]):
    BINDINGS = [
        Binding("escape", "close", "Close (download continues)"),
    ]

    def __init__(self, *, repo_id: str, expected_bytes: int) -> None:
        super().__init__()
        self._repo_id = repo_id
        self._expected = max(1, expected_bytes)
        self._state: DownloadState | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="download-box"):
            yield Static(f"Downloading {self._repo_id}", id="download-title", markup=False)
            yield ProgressBar(total=100, show_eta=False, id="download-bar")
            yield Static("starting…", id="download-note", markup=False)

    def on_mount(self) -> None:
        self._state = start_download(self._repo_id, self._expected)
        self.set_interval(POLL_S, self._tick)

    def _tick(self) -> None:
        if self._state is None:
            return
        done, error = self._state.snapshot()
        on_disk = bytes_on_disk_for(self._repo_id)
        percent = min(100.0, on_disk * 100.0 / self._expected)
        self.query_one("#download-bar", ProgressBar).update(progress=percent)
        note = f"{format_size(on_disk)} of {format_size(self._expected)}"
        if error:
            note = f"FAILED: {error}"
        elif done:
            note = f"done — {format_size(on_disk)} in the HF cache; vllm serve {self._repo_id}"
        self.query_one("#download-note", Static).update(note)

    def action_close(self) -> None:
        # The worker thread keeps going; snapshot_download resumes anyway if
        # the process dies. Closing the panel must not hold the bytes hostage.
        self.dismiss(None)
