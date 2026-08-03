"""Model detail: the files and the price tag, before any bytes move."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import DataTable, Static

from ..hf_client import (
    ModelDetail,
    ModelHit,
    detail_flags,
    fetch_architectures,
    fetch_model_detail,
    format_size,
)
from .download_progress import DownloadProgressScreen


class ModelDetailScreen(ModalScreen[None]):
    BINDINGS = [
        Binding("d", "download", "Download"),
        Binding("escape", "close", "Close"),
    ]

    def __init__(self, hit: ModelHit) -> None:
        super().__init__()
        self._hit = hit
        self._detail: ModelDetail | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="detail-box"):
            yield Static(self._hit.id, id="detail-title", markup=False)
            yield Static("loading file list…", id="detail-note", markup=False)
            yield DataTable(id="detail-table", cursor_type="row")

    def on_mount(self) -> None:
        table = self.query_one("#detail-table", DataTable)
        table.add_column("file")
        table.add_column("size")
        self.run_worker(self._load, thread=True, exclusive=True)

    def _load(self) -> None:
        try:
            detail = fetch_model_detail(self._hit.id)
        except Exception as exc:  # noqa: BLE001
            self.app.call_from_thread(
                self.query_one("#detail-note", Static).update,
                f"could not load: {type(exc).__name__}",
            )
            return
        architecture = fetch_architectures(self._hit.id)
        self.app.call_from_thread(self._show_detail, detail, architecture)

    def _show_detail(self, detail: ModelDetail, architecture: str) -> None:
        self._detail = detail
        table = self.query_one("#detail-table", DataTable)
        table.clear()
        for f in detail.files:
            table.add_row(f.name, format_size(f.size) if f.size else "-")
        note = f"{len(detail.files)} files • {format_size(detail.total_bytes)} total"
        if architecture:
            note += f" • {architecture}"
        flags = detail_flags(detail.files)
        if flags:
            note += f" • ⚠ {flags}"
        note += " • d to download • Esc to close"
        if self._hit.gated:
            note += " • GATED: needs an accepted license and a logged-in HF token"
        self.query_one("#detail-note", Static).update(note)

    def action_download(self) -> None:
        if self._detail is None:
            return
        detail = self._detail
        self.app.push_screen(
            DownloadProgressScreen(repo_id=detail.id, expected_bytes=detail.total_bytes)
        )

    def action_close(self) -> None:
        self.dismiss(None)
