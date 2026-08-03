"""Models tab: HF cache plus configured extra dirs, with delete."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import DataTable, Static

from ..hf_client import format_size
from ..local_models import LocalModel, delete_local_model, scan_extra_dirs, scan_hf_cache
from ..screens.confirm_dialog import ConfirmDialog


class ModelsView(Vertical):
    BINDINGS = [
        Binding("r", "refresh", "Refresh"),
        Binding("d", "delete", "Delete"),
    ]

    def __init__(self, *, extra_dirs: list[str]) -> None:
        super().__init__()
        self._extra_dirs = extra_dirs
        self._models: list[LocalModel] = []

    def compose(self) -> ComposeResult:
        yield Static("", id="models-note", markup=False)
        yield DataTable(id="models-table", cursor_type="row")

    def on_mount(self) -> None:
        table = self.query_one("#models-table", DataTable)
        for column in ("model", "size", "source", "last used"):
            table.add_column(column)
        self.action_refresh()

    def action_refresh(self) -> None:
        self.run_worker(self._scan, thread=True, exclusive=True, group="models")

    def _scan(self) -> None:
        models = scan_hf_cache() + scan_extra_dirs(self._extra_dirs)
        self.app.call_from_thread(self._show_models, models)

    def _show_models(self, models: list[LocalModel]) -> None:
        self._models = models
        table = self.query_one("#models-table", DataTable)
        table.clear()
        total = 0
        for model in models:
            total += model.size_bytes
            table.add_row(model.name, format_size(model.size_bytes),
                          model.source, model.last_used or "-")
        self.query_one("#models-note", Static).update(
            f"{len(models)} model(s) on disk • {format_size(total)} total • d to delete"
        )

    def action_delete(self) -> None:
        table = self.query_one("#models-table", DataTable)
        row = table.cursor_row
        if row is None or row < 0 or row >= len(self._models):
            return
        model = self._models[row]

        def on_confirm(confirmed: bool | None) -> None:
            if not confirmed:
                return
            error = delete_local_model(model)
            if error:
                self.query_one("#models-note", Static).update(f"delete failed: {error}")
            else:
                self.action_refresh()

        self.app.push_screen(
            ConfirmDialog(f"Delete {model.name} ({format_size(model.size_bytes)})?"),
            on_confirm,
        )
