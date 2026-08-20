"""Models tab: HF cache plus configured extra dirs, with delete."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import DataTable, Static

from ..hf_client import format_size
from ..lifecycle import plan_for_model
from ..local_models import LocalModel, delete_local_model, scan_extra_dirs, scan_hf_cache
from ..screens.confirm_dialog import ConfirmDialog
from ..screens.serve_plan import ServePlanScreen


class ModelsView(Vertical):
    BINDINGS = [
        Binding("r", "refresh", "Refresh"),
        Binding("d", "delete", "Delete"),
        Binding("a", "toggle_all", "All / serveable"),
        Binding("s", "serve_plan", "Serve plan"),
    ]

    def __init__(self, *, extra_dirs: list[str]) -> None:
        super().__init__()
        self._extra_dirs = extra_dirs
        self._all_models: list[LocalModel] = []
        self._models: list[LocalModel] = []
        # The tab answers "what can I serve" by default; `a` widens it to
        # "what is eating my disk", which needs the unserveable ones too.
        self._only_serveable = True

    def compose(self) -> ComposeResult:
        yield Static("", id="models-note", markup=False)
        yield DataTable(id="models-table", cursor_type="row")

    def on_mount(self) -> None:
        table = self.query_one("#models-table", DataTable)
        for column in ("model", "size", "vllm", "source", "last used"):
            table.add_column(column)
        self.action_refresh()

    def action_refresh(self) -> None:
        self.run_worker(self._scan, thread=True, exclusive=True, group="models")

    def _scan(self) -> None:
        models = scan_hf_cache() + scan_extra_dirs(self._extra_dirs)
        self.app.call_from_thread(self._show_models, models)

    def _show_models(self, models: list[LocalModel]) -> None:
        self._all_models = models
        shown = [m for m in models if m.serveable] if self._only_serveable else models
        self._models = shown
        table = self.query_one("#models-table", DataTable)
        table.clear()
        total = 0
        for model in shown:
            total += model.size_bytes
            table.add_row(model.name, format_size(model.size_bytes),
                          model.detail if model.serveable else f"no ({model.detail})",
                          model.source, model.last_used or "-")
        if self._only_serveable:
            hidden = len(models) - len(shown)
            note = (f"{len(shown)} serveable model(s) • {format_size(total)} • "
                    f"{hidden} hidden • a to show all • s for serve plan • d to delete")
        else:
            note = (f"{len(shown)} model(s) on disk • {format_size(total)} total • "
                    f"a to show serveable only • s for serve plan • d to delete")
        self.query_one("#models-note", Static).update(note)

    def action_toggle_all(self) -> None:
        self._only_serveable = not self._only_serveable
        self._show_models(self._all_models)

    def action_serve_plan(self) -> None:
        """Read-only preview of the exact `vllm serve` command for the row."""
        table = self.query_one("#models-table", DataTable)
        row = table.cursor_row
        if row is None or row < 0 or row >= len(self._models):
            return
        model = self._models[row]
        endpoints = self.app.config.endpoints
        endpoint = endpoints[0] if endpoints else "http://localhost:8000"
        self.app.push_screen(ServePlanScreen(plan_for_model(model, endpoint)))

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
