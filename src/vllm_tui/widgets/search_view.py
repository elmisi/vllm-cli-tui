"""Search tab: the HuggingFace Hub, filtered for things a vLLM can serve."""
from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import DataTable, Input, Static

from ..hf_client import ModelHit, hit_verdict, search_models
from ..screens.model_detail import ModelDetailScreen


class SearchView(Vertical):
    BINDINGS = [
        Binding("enter", "open_detail", "Details / download", priority=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._hits: list[ModelHit] = []

    def compose(self) -> ComposeResult:
        yield Input(placeholder="Search HuggingFace (e.g. qwen2.5 7b awq) — Enter to search",
                    id="search-input")
        yield Static("", id="search-note", markup=False)
        yield DataTable(id="search-table", cursor_type="row")

    def on_mount(self) -> None:
        table = self.query_one("#search-table", DataTable)
        for column in ("model", "vllm", "quant", "downloads", "likes", "gated"):
            table.add_column(column)

    @on(Input.Submitted, "#search-input")
    def _on_submit(self, event: Input.Submitted) -> None:
        query = event.value.strip()
        if not query:
            return
        self.query_one("#search-note", Static).update("searching…")
        self.run_worker(lambda: self._search(query), thread=True,
                        exclusive=True, group="search")

    def _search(self, query: str) -> None:
        try:
            hits = search_models(query)
        except Exception as exc:  # noqa: BLE001 — offline is a state, not a crash
            self.app.call_from_thread(
                self.query_one("#search-note", Static).update,
                f"search failed: {type(exc).__name__}",
            )
            return
        self.app.call_from_thread(self._show_hits, hits)

    def _show_hits(self, hits: list[ModelHit]) -> None:
        self._hits = hits
        table = self.query_one("#search-table", DataTable)
        table.clear()
        for hit in hits:
            table.add_row(hit.id, hit_verdict(hit), hit.quantization or "-",
                          f"{hit.downloads:,}", str(hit.likes),
                          "yes" if hit.gated else "")
        note = (f"{len(hits)} result(s) • vllm: ✓ supported, ? transformers with "
                f"unknown arch (newer vLLM may run it), no = unloadable • Enter for detail")
        if not hits:
            note = "no results"
        self.query_one("#search-note", Static).update(note)
        if hits:
            table.focus()

    @on(DataTable.RowSelected, "#search-table")
    def _on_row(self, event: DataTable.RowSelected) -> None:
        self.action_open_detail()

    def action_open_detail(self) -> None:
        table = self.query_one("#search-table", DataTable)
        row = table.cursor_row
        if row is None or row < 0 or row >= len(self._hits):
            return
        self.app.push_screen(ModelDetailScreen(self._hits[row]))
