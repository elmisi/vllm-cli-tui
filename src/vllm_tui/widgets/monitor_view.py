"""Monitor tab: one row per configured endpoint, refreshed every 5 seconds."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import DataTable, Static

from ..hf_client import format_size
from ..metrics_client import EndpointStatus, derive_rates, probe_endpoint

REFRESH_S = 5.0


class MonitorView(Vertical):
    def __init__(self, *, endpoints: list[str]) -> None:
        super().__init__()
        self._endpoints = endpoints
        self._previous: dict[str, object] = {}

    def compose(self) -> ComposeResult:
        yield Static("", id="monitor-note", markup=False)
        table = DataTable(id="monitor-table", cursor_type="row")
        yield table

    def on_mount(self) -> None:
        table = self.query_one("#monitor-table", DataTable)
        for column in ("endpoint", "model", "ctx", "run", "wait", "KV cache",
                       "gen tok/s", "prompt tok/s"):
            table.add_column(column)
        self._note(f"{len(self._endpoints)} endpoint(s) • refresh every {REFRESH_S:.0f}s")
        self.refresh_now()
        self.set_interval(REFRESH_S, self.refresh_now)

    def _note(self, text: str) -> None:
        self.query_one("#monitor-note", Static).update(text)

    def refresh_now(self) -> None:
        self.run_worker(self._poll_all, thread=True, exclusive=True, group="monitor")

    def _poll_all(self) -> None:
        statuses = [probe_endpoint(url) for url in self._endpoints]
        self.app.call_from_thread(self._show_statuses, statuses)

    def _show_statuses(self, statuses: list[EndpointStatus]) -> None:
        table = self.query_one("#monitor-table", DataTable)
        table.clear()
        for status in statuses:
            if not status.reachable:
                table.add_row(status.url, f"down ({status.error})", "-", "-", "-", "-", "-", "-")
                continue
            metrics = status.sample.metrics if status.sample else {}
            rates: dict[str, float] = {}
            previous = self._previous.get(status.url)
            if status.sample and previous:
                rates = derive_rates(previous, status.sample)  # type: ignore[arg-type]
            if status.sample:
                self._previous[status.url] = status.sample
            # V1 engines call it kv_cache_usage_perc, V0 called it gpu_cache_usage_perc.
            kv = metrics.get("vllm:kv_cache_usage_perc",
                             metrics.get("vllm:gpu_cache_usage_perc"))
            table.add_row(
                status.url,
                status.model or "?",
                f"{status.max_model_len:,}" if status.max_model_len else "-",
                f"{metrics.get('vllm:num_requests_running', 0):.0f}",
                f"{metrics.get('vllm:num_requests_waiting', 0):.0f}",
                f"{kv * 100:.0f}%" if kv is not None else "-",
                f"{rates.get('generation_tok_s', 0):.0f}" if rates else "-",
                f"{rates.get('prompt_tok_s', 0):.0f}" if rates else "-",
            )
