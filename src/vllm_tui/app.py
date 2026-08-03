"""Main vLLM TUI application."""
from __future__ import annotations

import argparse
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, TabbedContent, TabPane

from . import __version__
from .config import load_config
from .widgets.models_view import ModelsView
from .widgets.monitor_view import MonitorView
from .widgets.search_view import SearchView


class VllmTUI(App):
    """Three tabs: what the servers are doing, what is on disk, what to get."""

    CSS_PATH = Path(__file__).parent / "styles" / "app.tcss"
    TITLE = f"vLLM TUI v{__version__}"

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("1", "tab_monitor", "Monitor"),
        Binding("2", "tab_models", "Models"),
        Binding("3", "tab_search", "Search"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.config = load_config()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with TabbedContent(initial="tab-monitor"):
            with TabPane("Monitor [1]", id="tab-monitor"):
                yield MonitorView(endpoints=self.config.endpoints)
            with TabPane("Models [2]", id="tab-models"):
                yield ModelsView(extra_dirs=self.config.extra_model_dirs)
            with TabPane("Search [3]", id="tab-search"):
                yield SearchView()
        yield Footer()

    def _switch(self, tab_id: str) -> None:
        tabs = self.query_one(TabbedContent)
        tabs.active = tab_id
        # Move the focus into the freshly shown pane. Without this, keys keep
        # going to the hidden widget that had focus before — and the first
        # letter of a search that starts with "q" quits the app.
        pane = tabs.get_pane(tab_id)
        focusable = pane.query("Input, DataTable")
        if focusable:
            focusable.first().focus()

    def action_tab_monitor(self) -> None:
        self._switch("tab-monitor")

    def action_tab_models(self) -> None:
        self._switch("tab-models")

    def action_tab_search(self) -> None:
        self._switch("tab-search")


def main() -> None:
    parser = argparse.ArgumentParser(prog="vllm-tui")
    parser.add_argument("--version", action="version", version=f"vllm-tui {__version__}")
    parser.parse_args()
    VllmTUI().run()


if __name__ == "__main__":
    main()
