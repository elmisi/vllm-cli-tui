"""Serve plan preview: the exact command plus pre-flight checks, read-only.

The TUI shows what `vllm serve` would be and what could go wrong before any
process exists. It deliberately does not run the command — step 1 of the
lifecycle work is observation; start/stop (systemd / direct) comes later,
gated by the `lifecycle` config.
"""
from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

from ..lifecycle import ServePlan


class ServePlanScreen(ModalScreen[None]):
    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("enter", "close", "Close"),
    ]

    def __init__(self, plan: ServePlan) -> None:
        super().__init__()
        self.plan = plan

    def compose(self) -> ComposeResult:
        with Vertical(id="serve-plan-box"):
            yield Static(f"Serve plan — {self.plan.model_name}", id="serve-plan-title", markup=False)
            for index, check in enumerate(self.plan.checks):
                yield Static(
                    f"{check.mark} {check.text}",
                    classes=f"serve-plan-check serve-plan-{check.level}",
                    markup=False,
                )
            yield Static("", id="serve-plan-gap")
            yield Static(f"$ {self.plan.command}", id="serve-plan-command", markup=False)
            yield Static(
                "Read-only preview: the TUI shows the command, it does not run it. "
                "Copy it and run it with your usual tooling.",
                id="serve-plan-hint", markup=False,
            )

    def action_close(self) -> None:
        self.dismiss(None)