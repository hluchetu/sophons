"""Terminal UI for sophons examples and agents — panels, history, spinner.

Requires the ``cli`` extra: ``pip install 'sophons[cli]'``.

Primitives (the house style, one panel per event):

    from sophons.cli import ui

    ui.header("hybrid.py", subtitle="bm25 vs semantic vs RRF")
    ui.user("How much is FEE-WDR-021?")
    ui.tool("bm25: #1 · semantic: #3 · hybrid: #2")
    ui.agent("The fee is KES 110.", footer="sources: fees.md")

Loops built on the primitives:

- ``chat(title=..., answer=...)`` — bring your own answer function.
- ``chat_with_agent(agent, title=...)`` — wires a sophons Agent in;
  the footer shows run metrics automatically.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from sophons.errors import MissingDependencyError

_INSTALL_HINT = (
    "sophons.cli requires rich and prompt-toolkit. "
    "Install them with `pip install 'sophons[cli]'`."
)


class UI:
    """The panel vocabulary of the sophons terminal.

    Lazy: importing this module costs nothing; the first call pulls in
    rich (raising the friendly install hint if it's missing). Output
    degrades to plain text automatically when piped.
    """

    _BLUE = "#5f87ff"
    _GREEN = "#2dba4e"
    _YELLOW = "#e5c07b"
    _MAGENTA = "#c678dd"

    def __init__(self) -> None:
        self._console: Any = None

    @property
    def console(self) -> Any:
        if self._console is None:
            try:
                from rich.console import Console
            except ImportError as exc:
                raise MissingDependencyError(
                    _INSTALL_HINT, details={"extra": "cli"}
                ) from exc
            self._console = Console()
        return self._console

    def _panel(
        self, body: Any, *, title: str, color: str, footer: str = ""
    ) -> None:
        from rich.panel import Panel

        self.console.print(
            Panel(
                body,
                title=f"[bold {color}]{title}[/bold {color}]",
                subtitle=f"[dim]{footer}[/dim]" if footer else None,
                border_style=color,
                padding=(0, 1),
            )
        )

    def header(self, title: str, *, subtitle: str = "") -> None:
        from rich.panel import Panel

        self.console.print()
        self.console.print(
            Panel.fit(
                f"[bold white]{title}[/bold white]  [dim]{subtitle}[/dim]",
                border_style="bright_black",
                padding=(0, 2),
            )
        )
        self.console.print()

    def user(self, text: str) -> None:
        from rich.text import Text

        self._panel(Text(text), title="You", color=self._BLUE)

    def thinking(self, text: str) -> None:
        from rich.text import Text

        self._panel(Text(text), title="Thinking", color=self._MAGENTA)

    def tool(self, text: str) -> None:
        from rich.text import Text

        self._panel(Text(text), title="Tool", color=self._YELLOW)

    def agent(self, text: str, *, footer: str = "") -> None:
        from rich.markdown import Markdown

        self._panel(Markdown(text), title="Agent", color=self._GREEN, footer=footer)

    def status(self, text: str) -> Any:
        """Spinner context manager: ``with ui.status("Thinking..."):``"""
        return self.console.status(f"[dim]{text}[/dim]", spinner="dots")

    def note(self, text: str) -> None:
        """Dim one-liner between panels (counts, timings, asides)."""
        self.console.print(f"[dim]{text}[/dim]")


ui = UI()


def chat(
    *,
    title: str,
    subtitle: str = "",
    answer: Callable[[str], tuple[str, str]],
    history_name: str = "sophons_chat",
) -> None:
    """Run a chat loop in the terminal.

    ``answer(question)`` returns ``(text, footer)``; the footer shows in
    the agent panel (sources, metrics, anything).
    """
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.styles import Style
    except ImportError as exc:
        raise MissingDependencyError(
            _INSTALL_HINT, details={"extra": "cli"}
        ) from exc

    style = Style.from_dict({"prompt": f"bold {UI._BLUE}"})
    ui.header(title, subtitle=subtitle)
    ui.note("Type a question. exit or Ctrl+C to quit.")

    session = PromptSession(
        history=FileHistory(str(Path.home() / f".{history_name}")),
        style=style,
    )

    while True:
        try:
            question = session.prompt("  You › ", style=style).strip()
        except (KeyboardInterrupt, EOFError):
            ui.note("Goodbye.")
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", "/exit", "/quit"}:
            ui.note("Goodbye.")
            break

        ui.console.print()
        ui.user(question)
        ui.console.print()

        try:
            with ui.status("Thinking..."):
                text, footer = answer(question)
            ui.agent(text, footer=footer)
            ui.console.print()
        except KeyboardInterrupt:
            ui.note("Cancelled.")
        except Exception as exc:  # surface, keep chatting
            ui.console.print(f"\n[bold red]Error:[/bold red] {exc}\n")


def chat_with_agent(
    agent: Any,
    *,
    title: str,
    subtitle: str = "",
    session_id: str | None = "chat",
    history_name: str = "sophons_chat",
) -> None:
    """Chat with a sophons Agent; the footer shows run metrics."""

    def answer(question: str) -> tuple[str, str]:
        result = agent.run_sync(question, session_id=session_id)
        m = result.metrics
        footer = (
            f"steps={m.steps}  model_calls={m.model_calls}  "
            f"tool_calls={m.tool_calls}"
        )
        return result.message, footer

    chat(title=title, subtitle=subtitle, answer=answer, history_name=history_name)
