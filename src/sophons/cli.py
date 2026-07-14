"""Terminal chat for sophons agents — rich panels, prompt history, spinner.

Requires the ``cli`` extra: ``pip install 'sophons[cli]'``.

Two entry points:

- ``chat(title=..., answer=...)`` — bring your own answer function.
  ``answer(question)`` returns ``(text, footer)``; the footer shows in
  the agent panel (sources, metrics, anything).
- ``chat_with_agent(agent, title=...)`` — wires a sophons Agent in
  directly; the footer shows run metrics automatically.
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


def _ui():
    """Import the UI deps lazily, with the house install hint."""
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.styles import Style
        from rich.console import Console
        from rich.markdown import Markdown
        from rich.panel import Panel
        from rich.text import Text
    except ImportError as exc:
        raise MissingDependencyError(
            _INSTALL_HINT, details={"extra": "cli"}
        ) from exc
    return PromptSession, FileHistory, Style, Console, Markdown, Panel, Text


def chat(
    *,
    title: str,
    subtitle: str = "",
    answer: Callable[[str], tuple[str, str]],
    history_name: str = "sophons_chat",
) -> None:
    """Run a chat loop in the terminal.

    Args:
        title:        Shown in the header panel.
        subtitle:     Dim text next to the title (model name, mode, ...).
        answer:       Called with each question; returns (text, footer).
        history_name: Prompt history file name under the home directory —
                      share one across runs to get up-arrow recall.
    """
    PromptSession, FileHistory, Style, Console, Markdown, Panel, Text = _ui()

    console = Console()
    style = Style.from_dict({"prompt": "bold #5f87ff"})

    console.print()
    console.print(
        Panel.fit(
            f"[bold white]{title}[/bold white]  [dim]{subtitle}[/dim]\n"
            "[dim]Type a question. [bold]exit[/bold] or Ctrl+C to quit.[/dim]",
            border_style="bright_black",
            padding=(0, 2),
        )
    )
    console.print()

    session = PromptSession(
        history=FileHistory(str(Path.home() / f".{history_name}")),
        style=style,
    )

    while True:
        try:
            question = session.prompt("  You › ", style=style).strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", "/exit", "/quit"}:
            console.print("[dim]Goodbye.[/dim]")
            break

        console.print()
        console.print(
            Panel(
                Text(question),
                title="[bold #5f87ff]You[/bold #5f87ff]",
                border_style="#5f87ff",
                padding=(0, 1),
            )
        )
        console.print()

        try:
            with console.status("[dim]Thinking...[/dim]", spinner="dots"):
                text, footer = answer(question)
            console.print(
                Panel(
                    Markdown(text),
                    title="[bold #2dba4e]Agent[/bold #2dba4e]",
                    subtitle=f"[dim]{footer}[/dim]",
                    border_style="#2dba4e",
                    padding=(0, 1),
                )
            )
            console.print()
        except KeyboardInterrupt:
            console.print("\n[dim]Cancelled.[/dim]\n")
        except Exception as exc:  # surface, keep chatting
            console.print(f"\n[bold red]Error:[/bold red] {exc}\n")


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

    chat(
        title=title,
        subtitle=subtitle,
        answer=answer,
        history_name=history_name,
    )
