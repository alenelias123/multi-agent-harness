"""Interactive chat REPL for agentcli."""

from __future__ import annotations

import logging
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .model_router import ModelRouter, create_model_router

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "chat_system.txt"

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful AI coding assistant inside a terminal CLI tool called agentcli.
You help users with software engineering tasks: writing code, debugging, \
explaining concepts, planning architecture, and more.

Guidelines:
- Be concise and practical. Prefer actionable answers over verbose explanations.
- When writing code, use markdown fenced code blocks with the language specified.
- If a task is complex, suggest breaking it into steps.
- You can use any programming language or framework.
- If you're unsure, say so rather than guessing.
- Keep responses focused on what the user asked.
"""

SLASH_COMMANDS = {
    "/help": "Show available commands",
    "/quit": "Exit the chat",
    "/exit": "Exit the chat",
    "/clear": "Clear conversation history",
    "/model": "Show or change the active model",
    "/history": "Show conversation message count",
    "/system": "Show the current system prompt",
    "/export": "Export conversation to a file",
}


def _load_system_prompt(override_path: str | None = None) -> str:
    if override_path:
        from pathlib import Path as _P

        p = _P(override_path)
        if p.is_file():
            return p.read_text(encoding="utf-8")
        raise FileNotFoundError(f"System prompt file not found: {override_path}")
    try:
        return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return DEFAULT_SYSTEM_PROMPT


class ChatSession:
    """Manages an interactive chat session with an LLM."""

    def __init__(self, router: ModelRouter, system_prompt_path: str | None = None) -> None:
        self.router = router
        self.system_prompt = _load_system_prompt(override_path=system_prompt_path)
        self.messages: list[dict[str, str]] = []
        self.current_model: str | None = None
        self.console = Console()

    def _build_messages(self, user_input: str) -> list[dict[str, str]]:
        """Build the full message list for the API call."""
        msgs: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
        ]
        msgs.extend(self.messages)
        msgs.append({"role": "user", "content": user_input})
        return msgs

    async def send(self, user_input: str) -> str:
        """Send a user message and return the assistant's response."""
        api_messages = self._build_messages(user_input)

        response, model = await self.router.call(
            task_type="general",
            messages=api_messages,
            temperature=0.4,
            max_tokens=4000,
        )

        # Track the model used for the first successful call
        if self.current_model is None:
            self.current_model = model

        # Store the exchange in history
        self.messages.append({"role": "user", "content": user_input})
        self.messages.append({"role": "assistant", "content": response})

        return response

    def clear_history(self) -> None:
        """Reset conversation history."""
        self.messages.clear()
        self.current_model = None

    def message_count(self) -> int:
        """Return the number of user messages."""
        return sum(1 for m in self.messages if m["role"] == "user")


def _print_welcome(console: Console) -> None:
    """Print the welcome banner."""
    console.print()
    console.print(
        Panel(
            "[bold]agentcli chat[/bold] - Interactive coding assistant\n\n"
            "Type your message and press Enter. Commands:\n"
            "  [cyan]/help[/cyan]    Show all commands\n"
            "  [cyan]/clear[/cyan]   Clear history\n"
            "  [cyan]/model[/cyan]   Show active model\n"
            "  [cyan]/export[/cyan]  Save conversation to file\n"
            "  [cyan]/quit[/cyan]    Exit\n\n"
            "[dim]Press Ctrl+C to interrupt, Ctrl+D to exit.[/dim]",
            title="💬 Chat Mode",
            border_style="cyan",
        )
    )
    console.print()


def _print_response(console: Console, response: str, model: str | None) -> None:
    """Render the assistant's response with Rich markdown."""
    # Show model info as a small tag above the response
    if model:
        console.print(f"[dim]  ↳ {model}[/dim]")

    # Render as markdown for nice code blocks, lists, etc.
    console.print()
    console.print(Markdown(response))
    console.print()


def _handle_slash_command(
    command: str, session: ChatSession, console: Console
) -> bool:
    """Handle a slash command. Returns True if the REPL should continue."""
    parts = command.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    if cmd in ("/quit", "/exit"):
        console.print("[dim]Goodbye![/dim]")
        return False

    if cmd == "/help":
        table = Table(title="Commands", show_header=True, border_style="dim")
        table.add_column("Command", style="cyan")
        table.add_column("Description")
        for cmd_name, desc in SLASH_COMMANDS.items():
            table.add_row(cmd_name, desc)
        console.print(table)
        return True

    if cmd == "/clear":
        session.clear_history()
        console.print("[green]Conversation cleared.[/green]")
        return True

    if cmd == "/model":
        if session.current_model:
            console.print(f"[bold]Active model:[/bold] {session.current_model}")
        else:
            console.print("[dim]No model used yet. Send a message to activate.[/dim]")
        if args:
            console.print(
                "[yellow]Note: Model switching via /model is not yet supported.[/yellow]\n"
                "[dim]The router automatically selects the best available model.[/dim]"
            )
        return True

    if cmd == "/history":
        count = session.message_count()
        total = len(session.messages)
        console.print(
            f"[bold]Messages:[/bold] {count} user, "
            f"{total - count} assistant ({total} total)"
        )
        return True

    if cmd == "/system":
        console.print(Panel(session.system_prompt, title="System Prompt", border_style="dim"))
        return True

    if cmd == "/export":
        if not session.messages:
            console.print("[yellow]No messages to export.[/yellow]")
            return True

        filename = args if args else "chat_export.md"
        lines = ["# agentcli chat export\n"]
        for msg in session.messages:
            role = msg["role"].capitalize()
            lines.append(f"## {role}\n\n{msg['content']}\n")

        Path(filename).write_text("\n".join(lines), encoding="utf-8")
        console.print(f"[green]Exported to {filename}[/green]")
        return True

    console.print(f"[red]Unknown command: {cmd}[/red]. Type /help for available commands.")
    return True


async def run_chat(model: str | None = None, system_prompt_path: str | None = None) -> None:
    """Run the interactive chat REPL."""
    console = Console()

    async with create_model_router() as router:
        session = ChatSession(router, system_prompt_path=system_prompt_path)
        _print_welcome(console)

        if model:
            console.print(f"[dim]Requested model: {model}[/dim]\n")

        while True:
            try:
                user_input = console.input("[bold green]You>[/bold green] ")
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Goodbye![/dim]")
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            # Handle slash commands
            if user_input.startswith("/"):
                should_continue = _handle_slash_command(user_input, session, console)
                if not should_continue:
                    break
                continue

            # Send to LLM
            try:
                with console.status("[dim]Thinking...[/dim]", spinner="dots"):
                    response = await session.send(user_input)
                _print_response(console, response, session.current_model)
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted.[/yellow]")
                continue
            except Exception as e:
                console.print(f"\n[bold red]Error:[/bold red] {e}")
                console.print("[dim]Try again or type /quit to exit.[/dim]\n")
                continue
