from __future__ import annotations

import asyncio
import contextlib
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from ._version import __version__
from .chat import run_chat
from .config import APP_NAME, get_settings
from .context import get_context_bridge, get_shared_context
from .executor import create_executor
from .model_router import ModelRouter, create_model_router
from .planner import Planner
from .schemas import Run, RunStatus, TaskGraph, TaskResult, TaskStatus
from .storage import storage

app = typer.Typer(
    name="agentcli",
    help="Multi-agent development CLI powered by free-tier LLMs.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()

# ── sessions sub-app ─────────────────────────────────────────────────────────

sessions_app = typer.Typer(
    name="sessions",
    help="Manage parallel AI chat sessions in separate terminal windows.",
    no_args_is_help=True,
)
app.add_typer(sessions_app, name="sessions")


# ── version callback ────────────────────────────────────────────────────────


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"agentcli {__version__}")
        raise typer.Exit()


# ── signal handling ─────────────────────────────────────────────────────────


class GracefulExit(Exception):
    pass


def setup_signal_handlers() -> None:
    def handler(_signum: int, _frame: Any) -> None:
        raise GracefulExit()

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


# ── helper: check provider availability ─────────────────────────────────────


def _check_providers() -> None:
    """Fail fast with a clear message when no LLM provider is configured."""
    from .config import settings as s

    if s.openrouter_api_key or s.freebuff_enabled:
        return

    console.print(
        Panel(
            "[bold]No LLM provider configured.[/bold]\n\n"
            "To get started, choose one of:\n\n"
            "  [cyan]Option A - OpenRouter (recommended)[/cyan]\n"
            "    1. Get a free API key at [link]https://openrouter.ai/keys[/link]\n"
            "    2. Add to your .env file (in CWD or ~/.config/agentcli/):\n"
            "         OPENROUTER_API_KEY=sk-or-...\n\n"
            "  [cyan]Option B - Freebuff (zero-config)[/cyan]\n"
            "    1. Run: [bold]agentcli setup[/bold]\n"
            "    2. Or install manually: npm install -g freebuff\n\n"
            "  [cyan]Option C - Both (recommended for reliability)[/cyan]\n"
            "    1. Set OPENROUTER_API_KEY and FREEBUFF_ENABLED=true\n",
            title="⚠  Provider Setup Required",
            border_style="yellow",
        )
    )
    raise typer.Exit(1)


# ── core run logic ──────────────────────────────────────────────────────────


async def run_task(task_description: str) -> Run:
    run = Run(
        user_id=get_settings().default_user_id,
        task_description=task_description,
        status=RunStatus.RUNNING,
    )
    storage.save_run(run)

    async with create_model_router() as router:
        planner = Planner(router)
        console.print(f"[bold cyan]Planning task:[/bold cyan] {task_description}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            plan_task = progress.add_task("Planning...", total=None)
            try:
                graph = await planner.plan(task_description)
            except Exception as e:
                run.status = RunStatus.FAILED
                run.error = str(e)
                run.completed_at = datetime.utcnow()
                storage.save_run(run)
                console.print(f"[bold red]Planning failed:[/bold red] {e}")
                raise

            progress.update(plan_task, description="Planning complete")

        run.task_graph = graph
        run.status = RunStatus.RUNNING
        storage.save_run(run)

        console.print(f"[bold green]Plan created:[/bold green] {len(graph.tasks)} tasks")
        _print_task_graph(graph)

        async with create_executor(router, run.run_id) as executor:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task_map = {
                    t.id: progress.add_task(
                        f"[{t.id}] waiting...", total=None
                    )
                    for t in graph.tasks
                }

                async def progress_callback() -> None:
                    while True:
                        await asyncio.sleep(0.5)
                        for tid, result in executor.results.items():
                            if result.status == TaskStatus.RUNNING:
                                model = result.model_used or '?'
                                desc = f"[{tid}] running on {model}..."
                                progress.update(task_map[tid], description=desc)
                            elif result.status == TaskStatus.SUCCESS:
                                desc = f"[green][{tid}] done[/green]"
                                progress.update(task_map[tid], description=desc)
                            elif result.status == TaskStatus.FAILED:
                                desc = f"[red][{tid}] failed[/red]"
                                progress.update(task_map[tid], description=desc)
                            elif result.status == TaskStatus.SKIPPED:
                                desc = f"[yellow][{tid}] skipped[/yellow]"
                                progress.update(task_map[tid], description=desc)

                progress_task = asyncio.create_task(progress_callback())

                try:
                    results = await executor.execute(graph)
                except GracefulExit:
                    run.status = RunStatus.INTERRUPTED
                    run.error = "Interrupted by user"
                    raise
                finally:
                    progress_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await progress_task

    run.results = list(results.values())
    run.completed_at = datetime.utcnow()

    failed_count = sum(1 for r in results.values() if r.status == TaskStatus.FAILED)
    if failed_count > 0:
        run.status = RunStatus.FAILED
        run.error = f"{failed_count} task(s) failed"
    else:
        run.status = RunStatus.COMPLETED
        run.final_output = await _aggregate_results(graph, results, router)

    storage.save_run(run)
    for result in results.values():
        task = graph.get_task(result.task_id)
        if task:
            storage.save_task_result(run.run_id, result, task.description, task.depends_on)

    return run


async def _aggregate_results(
    graph: TaskGraph,
    results: dict[str, TaskResult],
    router: ModelRouter,
) -> str:
    leaf_tasks = graph.leaf_tasks()
    leaf_outputs = []

    for task in leaf_tasks:
        result = results.get(task.id)
        if result and result.status == TaskStatus.SUCCESS and result.output:
            leaf_outputs.append(f"## {task.id}: {task.description}\n\n{result.output}")

    if not leaf_outputs:
        return "No successful task outputs to aggregate."

    if len(leaf_outputs) == 1:
        return leaf_outputs[0]

    combined = "\n\n---\n\n".join(leaf_outputs)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a technical editor. Combine multiple task outputs into "
                "a single coherent, well-structured final deliverable. Remove "
                "redundancy, resolve conflicts, and ensure flow."
            ),
        },
        {
            "role": "user",
            "content": f"Combine these task outputs:\n\n{combined}",
        },
    ]

    try:
        response, _ = await router.call(
            task_type="writing",
            messages=messages,
            temperature=0.3,
            max_tokens=4000,
        )
        return response
    except Exception as e:
        console.print(
            f"[yellow]Aggregation LLM failed, returning concatenated: {e}[/yellow]"
        )
        return combined


def _print_task_graph(graph: TaskGraph) -> None:
    table = Table(title="Task Graph")
    table.add_column("ID", style="cyan")
    table.add_column("Description")
    table.add_column("Depends On", style="yellow")
    table.add_column("Type", style="magenta")
    table.add_column("Complexity", style="green")

    for task in graph.tasks:
        table.add_row(
            task.id,
            task.description[:60] + ("..." if len(task.description) > 60 else ""),
            ", ".join(task.depends_on) if task.depends_on else "—",
            task.task_type.value,
            task.complexity.value,
        )
    console.print(table)


# ── CLI commands ────────────────────────────────────────────────────────────


@app.callback(invoke_without_command=True)
def main(
    version: Annotated[
        bool, typer.Option(
            "--version", "-v", callback=_version_callback,
            is_eager=True, help="Show version and exit."
        )
    ] = False,
) -> None:
    """Multi-agent development CLI powered by free-tier LLMs."""


@app.command()
def run(
    task: Annotated[str, typer.Argument(help="Natural language development task")],
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Save final output to a file")] = None,
    db_path: Annotated[Path | None, typer.Option("--db-path", help="Override database location")] = None,
) -> None:
    """Run a development task through the multi-agent pipeline."""
    _check_providers()
    setup_signal_handlers()

    if db_path:
        storage.db_path = db_path
        storage._init_db()

    try:
        run_result = asyncio.run(run_task(task))
    except GracefulExit:
        console.print("\n[yellow]Interrupted. Run marked as interrupted.[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        sys.exit(1)

    if run_result.status == RunStatus.COMPLETED:
        output_text = run_result.final_output or "No output"
        console.print(Panel(output_text, title="Final Result", border_style="green"))
        if output:
            output.write_text(output_text, encoding="utf-8")
            console.print(f"[dim]Output saved to {output}[/dim]")
    else:
        error = run_result.error or "Unknown error"
        console.print(Panel(error, title="Run Failed", border_style="red"))
        sys.exit(1)


@app.command()
def history(
    limit: Annotated[int, typer.Option("--limit", "-n", help="Number of runs to show")] = 20,
    db_path: Annotated[Path | None, typer.Option("--db-path", help="Override database location")] = None,
) -> None:
    """Show recent runs."""
    if db_path:
        storage.db_path = db_path
        storage._init_db()

    runs = storage.list_runs(user_id=get_settings().default_user_id, limit=limit)
    if not runs:
        console.print("[yellow]No runs found.[/yellow]")
        return

    table = Table(title="Run History")
    table.add_column("Run ID", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Status", style="bold")
    table.add_column("Created", style="dim")
    table.add_column("Tasks", style="dim")

    for run in runs:
        status_style = {
            RunStatus.COMPLETED: "green",
            RunStatus.FAILED: "red",
            RunStatus.INTERRUPTED: "yellow",
            RunStatus.RUNNING: "blue",
            RunStatus.PENDING: "dim",
        }.get(run.status, "white")

        task_count = len(run.task_graph.tasks) if run.task_graph else 0
        table.add_row(
            run.run_id,
            run.task_description[:50] + ("..." if len(run.task_description) > 50 else ""),
            Text(run.status.value, style=status_style),
            run.created_at.strftime("%Y-%m-%d %H:%M"),
            str(task_count),
        )
    console.print(table)


@app.command()
def show(
    run_id: Annotated[str, typer.Argument(help="Run ID to display")],
    db_path: Annotated[Path | None, typer.Option("--db-path", help="Override database location")] = None,
) -> None:
    """Show details of a specific run."""
    if db_path:
        storage.db_path = db_path
        storage._init_db()

    run = storage.get_run(run_id)
    if not run:
        console.print(f"[red]Run {run_id} not found.[/red]")
        raise typer.Exit(1)

    info = (
        f"[bold]Run ID:[/bold] {run.run_id}\n"
        f"[bold]Description:[/bold] {run.task_description}\n"
        f"[bold]Status:[/bold] {run.status.value}\n"
        f"[bold]Created:[/bold] {run.created_at}"
    )
    console.print(Panel(info, title="Run Info"))

    if run.task_graph:
        _print_task_graph(run.task_graph)

    results = storage.get_task_results(run_id)
    if results:
        table = Table(title="Task Results")
        table.add_column("Task ID", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Model", style="dim")
        table.add_column("Output", style="white")

        for result in results:
            status_style = {
                TaskStatus.SUCCESS: "green",
                TaskStatus.FAILED: "red",
                TaskStatus.SKIPPED: "yellow",
                TaskStatus.INTERRUPTED: "blue",
            }.get(result.status, "white")

            output_preview = (result.output or result.error or "")[:100]
            if len(output_preview) == 100:
                output_preview += "..."

            table.add_row(
                result.task_id,
                Text(result.status.value, style=status_style),
                result.model_used or "—",
                output_preview,
            )
        console.print(table)

    if run.final_output:
        console.print(Panel(run.final_output, title="Final Output", border_style="green"))


@app.command()
def config(
    action: Annotated[
        str,
        typer.Argument(help="Config action: 'path' or 'init'"),
    ] = "path",
) -> None:
    """Show or create configuration paths."""
    from platformdirs import user_config_dir, user_data_dir

    config_dir = Path(user_config_dir(APP_NAME))
    data_dir = Path(user_data_dir(APP_NAME))

    if action == "init":
        config_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)

        env_path = config_dir / ".env"
        if not env_path.exists():
            env_path.write_text(
                "# AgentCLI configuration\n"
                "# Get a free API key at https://openrouter.ai/keys\n"
                "OPENROUTER_API_KEY=\n\n"
                "# Optional: enable Freebuff provider (requires: npm install -g freebuff)\n"
                "# FREEBUFF_ENABLED=true\n\n"
                "# Optional: Freebuff auth token (auto-detected if installed)\n"
                "# FREEBUFF_TOKEN=\n",
                encoding="utf-8",
            )
            console.print(f"[green]Created {env_path}[/green]")
            console.print("[dim]Edit it and add your OPENROUTER_API_KEY.[/dim]")
        else:
            console.print(f"[yellow]{env_path} already exists[/yellow]")

        console.print(f"\n[bold]Config dir:[/bold]  {config_dir}")
        console.print(f"[bold]Data dir:[/bold]    {data_dir}")
        console.print(f"[bold]Database:[/bold]    {data_dir / 'agentcli.db'}")
    elif action == "path":
        console.print(f"[bold]Config dir:[/bold]  {config_dir}")
        console.print(f"[bold]Data dir:[/bold]    {data_dir}")
        console.print(f"[bold]Database:[/bold]    {data_dir / 'agentcli.db'}")
        console.print(f"[bold]Env file:[/bold]    {config_dir / '.env'}")

        # Show where .env was found
        from .config import _find_dotenv

        found = _find_dotenv()
        if found:
            console.print(f"\n[green]Using env file:[/green] {found}")
        else:
            console.print(
                "\n[yellow]No .env file found.[/yellow] Run "
                "[bold]agentcli config init[/bold] to create one."
            )
    else:
        console.print(f"[red]Unknown action: {action}. Use 'path' or 'init'.[/red]")
        raise typer.Exit(1)


@app.command()
def chat(
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Model to use (overrides default routing)"),
    ] = None,
    system_prompt: Annotated[
        str | None,
        typer.Option("--system-prompt", help="Path to a custom system prompt file"),
    ] = None,
) -> None:
    """Start an interactive chat session."""
    _check_providers()
    setup_signal_handlers()

    try:
        asyncio.run(run_chat(model=model, system_prompt_path=system_prompt))
    except GracefulExit:
        console.print("\n[dim]Goodbye![/dim]")
    except KeyboardInterrupt:
        console.print("\n[dim]Goodbye![/dim]")


@app.command()
def setup(
    auth: Annotated[
        bool, typer.Option("--auth", help="Run the interactive auth flow after install")
    ] = False,
    token: Annotated[
        str | None, typer.Option("--token", help="Set a freebuff auth token directly")
    ] = None,
) -> None:
    """Install and configure Freebuff CLI for agentcli.

    Checks if freebuff is installed, installs it if missing, and
    configures authentication.
    """
    from .setup import (
        get_freebuff_token,
        get_freebuff_version,
        is_freebuff_installed,
        install_freebuff,
        run_auth_flow,
        save_token_to_env,
    )

    console.print(
        Panel(
            "[bold]Freebuff Setup[/bold]\n\n"
            "This will install and configure the freebuff CLI for use "
            "with agentcli's multi-agent pipeline.",
            title="🔧 Setup",
            border_style="cyan",
        )
    )

    # Step 1: Check/install
    version = get_freebuff_version()
    if is_freebuff_installed():
        console.print(f"[green]✓ freebuff {version} is already installed[/green]")
    else:
        console.print("[yellow]→ Installing freebuff CLI...[/yellow]")
        if not install_freebuff():
            console.print("[red]✗ Installation failed. Install manually:[/red]")
            console.print("  [cyan]npm install -g freebuff[/cyan]")
            raise typer.Exit(1)

    # Step 2: Token
    if token:
        console.print(f"[cyan]→ Saving provided token to .env...[/cyan]")
        env_path = save_token_to_env(token)
        console.print(f"[green]✓ Token saved to {env_path}[/green]")
    elif auth:
        console.print("[cyan]→ Running auth flow...[/cyan]")
        if not run_auth_flow():
            console.print("[yellow]⚠ Auth may have been skipped. You can set it later:[/yellow]")
            console.print("  [cyan]agentcli setup --token YOUR_TOKEN[/cyan]")
    else:
        existing_token = get_freebuff_token()
        if existing_token:
            console.print("[green]✓ Auth token already configured[/green]")
        else:
            console.print("[yellow]⚠ No auth token found.[/yellow]")
            console.print("  Options:")
            console.print("    1. Run: [cyan]agentcli setup --auth[/cyan]")
            console.print("    2. Or:  [cyan]agentcli setup --token YOUR_TOKEN[/cyan]")
            console.print("    3. Or:  [cyan]export FREEBUFF_TOKEN=your_token[/cyan]")

    # Step 3: Summary
    console.print()
    token = get_freebuff_token()
    if token:
        console.print("[green]✓ Freebuff is ready to use![/green]")
        console.print()
        console.print("[cyan]To enable in agentcli, add to your .env:[/cyan]")
        console.print("  [green]FREEBUFF_ENABLED=true[/green]")
    else:
        console.print("[yellow]Freebuff installed but needs authentication.[/yellow]")
        console.print("  Run: [cyan]agentcli setup --auth[/cyan]")


@app.command("context")
def context_cmd(
    action: Annotated[
        str, typer.Argument(help="Action: 'ls', 'get', 'stats', 'clear'")
    ] = "stats",
    key: Annotated[
        str | None, typer.Option("--key", "-k", help="Context key to read/write")
    ] = None,
    value: Annotated[
        str | None, typer.Option("--value", "-v", help="Value to write")
    ] = None,
    namespace: Annotated[
        str | None, typer.Option("--namespace", "-n", help="Context namespace")
    ] = None,
    run_id: Annotated[
        str | None, typer.Option("--run-id", "-r", help="Run ID (for run-scoped context)")
    ] = None,
    limit: Annotated[int, typer.Option("--limit", help="Max entries to show")] = 20,
) -> None:
    """Manage shared context for multi-agent coordination."""
    from .context import SharedContext

    ctx = SharedContext()

    if action == "stats":
        stats = ctx.get_stats()
        table = Table(title="Context Store Stats")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Total Entries", str(stats["total_entries"]))
        table.add_row("Namespaces", str(stats["total_namespaces"]))
        table.add_row("Unique Tags", ", ".join(stats["unique_tags"]) or "—")
        console.print(table)

    elif action == "ls":
        entries = ctx.read_all(namespace=namespace, limit=limit)
        if not entries:
            console.print("[yellow]No context entries found.[/yellow]")
            return

        table = Table(title="Context Entries")
        table.add_column("Namespace", style="cyan")
        table.add_column("Key", style="green")
        table.add_column("Source", style="dim")
        table.add_column("Tags", style="magenta")
        table.add_column("Value (preview)", style="white")

        for entry in entries:
            value_preview = entry.value[:60] + ("..." if len(entry.value) > 60 else "")
            table.add_row(
                entry.namespace,
                entry.key,
                entry.source_agent or "—",
                ", ".join(entry.tags[:3]) or "—",
                value_preview,
            )
        console.print(table)

    elif action == "get":
        if not key:
            console.print("[red]--key is required for 'get'[/red]")
            raise typer.Exit(1)

        ns = namespace or f"run:{run_id}" if run_id else "global"
        val = ctx.read(key=key, namespace=ns)
        if val:
            console.print(Panel(val, title=f"[{ns}] {key}", border_style="green"))
        else:
            console.print(f"[yellow]Key '{key}' not found in namespace '{ns}'[/yellow]")

    elif action == "set":
        if not key or not value:
            console.print("[red]--key and --value are required for 'set'[/red]")
            raise typer.Exit(1)

        ns = namespace or f"run:{run_id}" if run_id else "global"
        ctx.write(key=key, value=value, namespace=ns)
        console.print(f"[green]✓ Stored [{ns}] {key}[/green]")

    elif action == "clear":
        if namespace:
            count = ctx.clear_namespace(namespace)
            console.print(f"[green]✓ Cleared {count} entries from '{namespace}'[/green]")
        else:
            console.print("[yellow]Use --namespace to specify which namespace to clear[/yellow]")

    elif action == "search":
        if not key:
            console.print("[red]--key is required as search query[/red]")
            raise typer.Exit(1)

        results = ctx.search(query=key, namespace=namespace)
        if not results:
            console.print(f"[yellow]No results for '{key}'[/yellow]")
            return

        table = Table(title=f"Search Results: '{key}'")
        table.add_column("Namespace", style="cyan")
        table.add_column("Key", style="green")
        table.add_column("Value (preview)", style="white")

        for entry in results:
            value_preview = entry.value[:80] + ("..." if len(entry.value) > 80 else "")
            table.add_row(entry.namespace, entry.key, value_preview)
        console.print(table)

    else:
        console.print(f"[red]Unknown action: {action}. Use ls|get|set|clear|search|stats[/red]")
        raise typer.Exit(1)


# ── sessions commands ───────────────────────────────────────────────────────


def _get_session_manager() -> Any:
    """Lazy-import and instantiate the SessionManager."""
    from .sessions import SessionManager, TmuxError, is_tmux_available

    if not is_tmux_available():
        console.print(
            "[red]tmux is not installed.[/red]\n\n"
            "Install it with:\n"
            "  Ubuntu/Debian: [cyan]sudo apt install tmux[/cyan]\n"
            "  macOS:         [cyan]brew install tmux[/cyan]\n"
            "  Arch:          [cyan]sudo pacman -S tmux[/cyan]"
        )
        raise typer.Exit(1)

    return SessionManager()


@sessions_app.command("create")
def session_create(
    name: Annotated[
        str | None,
        typer.Option("--name", "-n", help="Human-readable session name"),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Model to use for this session"),
    ] = None,
    system_prompt: Annotated[
        str | None,
        typer.Option("--system-prompt", help="Path to a custom system prompt file"),
    ] = None,
) -> None:
    """Create a new AI chat session in a separate terminal window.

    Each session runs ``agentcli chat`` in its own tmux window.  You can
    later attach to it, send messages, or view its output.
    """
    manager = _get_session_manager()

    with console.status("[dim]Creating session...[/dim]", spinner="dots"):
        session = manager.create_session(name=name, model=model, system_prompt=system_prompt)

    console.print(
        Panel(
            f"[bold]Session created![/bold]\n\n"
            f"  [cyan]ID:[/cyan]     {session.session_id}\n"
            f"  [cyan]Name:[/cyan]   {session.name}\n"
            f"  [cyan]Model:[/cyan]  {session.model or '(default)'}\n"
            f"  [cyan]tmux:[/cyan]   {session.tmux_window}\n\n"
            f"[dim]Quick actions:[/dim]\n"
            f"  Attach:     [green]agentcli sessions attach {session.session_id}[/green]\n"
            f"  Send input: [green]agentcli sessions send {session.session_id} -m 'your message'[/green]\n"
            f"  View logs:  [green]agentcli sessions logs {session.session_id}[/green]",
            title="🚀 New Session",
            border_style="green",
        )
    )


@sessions_app.command("list")
def session_list() -> None:
    """List all AI sessions (running and stopped)."""
    manager = _get_session_manager()
    sessions = manager.list_sessions()

    if not sessions:
        console.print("[yellow]No sessions found. Create one with:[/yellow]")
        console.print("  [cyan]agentcli sessions create[/cyan]")
        return

    table = Table(title="AI Sessions")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Status", style="bold")
    table.add_column("Model", style="dim")
    table.add_column("tmux Window", style="dim")

    for s in sessions:
        status_style = "green" if s.status == "running" else "red"
        status_icon = "●" if s.status == "running" else "○"
        table.add_row(
            s.session_id,
            s.name,
            Text(f"{status_icon} {s.status}", style=status_style),
            s.model or "(default)",
            s.tmux_window,
        )
    console.print(table)
    console.print(
        "\n[dim]Tip: Use [bold]tmux attach -t <tmux-window>[/bold] or "
        "[bold]agentcli sessions attach <id>[/bold] to enter a session.[/dim]"
    )


@sessions_app.command("attach")
def session_attach(
    session_id: Annotated[
        str, typer.Argument(help="Session ID to attach to")
    ],
) -> None:
    """Attach your terminal to a running session.

    This switches your terminal into the session's tmux window.  Press
    ``Ctrl-B d`` to detach and return to the main terminal.
    """
    manager = _get_session_manager()

    try:
        manager.attach_session(session_id)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@sessions_app.command("send")
def session_send(
    session_id: Annotated[
        str, typer.Argument(help="Session ID to send to")
    ],
    message: Annotated[
        str, typer.Option("--message", "-m", help="Message to send to the session")
    ],
) -> None:
    """Send a message to a session without attaching.

    The message is injected into the running chat REPL as if the user
    had typed it.  This lets you control multiple sessions in parallel
    from a single terminal.
    """
    manager = _get_session_manager()

    try:
        manager.send_message(session_id, message)
        console.print(f"[green]✓ Message sent to session {session_id}[/green]")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@sessions_app.command("logs")
def session_logs(
    session_id: Annotated[
        str, typer.Argument(help="Session ID to view logs for")
    ],
    lines: Annotated[
        int, typer.Option("--lines", "-l", help="Number of lines to show")
    ] = 50,
) -> None:
    """View the recent output from a session's terminal."""
    manager = _get_session_manager()

    try:
        output = manager.get_logs(session_id, lines=lines)
        console.print(
            Panel(output, title=f"Session {session_id} — Last {lines} lines", border_style="dim")
        )
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@sessions_app.command("kill")
def session_kill(
    session_id: Annotated[
        str, typer.Argument(help="Session ID to kill")
    ],
) -> None:
    """Terminate a running session and remove it."""
    manager = _get_session_manager()

    try:
        manager.kill_session(session_id)
        console.print(f"[green]✓ Session {session_id} killed[/green]")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@sessions_app.command("kill-all")
def session_kill_all() -> None:
    """Terminate all tracked sessions."""
    manager = _get_session_manager()
    count = manager.kill_all_sessions()
    if count == 0:
        console.print("[yellow]No sessions to kill.[/yellow]")
    else:
        console.print(f"[green]✓ Killed {count} session(s)[/green]")


if __name__ == "__main__":
    app()
