from __future__ import annotations

import asyncio
import contextlib
import signal
import sys
from datetime import datetime
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from .config import settings
from .executor import create_executor
from .model_router import ModelRouter, create_model_router
from .planner import Planner
from .schemas import Run, RunStatus, TaskGraph, TaskResult, TaskStatus
from .storage import storage

app = typer.Typer(
    name="agentcli",
    help="Multi-agent development CLI (Phase 1: single-user)",
    add_completion=False,
)
console = Console()


class GracefulExit(Exception):
    pass


def setup_signal_handlers() -> None:
    def handler(_signum: int, _frame: Any) -> None:
        raise GracefulExit()

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


async def run_task(task_description: str) -> Run:
    run = Run(
        user_id=settings.default_user_id,
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


@app.command()
def run(
    task: Annotated[str, typer.Argument(help="Natural language development task")],
) -> None:
    """Run a development task through the multi-agent pipeline."""
    setup_signal_handlers()
    try:
        run_result = asyncio.run(run_task(task))
    except GracefulExit:
        console.print("\n[yellow]Interrupted. Run marked as interrupted.[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        sys.exit(1)

    if run_result.status == RunStatus.COMPLETED:
        output = run_result.final_output or "No output"
        console.print(Panel(output, title="Final Result", border_style="green"))
    else:
        error = run_result.error or "Unknown error"
        console.print(Panel(error, title="Run Failed", border_style="red"))
        sys.exit(1)


@app.command()
def history(
    limit: Annotated[int, typer.Option("--limit", "-n", help="Number of runs to show")] = 20,
) -> None:
    """Show recent runs."""
    runs = storage.list_runs(user_id=settings.default_user_id, limit=limit)
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
) -> None:
    """Show details of a specific run."""
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


if __name__ == "__main__":
    app()