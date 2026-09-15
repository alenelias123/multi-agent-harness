"""Textual TUI dashboard for agentcli with inline plan editing.

Provides a full-screen interactive dashboard that mirrors the CLI:
  - Plan:      generate plans from natural language, edit tasks, approve & run
  - Logs:      live output from running tasks
  - History:   recent runs with full detail (task results + final output)
  - Chat:      interactive LLM chat with slash commands
  - Context:   shared context store management (set / search / clear)
  - Sessions:  create / send / logs / kill parallel tmux chat sessions
  - Config:    config paths + provider status

Usage:
    agentcli dashboard              # Launch dashboard
    agentcli dashboard --run <id>   # Focus on a specific run
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any, ClassVar

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Select,
    Static,
    TabbedContent,
)

# ── Status helpers ──────────────────────────────────────────────────────────

STATUS_ICONS = {
    "pending": ("○", "dim"),
    "ready": ("○", "dim"),
    "running": ("●", "yellow"),
    "success": ("✓", "green"),
    "failed": ("✗", "red"),
    "skipped": ("-", "yellow"),
    "interrupted": ("!", "blue"),
}

RUN_STATUS_STYLES = {
    "completed": "green",
    "failed": "red",
    "interrupted": "yellow",
    "running": "blue",
    "pending": "dim",
}


def _status_text(status: str) -> Text:
    icon, style = STATUS_ICONS.get(status, ("?", "white"))
    return Text(f"{icon} {status}", style=style)


# ── Task Graph Widget ──────────────────────────────────────────────────────


class TaskGraphView(Static):
    """ASCII DAG visualization of the task graph."""

    graph_text: reactive[str] = reactive(
        "No active task graph.\nPress [e] to enter edit mode and create a plan, "
        "or type a task above and press Generate Plan."
    )

    def render(self) -> Text:
        return Text(self.graph_text)


# ── Log Panel ──────────────────────────────────────────────────────────────


class LogPanel(RichLog):
    """Live log output panel."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs, highlight=True, markup=True)


# ── Edit Modal ─────────────────────────────────────────────────────────────


class TaskEditModal(Static):
    """Inline modal for editing a single task."""

    edit_visible: reactive[bool] = reactive(False)

    def compose(self) -> ComposeResult:
        with Vertical(id="edit-modal"):
            yield Label("Edit Task", id="edit-title")
            yield Input(placeholder="Task ID (e.g. t1)", id="edit-id")
            yield Input(placeholder="Description", id="edit-desc")
            yield Select(
                [
                    ("planning", "planning"),
                    ("coding", "coding"),
                    ("analysis", "analysis"),
                    ("writing", "writing"),
                    ("general", "general"),
                ],
                value="coding",
                id="edit-type",
                prompt="Type",
            )
            yield Select(
                [
                    ("low", "low"),
                    ("medium", "medium"),
                    ("high", "high"),
                ],
                value="medium",
                id="edit-complexity",
                prompt="Complexity",
            )
            yield Input(placeholder="Depends on (comma-separated, e.g. t1,t2)", id="edit-deps")
            with Horizontal(id="edit-buttons"):
                yield Static("[green]⏎[/green] Save  ", id="btn-save")
                yield Static("[red]esc[/red] Cancel", id="btn-cancel")


class ContextModal(Static):
    """Inline modal for writing a context entry."""

    def compose(self) -> ComposeResult:
        with Vertical(id="context-modal"):
            yield Label("Set Context", id="context-modal-title")
            yield Input(placeholder="Namespace (default: global)", id="ctx-namespace")
            yield Input(placeholder="Key (e.g. arch, decision, note)", id="ctx-key")
            yield Input(placeholder="Value", id="ctx-value")
            with Horizontal(id="context-modal-buttons"):
                yield Static("[green]⏎[/green] Save  ", id="btn-ctx-save")
                yield Static("[red]esc[/red] Cancel", id="btn-ctx-cancel")


# ── Main Dashboard ─────────────────────────────────────────────────────────


class Dashboard(App):
    """agentcli full-screen TUI dashboard.

    Tabs:
      - Plan:      task table + DAG visualization + edit controls + run
      - Logs:      live output from running tasks
      - History:   recent runs + run detail viewer
      - Chat:      interactive LLM chat
      - Context:   shared context store
      - Sessions:  session management
      - Config:    paths + provider status
    """

    TITLE = "agentcli dashboard"
    SUB_TITLE = "Multi-agent development CLI"

    CSS = """
    Screen {
        layout: vertical;
    }

    TabbedContent {
        height: 1fr;
    }

    TabPane {
        height: 1fr;
    }

    /* ── Plan tab ── */
    #plan-tab {
        height: 1fr;
    }

    #plan-bar {
        height: 3;
        dock: top;
        background: $surface;
        padding: 0 1;
    }

    #plan-bar Horizontal {
        height: 3;
        align: center middle;
    }

    #task-input {
        width: 1fr;
    }

    #plan-split {
        height: 1fr;
    }

    #task-table {
        width: 3fr;
        height: 1fr;
        border: solid $primary;
    }

    #dag-view {
        width: 2fr;
        height: 1fr;
        border: solid $accent;
        padding: 0 1;
        overflow-y: auto;
    }

    /* ── Edit modal ── */
    #edit-modal, #context-modal {
        layer: modal;
        dock: bottom;
        height: auto;
        max-height: 18;
        background: $surface;
        border: tall $accent;
        padding: 1 2;
        display: none;
    }

    #edit-modal.visible, #context-modal.visible {
        display: block;
    }

    #edit-title, #context-modal-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }

    #edit-modal Input, #context-modal Input {
        margin-bottom: 0;
    }

    #edit-modal Select {
        margin-bottom: 0;
    }

    #edit-buttons, #context-modal-buttons {
        margin-top: 1;
        height: 1;
    }

    /* ── Action bar ── */
    #action-bar {
        height: 3;
        dock: bottom;
        background: $surface;
        padding: 0 1;
    }

    #action-bar Horizontal {
        height: 3;
        align: center middle;
    }

    .action-btn {
        width: auto;
        height: 3;
        min-width: 20;
        content-align: center middle;
        text-align: center;
        margin: 0 1;
        padding: 0 2;
    }

    .small-btn {
        width: auto;
        height: 3;
        min-width: 10;
        content-align: center middle;
        text-align: center;
        margin: 0 1;
        padding: 0 2;
    }

    #btn-approve {
        background: $success;
        color: $text;
        text-style: bold;
    }

    #btn-cancel-plan {
        background: $error;
        color: $text;
    }

    /* ── Log tab ── */
    #log-tab {
        height: 1fr;
    }

    #log-panel {
        height: 1fr;
        border: solid $primary;
    }

    /* ── History tab ── */
    #history-tab {
        height: 1fr;
    }

    #history-bar {
        height: 3;
        background: $surface;
        padding: 0 1;
    }

    #history-table {
        height: 1fr;
        border: solid $primary;
    }

    #history-detail {
        height: 1fr;
        border: solid $accent;
        padding: 0 1;
        overflow-y: auto;
    }

    /* ── Chat tab ── */
    #chat-tab {
        height: 1fr;
    }

    #chat-log {
        height: 1fr;
        border: solid $primary;
    }

    #chat-input {
        height: 3;
        dock: bottom;
        background: $surface;
        padding: 0 1;
    }

    /* ── Context tab ── */
    #context-tab {
        height: 1fr;
    }

    #context-stats {
        height: 2;
        background: $surface;
        padding: 0 1;
    }

    #context-bar {
        height: 3;
        background: $surface;
        padding: 0 1;
    }

    #ctx-ns-filter {
        width: 20;
    }

    #ctx-search {
        width: 1fr;
    }

    #context-table {
        height: 1fr;
        border: solid $primary;
    }

    /* ── Sessions tab ── */
    #sessions-tab {
        height: 1fr;
    }

    #session-bar {
        height: 3;
        background: $surface;
        padding: 0 1;
    }

    #session-name {
        width: 16;
    }

    #session-msg {
        width: 1fr;
    }

    #session-list {
        height: 1fr;
        border: solid $primary;
    }

    #session-detail {
        height: 6;
        border: solid $accent;
        padding: 0 1;
        overflow-y: auto;
    }

    /* ── Config tab ── */
    #config-tab {
        height: 1fr;
    }

    #config-info {
        height: 1fr;
        border: solid $primary;
        padding: 1 2;
        overflow-y: auto;
    }

    /* ── Status bar ── */
    #status-bar {
        height: 1;
        dock: bottom;
        background: $surface;
        padding: 0 1;
    }
    """

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("e", "toggle_edit", "Edit Mode"),
        Binding("a", "add_task", "Add Task"),
        Binding("x", "delete_task", "Del Task"),
        Binding("d", "approve_plan", "Approve & Run", show=False),
        Binding("escape", "close_modal", "Close"),
        Binding("1", "show_tab('plan')", "Plan", show=False),
        Binding("2", "show_tab('logs')", "Logs", show=False),
        Binding("3", "show_tab('history')", "History", show=False),
        Binding("4", "show_tab('chat')", "Chat", show=False),
        Binding("5", "show_tab('context')", "Context", show=False),
        Binding("6", "show_tab('sessions')", "Sessions", show=False),
        Binding("7", "show_tab('config')", "Config", show=False),
    ]

    # Reactive state
    run_id: reactive[str | None] = reactive(None)
    edit_mode: reactive[bool] = reactive(False)
    editing_task_id: reactive[str | None] = reactive(None)

    # In-memory plan (list of dicts) — used in edit mode
    plan: reactive[list[dict[str, Any]]] = reactive(list)

    def __init__(self, run_id: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.run_id = run_id
        self._start_time = time.time()
        self._executing = False
        self._task_description = ""
        self._chat_session: Any = None
        self._chat_router_cm: Any = None
        self._chat_close_task: Any = None
        self._session_manager: Any = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        from textual.widgets import TabPane

        with TabbedContent(id="tabs"):
            # ── Plan tab ────────────────────────────────────────────
            with TabPane("Plan", id="plan"), Vertical(id="plan-tab"):
                with Horizontal(id="plan-bar"):
                    yield Input(placeholder="Describe a development task...", id="task-input")
                    yield Static("⚡ Generate Plan", id="btn-generate", classes="action-btn")
                with Horizontal(id="plan-split"):
                    yield DataTable(id="task-table")
                    yield TaskGraphView(id="dag-view")

            # ── Logs tab ────────────────────────────────────────────
            with TabPane("Logs", id="logs"), Vertical(id="log-tab"):
                yield LogPanel(id="log-panel")

            # ── History tab ─────────────────────────────────────────
            with TabPane("History", id="history"), Vertical(id="history-tab"):
                with Horizontal(id="history-bar"):
                    yield Static("↺ Refresh", id="btn-history-refresh", classes="small-btn")
                    yield Static("◔ Load into Plan", id="btn-history-load", classes="small-btn")
                yield DataTable(id="history-table")
                yield Static(id="history-detail")

            # ── Chat tab ────────────────────────────────────────────
            with TabPane("Chat", id="chat"), Vertical(id="chat-tab"):
                yield LogPanel(id="chat-log")
                yield Input(placeholder="Message... (/help for commands)", id="chat-input")

            # ── Context tab ─────────────────────────────────────────
            with TabPane("Context", id="context"), Vertical(id="context-tab"):
                yield Static(id="context-stats")
                with Horizontal(id="context-bar"):
                    yield Input(placeholder="Namespace filter", id="ctx-ns-filter")
                    yield Input(placeholder="Search...", id="ctx-search")
                    yield Static("Set", id="btn-ctx-set", classes="small-btn")
                    yield Static("Clear NS", id="btn-ctx-clear", classes="small-btn")
                    yield Static("↺ Refresh", id="btn-ctx-refresh", classes="small-btn")
                yield DataTable(id="context-table")

            # ── Sessions tab ────────────────────────────────────────
            with TabPane("Sessions", id="sessions"), Vertical(id="sessions-tab"):
                with Horizontal(id="session-bar"):
                    yield Input(placeholder="Name", id="session-name")
                    yield Static("Create", id="btn-session-create", classes="small-btn")
                    yield Input(placeholder="Message to send...", id="session-msg")
                    yield Static("Send", id="btn-session-send", classes="small-btn")
                    yield Static("Logs", id="btn-session-logs", classes="small-btn")
                    yield Static("Kill", id="btn-session-kill", classes="small-btn")
                    yield Static("Kill All", id="btn-session-killall", classes="small-btn")
                yield DataTable(id="session-list")
                yield Static(id="session-detail")

            # ── Config tab ──────────────────────────────────────────
            with TabPane("Config", id="config"), Vertical(id="config-tab"):
                yield Static(id="config-info")

        # ── Modals (hidden by default) ──────────────────────────────
        yield TaskEditModal(id="edit-modal")
        yield ContextModal(id="context-modal")

        # ── Action bar ──────────────────────────────────────────────
        with Horizontal(id="action-bar"):
            yield Static("[green]✓[/green] Approve & Run  ", id="btn-approve", classes="action-btn")
            yield Static("[red]✗[/red] Cancel Plan  ", id="btn-cancel-plan", classes="action-btn")

        yield Footer()

    def on_mount(self) -> None:
        """Initialize the dashboard on mount."""
        self.title = "agentcli dashboard"
        self.sub_title = f"Run: {self.run_id or 'new plan'}"

        # Set up tables
        table = self.query_one("#task-table", DataTable)
        table.add_columns("Status", "ID", "Description", "Type", "Complexity", "Depends On")
        table.cursor_type = "row"

        session_table = self.query_one("#session-list", DataTable)
        session_table.add_columns("Status", "ID", "Name", "Model", "tmux Window")
        session_table.cursor_type = "row"

        history_table = self.query_one("#history-table", DataTable)
        history_table.add_columns("Run ID", "Description", "Status", "Created", "Tasks")
        history_table.cursor_type = "row"

        ctx_table = self.query_one("#context-table", DataTable)
        ctx_table.add_columns("Namespace", "Key", "Source", "Tags", "Value (preview)")
        ctx_table.cursor_type = "row"

        # If we have a run, load its graph
        if self.run_id:
            self._load_run_graph()
        else:
            # Start in edit mode for new plans
            self.edit_mode = True
            self._refresh_plan_table()

        self._refresh_history_table()
        self._refresh_context_table()
        self._refresh_sessions_table()
        self._refresh_config_info()
        self._append_log(
            "Welcome to agentcli dashboard. Type a task and press Generate Plan to start."
        )

        # Start background monitor for live runs
        self._start_monitor()

    # ── Data loading helpers ──────────────────────────────────────────────

    def _load_run_graph(self, run_id: str | None = None) -> None:
        """Load task graph from an existing run."""
        rid = run_id or self.run_id
        if not rid:
            return
        try:
            from .storage import storage

            run = storage.get_run(rid)
            if run and run.task_graph:
                results = storage.get_task_results(rid)
                result_map = {r.task_id: r.status.value for r in results}
                self.plan = [
                    {
                        "id": t.id,
                        "description": t.description,
                        "task_type": t.task_type.value,
                        "complexity": t.complexity.value,
                        "depends_on": t.depends_on,
                        "status": result_map.get(t.id, "pending"),
                    }
                    for t in run.task_graph.tasks
                ]
                self.run_id = rid
                self._refresh_plan_table()
                self._refresh_dag()
        except Exception:
            pass

    def _refresh_plan_table(self) -> None:
        """Refresh the task table from self.plan."""
        table = self.query_one("#task-table", DataTable)
        table.clear()

        for task in self.plan:
            deps = ", ".join(task.get("depends_on", [])) or "—"
            status = task.get("status", "pending")
            table.add_row(
                _status_text(status),
                task["id"],
                task["description"][:60],
                task.get("task_type", "general"),
                task.get("complexity", "medium"),
                deps,
            )

    def _build_graph(self) -> Any:
        """Build a TaskGraph from the current plan."""
        from .schemas import Complexity, Task, TaskGraph, TaskType

        graph = TaskGraph(max_tasks=20)
        for t in self.plan:
            try:
                task_type = TaskType(t.get("task_type", "general"))
            except ValueError:
                task_type = TaskType.GENERAL
            try:
                complexity = Complexity(t.get("complexity", "medium"))
            except ValueError:
                complexity = Complexity.MEDIUM

            graph.add_task(
                Task(
                    id=t["id"],
                    description=t["description"],
                    depends_on=t.get("depends_on", []),
                    task_type=task_type,
                    complexity=complexity,
                )
            )
        return graph

    def _refresh_dag(self) -> None:
        """Refresh the DAG visualization from self.plan + live statuses."""
        dag_view = self.query_one("#dag-view", TaskGraphView)

        if not self.plan:
            dag_view.graph_text = (
                "No tasks yet. Type a task above and press Generate Plan, "
                "or press [a] to add a task manually."
            )
            return

        try:
            from .graph import render_dag_ascii

            graph = self._build_graph()

            completed = {t["id"] for t in self.plan if t.get("status") == "success"}
            failed = {t["id"] for t in self.plan if t.get("status") == "failed"}
            running = next(
                (t["id"] for t in self.plan if t.get("status") == "running"), None
            )

            dag_view.graph_text = render_dag_ascii(
                graph, completed=completed, failed=failed, running=running
            )

        except Exception as e:
            dag_view.graph_text = f"Error rendering DAG: {e}"

    # ── Plan generation & execution ───────────────────────────────────────

    def action_generate_plan(self) -> None:
        """Generate a plan from the task input using the planner."""
        desc = self.query_one("#task-input", Input).value.strip()
        if not desc:
            self.notify("Enter a task description first.", severity="warning")
            return
        self._task_description = desc
        self._generate_plan_worker(desc)

    @work(exclusive=True, group="plan")
    async def _generate_plan_worker(self, desc: str) -> None:
        """Background planner call."""
        from .model_router import create_model_router
        from .planner import Planner

        self._append_log(f"[bold cyan]Planning:[/bold cyan] {desc}")
        try:
            async with create_model_router() as router:
                planner = Planner(router)
                graph = await planner.plan(desc)
        except Exception as e:
            self.notify(f"Planning failed: {e}", severity="error")
            self._append_log(f"[red]Planning failed: {e}[/red]")
            return

        self.plan = [
            {
                "id": t.id,
                "description": t.description,
                "task_type": t.task_type.value,
                "complexity": t.complexity.value,
                "depends_on": t.depends_on,
                "status": "pending",
            }
            for t in graph.tasks
        ]
        self._refresh_plan_table()
        self._refresh_dag()
        self.notify(
            f"Plan created with {len(graph.tasks)} tasks. Press [d] to run.",
            severity="information",
        )
        self._append_log(f"[green]Plan created with {len(graph.tasks)} tasks.[/green]")

    def action_approve_plan(self) -> None:
        """Approve the plan and start execution in-app."""
        if not self.plan:
            self.notify("No tasks in the plan yet.", severity="warning")
            return
        if self._executing:
            self.notify("A run is already in progress.", severity="warning")
            return
        self._run_worker()

    @work(exclusive=True, group="run")
    async def _run_worker(self) -> None:
        """Execute the current plan with live status updates."""
        from .config import get_settings
        from .executor import create_executor
        from .model_router import create_model_router
        from .schemas import Run, RunStatus, TaskStatus
        from .storage import storage

        self._executing = True
        try:
            graph = self._build_graph()
            task_desc = self._task_description or "Manually built plan"
            run = Run(
                user_id=get_settings().default_user_id,
                task_description=task_desc,
                status=RunStatus.RUNNING,
            )
            storage.save_run(run)
            self.run_id = run.run_id
            self.sub_title = f"Running: {run.run_id}"

            # Mark all tasks pending
            for t in self.plan:
                t["status"] = "pending"
            self._refresh_plan_table()
            self._refresh_dag()

            self._append_log(f"[bold cyan]Run started:[/bold cyan] {run.run_id}")
            self.notify(f"Run started: {run.run_id}", severity="information")

            async with create_model_router() as router:
                async with create_executor(router, run.run_id) as executor:
                    exec_task = asyncio.create_task(executor.execute(graph))
                    while not exec_task.done():
                        await asyncio.sleep(0.3)
                        self._sync_executor_state(executor)
                    results = await exec_task
                    self._sync_executor_state(executor)

                # Aggregate final output
                failed_count = sum(
                    1 for r in results.values() if r.status == TaskStatus.FAILED
                )
                run.results = list(results.values())
                run.completed_at = datetime.utcnow()
                if failed_count > 0:
                    run.status = RunStatus.FAILED
                    run.error = f"{failed_count} task(s) failed"
                else:
                    run.status = RunStatus.COMPLETED
                    try:
                        run.final_output = await self._aggregate_results(
                            graph, results, router
                        )
                    except Exception as e:
                        self._append_log(f"[yellow]Aggregation failed: {e}[/yellow]")

                storage.save_run(run)
                for result in results.values():
                    task = graph.get_task(result.task_id)
                    if task:
                        storage.save_task_result(
                            run.run_id, result, task.description, task.depends_on
                        )

            self.sub_title = f"Run: {self.run_id or 'new plan'}"
            self._show_final_result(run)
            self._refresh_history_table()
            self.notify(f"Run {run.status.value}: {run.run_id}", severity="information")
        except Exception as e:
            self.notify(f"Run failed: {e}", severity="error")
            self._append_log(f"[red]Run failed: {e}[/red]")
        finally:
            self._executing = False

    def _sync_executor_state(self, executor: Any) -> None:
        """Merge live executor results into the plan table / DAG."""
        changed = False
        for tid, result in executor.results.items():
            task = next((t for t in self.plan if t["id"] == tid), None)
            if not task:
                continue
            status = result.status.value
            if task.get("status") != status:
                task["status"] = status
                changed = True

            # Stream output to logs once per task
            if result.status.value == "success" and not task.get("_streamed") and result.output:
                task["_streamed"] = True
                self._stream_output(tid, result)
            elif result.status.value == "failed" and not task.get("_streamed") and result.error:
                task["_streamed"] = True
                self._append_log(f"[red][{tid}] ERROR: {result.error[:300]}[/red]")
        if changed:
            self._refresh_plan_table()
            self._refresh_dag()

    def _stream_output(self, task_id: str, result: Any) -> None:
        """Write a task's output to the live log panel."""
        if not result.output:
            return
        output = result.output
        lines = output.split("\n")
        self._append_log(
            f"[bold green]── {task_id} output (on {result.model_used or '?'}) ──[/bold green]"
        )
        if len(lines) > 20:
            self._append_log("\n".join(lines[:20]))
            self._append_log(f"[dim]... ({len(lines) - 20} more lines)[/dim]")
        else:
            self._append_log(output)
        self._append_log("")

    async def _aggregate_results(self, graph: Any, results: dict[str, Any], router: Any) -> str:
        """Combine leaf task outputs into a single final deliverable."""
        from .schemas import TaskStatus

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
            return str(response)
        except Exception as e:
            self._append_log(
                f"[yellow]Aggregation LLM failed, using concatenated output: {e}[/yellow]"
            )
            return combined

    def _show_final_result(self, run: Any) -> None:
        """Display the final run output in the Logs tab."""
        self._append_log(f"\n[bold green]═══ Final Result ({run.status.value}) ═══[/bold green]")
        if run.final_output:
            self._append_log(run.final_output)
        elif run.error:
            self._append_log(f"[red]{run.error}[/red]")
        else:
            self._append_log("[dim]No output.[/dim]")
        self._append_log("[bold green]═══ End of run ═══[/bold green]\n")
        self.query_one(TabbedContent).active = "logs"

    # ── Edit mode actions ───────────────────────────────────────────────

    def action_toggle_edit(self) -> None:
        """Toggle edit mode on/off."""
        self.edit_mode = not self.edit_mode
        if self.edit_mode:
            self.sub_title = "EDIT MODE — [a]dd [x]del [⏎]edit [d]approve"
            modal = self.query_one("#edit-modal")
            modal.add_class("visible")
        else:
            self.sub_title = f"Run: {self.run_id or 'view'}"
            modal = self.query_one("#edit-modal")
            modal.remove_class("visible")

    def action_add_task(self) -> None:
        """Open the edit modal pre-filled for adding a new task."""
        if not self.edit_mode:
            return

        existing_ids = {t["id"] for t in self.plan}
        next_num = 1
        while f"t{next_num}" in existing_ids:
            next_num += 1
        new_id = f"t{next_num}"

        modal = self.query_one("#edit-modal")
        modal.add_class("visible")

        self.query_one("#edit-id", Input).value = new_id
        self.query_one("#edit-desc", Input).value = ""
        self.query_one("#edit-desc", Input).focus()
        self.query_one("#edit-deps", Input).value = ""

        self.editing_task_id = None  # None = adding new

    def action_delete_task(self) -> None:
        """Delete the currently selected task."""
        if not self.edit_mode or not self.plan:
            return

        table = self.query_one("#task-table", DataTable)
        cursor_row = table.cursor_row

        if cursor_row < 0 or cursor_row >= len(self.plan):
            return

        task_id = self.plan[cursor_row]["id"]
        self.plan = [t for t in self.plan if t["id"] != task_id]

        for t in self.plan:
            t["depends_on"] = [d for d in t.get("depends_on", []) if d != task_id]

        self._refresh_plan_table()
        self._refresh_dag()

    def action_close_modal(self) -> None:
        """Close any open modal."""
        for modal_id in ("#edit-modal", "#context-modal"):
            modal = self.query_one(modal_id)
            modal.remove_class("visible")
        self.editing_task_id = None

    # ── Event handlers ──────────────────────────────────────────────────

    @on(DataTable.RowSelected, "#task-table")
    def on_task_selected(self, event: DataTable.RowSelected) -> None:
        """When a row is selected, load it into the edit modal."""
        if not self.edit_mode:
            return

        row_idx = event.cursor_row
        if row_idx < 0 or row_idx >= len(self.plan):
            return

        task = self.plan[row_idx]
        self.editing_task_id = task["id"]

        self.query_one("#edit-id", Input).value = task["id"]
        self.query_one("#edit-desc", Input).value = task["description"]
        self.query_one("#edit-deps", Input).value = ", ".join(task.get("depends_on", []))

        type_select = self.query_one("#edit-type", Select)
        for option in type_select._options:
            if option[1] == task.get("task_type", "general"):
                type_select.value = option[1]
                break

        complexity_select = self.query_one("#edit-complexity", Select)
        for option in complexity_select._options:
            if option[1] == task.get("complexity", "medium"):
                complexity_select.value = option[1]
                break

        modal = self.query_one("#edit-modal")
        modal.add_class("visible")

    @on(Input.Submitted, "#edit-desc")
    def on_desc_submitted(self, _event: Input.Submitted) -> None:
        """Save the task when description is submitted."""
        self._save_task_from_modal()

    @on(Input.Submitted, "#edit-deps")
    def on_deps_submitted(self, _event: Input.Submitted) -> None:
        """Save the task when deps is submitted."""
        self._save_task_from_modal()

    @on(Input.Submitted, "#task-input")
    def on_task_input_submitted(self, _event: Input.Submitted) -> None:
        """Generate a plan when the user hits Enter on the task input."""
        self.action_generate_plan()

    def _save_task_from_modal(self) -> None:
        """Read the modal fields and save/update the task in self.plan."""
        task_id = self.query_one("#edit-id", Input).value.strip()
        desc = self.query_one("#edit-desc", Input).value.strip()
        task_type = self.query_one("#edit-type", Select).value or "general"
        complexity = self.query_one("#edit-complexity", Select).value or "medium"
        deps_raw = self.query_one("#edit-deps", Input).value.strip()
        deps = [d.strip() for d in deps_raw.split(",") if d.strip()] if deps_raw else []

        if not task_id or not desc:
            self.notify("Task ID and description are required.", severity="warning")
            return

        existing = next((t for t in self.plan if t["id"] == task_id), None)

        task_data = {
            "id": task_id,
            "description": desc,
            "task_type": task_type,
            "complexity": complexity,
            "depends_on": deps,
            "status": existing.get("status", "pending") if existing else "pending",
        }
        if existing and existing.get("_streamed"):
            task_data["_streamed"] = True

        if existing:
            idx = self.plan.index(existing)
            self.plan[idx] = task_data
        else:
            self.plan.append(task_data)

        self._refresh_plan_table()
        self._refresh_dag()

        modal = self.query_one("#edit-modal")
        modal.remove_class("visible")
        self.editing_task_id = None

    # ── History tab ─────────────────────────────────────────────────────

    def _refresh_history_table(self) -> None:
        """Reload the history table from storage."""
        from .config import get_settings
        from .storage import storage

        table = self.query_one("#history-table", DataTable)
        table.clear()

        runs = storage.list_runs(user_id=get_settings().default_user_id, limit=50)
        for run in runs:
            style = RUN_STATUS_STYLES.get(run.status.value, "white")
            task_count = len(run.task_graph.tasks) if run.task_graph else 0
            table.add_row(
                run.run_id,
                run.task_description[:40] + ("..." if len(run.task_description) > 40 else ""),
                Text(run.status.value, style=style),
                run.created_at.strftime("%Y-%m-%d %H:%M"),
                str(task_count),
            )

    @on(DataTable.RowSelected, "#history-table")
    def on_history_selected(self, event: DataTable.RowSelected) -> None:
        """Show full detail for the selected run."""
        if event.cursor_row < 0:
            return
        table = self.query_one("#history-table", DataTable)
        row = table.get_row_at(event.cursor_row)
        if not row:
            return
        self._show_run_detail(str(row[0]))

    def _show_run_detail(self, run_id: str) -> None:
        """Render run details (info, task results, final output)."""
        from .storage import storage

        detail = self.query_one("#history-detail", Static)
        run = storage.get_run(run_id)
        if not run:
            detail.update(f"[red]Run {run_id} not found.[/red]")
            return

        lines = [
            f"[bold cyan]Run ID:[/bold cyan] {run.run_id}",
            f"[bold]Description:[/bold] {run.task_description}",
            f"[bold]Status:[/bold] {run.status.value}",
            f"[bold]Created:[/bold] {run.created_at}",
        ]
        if run.error:
            lines.append(f"[bold red]Error:[/bold red] {run.error}")

        results = storage.get_task_results(run_id)
        if results:
            lines.append("")
            lines.append("[bold]Task Results:[/bold]")
            for r in results:
                style = {"success": "green", "failed": "red", "skipped": "yellow"}.get(
                    r.status.value, "white"
                )
                preview = (r.output or r.error or "")[:80]
                lines.append(
                    f"  [{style}]{r.task_id} {r.status.value}[/{style}] "
                    f"(model: {r.model_used or '—'}) {preview}"
                )

        if run.final_output:
            lines.append("")
            lines.append("[bold green]Final Output:[/bold green]")
            output = run.final_output
            if len(output) > 2000:
                output = output[:2000] + "\n[dim]... (truncated)[/dim]"
            lines.append(output)

        detail.update("\n".join(lines))

    # ── Chat tab ────────────────────────────────────────────────────────

    @on(Input.Submitted, "#chat-input")
    def on_chat_input(self, event: Input.Submitted) -> None:
        """Handle a chat message or slash command."""
        text = event.value.strip()
        self.query_one("#chat-input", Input).value = ""
        if not text:
            return
        if text.startswith("/"):
            self._chat_slash(text)
            return
        self._chat_send(text)

    def _chat_slash(self, text: str) -> None:
        """Handle chat slash commands."""
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        log = self.query_one("#chat-log", LogPanel)

        if cmd in ("/quit", "/exit"):
            self.notify("Chat closed. Press 1 for Plan or q to quit.", severity="information")
            log.write("[dim]Goodbye![/dim]")
        elif cmd == "/help":
            from .chat import SLASH_COMMANDS

            log.write("[bold]Commands:[/bold]")
            for name, desc in SLASH_COMMANDS.items():
                log.write(f"  [cyan]{name}[/cyan] — {desc}")
        elif cmd == "/clear":
            if self._chat_session:
                self._chat_session.clear_history()
            log.write("[green]Conversation cleared.[/green]")
        elif cmd == "/model":
            if self._chat_session and self._chat_session.current_model:
                log.write(f"[bold]Active model:[/bold] {self._chat_session.current_model}")
            else:
                log.write("[dim]No model used yet. Send a message to activate.[/dim]")
        elif cmd == "/history":
            if self._chat_session:
                count = self._chat_session.message_count()
                log.write(f"[bold]Messages:[/bold] {count} user messages")
            else:
                log.write("[dim]No messages yet.[/dim]")
        elif cmd == "/export":
            if self._chat_session and self._chat_session.messages:
                filename = args if args else "chat_export.md"
                lines = ["# agentcli chat export\n"]
                for msg in self._chat_session.messages:
                    role = msg["role"].capitalize()
                    lines.append(f"## {role}\n\n{msg['content']}\n")
                try:
                    from pathlib import Path

                    Path(filename).write_text("\n".join(lines), encoding="utf-8")
                    log.write(f"[green]Exported to {filename}[/green]")
                except Exception as e:
                    log.write(f"[red]Export failed: {e}[/red]")
            else:
                log.write("[yellow]No messages to export.[/yellow]")
        else:
            log.write(f"[red]Unknown command: {cmd}[/red]. Type /help for commands.")

    def _chat_send(self, text: str) -> None:
        """Queue a chat message to be sent to the LLM."""
        log = self.query_one("#chat-log", LogPanel)
        log.write(f"[bold green]You>[/bold green] {text}")
        self._chat_send_worker(text)

    @work(exclusive=True, group="chat")
    async def _chat_send_worker(self, text: str) -> None:
        """Send a chat message via the model router."""
        from .chat import ChatSession
        from .model_router import create_model_router

        log = self.query_one("#chat-log", LogPanel)
        log.write("[dim]Thinking...[/dim]")

        try:
            if self._chat_session is None:
                cm = create_model_router()
                router = await cm.__aenter__()
                self._chat_router_cm = cm
                self._chat_session = ChatSession(router)

            response = await self._chat_session.send(text)
            if self._chat_session.current_model:
                log.write(f"[dim]↳ {self._chat_session.current_model}[/dim]")
            log.write(response)
        except Exception as e:
            log.write(f"[bold red]Error:[/bold red] {e}")
            log.write(
                "[dim]Try again, or check your provider configuration in the Config tab.[/dim]"
            )

    # ── Context tab ─────────────────────────────────────────────────────

    def _get_context(self) -> Any:
        from .context import get_shared_context

        return get_shared_context()

    def _refresh_context_table(self) -> None:
        """Reload the context entries table."""
        ctx = self._get_context()
        table = self.query_one("#context-table", DataTable)
        table.clear()

        ns_filter = self.query_one("#ctx-ns-filter", Input).value.strip()
        search = self.query_one("#ctx-search", Input).value.strip()

        try:
            entries = ctx.read_all(namespace=ns_filter or None, limit=100)
            if search:
                entries = [
                    e for e in entries
                    if search.lower() in e.value.lower() or search.lower() in e.key.lower()
                ]
        except Exception:
            entries = []

        for entry in entries:
            value_preview = entry.value[:60] + ("..." if len(entry.value) > 60 else "")
            table.add_row(
                entry.namespace,
                entry.key,
                entry.source_agent or "—",
                ", ".join(entry.tags[:3]) or "—",
                value_preview,
            )

        try:
            stats = ctx.get_stats()
            stats_label = self.query_one("#context-stats", Static)
            stats_label.update(
                f"Entries: [green]{stats['total_entries']}[/green]  "
                f"Namespaces: [cyan]{stats['total_namespaces']}[/cyan]  "
                f"Tags: {', '.join(stats['unique_tags'][:8]) or '—'}"
            )
        except Exception:
            pass

    def action_context_set(self) -> None:
        """Open the context modal to write an entry."""
        modal = self.query_one("#context-modal")
        modal.add_class("visible")
        self.query_one("#ctx-key", Input).value = ""
        self.query_one("#ctx-value", Input).value = ""
        self.query_one("#ctx-namespace", Input).value = "global"
        self.query_one("#ctx-key", Input).focus()

    def _save_context_from_modal(self) -> None:
        """Read the context modal and write the entry."""
        namespace = self.query_one("#ctx-namespace", Input).value.strip() or "global"
        key = self.query_one("#ctx-key", Input).value.strip()
        value = self.query_one("#ctx-value", Input).value.strip()

        if not key or not value:
            self.notify("Key and value are required.", severity="warning")
            return

        try:
            self._get_context().write(key=key, value=value, namespace=namespace)
            self.notify(f"Stored [{namespace}] {key}", severity="information")
        except Exception as e:
            self.notify(f"Failed to store context: {e}", severity="error")

        self.action_close_modal()
        self._refresh_context_table()

    def action_context_clear(self) -> None:
        """Clear a namespace from the filter input."""
        ns = self.query_one("#ctx-ns-filter", Input).value.strip()
        if not ns:
            self.notify("Enter a namespace in the filter box to clear it.", severity="warning")
            return
        try:
            count = self._get_context().clear_namespace(ns)
            self.notify(f"Cleared {count} entries from '{ns}'", severity="information")
        except Exception as e:
            self.notify(f"Failed to clear: {e}", severity="error")
        self._refresh_context_table()

    # ── Sessions tab ────────────────────────────────────────────────────

    def _get_session_manager(self) -> Any | None:
        from .sessions import SessionManager, is_tmux_available

        if self._session_manager is None:
            if not is_tmux_available():
                self.notify("tmux is not available. Sessions require tmux.", severity="warning")
                return None
            self._session_manager = SessionManager()
        return self._session_manager

    def _refresh_sessions_table(self) -> None:
        """Reload the sessions table."""
        from .sessions import is_tmux_available

        table = self.query_one("#session-list", DataTable)
        table.clear()

        if not is_tmux_available():
            self.query_one("#session-detail", Static).update(
                "[yellow]tmux is not available.[/yellow] Install it with: "
                "sudo apt install tmux / brew install tmux"
            )
            return

        manager = self._get_session_manager()
        if manager is None:
            return

        try:
            sessions = manager.list_sessions()
            for s in sessions:
                status_style = "green" if s.status == "running" else "red"
                status_icon = "●" if s.status == "running" else "○"
                table.add_row(
                    Text(f"{status_icon} {s.status}", style=status_style),
                    s.session_id,
                    s.name,
                    s.model or "(default)",
                    s.tmux_window,
                )
        except Exception as e:
            self.query_one("#session-detail", Static).update(f"[red]{e}[/red]")

    def action_session_create(self) -> None:
        """Create a new session."""
        name = self.query_one("#session-name", Input).value.strip()
        manager = self._get_session_manager()
        if manager is None:
            return
        try:
            session = manager.create_session(name=name or None)
            self.notify(f"Session created: {session.session_id}", severity="information")
            self._refresh_sessions_table()
        except Exception as e:
            self.notify(f"Failed to create session: {e}", severity="error")

    def _selected_session_id(self) -> str | None:
        table = self.query_one("#session-list", DataTable)
        row = table.get_row_at(table.cursor_row) if table.cursor_row >= 0 else None
        if not row:
            self.notify("Select a session first.", severity="warning")
            return None
        return str(row[1])

    def action_session_send(self) -> None:
        """Send a message to the selected session."""
        session_id = self._selected_session_id()
        if not session_id:
            return
        msg = self.query_one("#session-msg", Input).value.strip()
        if not msg:
            self.notify("Enter a message to send.", severity="warning")
            return
        manager = self._get_session_manager()
        if manager is None:
            return
        try:
            manager.send_message(session_id, msg)
            self.notify(f"Message sent to {session_id}", severity="information")
            self.query_one("#session-msg", Input).value = ""
        except Exception as e:
            self.notify(f"Failed to send: {e}", severity="error")

    def action_session_logs(self) -> None:
        """Show recent output from the selected session."""
        session_id = self._selected_session_id()
        if not session_id:
            return
        manager = self._get_session_manager()
        if manager is None:
            return
        try:
            logs = manager.get_logs(session_id, lines=40)
            self.query_one("#session-detail", Static).update(
                f"[bold cyan]Session {session_id} logs:[/bold cyan]\n{logs}"
            )
        except Exception as e:
            self.notify(f"Failed to get logs: {e}", severity="error")

    def action_session_kill(self) -> None:
        """Kill the selected session."""
        session_id = self._selected_session_id()
        if not session_id:
            return
        manager = self._get_session_manager()
        if manager is None:
            return
        try:
            manager.kill_session(session_id)
            self.notify(f"Session {session_id} killed", severity="information")
            self._refresh_sessions_table()
        except Exception as e:
            self.notify(f"Failed to kill: {e}", severity="error")

    def action_session_killall(self) -> None:
        """Kill all sessions."""
        manager = self._get_session_manager()
        if manager is None:
            return
        try:
            count = manager.kill_all_sessions()
            self.notify(f"Killed {count} session(s)", severity="information")
            self._refresh_sessions_table()
        except Exception as e:
            self.notify(f"Failed: {e}", severity="error")

    # ── Config tab ──────────────────────────────────────────────────────

    def _refresh_config_info(self) -> None:
        """Render config paths and provider status."""
        from platformdirs import user_config_dir, user_data_dir

        from ._version import __version__
        from .config import APP_NAME, _find_dotenv, settings

        config_dir = user_config_dir(APP_NAME)
        data_dir = user_data_dir(APP_NAME)
        env_file = _find_dotenv()
        db_path = settings.db_path

        provider_lines = [
            (
                "[green]✓ configured[/green]" if settings.openrouter_api_key
                else "[dim]not set[/dim]"
            ),
            (
                "[green]✓ enabled[/green]" if settings.freebuff_enabled
                else "[dim]disabled[/dim]"
            ),
            (
                "[green]✓ configured[/green]" if settings.opencode_api_key
                else "[dim]not set[/dim]"
            ),
        ]

        info = (
            f"[bold]agentcli {__version__}[/bold]\n\n"
            f"[bold cyan]Paths[/bold cyan]\n"
            f"  Config dir:  {config_dir}\n"
            f"  Data dir:    {data_dir}\n"
            f"  Database:    {db_path}\n"
            f"  Env file:    {env_file or '(none found — run: agentcli config init)'}\n\n"
            f"[bold cyan]Providers[/bold cyan]\n"
            f"  OpenRouter:  {provider_lines[0]}\n"
            f"  Freebuff:    {provider_lines[1]}\n"
            f"  OpenCode:    {provider_lines[2]}\n\n"
            f"[dim]Provider setup hints:[/dim]\n"
            f"  OpenRouter: set OPENROUTER_API_KEY in your .env (https://openrouter.ai/keys)\n"
            f"  Freebuff:   run [bold]agentcli setup[/bold] then set FREEBUFF_ENABLED=true\n"
            f"  OpenCode:   run [bold]agentcli opencode[/bold] to configure interactively\n\n"
            f"[dim]Max parallelism:[/dim] {settings.max_parallelism}\n"
            f"[dim]Max tasks:[/dim] {settings.max_tasks}\n"
            f"[dim]Task retries:[/dim] {settings.task_max_retries}"
        )
        self.query_one("#config-info", Static).update(info)

    # ── Logging helper ──────────────────────────────────────────────────

    def _append_log(self, message: str) -> None:
        """Append a line to the live log panel."""
        log = self.query_one("#log-panel", LogPanel)
        log.write(message)

    # ── Button clicks ───────────────────────────────────────────────────

    def on_click(self, event: Any) -> None:
        """Handle click events on action buttons."""
        target = getattr(event, "control", None)
        if target is None:
            return
        target_id = getattr(target, "id", "")
        handlers = {
            "btn-approve": self.action_approve_plan,
            "btn-cancel-plan": self.action_cancel_plan,
            "btn-generate": self.action_generate_plan,
            "btn-history-refresh": self._refresh_history_table,
            "btn-history-load": self.action_history_load,
            "btn-ctx-set": self.action_context_set,
            "btn-ctx-clear": self.action_context_clear,
            "btn-ctx-refresh": self._refresh_context_table,
            "btn-session-create": self.action_session_create,
            "btn-session-send": self.action_session_send,
            "btn-session-logs": self.action_session_logs,
            "btn-session-kill": self.action_session_kill,
            "btn-session-killall": self.action_session_killall,
            "btn-save": self._save_task_from_modal,
            "btn-cancel": self.action_close_modal,
            "btn-ctx-save": self._save_context_from_modal,
            "btn-ctx-cancel": self.action_close_modal,
        }
        handler = handlers.get(target_id)
        if handler:
            handler()

    def action_cancel_plan(self) -> None:
        """Cancel the current plan (clear tasks)."""
        if self._executing:
            self.notify(
                "A run is in progress — press q to quit after it finishes.", severity="warning"
            )
            return
        self.plan = []
        self._task_description = ""
        self._refresh_plan_table()
        self._refresh_dag()
        self.notify("Plan cancelled.", severity="warning")

    def action_history_load(self) -> None:
        """Load the selected history run into the Plan tab."""
        table = self.query_one("#history-table", DataTable)
        row = table.get_row_at(table.cursor_row) if table.cursor_row >= 0 else None
        if not row:
            self.notify("Select a run in the history table first.", severity="warning")
            return
        run_id = str(row[0])
        self._load_run_graph(run_id)
        self.notify(f"Loaded run {run_id} into Plan. Press [d] to re-run.", severity="information")
        self.query_one(TabbedContent).active = "plan"

    # ── Background monitor ──────────────────────────────────────────────

    def _start_monitor(self) -> None:
        """Start the background task monitoring loop."""
        self._monitor_task()

    @work(exclusive=True, group="monitor")
    async def _monitor_task(self) -> None:
        """Background loop that updates the dashboard."""
        while True:
            await asyncio.sleep(1.0)
            await self._update_live_state()

    async def _update_live_state(self) -> None:
        """Fetch latest state from storage for live runs."""
        try:
            from .storage import storage

            # Refresh live run statuses from storage (external runs)
            if self.run_id and not self._executing:
                results = storage.get_task_results(self.run_id)
                if results:
                    result_map = {r.task_id: r.status.value for r in results}
                    changed = False
                    for t in self.plan:
                        status = result_map.get(t["id"])
                        if status and t.get("status") != status:
                            t["status"] = status
                            changed = True
                    if changed:
                        self._refresh_plan_table()
                        self._refresh_dag()

            # Refresh sessions table
            tabs = self.query_one(TabbedContent)
            if tabs.active == "sessions":
                self._refresh_sessions_table()
            elif tabs.active == "history":
                self._refresh_history_table()
            elif tabs.active == "context":
                self._refresh_context_table()
        except Exception:
            pass

    # ── Navigation ──────────────────────────────────────────────────────

    def action_show_tab(self, tab_id: str) -> None:
        """Switch to a specific tab."""
        self.query_one(TabbedContent).active = tab_id

    def action_refresh(self) -> None:
        """Force refresh the dashboard."""
        self._refresh_plan_table()
        self._refresh_dag()
        self._refresh_history_table()
        self._refresh_context_table()
        self._refresh_sessions_table()
        self._refresh_config_info()

    def on_unmount(self) -> None:
        """Clean up resources when the app closes."""
        cm = self._chat_router_cm
        if cm is not None:
            self._chat_router_cm = None

            async def _close() -> None:
                await cm.__aexit__(None, None, None)

            import contextlib

            with contextlib.suppress(Exception):
                self._chat_close_task = asyncio.create_task(_close())

    async def action_quit(self) -> None:
        """Quit the dashboard."""
        self.exit(None)


# ── Standalone entry point ─────────────────────────────────────────────────


def run_dashboard(run_id: str | None = None) -> object | None:
    """Launch the TUI dashboard."""
    app = Dashboard(run_id=run_id)
    result: object | None = app.run()

    # If the user approved a plan, return it
    return result