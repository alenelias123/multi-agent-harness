from collections import deque
from typing import Any

from .schemas import Complexity, Task, TaskGraph, TaskType


class GraphError(Exception):
    pass


class CycleDetectedError(GraphError):
    pass


class MaxTasksExceededError(GraphError):
    pass


class UnknownDependencyError(GraphError):
    pass


def _coerce_str_list(value: object) -> list[str]:
    """Coerce a contract field into a list of strings, tolerantly.

    Accepts a proper list of strings, a single string, a comma-separated
    string, or None. Anything else becomes an empty list.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [p.strip() for p in value.split(",") if p.strip()]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                # Tolerate {"description": "..."} style entries
                text = str(
                    item.get("description") or item.get("text") or item.get("name") or ""
                ).strip()
                if text:
                # fall through to str() of a bare value
                    out.append(text)
            elif item is not None:
                text = str(item).strip()
                if text:
                    out.append(text)
        return out
    return []


def _coerce_str_dict(value: object) -> dict[str, str]:
    """Coerce an optional dictionary of strings (e.g. assumptions)."""
    if not isinstance(value, dict):
        return {}
    return {str(k): str(v) for k, v in value.items() if v is not None}


def build_graph_from_planner_output(
    planner_json: dict[str, Any],
    max_tasks: int = 20,
) -> TaskGraph:
    graph = TaskGraph(max_tasks=max_tasks)

    tasks_data = planner_json.get("tasks")
    if tasks_data is None:
        raise GraphError("Planner output must contain a 'tasks' list")
    if not isinstance(tasks_data, list):
        raise GraphError("Planner output must contain a 'tasks' list")

    for i, task_data in enumerate(tasks_data):
        if not isinstance(task_data, dict):
            raise GraphError(f"Task {i} must be an object")

        task_id = task_data.get("id") or f"t{i+1}"
        description = task_data.get("description")
        if not description:
            raise GraphError(f"Task {task_id} missing description")

        depends_on = task_data.get("depends_on", [])
        if not isinstance(depends_on, list):
            raise GraphError(f"Task {task_id} depends_on must be a list")

        task_type_str = task_data.get("task_type", "general").lower()
        try:
            task_type = TaskType(task_type_str)
        except ValueError:
            task_type = TaskType.GENERAL

        complexity_str = task_data.get("complexity", "medium").lower()
        try:
            complexity = Complexity(complexity_str)
        except ValueError:
            complexity = Complexity.MEDIUM

        task = Task(
            id=task_id,
            description=description,
            depends_on=depends_on,
            task_type=task_type,
            complexity=complexity,
            expected_inputs=_coerce_str_list(task_data.get("expected_inputs")),
            expected_outputs=_coerce_str_list(task_data.get("expected_outputs")),
            validation_criteria=_coerce_str_list(task_data.get("validation_criteria")),
        )
        graph.add_task(task)

    graph.validate_dependencies()
    graph.detect_cycles()

    return graph


def topological_sort(graph: TaskGraph) -> list[Task]:
    in_degree: dict[str, int] = {}
    adj: dict[str, list[str]] = {}

    for task in graph.tasks:
        in_degree[task.id] = 0
        adj[task.id] = []

    for task in graph.tasks:
        for dep_id in task.depends_on:
            adj[dep_id].append(task.id)
            in_degree[task.id] += 1

    queue = deque([tid for tid, deg in in_degree.items() if deg == 0])
    result = []

    while queue:
        node_id = queue.popleft()
        found_task = graph.get_task(node_id)
        if found_task is not None:
            result.append(found_task)
        for neighbor in adj[node_id]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(result) != len(graph.tasks):
        raise CycleDetectedError("Graph has cycles")

    return result


def compute_execution_order(graph: TaskGraph) -> list[list[Task]]:
    sorted_tasks = topological_sort(graph)
    levels: list[list[Task]] = []
    task_level: dict[str, int] = {}

    for task in sorted_tasks:
        level = 0 if not task.depends_on else max(task_level[dep] for dep in task.depends_on) + 1
        task_level[task.id] = level
        while len(levels) <= level:
            levels.append([])
        levels[level].append(task)

    return levels


def render_dag_ascii(
    graph: TaskGraph,
    statuses: dict[str, str] | None = None,
    completed: set[str] | None = None,
    failed: set[str] | None = None,
    running: str | None = None,
) -> str:
    """Render the task graph as an ASCII art DAG.

    Example output::

        ┌─ t1 ──────────────┐
        │ Design DB schema   │
        └────────┬───────────┘
           ┌─────┴─────┐
           ▼           ▼
        ┌─ t2 ─┐   ┌─ t3 ─┐
        │Models│   │Routes│
        └──┬───┘   └──┬───┘
           └────┬─────┘
                ▼
           ┌─ t5 ─┐
           │Tests │
           └──────┘

    Parameters
    ----------
    graph:
        The task graph to render.
    completed:
        Set of task IDs that have completed successfully.
    failed:
        Set of task IDs that have failed.
    running:
        The task ID currently running, or None.
    """
    completed = completed or set()
    failed = failed or set()
    statuses = statuses or {}

    if not graph.tasks:
        return "[empty graph]"

    levels = compute_execution_order(graph)

    # Build a mapping from task ID to its level and position
    task_level: dict[str, int] = {}
    for level_idx, level_tasks in enumerate(levels):
        for task in level_tasks:
            task_level[task.id] = level_idx

    # Build adjacency: parent -> children
    children: dict[str, list[str]] = {t.id: [] for t in graph.tasks}
    for task in graph.tasks:
        for dep_id in task.depends_on:
            children.setdefault(dep_id, []).append(task.id)

    # Assign column positions using a simple BFS layout
    task_col: dict[str, int] = {}
    col_counter = 0
    for level_tasks in levels:
        for task in level_tasks:
            task_col[task.id] = col_counter
            col_counter += 1

    # Status symbols
    def _symbol(task_id: str) -> str:
        if task_id in failed:
            return "✗"
        if task_id in completed:
            return "✓"
        if task_id == running:
            return "●"
        return "○"

    def _color(task_id: str) -> str:
        if task_id in failed:
            return "red"
        if task_id in completed:
            return "green"
        if task_id == running:
            return "yellow"
        return "dim"

    lines: list[str] = []
    lines.append("")

    # Render level by level
    for level_idx, level_tasks in enumerate(levels):
        # Node boxes
        box_lines: list[list[str]] = []
        for task in level_tasks:
            sym = _symbol(task.id)
            desc = task.description[:22]
            if len(task.description) > 22:
                desc += "…"
            box_lines.append([
                f"┌─ {task.id} ─{'─' * max(0, 18 - len(task.id))}┐",
                f"│ {sym} {desc:<20} │",
                f"└{'─' * (len(task.id) + 20)}┘",
            ])

        # Interleave boxes horizontally
        max_height = 3
        for row_idx in range(max_height):
            parts: list[str] = []
            for bl in box_lines:
                parts.append(bl[row_idx])
            lines.append("  ".join(parts))

        # Draw arrows to next level
        if level_idx < len(levels) - 1:
            next_tasks = levels[level_idx + 1]
            arrow_parts: list[str] = []
            for _ in level_tasks:
                arrow_parts.append("        │")
            lines.append("  ".join(arrow_parts))

            # Show branching
            has_branch = any(
                len([c for c in children.get(t.id, []) if c in {nt.id for nt in next_tasks}]) > 1
                for t in level_tasks
            )
            if has_branch:
                merge_parts: list[str] = []
                for t in level_tasks:
                    child_count = len(
                        [c for c in children.get(t.id, []) if c in {nt.id for nt in next_tasks}]
                    )
                    if child_count > 1:
                        merge_parts.append("    ┌───┴───┐")
                    else:
                        merge_parts.append("        │")
                lines.append("  ".join(merge_parts))

            # Down arrows
            down_parts: list[str] = []
            for _ in level_tasks:
                down_parts.append("        ▼")
            lines.append("  ".join(down_parts))

    lines.append("")

    # Legend
    lines.append("  ○ pending  ● running  ✓ done  ✗ failed")
    lines.append("")

    return "\n".join(lines)