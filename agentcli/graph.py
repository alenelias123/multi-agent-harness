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