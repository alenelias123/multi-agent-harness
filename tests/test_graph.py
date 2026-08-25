import pytest

from agentcli.graph import (
    CycleDetectedError,
    GraphError,
    build_graph_from_planner_output,
    compute_execution_order,
    topological_sort,
)
from agentcli.schemas import Complexity, Task, TaskGraph, TaskType


class TestTaskGraph:
    def test_add_task(self) -> None:
        graph = TaskGraph(max_tasks=5)
        task = Task(id="t1", description="Test task")
        graph.add_task(task)
        assert len(graph.tasks) == 1
        assert graph.get_task("t1") == task

    def test_duplicate_task_id_raises(self) -> None:
        graph = TaskGraph(max_tasks=5)
        graph.add_task(Task(id="t1", description="Task 1"))
        with pytest.raises(ValueError, match="already exists"):
            graph.add_task(Task(id="t1", description="Task 2"))

    def test_max_tasks_limit(self) -> None:
        graph = TaskGraph(max_tasks=2)
        graph.add_task(Task(id="t1", description="Task 1"))
        graph.add_task(Task(id="t2", description="Task 2"))
        with pytest.raises(ValueError, match="Max task limit"):
            graph.add_task(Task(id="t3", description="Task 3"))

    def test_validate_dependencies_success(self) -> None:
        graph = TaskGraph(max_tasks=10)
        graph.add_task(Task(id="t1", description="Task 1"))
        graph.add_task(Task(id="t2", description="Task 2", depends_on=["t1"]))
        graph.validate_dependencies()

    def test_validate_dependencies_unknown_raises(self) -> None:
        graph = TaskGraph(max_tasks=10)
        graph.add_task(Task(id="t1", description="Task 1", depends_on=["t99"]))
        with pytest.raises(ValueError, match="unknown task"):
            graph.validate_dependencies()

    def test_detect_cycles_simple(self) -> None:
        graph = TaskGraph(max_tasks=10)
        graph.add_task(Task(id="t1", description="Task 1", depends_on=["t2"]))
        graph.add_task(Task(id="t2", description="Task 2", depends_on=["t1"]))
        with pytest.raises(ValueError, match="Circular dependency"):
            graph.detect_cycles()

    def test_detect_cycles_complex(self) -> None:
        graph = TaskGraph(max_tasks=10)
        graph.add_task(Task(id="t1", description="Task 1", depends_on=["t3"]))
        graph.add_task(Task(id="t2", description="Task 2", depends_on=["t1"]))
        graph.add_task(Task(id="t3", description="Task 3", depends_on=["t2"]))
        with pytest.raises(ValueError, match="Circular dependency"):
            graph.detect_cycles()

    def test_no_cycles_passes(self) -> None:
        graph = TaskGraph(max_tasks=10)
        graph.add_task(Task(id="t1", description="Task 1"))
        graph.add_task(Task(id="t2", description="Task 2", depends_on=["t1"]))
        graph.add_task(
            Task(id="t3", description="Task 3", depends_on=["t1", "t2"])
        )
        graph.detect_cycles()

    def test_ready_set_empty(self) -> None:
        graph = TaskGraph(max_tasks=10)
        graph.add_task(Task(id="t1", description="Task 1"))
        graph.add_task(Task(id="t2", description="Task 2", depends_on=["t1"]))
        ready = graph.ready_set(set())
        assert len(ready) == 1
        assert ready[0].id == "t1"

    def test_ready_set_after_completion(self) -> None:
        graph = TaskGraph(max_tasks=10)
        graph.add_task(Task(id="t1", description="Task 1"))
        graph.add_task(Task(id="t2", description="Task 2", depends_on=["t1"]))
        ready = graph.ready_set({"t1"})
        assert len(ready) == 1
        assert ready[0].id == "t2"

    def test_ready_set_excludes_completed(self) -> None:
        graph = TaskGraph(max_tasks=10)
        graph.add_task(Task(id="t1", description="Task 1"))
        ready = graph.ready_set({"t1"})
        assert len(ready) == 0

    def test_leaf_tasks(self) -> None:
        graph = TaskGraph(max_tasks=10)
        graph.add_task(Task(id="t1", description="Task 1"))
        graph.add_task(Task(id="t2", description="Task 2", depends_on=["t1"]))
        graph.add_task(Task(id="t3", description="Task 3", depends_on=["t1"]))
        leaves = graph.leaf_tasks()
        assert len(leaves) == 2
        assert {t.id for t in leaves} == {"t2", "t3"}


class TestBuildGraphFromPlannerOutput:
    def test_valid_graph(self) -> None:
        planner_json = {
            "tasks": [
                {
                    "id": "t1",
                    "description": "Task 1",
                    "depends_on": [],
                    "task_type": "planning",
                    "complexity": "low",
                },
                {
                    "id": "t2",
                    "description": "Task 2",
                    "depends_on": ["t1"],
                    "task_type": "coding",
                    "complexity": "medium",
                },
            ]
        }
        graph = build_graph_from_planner_output(planner_json)
        assert len(graph.tasks) == 2

    def test_missing_tasks_key_raises(self) -> None:
        with pytest.raises(GraphError, match="must contain a 'tasks' list"):
            build_graph_from_planner_output({})

    def test_invalid_task_type_defaults_to_general(self) -> None:
        planner_json = {
            "tasks": [
                {
                    "id": "t1",
                    "description": "Task 1",
                    "depends_on": [],
                    "task_type": "invalid",
                    "complexity": "low",
                }
            ]
        }
        graph = build_graph_from_planner_output(planner_json)
        assert graph.tasks[0].task_type == TaskType.GENERAL

    def test_invalid_complexity_defaults_to_medium(self) -> None:
        planner_json = {
            "tasks": [
                {
                    "id": "t1",
                    "description": "Task 1",
                    "depends_on": [],
                    "task_type": "planning",
                    "complexity": "invalid",
                }
            ]
        }
        graph = build_graph_from_planner_output(planner_json)
        assert graph.tasks[0].complexity == Complexity.MEDIUM

    def test_max_tasks_enforced(self) -> None:
        tasks = [
            {
                "id": f"t{i}",
                "description": f"Task {i}",
                "depends_on": [],
                "task_type": "general",
                "complexity": "low",
            }
            for i in range(25)
        ]
        planner_json = {"tasks": tasks}
        with pytest.raises(ValueError, match="Max task limit"):
            build_graph_from_planner_output(planner_json, max_tasks=20)


class TestTopologicalSort:
    def test_linear_chain(self) -> None:
        graph = TaskGraph(max_tasks=10)
        graph.add_task(Task(id="t1", description="Task 1"))
        graph.add_task(Task(id="t2", description="Task 2", depends_on=["t1"]))
        graph.add_task(Task(id="t3", description="Task 3", depends_on=["t2"]))
        sorted_tasks = topological_sort(graph)
        assert [t.id for t in sorted_tasks] == ["t1", "t2", "t3"]

    def test_diamond_dependencies(self) -> None:
        graph = TaskGraph(max_tasks=10)
        graph.add_task(Task(id="t1", description="Task 1"))
        graph.add_task(Task(id="t2", description="Task 2", depends_on=["t1"]))
        graph.add_task(Task(id="t3", description="Task 3", depends_on=["t1"]))
        graph.add_task(
            Task(id="t4", description="Task 4", depends_on=["t2", "t3"])
        )
        sorted_tasks = topological_sort(graph)
        ids = [t.id for t in sorted_tasks]
        assert ids[0] == "t1"
        assert ids[-1] == "t4"
        assert ids.index("t2") < ids.index("t4")
        assert ids.index("t3") < ids.index("t4")

    def test_cycle_raises(self) -> None:
        graph = TaskGraph(max_tasks=10)
        graph.add_task(Task(id="t1", description="Task 1", depends_on=["t2"]))
        graph.add_task(Task(id="t2", description="Task 2", depends_on=["t1"]))
        with pytest.raises(CycleDetectedError):
            topological_sort(graph)


class TestComputeExecutionOrder:
    def test_levels(self) -> None:
        graph = TaskGraph(max_tasks=10)
        graph.add_task(Task(id="t1", description="Task 1"))
        graph.add_task(Task(id="t2", description="Task 2", depends_on=["t1"]))
        graph.add_task(Task(id="t3", description="Task 3", depends_on=["t1"]))
        graph.add_task(
            Task(id="t4", description="Task 4", depends_on=["t2", "t3"])
        )
        levels = compute_execution_order(graph)
        assert len(levels) == 3
        assert [t.id for t in levels[0]] == ["t1"]
        assert {t.id for t in levels[1]} == {"t2", "t3"}
        assert [t.id for t in levels[2]] == ["t4"]
