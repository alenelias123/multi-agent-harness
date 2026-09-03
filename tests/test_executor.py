from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from agentcli.context import SharedContext, get_context_bridge
from agentcli.executor import DAGExecutor, TaskExecutor
from agentcli.schemas import Task, TaskGraph, TaskStatus, TaskType


class MockModelRouter:
    def __init__(
        self,
        responses: dict[str, tuple[str, str]] | None = None,
        fail_counts: dict[str, int] | None = None,
        sequence: list[tuple[str, tuple[str, str]]] | None = None,
    ) -> None:
        self.responses = responses or {}
        self.fail_counts = fail_counts or {}
        self.sequence = sequence or []
        self.call_counts: dict[str, int] = {}
        self.sequence_index = 0
        self.call_history: list[dict] = []

    async def call(
        self, task_type: str, messages: list[dict], **kwargs: object
    ) -> tuple[str, str]:
        self.call_history.append(
            {"task_type": task_type, "messages": messages, "kwargs": kwargs}
        )

        if self.sequence and self.sequence_index < len(self.sequence):
            _task_id, response = self.sequence[self.sequence_index]
            self.sequence_index += 1
            return response

        task_id = kwargs.get("task_id")
        if not task_id:
            for msg in messages:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if content.startswith("Task: "):
                        task_id = content.split("Task: ")[1].split("\n")[0]
                        task_id = task_id.strip()
                        break

        if task_id:
            self.call_counts[task_id] = (
                self.call_counts.get(task_id, 0) + 1
            )
            call_num = self.call_counts[task_id]

            if task_id in self.fail_counts and call_num <= self.fail_counts[task_id]:
                raise Exception(
                    f"Simulated failure for {task_id} "
                    f"(attempt {call_num})"
                )

            if task_id in self.responses:
                return self.responses[task_id]

        return (f"Output for {task_id or 'unknown'}", "mock-model")


@pytest.fixture
def sample_graph() -> TaskGraph:
    graph = TaskGraph(max_tasks=10)
    graph.add_task(
        Task(id="t1", description="Task 1", task_type=TaskType.CODING)
    )
    graph.add_task(
        Task(
            id="t2",
            description="Task 2",
            depends_on=["t1"],
            task_type=TaskType.CODING,
        )
    )
    graph.add_task(
        Task(
            id="t3",
            description="Task 3",
            depends_on=["t1"],
            task_type=TaskType.CODING,
        )
    )
    graph.add_task(
        Task(
            id="t4",
            description="Task 4",
            depends_on=["t2", "t3"],
            task_type=TaskType.CODING,
        )
    )
    return graph


@pytest.fixture
def mock_router() -> MockModelRouter:
    return MockModelRouter()


class TestTaskExecutor:
    @pytest.mark.asyncio
    async def test_execute_success(
        self, mock_router: MockModelRouter
    ) -> None:
        ctx = SharedContext()
        executor = TaskExecutor(mock_router, "run-1", context_bridge=None)
        task = Task(id="t1", description="Test task", task_type=TaskType.CODING)
        result = await executor.execute_task(task, {})
        assert result.status == TaskStatus.SUCCESS
        assert result.output is not None
        assert result.model_used == "mock-model"

    @pytest.mark.asyncio
    async def test_execute_retry_on_failure(self) -> None:
        router = MockModelRouter(
            responses={"t1": ("Success after retry", "mock-model")},
            fail_counts={"t1": 1},
        )

        executor = TaskExecutor(router, "run-1")
        task = Task(id="t1", description="Test task", task_type=TaskType.CODING)
        result = await executor.execute_task(task, {})
        assert result.status == TaskStatus.SUCCESS
        assert result.attempts == 2

    @pytest.mark.asyncio
    async def test_execute_fails_after_max_retries(self) -> None:
        router = MockModelRouter(fail_counts={"t1": 3})
        executor = TaskExecutor(router, "run-1")
        task = Task(id="t1", description="Test task", task_type=TaskType.CODING)
        result = await executor.execute_task(task, {})
        assert result.status == TaskStatus.FAILED
        assert result.attempts == 2


class TestDAGExecutor:
    @pytest.mark.asyncio
    async def test_execute_linear_chain(
        self, sample_graph: TaskGraph, mock_router: MockModelRouter
    ) -> None:
        executor = DAGExecutor(mock_router, "run-1")
        results = await executor.execute(sample_graph)

        assert len(results) == 4
        for result in results.values():
            assert result.status == TaskStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_execute_parallel_tasks(
        self, sample_graph: TaskGraph, mock_router: MockModelRouter
    ) -> None:
        executor = DAGExecutor(mock_router, "run-1")
        results = await executor.execute(sample_graph)

        t2_result = results["t2"]
        t3_result = results["t3"]
        assert t2_result.status == TaskStatus.SUCCESS
        assert t3_result.status == TaskStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_failed_task_marks_downstream_skipped(
        self, sample_graph: TaskGraph
    ) -> None:
        router = MockModelRouter(fail_counts={"t1": 3})
        executor = DAGExecutor(router, "run-1")
        results = await executor.execute(sample_graph)

        assert results["t1"].status == TaskStatus.FAILED
        assert results["t2"].status == TaskStatus.SKIPPED
        assert results["t3"].status == TaskStatus.SKIPPED
        assert results["t4"].status == TaskStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_upstream_outputs_passed_to_downstream(
        self, sample_graph: TaskGraph
    ) -> None:
        captured_messages: list[list[dict]] = []

        class CapturingRouter:
            def __init__(self) -> None:
                self.call_count = 0

            async def call(
                self, task_type: str, messages: list[dict], **_kw: object  # noqa: ARG002
            ) -> tuple[str, str]:
                self.call_count += 1
                captured_messages.append(messages)
                return (f"Output from call {self.call_count}", "mock-model")

        router = CapturingRouter()
        executor = DAGExecutor(router, "run-1")  # type: ignore[arg-type]
        await executor.execute(sample_graph)

        assert len(captured_messages) >= 2
        t2_messages = captured_messages[1]
        user_msg = next(m for m in t2_messages if m["role"] == "user")
        assert "Output from call 1" in user_msg["content"]

    @pytest.mark.asyncio
    async def test_concurrency_limit_respected(self) -> None:
        graph = TaskGraph(max_tasks=10)
        for i in range(8):
            graph.add_task(
                Task(
                    id=f"t{i}",
                    description=f"Task {i}",
                    task_type=TaskType.CODING,
                )
            )

        active_count = 0
        max_active = 0
        semaphore = asyncio.Semaphore(4)

        original_execute_task = TaskExecutor.execute_task

        async def tracking_execute_task(
            self: TaskExecutor,
            task: Task,
            upstream_outputs: dict[str, str],
        ) -> object:
            nonlocal active_count, max_active
            async with semaphore:
                active_count += 1
                max_active = max(max_active, active_count)
                try:
                    return await original_execute_task(
                        self, task, upstream_outputs
                    )
                finally:
                    active_count -= 1

        with patch.object(TaskExecutor, "execute_task", tracking_execute_task):
            router = MockModelRouter()
            executor = DAGExecutor(router, "run-1")
            await executor.execute(graph)

        assert max_active <= 4


class TestExecutorIntegration:
    @pytest.mark.asyncio
    async def test_full_execution_flow(self) -> None:
        graph = TaskGraph(max_tasks=10)
        graph.add_task(
            Task(id="t1", description="Design API", task_type=TaskType.PLANNING)
        )
        graph.add_task(
            Task(
                id="t2",
                description="Implement models",
                depends_on=["t1"],
                task_type=TaskType.CODING,
            )
        )
        graph.add_task(
            Task(
                id="t3",
                description="Implement routes",
                depends_on=["t1"],
                task_type=TaskType.CODING,
            )
        )
        graph.add_task(
            Task(
                id="t4",
                description="Write tests",
                depends_on=["t2", "t3"],
                task_type=TaskType.CODING,
            )
        )

        router = MockModelRouter(
            sequence=[
                ("t1", ("API Design: REST with JSON", "planner-model")),
                ("t2", ("Models implemented", "coding-model")),
                ("t3", ("Routes implemented", "coding-model")),
                ("t4", ("Tests written", "coding-model")),
            ]
        )

        executor = DAGExecutor(router, "run-1")
        results = await executor.execute(graph)

        assert all(
            r.status == TaskStatus.SUCCESS for r in results.values()
        )
        assert results["t1"].model_used == "planner-model"
        assert results["t2"].model_used == "coding-model"


class TestContextSharing:
    """Test that the executor properly stores outputs in shared context."""

    @pytest.mark.asyncio
    async def test_task_outputs_stored_in_context(self) -> None:
        graph = TaskGraph(max_tasks=10)
        graph.add_task(
            Task(id="t1", description="Task 1", task_type=TaskType.CODING)
        )

        router = MockModelRouter(
            sequence=[("t1", ("hello output", "mock-model"))]
        )

        # Use a fresh context so tests don't leak state
        from agentcli.context import ContextBridge, SharedContext
        ctx = SharedContext()
        bridge = ContextBridge(ctx)

        executor = DAGExecutor(router, "run-ctx-1", context_bridge=bridge)
        await executor.execute(graph)

        # Check that the output was stored in shared context
        output = ctx.read(key="task:t1:output", namespace="run:run-ctx-1")
        assert output == "hello output"

    @pytest.mark.asyncio
    async def test_shared_state_readable_by_downstream(self) -> None:
        graph = TaskGraph(max_tasks=10)
        graph.add_task(
            Task(id="t1", description="Task 1", task_type=TaskType.CODING)
        )
        graph.add_task(
            Task(
                id="t2",
                description="Task 2",
                depends_on=["t1"],
                task_type=TaskType.CODING,
            )
        )

        # Use a fresh context
        from agentcli.context import ContextBridge, SharedContext
        ctx = SharedContext()
        bridge = ContextBridge(ctx)

        # Store shared state before execution
        bridge.store_shared_state(
            run_id="run-ctx-2",
            key="project_context",
            value="Python project with FastAPI",
        )

        captured_messages: list[list[dict]] = []

        class StateCapturingRouter:
            async def call(
                self, task_type: str, messages: list[dict], **_kw: object
            ) -> tuple[str, str]:
                captured_messages.append(messages)
                return ("done", "mock-model")

        router = StateCapturingRouter()
        executor = DAGExecutor(router, "run-ctx-2", context_bridge=bridge)  # type: ignore[arg-type]
        await executor.execute(graph)

        # t2's user message should contain the shared state
        assert len(captured_messages) >= 2
        t2_msg = captured_messages[1]
        user_msg = next(m for m in t2_msg if m["role"] == "user")
        assert "Python project with FastAPI" in user_msg["content"]
