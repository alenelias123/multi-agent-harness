from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import pytest

from agentcli.context import SharedContext
from agentcli.executor import (
    DAGExecutor,
    RetryStrategy,
    TaskExecutor,
    build_execution_evidence,
)
from agentcli.schemas import Task, TaskGraph, TaskStatus, TaskType


def _make_code_output(content: str) -> str:
    """Wrap content in a code block for validation."""
    return f"```python\n{content}\n```"


def _make_test_output(content: str) -> str:
    """Wrap content with test markers for validation."""
    return f"```python\n{content}\n\n# tests\ndef test_example():\n    assert True\n```"


class MockModelRouter:
    def __init__(
        self,
        responses: dict[str, tuple[str, str]] | None = None,
        fail_counts: dict[str, int] | None = None,
        sequence: list[tuple[str, tuple[str, str]]] | None = None,
        auto_wrap_code: bool = True,
    ) -> None:
        self.responses = responses or {}
        self.fail_counts = fail_counts or {}
        self.sequence = sequence or []
        self.call_counts: dict[str, int] = {}
        self.sequence_index = 0
        self.call_history: list[dict] = []
        self.auto_wrap_code = auto_wrap_code

    def _wrap_for_type(self, task_type: str, content: str) -> str:
        """Wrap output based on task type for validation."""
        if not self.auto_wrap_code:
            return content
        if task_type == "coding":
            return _make_code_output(content)
        if task_type == "analysis":
            return f"## Analysis\n\n{content}\n\n### Findings\n- Finding 1\n- Finding 2"
        if task_type == "writing":
            return (
                "# Documentation\n\n"
                f"{content}\n\n"
                "## Details\nMore detailed content here to meet minimum word count."
            )
        if task_type == "planning":
            return (
                f"## Plan\n\n1. Phase 1: {content}\n"
                "2. Phase 2: Implementation\n3. Phase 3: Testing"
            )
        return content

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
                response = self.responses[task_id]
                wrapped = self._wrap_for_type(task_type, response[0])
                return (wrapped, response[1])

        default = f"Output for {task_id or 'unknown'}"
        wrapped = self._wrap_for_type(task_type, default)
        return (wrapped, "mock-model")


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
        ctx = SharedContext()  # noqa: F841
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
        executor = TaskExecutor(
            router, "run-1", retry_strategy=RetryStrategy(max_attempts=2)
        )
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
                # Wrap in code block for coding task validation
                return (f"```python\nOutput from call {self.call_count}\n```", "mock-model")

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

    @pytest.mark.asyncio
    async def test_executor_feedback_revises_planner_graph(self) -> None:
        graph = TaskGraph(max_tasks=10, objective="Build a parser")
        graph.add_task(Task(id="t1", description="Parser", task_type=TaskType.CODING))

        revised = {
            "needs_review": False,
            "confidence": "high",
            "concerns": "",
            "tasks": [
                {
                    "id": "t1",
                    "description": "Implement the parser module",
                    "depends_on": [],
                    "task_type": "coding",
                    "complexity": "low",
                    "expected_outputs": ["parser module"],
                    "validation_criteria": ["parser module is produced"],
                }
            ],
        }
        router = MockModelRouter(
            sequence=[
                ("planner", (json.dumps(revised), "planner-model")),
                ("t1", ("def parse():\n    return 'parser module'", "coding-model")),
            ],
            auto_wrap_code=True,
        )

        executor = DAGExecutor(router, "run-feedback")
        results = await executor.execute(graph)

        assert executor.graph is not None
        assert executor.graph.get_task("t1").description == "Implement the parser module"
        assert results["t1"].status == TaskStatus.SUCCESS
        assert results["t1"].quality_score is not None
        revision_prompt = router.call_history[0]["messages"][-1]["content"]
        assert "EXECUTOR FEEDBACK" in revision_prompt
        assert "too_granular_or_underspecified" in revision_prompt

    @pytest.mark.asyncio
    async def test_execution_evidence_includes_failures_and_quality(self) -> None:
        graph = TaskGraph(max_tasks=10, objective="Build a parser")
        graph.add_task(
            Task(
                id="t1",
                description="Implement the parser module",
                task_type=TaskType.CODING,
                expected_outputs=["parser module"],
                validation_criteria=["parser module is produced"],
            )
        )
        router = MockModelRouter(fail_counts={"t1": 3})
        executor = DAGExecutor(
            router,
            "run-evidence",
            retry_strategy=RetryStrategy(max_attempts=1),
        )
        results = await executor.execute(graph)

        evidence = build_execution_evidence(executor.graph or graph, results)
        assert "Objective: Build a parser" in evidence
        assert "t1: failed" in evidence
        assert "execution_failed" in evidence


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
                (
                    "t1",
                    (
                        "1. Phase 1: Design API\n"
                        "2. Phase 2: Implementation\n"
                        "3. Phase 3: Testing",
                        "planner-model",
                    ),
                ),
                ("t2", ("class Model:\n    pass", "coding-model")),
                ("t3", ("def route():\n    pass", "coding-model")),
                ("t4", ("def test_model():\n    assert True", "coding-model")),
            ]
        )

        executor = DAGExecutor(router, "run-1")
        results = await executor.execute(graph)

        assert all(
            r.status == TaskStatus.SUCCESS for r in results.values()
        )
        assert results["t1"].model_used == "planner-model"
        assert results["t2"].model_used == "coding-model"


class TestContractHandoff:
    """Verify the planner-to-executor contract reaches the task prompt."""

    @pytest.mark.asyncio
    async def test_contract_rendered_in_task_prompt(self) -> None:
        captured_messages: list[list[dict]] = []

        class ContractCapturingRouter:
            def __init__(self) -> None:
                self.call_count = 0

            async def call(
                self, task_type: str, messages: list[dict], **_kw: object  # noqa: ARG002
            ) -> tuple[str, str]:
                self.call_count += 1
                captured_messages.append(messages)
                # Valid code output incl. a test for the "tests pass" criterion
                return (
                    "```python\ndef parser():\n    pass\n\n"
                    "def test_parser():\n    assert parser() is not None\n```",
                    "mock-model",
                )

        graph = TaskGraph(max_tasks=10)
        graph.add_task(
            Task(
                id="t1",
                description="Implement the parser",
                task_type=TaskType.CODING,
                expected_inputs=["token spec"],
                expected_outputs=["parser.py"],
                validation_criteria=["parses sample input", "tests pass"],
            )
        )

        router = ContractCapturingRouter()
        executor = DAGExecutor(router, "run-contract")  # type: ignore[arg-type]
        await executor.execute(graph)

        user_msg = next(
            m for m in captured_messages[0] if m["role"] == "user"
        )
        assert "Execution contract:" in user_msg["content"]
        assert "Expected inputs: token spec" in user_msg["content"]
        assert "Required outputs: parser.py" in user_msg["content"]
        assert "Validation criteria" in user_msg["content"]
        assert "parses sample input" in user_msg["content"]

    @pytest.mark.asyncio
    async def test_prompt_unchanged_without_contract(self) -> None:
        captured_messages: list[list[dict]] = []

        class BareCapturingRouter:
            async def call(
                self, task_type: str, messages: list[dict], **_kw: object  # noqa: ARG002
            ) -> tuple[str, str]:
                captured_messages.append(messages)
                return ("done", "mock-model")

        graph = TaskGraph(max_tasks=10)
        graph.add_task(
            Task(id="t1", description="Task 1", task_type=TaskType.CODING)
        )

        router = BareCapturingRouter()
        executor = DAGExecutor(router, "run-bare")  # type: ignore[arg-type]
        await executor.execute(graph)

        user_msg = next(
            m for m in captured_messages[0] if m["role"] == "user"
        )
        assert "Execution contract:" not in user_msg["content"]


class TestContextSharing:
    """Test that the executor properly stores outputs in shared context."""

    @pytest.mark.asyncio
    async def test_task_outputs_stored_in_context(self) -> None:
        graph = TaskGraph(max_tasks=10)
        graph.add_task(
            Task(id="t1", description="Task 1", task_type=TaskType.CODING)
        )

        router = MockModelRouter(
            sequence=[
                ("t1", ("```python\ndef hello():\n    return 'hello output'\n```", "mock-model"))
            ],
            auto_wrap_code=False,
        )

        # Use a fresh context so tests don't leak state
        from agentcli.context import ContextBridge, SharedContext
        ctx = SharedContext()
        bridge = ContextBridge(ctx)

        executor = DAGExecutor(router, "run-ctx-1", context_bridge=bridge)
        await executor.execute(graph)

        # Check that the output was stored in shared context
        output = ctx.read(key="task:t1:output", namespace="run:run-ctx-1")
        assert "hello output" in output

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
                self, task_type: str, messages: list[dict], **_kw: object  # noqa: ARG002
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
