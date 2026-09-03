import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from .config import settings
from .context import ContextBridge, SharedContext, get_context_bridge, get_shared_context
from .model_router import ModelRouter
from .schemas import Task, TaskGraph, TaskResult, TaskStatus

logger = logging.getLogger(__name__)


@dataclass
class ExecutionContext:
    task: Task
    upstream_outputs: dict[str, str]
    run_id: str


class TaskExecutor:
    def __init__(self, router: ModelRouter, run_id: str,
                 context_bridge: ContextBridge | None = None):
        self.router = router
        self.run_id = run_id
        self.semaphore = asyncio.Semaphore(settings.max_parallelism)
        self.context_bridge = context_bridge or get_context_bridge()

    async def execute_task(self, task: Task, upstream_outputs: dict[str, str]) -> TaskResult:
        async with self.semaphore:
            # Gather additional context from shared store
            shared_context = await self._gather_shared_context(task, upstream_outputs)

            # Merge upstream outputs with shared context
            merged_upstream = {**upstream_outputs, **shared_context}

            return await self._execute_with_retries(task, merged_upstream)

    async def _gather_shared_context(
        self, task: Task, upstream_outputs: dict[str, str]
    ) -> dict[str, str]:
        """Read relevant context from the shared store for this task."""
        extra: dict[str, str] = {}

        try:
            # Get all completed task outputs from this run (may include
            # tasks from other parallel branches)
            all_outputs = self.context_bridge.get_all_task_outputs(self.run_id)
            for task_id, output in all_outputs.items():
                # Skip duplicates with already-passed upstream_outputs
                if task_id not in upstream_outputs:
                    extra[task_id] = output

            # Get shared state for this run
            shared_state_keys = [
                "project_context", "architecture", "conventions",
            ]
            for state_key in shared_state_keys:
                val = self.context_bridge.get_shared_state(self.run_id, state_key)
                if val:
                    extra[f"state:{state_key}"] = val

        except Exception as e:
            logger.debug(f"Could not gather shared context: {e}")

        return extra

    async def _execute_with_retries(
        self, task: Task, upstream_outputs: dict[str, str]
    ) -> TaskResult:
        last_error: Exception | None = None

        for attempt in range(1, settings.task_max_retries + 1):
            try:
                output, model = await self._call_llm(task, upstream_outputs)

                # Store output in shared context for downstream tasks
                self.context_bridge.store_task_output(
                    run_id=self.run_id,
                    task_id=task.id,
                    output=output,
                    tags=[task.task_type.value, f"attempt:{attempt}"],
                )

                return TaskResult(
                    task_id=task.id,
                    status=TaskStatus.SUCCESS,
                    output=output,
                    model_used=model,
                    attempts=attempt,
                )
            except Exception as e:
                last_error = e
                logger.warning(f"Task {task.id} attempt {attempt} failed: {e}")
                if attempt < settings.task_max_retries:
                    await asyncio.sleep(1 * attempt)
                    continue

        return TaskResult(
            task_id=task.id,
            status=TaskStatus.FAILED,
            error=str(last_error),
            attempts=settings.task_max_retries,
        )

    async def _call_llm(self, task: Task, upstream_outputs: dict[str, str]) -> tuple[str, str]:
        context_parts = []
        for dep_id, output in upstream_outputs.items():
            # Distinguish between task outputs and shared state
            if dep_id.startswith("state:"):
                state_key = dep_id[len("state:"):]
                context_parts.append(f"--- Shared State: {state_key} ---\n{output}")
            else:
                context_parts.append(f"--- Output from {dep_id} ---\n{output}")

        context = "\n\n".join(context_parts) if context_parts else "No upstream context."

        system_prompt = self._get_system_prompt(task.task_type)
        user_prompt = f"""Task: {task.description}

Context from upstream tasks and shared state:
{context}

Please complete this task and provide your output."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response, model = await self.router.call(
            task_type=task.task_type.value,
            messages=messages,
            temperature=0.3,
            max_tokens=4000,
            task_id=task.id,
        )

        return response, model

    def _get_system_prompt(self, task_type: str) -> str:
        prompts: dict[str, str] = {
            "coding": (
                "You are an expert software engineer. Write clean, "
                "well-documented, production-ready code. Follow best "
                "practices for the language/framework. When multiple "
                "task outputs are provided, build upon them coherently."
            ),
            "analysis": (
                "You are a senior engineer analyzing code or systems. "
                "Provide thorough, structured analysis with clear "
                "findings and recommendations."
            ),
            "writing": (
                "You are a technical writer. Produce clear, "
                "well-structured documentation or explanations."
            ),
            "planning": (
                "You are a project planner. Break down tasks into "
                "clear, actionable steps."
            ),
            "general": (
                "You are a helpful AI assistant. Provide accurate, "
                "useful responses."
            ),
        }
        return prompts.get(task_type, prompts["general"])


class DAGExecutor:
    def __init__(self, router: ModelRouter, run_id: str,
                 context_bridge: ContextBridge | None = None):
        self.router = router
        self.run_id = run_id
        self.context_bridge = context_bridge or get_context_bridge()
        self.task_executor = TaskExecutor(router, run_id, self.context_bridge)
        self.results: dict[str, TaskResult] = {}
        self.completed: set[str] = set()
        self.failed: set[str] = set()
        self.skipped: set[str] = set()

    async def execute(self, graph: TaskGraph) -> dict[str, TaskResult]:
        self._graph = graph

        # Store the task graph in shared context for other agents to read
        self.context_bridge.store_shared_state(
            run_id=self.run_id,
            key="task_graph",
            value=graph.model_dump_json(),
        )

        pending_tasks = {t.id: t for t in graph.tasks}
        running_tasks: dict[str, asyncio.Task] = {}

        while pending_tasks or running_tasks:
            ready_tasks = self._get_ready_tasks(pending_tasks)

            for task in ready_tasks:
                upstream_outputs = self._get_upstream_outputs(task)
                coro = self.task_executor.execute_task(task, upstream_outputs)
                running_tasks[task.id] = asyncio.create_task(coro)
                del pending_tasks[task.id]

            if not running_tasks:
                break

            done, _ = await asyncio.wait(
                running_tasks.values(),
                return_when=asyncio.FIRST_COMPLETED,
            )

            for done_task in done:
                task_id = next(k for k, v in running_tasks.items() if v is done_task)
                result = await done_task
                self.results[task_id] = result
                del running_tasks[task_id]

                if result.status == TaskStatus.SUCCESS:
                    self.completed.add(task_id)
                elif result.status == TaskStatus.FAILED:
                    self.failed.add(task_id)
                    self._mark_downstream_skipped(
                        task_id, pending_tasks
                    )

        self._mark_remaining_skipped(pending_tasks)
        return self.results

    def _get_ready_tasks(
        self, pending: dict[str, Task]
    ) -> list[Task]:
        ready: list[Task] = []
        for task in pending.values():
            if all(dep in self.completed for dep in task.depends_on):
                if any(dep in self.failed for dep in task.depends_on):
                    continue
                ready.append(task)
        return ready

    def _get_upstream_outputs(self, task: Task) -> dict[str, str]:
        outputs = {}
        for dep_id in task.depends_on:
            result = self.results.get(dep_id)
            if result and result.status == TaskStatus.SUCCESS and result.output:
                outputs[dep_id] = result.output
        return outputs

    def _mark_downstream_skipped(
        self, failed_task_id: str, pending: dict[str, Task]
    ) -> None:
        dependents = self._get_all_dependents(failed_task_id)
        for dep_id in dependents:
            if dep_id in pending:
                self.results[dep_id] = TaskResult(
                    task_id=dep_id,
                    status=TaskStatus.SKIPPED,
                    error=f"Upstream task {failed_task_id} failed",
                )
                self.skipped.add(dep_id)
                del pending[dep_id]

    def _get_all_dependents(self, task_id: str) -> set[str]:
        dependents: set[str] = set()
        adj: dict[str, list[str]] = {}
        for task in self._graph.tasks:
            for dep in task.depends_on:
                adj.setdefault(dep, []).append(task.id)

        queue = [task_id]
        while queue:
            current = queue.pop()
            for child in adj.get(current, []):
                if child not in dependents:
                    dependents.add(child)
                    queue.append(child)
        return dependents

    def _mark_remaining_skipped(self, pending: dict[str, Task]) -> None:
        for task_id, _task in pending.items():
            self.results[task_id] = TaskResult(
                task_id=task_id,
                status=TaskStatus.SKIPPED,
                error="Execution interrupted or dependencies not met",
            )
            self.skipped.add(task_id)


@asynccontextmanager
async def create_executor(router: ModelRouter, run_id: str) -> AsyncGenerator[DAGExecutor, None]:
    ctx_bridge = get_context_bridge()
    yield DAGExecutor(router, run_id, ctx_bridge)
