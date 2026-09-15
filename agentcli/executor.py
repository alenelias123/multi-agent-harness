import asyncio
import logging
import re
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .config import settings
from .context import ContextBridge, get_context_bridge
from .model_router import ModelRouter
from .schemas import Complexity, Task, TaskGraph, TaskResult, TaskStatus, TaskType

logger = logging.getLogger(__name__)


# ── Failure Classification ────────────────────────────────────────────────

class FailureType(Enum):
    """Types of failures for adaptive retry strategy."""
    NETWORK = "network"
    RATE_LIMIT = "rate_limit"
    INVALID_OUTPUT = "invalid_output"
    VALIDATION_FAILED = "validation_failed"
    TIMEOUT = "timeout"
    MODEL_ERROR = "model_error"
    UNKNOWN = "unknown"


def classify_failure(error: Exception, output: str | None = None) -> FailureType:
    """Classify the failure type for adaptive retry."""
    error_str = str(error).lower()
    if "rate limit" in error_str or "429" in error_str:
        return FailureType.RATE_LIMIT
    if "timeout" in error_str or "timed out" in error_str:
        return FailureType.TIMEOUT
    if "connection" in error_str or "network" in error_str:
        return FailureType.NETWORK
    if "model" in error_str or "provider" in error_str:
        return FailureType.MODEL_ERROR
    if output is not None and (not output.strip() or len(output.strip()) < 10):
        return FailureType.INVALID_OUTPUT
    return FailureType.UNKNOWN


# ── Retry Strategy ────────────────────────────────────────────────────────

@dataclass
class RetryStrategy:
    """Strategy for retrying a failed task."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: float = 0.1
    # Adaptive: change model/prompt on specific failures
    switch_model_on: list[FailureType] = field(default_factory=lambda: [
        FailureType.RATE_LIMIT, FailureType.MODEL_ERROR
    ])
    switch_prompt_on: list[FailureType] = field(default_factory=lambda: [
        FailureType.VALIDATION_FAILED, FailureType.INVALID_OUTPUT
    ])
    escalate_on: list[FailureType] = field(default_factory=lambda: [
        FailureType.RATE_LIMIT, FailureType.MODEL_ERROR
    ])

    def get_delay(self, attempt: int) -> float:
        delay = min(self.base_delay * (self.exponential_base ** (attempt - 1)), self.max_delay)
        jitter_amount = delay * self.jitter
        return delay + (jitter_amount * (0.5 - hash(str(time.time())) % 1000 / 1000))


DEFAULT_RETRY_STRATEGY = RetryStrategy(max_attempts=settings.task_max_retries)


# ── Output Validators ─────────────────────────────────────────────────────

class OutputValidator:
    """Validate task outputs against expected criteria."""

    @staticmethod
    def validate_coding(output: str, criteria: list[str]) -> tuple[bool, str | None]:
        """Validate coding task output."""
        if not output or not output.strip():
            return False, "Empty output"

        # Check for code blocks
        has_code = (
            bool(re.search(r'```[\w]*\n', output))
            or bool(
                re.search(r'^\s*(def|class|function|const|let|var)\s+\w+', output, re.MULTILINE)
            )
        )
        if not has_code:
            return False, "No code found in output"

        # Check validation criteria
        for criterion in criteria:
            criterion_lower = criterion.lower()
            if "test" in criterion_lower and "test" not in output.lower():
                return False, f"Missing tests: {criterion}"
            if "docstring" in criterion_lower and '"""' not in output and "'''" not in output:
                return False, f"Missing docstrings: {criterion}"
            if ("type hint" in criterion_lower or "typing" in criterion_lower) and not (
                re.search(r':\s*\w+', output) or re.search(r'->\s*\w+', output)
            ):
                return False, f"Missing type hints: {criterion}"

        return True, None

    @staticmethod
    def validate_analysis(output: str, criteria: list[str]) -> tuple[bool, str | None]:  # noqa: ARG004
        """Validate analysis task output."""
        if not output or not output.strip():
            return False, "Empty output"

        # Analysis should have structure
        has_structure = bool(re.search(r'(##|###|\*\*|\d+\.)', output))
        if not has_structure:
            return False, "Analysis lacks structure (headers, bullets, numbered lists)"

        return True, None

    @staticmethod
    def validate_writing(output: str, criteria: list[str]) -> tuple[bool, str | None]:  # noqa: ARG004
        """Validate writing task output."""
        if not output or not output.strip():
            return False, "Empty output"

        min_words = 50
        word_count = len(output.split())
        if word_count < min_words:
            return False, f"Output too short ({word_count} words, minimum {min_words})"

        return True, None

    @staticmethod
    def validate_planning(output: str, criteria: list[str]) -> tuple[bool, str | None]:  # noqa: ARG004
        """Validate planning task output."""
        if not output or not output.strip():
            return False, "Empty output"

        # Should have tasks or steps
        has_tasks = bool(re.search(r'(task|step|phase|milestone)\s*\d+', output, re.IGNORECASE))
        if not has_tasks:
            return False, "Plan lacks structured tasks/steps"

        return True, None

    @staticmethod
    def validate_general(output: str, criteria: list[str]) -> tuple[bool, str | None]:  # noqa: ARG004
        """Validate general task output."""
        if not output or not output.strip():
            return False, "Empty output"
        return True, None

    @classmethod
    def validate(cls, task_type: TaskType, output: str, criteria: list[str]) -> tuple[bool, str | None]:  # noqa: E501
        """Route to appropriate validator."""
        validators = {
            TaskType.CODING: cls.validate_coding,
            TaskType.ANALYSIS: cls.validate_analysis,
            TaskType.WRITING: cls.validate_writing,
            TaskType.PLANNING: cls.validate_planning,
            TaskType.GENERAL: cls.validate_general,
        }
        validator = validators.get(task_type, cls.validate_general)
        return validator(output, criteria)


# ── Dependency Semantics ──────────────────────────────────────────────────

class DependencyKind(Enum):
    """Types of dependencies between tasks."""
    STANDARD = "standard"          # Normal depends_on
    ARTIFACT = "artifact"          # Must produce specific artifact
    PARALLEL_GROUP = "parallel"    # Run in parallel with others
    APPROVAL = "approval"          # Needs human approval before execution
    OPTIONAL = "optional"          # Failure doesn't block downstream


@dataclass
class DependencySpec:
    """Rich dependency specification."""
    task_id: str
    kind: DependencyKind = DependencyKind.STANDARD
    required_artifact: str | None = None  # For ARTIFACT kind
    parallel_group: str | None = None   # For PARALLEL_GROUP kind
    approval_required: bool = False     # For APPROVAL kind


# ── Critical Path Analysis ────────────────────────────────────────────────

def compute_critical_path(graph: TaskGraph) -> dict[str, int]:
    """Compute critical path length for each task (longest path to leaf)."""
    # Build adjacency
    children: dict[str, list[str]] = {}
    for task in graph.tasks:
        for dep in task.depends_on:
            children.setdefault(dep, []).append(task.id)

    # Memoized DFS
    memo: dict[str, int] = {}

    def dfs(task_id: str) -> int:
        if task_id in memo:
            return memo[task_id]
        max_child = 0
        for child in children.get(task_id, []):
            max_child = max(max_child, dfs(child))
        memo[task_id] = 1 + max_child
        return memo[task_id]

    for task in graph.tasks:
        dfs(task.id)

    return memo


def compute_task_priority(graph: TaskGraph) -> dict[str, float]:
    """Compute priority score for each task (higher = more important to run early)."""
    critical_path = compute_critical_path(graph)

    # Weight factors
    complexity_weight: dict[Complexity, int] = {
        Complexity.LOW: 1, Complexity.MEDIUM: 2, Complexity.HIGH: 3
    }
    type_weight = {
        TaskType.PLANNING: 10,
        TaskType.ANALYSIS: 8,
        TaskType.CODING: 6,
        TaskType.WRITING: 4,
        TaskType.GENERAL: 2,
    }

    priorities: dict[str, float] = {}
    for task in graph.tasks:
        priority = (
            critical_path.get(task.id, 1) * 5 +
            complexity_weight.get(task.complexity, 2) * 3 +
            type_weight.get(task.task_type, 2)
        )
        priorities[task.id] = priority

    return priorities


# ── Execution Context ──────────────────────────────────────────────────────

@dataclass
class ExecutionContext:
    task: Task
    upstream_outputs: dict[str, str]
    run_id: str
    attempt: int = 1
    failure_type: FailureType | None = None
    previous_output: str | None = None


# ── Task Executor ─────────────────────────────────────────────────────────

class TaskExecutor:
    def __init__(
        self,
        router: ModelRouter,
        run_id: str,
        context_bridge: ContextBridge | None = None,
        retry_strategy: RetryStrategy | None = None,
    ):
        self.router = router
        self.run_id = run_id
        self.context_bridge = context_bridge or get_context_bridge()
        self.retry_strategy = retry_strategy or DEFAULT_RETRY_STRATEGY
        # Semaphore for global concurrency control
        self.semaphore = asyncio.Semaphore(settings.max_parallelism)

    async def execute_task(self, task: Task, upstream_outputs: dict[str, str]) -> TaskResult:
        async with self.semaphore:
            shared_context = await self._gather_shared_context(task, upstream_outputs)
            merged_upstream = {**upstream_outputs, **shared_context}

            return await self._execute_with_adaptive_retries(task, merged_upstream)

    async    def _gather_shared_context(
        self, task: Task, upstream_outputs: dict[str, str]  # noqa: ARG002
    ) -> dict[str, str]:
        extra: dict[str, str] = {}

        try:
            all_outputs = self.context_bridge.get_all_task_outputs(self.run_id)
            for task_id, output in all_outputs.items():
                if task_id not in upstream_outputs:
                    extra[task_id] = output

            shared_state_keys = ["project_context", "architecture", "conventions"]
            for state_key in shared_state_keys:
                val = self.context_bridge.get_shared_state(self.run_id, state_key)
                if val:
                    extra[f"state:{state_key}"] = val

        except Exception as e:
            logger.debug(f"Could not gather shared context: {e}")

        return extra

    async def _execute_with_adaptive_retries(
        self, task: Task, upstream_outputs: dict[str, str]
    ) -> TaskResult:
        last_error: Exception | None = None
        last_output: str | None = None
        current_model: str | None = None
        failure_type: FailureType = FailureType.UNKNOWN

        for attempt in range(1, self.retry_strategy.max_attempts + 1):
            ctx = ExecutionContext(
                task=task,
                upstream_outputs=upstream_outputs,
                run_id=self.run_id,
                attempt=attempt,
                failure_type=failure_type,
                previous_output=last_output,
            )

            try:
                output, model = await self._call_llm(ctx, current_model)
                current_model = model

                # Validate output
                valid, validation_error = OutputValidator.validate(
                    task.task_type, output, task.validation_criteria
                )
                if not valid:
                    failure_type = FailureType.VALIDATION_FAILED
                    raise ValueError(f"Validation failed: {validation_error}")

                # Store output
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
                    started_at=self._get_start_time(),
                    completed_at=self._get_start_time(),
                )

            except Exception as e:
                last_error = e
                failure_type = classify_failure(e, last_output)
                logger.warning(
                    f"Task {task.id} attempt {attempt} failed ({failure_type.value}): {e}"
                )

                if attempt < self.retry_strategy.max_attempts:
                    delay = self.retry_strategy.get_delay(attempt)
                    logger.info(f"Retrying task {task.id} in {delay:.1f}s (attempt {attempt + 1})")
                    await asyncio.sleep(delay)
                    continue

        return TaskResult(
            task_id=task.id,
            status=TaskStatus.FAILED,
            error=f"{failure_type.value}: {last_error}",
            attempts=self.retry_strategy.max_attempts,
            model_used=current_model,
        )

    def _get_start_time(self) -> datetime:
        return datetime.utcnow()

    async def _call_llm(
        self,
        ctx: ExecutionContext,
        preferred_model: str | None = None,
    ) -> tuple[str, str]:
        """Call LLM with context-aware prompt adaptation."""
        task = ctx.task
        upstream_outputs = ctx.upstream_outputs

        # Build context from upstream
        context_parts = []
        for dep_id, output in upstream_outputs.items():
            if dep_id.startswith("state:"):
                state_key = dep_id[len("state:"):]
                context_parts.append(f"--- Shared State: {state_key} ---\n{output}")
            else:
                context_parts.append(f"--- Output from {dep_id} ---\n{output}")

        context = "\n\n".join(context_parts) if context_parts else "No upstream context."

        # Add execution contract
        contract = task.render_contract()
        if contract:
            context = f"{context}\n\nExecution contract:\n{contract}"

        # Adaptive prompt based on failure
        system_prompt = self._get_system_prompt(task.task_type)
        if ctx.attempt > 1:
            if ctx.failure_type == FailureType.VALIDATION_FAILED:
                system_prompt += (
                    "\n\nPREVIOUS ATTEMPT FAILED VALIDATION. "
                    "Ensure output meets all criteria."
                )
            elif ctx.failure_type == FailureType.INVALID_OUTPUT:
                system_prompt += (
                    "\n\nPREVIOUS ATTEMPT PRODUCED INVALID OUTPUT. "
                    "Provide complete, well-formed response."
                )
            elif ctx.failure_type == FailureType.RATE_LIMIT:
                system_prompt += "\n\nRATE LIMITED. Be concise but complete."

            if ctx.previous_output:
                system_prompt += (
                    f"\n\nPrevious output (for reference):\n{ctx.previous_output[:2000]}"
                )

        user_prompt = f"""Task: {task.description}

Context from upstream tasks and shared state:
{context}

Please complete this task and provide your output."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Select model - could be adaptive based on failure
        task_type_for_router = task.task_type.value
        if ctx.failure_type in self.retry_strategy.switch_model_on and preferred_model:
            # Try a different model on rate limit / model errors
            pass  # Router handles fallback

        response, model = await self.router.call(
            task_type=task_type_for_router,
            messages=messages,
            temperature=0.3,
            max_tokens=4000,
            task_id=task.id,
        )

        return response, model

    def _get_system_prompt(self, task_type: TaskType) -> str:
        prompts: dict[TaskType, str] = {
            TaskType.CODING: (
                "You are an expert software engineer. Write clean, "
                "well-documented, production-ready code. Follow best "
                "practices for the language/framework. When multiple "
                "task outputs are provided, build upon them coherently. "
                "Always include proper error handling, type hints, and tests where appropriate."
            ),
            TaskType.ANALYSIS: (
                "You are a senior engineer analyzing code or systems. "
                "Provide thorough, structured analysis with clear "
                "findings and recommendations. Use headers and bullet points."
            ),
            TaskType.WRITING: (
                "You are a technical writer. Produce clear, "
                "well-structured documentation or explanations. "
                "Use markdown formatting appropriately."
            ),
            TaskType.PLANNING: (
                "You are a project planner. Break down tasks into "
                "clear, actionable steps with dependencies. "
                "Provide structured plans with numbered phases."
            ),
            TaskType.GENERAL: (
                "You are a helpful AI assistant. Provide accurate, "
                "useful responses."
            ),
        }
        return prompts.get(task_type, prompts[TaskType.GENERAL])


# ── DAG Executor with Advanced Orchestration ──────────────────────────────

class DAGExecutor:
    def __init__(
        self,
        router: ModelRouter,
        run_id: str,
        context_bridge: ContextBridge | None = None,
        retry_strategy: RetryStrategy | None = None,
        enable_partial_recovery: bool = True,
    ):
        self.router = router
        self.run_id = run_id
        self.context_bridge = context_bridge or get_context_bridge()
        self.retry_strategy = retry_strategy or DEFAULT_RETRY_STRATEGY
        self.enable_partial_recovery = enable_partial_recovery

        self.task_executor = TaskExecutor(router, run_id, self.context_bridge, self.retry_strategy)
        self.results: dict[str, TaskResult] = {}
        self.completed: set[str] = set()
        self.failed: set[str] = set()
        self.skipped: set[str] = set()
        self.running: dict[str, asyncio.Task] = {}
        self._graph: TaskGraph | None = None
        self._priorities: dict[str, float] = {}
        self._approval_pending: set[str] = set()

    async def execute(self, graph: TaskGraph) -> dict[str, TaskResult]:
        self._graph = graph
        self._priorities = compute_task_priority(graph)

        # Store the task graph
        self.context_bridge.store_shared_state(
            run_id=self.run_id,
            key="task_graph",
            value=graph.model_dump_json(),
        )

        pending_tasks = {t.id: t for t in graph.tasks}
        running_tasks: dict[str, asyncio.Task] = {}

        while pending_tasks or running_tasks:
            # Get ready tasks sorted by priority
            ready_tasks = self._get_ready_tasks(pending_tasks)

            # Limit concurrent tasks - don't start all at once
            max_start = settings.max_parallelism - len(running_tasks)
            for task in ready_tasks[:max_start]:
                # Check for approval requirement
                if self._requires_approval(task):
                    self._approval_pending.add(task.id)
                    logger.info(f"Task {task.id} waiting for approval")
                    continue

                upstream_outputs = self._get_upstream_outputs(task)
                coro = self.task_executor.execute_task(task, upstream_outputs)
                running_tasks[task.id] = asyncio.create_task(coro)
                del pending_tasks[task.id]

            if not running_tasks:
                if self._approval_pending and pending_tasks:
                    # Wait for approval or handle deadlock
                    logger.warning("All ready tasks need approval, waiting...")
                    await asyncio.sleep(5)
                    continue
                break

            # Wait for at least one completion
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
                    if self.enable_partial_recovery:
                        self._mark_dependent_skipped(task_id, pending_tasks)
                    else:
                        # Legacy behavior: mark all downstream as skipped
                        self._mark_downstream_skipped(task_id, pending_tasks)

        self._mark_remaining_skipped(pending_tasks)
        return self.results

    def _get_ready_tasks(self, pending: dict[str, Task]) -> list[Task]:
        ready: list[Task] = []
        for task in pending.values():
            # Check standard dependencies
            deps_met = all(dep in self.completed for dep in task.depends_on)
            if not deps_met:
                continue

            # Check if any required dependency failed
            if any(dep in self.failed for dep in task.depends_on):
                # Check if dependency is optional
                if self._is_optional_dependency(task):
                    continue  # Will be handled as optional
                continue  # Skip if required dependency failed

            # Check artifact dependencies
            if not self._check_artifact_dependencies(task):
                continue

            ready.append(task)

        # Sort by priority (highest first)
        ready.sort(key=lambda t: -self._priorities.get(t.id, 0))
        return ready

    def _is_optional_dependency(self, task: Task) -> bool:  # noqa: ARG002
        """Check if a failed dependency is optional."""
        # For now, check if validation_criteria mentions optional
        # In future, this could be a field on Task
        return False

    def _check_artifact_dependencies(self, task: Task) -> bool:
        """Check if artifact dependencies are satisfied.

        Only meaningful for tasks that actually have upstream dependencies:
        ``expected_inputs`` on a dependency-free task describe what the task
        consumes from the *user* request, not from other tasks, so they can
        never block readiness.
        """
        if not task.depends_on:
            return True
        # Check expected_inputs against available artifacts
        for expected in task.expected_inputs:
            found = False
            for dep_id in task.depends_on:
                result = self.results.get(dep_id)
                if (
                    result
                    and result.status == TaskStatus.SUCCESS
                    and result.output
                    and expected.lower() in result.output.lower()
                ):
                    found = True
                    break
            if not found:
                logger.debug(f"Task {task.id} waiting for artifact: {expected}")
                return False
        return True

    def _requires_approval(self, task: Task) -> bool:
        """Check if task requires human approval."""
        # Check for approval marker in validation criteria
        return any("approval" in c.lower() for c in task.validation_criteria)

    def _get_upstream_outputs(self, task: Task) -> dict[str, str]:
        outputs = {}
        for dep_id in task.depends_on:
            result = self.results.get(dep_id)
            if result and result.status == TaskStatus.SUCCESS and result.output:
                outputs[dep_id] = result.output
        return outputs

    def _mark_dependent_skipped(self, failed_task_id: str, pending: dict[str, Task]) -> None:
        """Mark only direct dependents as skipped, allowing parallel branches to continue."""
        dependents = self._get_direct_dependents(failed_task_id)
        for dep_id in dependents:
            if dep_id in pending:
                task = pending[dep_id]
                # Check if task has other successful dependencies (partial recovery)
                other_deps = [d for d in task.depends_on if d != failed_task_id]
                other_succeeded = all(d in self.completed for d in other_deps)

                if other_succeeded and len(other_deps) > 0:
                    # Task can still run with other inputs - don't skip
                    logger.info(
                        f"Task {dep_id} continuing despite {failed_task_id} failure "
                        "(has other deps)"
                    )
                    continue

                self.results[dep_id] = TaskResult(
                    task_id=dep_id,
                    status=TaskStatus.SKIPPED,
                    error=f"Required upstream task {failed_task_id} failed",
                )
                self.skipped.add(dep_id)
                del pending[dep_id]

    def _mark_downstream_skipped(
        self, failed_task_id: str, pending: dict[str, Task]
    ) -> None:
        """Legacy: mark all downstream as skipped."""
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

    def _get_direct_dependents(self, task_id: str) -> set[str]:
        """Get immediate dependents only."""
        dependents: set[str] = set()
        for task in self._graph.tasks if self._graph else []:
            if task_id in task.depends_on:
                dependents.add(task.id)
        return dependents

    def _get_all_dependents(self, task_id: str) -> set[str]:
        """Get all transitive dependents."""
        dependents: set[str] = set()
        if not self._graph:
            return dependents

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
async def create_executor(
    router: ModelRouter,
    run_id: str,
    retry_strategy: RetryStrategy | None = None,
    enable_partial_recovery: bool = True,
) -> AsyncGenerator[DAGExecutor, None]:
    ctx_bridge = get_context_bridge()
    yield DAGExecutor(
        router,
        run_id,
        ctx_bridge,
        retry_strategy=retry_strategy,
        enable_partial_recovery=enable_partial_recovery,
    )