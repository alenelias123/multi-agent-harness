from datetime import datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class TaskStatus(StrEnum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    INTERRUPTED = "interrupted"


class TaskType(StrEnum):
    PLANNING = "planning"
    CODING = "coding"
    ANALYSIS = "analysis"
    WRITING = "writing"
    GENERAL = "general"


class Complexity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Task(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:8])
    description: str
    depends_on: list[str] = Field(default_factory=list)
    task_type: TaskType = TaskType.GENERAL
    complexity: Complexity = Complexity.MEDIUM

    @field_validator("id", mode="before")
    @classmethod
    def generate_id(cls, v: str | None) -> str:
        return v or uuid4().hex[:8]


class TaskGraph(BaseModel):
    tasks: list[Task] = Field(default_factory=list)
    max_tasks: int = 20

    def add_task(self, task: Task) -> None:
        if len(self.tasks) >= self.max_tasks:
            raise ValueError(f"Max task limit ({self.max_tasks}) reached")
        if any(t.id == task.id for t in self.tasks):
            raise ValueError(f"Task with id {task.id} already exists")
        self.tasks.append(task)

    def get_task(self, task_id: str) -> Task | None:
        return next((t for t in self.tasks if t.id == task_id), None)

    def validate_dependencies(self) -> None:
        task_ids = {t.id for t in self.tasks}
        for task in self.tasks:
            for dep_id in task.depends_on:
                if dep_id not in task_ids:
                    raise ValueError(f"Task {task.id} depends on unknown task {dep_id}")

    def detect_cycles(self) -> None:
        visited = set()
        rec_stack = set()

        def dfs(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            task = self.get_task(node_id)
            if task:
                for dep_id in task.depends_on:
                    if dep_id not in visited:
                        if dfs(dep_id):
                            return True
                    elif dep_id in rec_stack:
                        return True
            rec_stack.remove(node_id)
            return False

        for task in self.tasks:
            if task.id not in visited and dfs(task.id):
                raise ValueError("Circular dependency detected in task graph")

    def ready_set(self, completed: set[str]) -> list[Task]:
        ready = []
        for task in self.tasks:
            if task.id in completed:
                continue
            if all(dep in completed for dep in task.depends_on):
                ready.append(task)
        return ready

    def leaf_tasks(self) -> list[Task]:
        dependents: dict[str, list[str]] = {}
        for task in self.tasks:
            for dep_id in task.depends_on:
                dependents.setdefault(dep_id, []).append(task.id)
        return [t for t in self.tasks if t.id not in dependents]


class TaskResult(BaseModel):
    task_id: str
    status: TaskStatus
    output: str | None = None
    error: str | None = None
    model_used: str | None = None
    attempts: int = 1
    started_at: datetime | None = None
    completed_at: datetime | None = None


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class Run(BaseModel):
    run_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    user_id: str = "local"
    task_description: str
    status: RunStatus = RunStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    task_graph: TaskGraph | None = None
    results: list[TaskResult] = Field(default_factory=list)
    final_output: str | None = None
    error: str | None = None

    def get_result(self, task_id: str) -> TaskResult | None:
        return next((r for r in self.results if r.task_id == task_id), None)

    def all_completed(self) -> bool:
        if not self.task_graph:
            return False
        task_ids = {t.id for t in self.task_graph.tasks}
        active_statuses = {
            TaskStatus.SUCCESS,
            TaskStatus.FAILED,
            TaskStatus.SKIPPED,
        }
        result_ids = {
            r.task_id
            for r in self.results
            if r.status in active_statuses
        }
        return task_ids == result_ids