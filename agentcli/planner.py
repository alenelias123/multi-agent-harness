import json
import logging
from pathlib import Path

from .config import settings
from .graph import GraphError, build_graph_from_planner_output
from .model_router import ModelRouter, ModelRoutingError
from .schemas import TaskGraph

logger = logging.getLogger(__name__)


PLANNER_PROMPT_PATH = Path(__file__).parent / "prompts" / "planner_system.txt"


def load_planner_prompt() -> str:
    try:
        return PLANNER_PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return DEFAULT_PLANNER_PROMPT


DEFAULT_PLANNER_PROMPT = """You are a task decomposition planner.
Given a natural-language development task, break it down into a dependency graph
(DAG) of smaller subtasks.

OUTPUT FORMAT: You MUST output ONLY valid JSON matching this exact schema:
{
  "tasks": [
    {
      "id": "string (unique, short, e.g., t1, t2)",
      "description": "string (clear, actionable subtask description)",
      "depends_on": ["string"] (list of task IDs this task depends on),
      "task_type": "planning|coding|analysis|writing|general",
      "complexity": "low|medium|high"
    }
  ]
}

RULES:
1. Output ONLY the JSON object. No markdown, no prose, no explanations.
2. Each task must have a unique ID (e.g., t1, t2, t3...).
3. depends_on lists IDs of tasks that must complete before this task can start.
4. The graph MUST be acyclic (no circular dependencies).
5. Max 20 tasks total.
6. task_type must be one of: planning, coding, analysis, writing, general.
7. complexity must be one of: low, medium, high.
8. Prefer fewer, well-scoped tasks over many tiny ones.

EXAMPLE:
Input: "Create a REST API for a todo app with CRUD operations, using FastAPI and SQLite"

Output:
{
  "tasks": [
    {
      "id": "t1",            "description": "Design the database schema for todos",
      "depends_on": [],
      "task_type": "planning",
      "complexity": "low"
    },
    {
      "id": "t2",
      "description": "Create SQLAlchemy models and database connection setup",
      "depends_on": ["t1"],
      "task_type": "coding",
      "complexity": "medium"
    },
    {
      "id": "t3",
      "description": "Implement FastAPI routes for CRUD operations (create, read, update, delete)",
      "depends_on": ["t2"],
      "task_type": "coding",
      "complexity": "medium"
    },
    {
      "id": "t4",
      "description": "Add Pydantic schemas for request/response validation",
      "depends_on": ["t2"],
      "task_type": "coding",
      "complexity": "low"
    },
    {
      "id": "t5",
      "description": "Write unit tests for the API endpoints",
      "depends_on": ["t3", "t4"],
      "task_type": "coding",
      "complexity": "medium"
    }
  ]
}"""


class Planner:
    def __init__(self, router: ModelRouter):
        self.router = router
        self.system_prompt = load_planner_prompt()

    async def plan(self, task_description: str) -> TaskGraph:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task_description},
        ]

        last_error: Exception | None = None

        for attempt in range(settings.planner_max_retries + 1):
            try:
                response, model = await self.router.call(
                    task_type="planning",
                    messages=messages,
                    temperature=0.1,
                    max_tokens=4000,
                    response_format={"type": "json_object"},
                )
                logger.info(f"Planner succeeded on attempt {attempt + 1} using {model}")

                try:
                    planner_json = json.loads(response)
                except json.JSONDecodeError as e:
                    raise GraphError(
                        f"Planner returned invalid JSON: {e}"
                    ) from e

                return build_graph_from_planner_output(planner_json, max_tasks=settings.max_tasks)

            except (GraphError, ModelRoutingError) as e:
                last_error = e
                logger.warning(f"Planner attempt {attempt + 1} failed: {e}")
                if attempt < settings.planner_max_retries:
                    last_response = (
                        response if 'response' in locals() else ""
                    )
                    messages.append(
                        {"role": "assistant", "content": last_response}
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Previous attempt failed: {e}. "
                                "Output ONLY valid JSON matching"
                                " the schema."
                            ),
                        }
                    )
                    continue
                break

        max_retries = settings.planner_max_retries + 1
        raise PlannerError(
            f"Planning failed after {max_retries} attempts: {last_error}"
        )


class PlannerError(Exception):
    pass