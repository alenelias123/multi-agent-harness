import json
import logging
import re
from pathlib import Path
from typing import Any

from .config import settings
from .graph import GraphError, build_graph_from_planner_output
from .model_router import ModelRouter, ModelRoutingError
from .plan_review import (
    build_critique_prompt,
    estimate_plan_cost,
    find_issues,
    prune_plan,
    score_plan,
)
from .schemas import TaskGraph

logger = logging.getLogger(__name__)


PLANNER_PROMPT_PATH = Path(__file__).parent / "prompts" / "planner_system.txt"

# Planner LLM-call budget per plan() invocation: initial attempt + critiques.
MAX_PLANNER_ATTEMPTS = 3


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
  "needs_review": true|false,
  "confidence": "low|medium|high",
  "concerns": "string (what you are unsure about, or empty)",
  "tasks": [
    {
      "id": "string (unique, short, e.g., t1, t2)",
      "description": "string (one clear objective per task, starting with a verb)",
      "depends_on": ["string"] (list of task IDs this task depends on),
      "task_type": "planning|coding|analysis|writing|general",
      "complexity": "low|medium|high",
      "expected_inputs": ["string"] (what this task needs from dependencies or the user),
      "expected_outputs": ["string"] (concrete artifacts this task must produce),
      "validation_criteria": ["string"] (how to tell the output is correct/testable)
    }
  ]
}

RULES:
1. Output ONLY the JSON object. No markdown, no prose, no explanations.
2. Each task must have a unique ID (e.g., t1, t2, t3...).
3. depends_on lists IDs of tasks that must complete before this task can start.
4. The graph MUST be acyclic (no circular dependencies) and reference only defined IDs.
5. Max 20 tasks total.
6. task_type must be one of: planning, coding, analysis, writing, general.
7. complexity must be one of: low, medium, high.
8. One objective per task. Split tasks that combine multiple goals.
9. Tasks must be small, independently verifiable, and start with an action verb.
10. Every task MUST declare expected_inputs, expected_outputs and validation_criteria.
11. If the request is ambiguous or under-specified, set needs_review=true,
    confidence="low", and explain in concerns.

EXAMPLE:
Input: "Create a REST API for a todo app with CRUD operations, using FastAPI and SQLite"

Output:
{
  "needs_review": false,
  "confidence": "high",
  "concerns": "",
  "tasks": [
    {
      "id": "t1",
      "description": "Design the database schema for todos",
      "_note": "columns: id, title, description, completed, created_at",
      "depends_on": [],
      "task_type": "planning",
      "complexity": "low",
      "expected_inputs": ["list of todo attributes from the request"],
      "expected_outputs": ["schema definition with table name, columns and types"],
      "validation_criteria": ["schema includes all todo attributes and a primary key"]
    },
    {
      "id": "t2",
      "description": "Create SQLAlchemy models and database connection setup",
      "depends_on": ["t1"],
      "task_type": "coding",
      "complexity": "medium",
      "expected_inputs": ["schema definition from t1"],
      "expected_outputs": ["models.py with Todo model + session factory"],
      "validation_criteria": ["model fields match the schema", "session connects to SQLite"]
    },
    {
      "id": "t3",
      "description": "Implement FastAPI routes for CRUD operations (create, read, update, delete)",
      "depends_on": ["t2"],
      "task_type": "coding",
      "complexity": "medium",
      "expected_inputs": ["Todo model and session factory from t2"],
      "expected_outputs": ["router with POST/GET/PUT/DELETE endpoints"],
      "validation_criteria": ["endpoints return correct status codes", "CRUD persists"]
    },
    {
      "id": "t4",
      "description": "Add Pydantic schemas for request/response validation",
      "depends_on": ["t2"],
      "task_type": "coding",
      "complexity": "low",
      "expected_inputs": ["Todo model from t2"],
      "expected_outputs": ["Pydantic models for TodoCreate, TodoUpdate and TodoRead"],
      "validation_criteria": ["schemas validate sample payloads without errors"]
    },
    {
      "id": "t5",
      "description": "Write unit tests for the API endpoints",
      "depends_on": ["t3", "t4"],
      "task_type": "coding",
      "complexity": "medium",
      "expected_inputs": ["routes from t3 and validation schemas from t4"],
      "expected_outputs": ["test suite covering all CRUD endpoints"],
      "validation_criteria": ["all tests pass", "each endpoint has at least one test"]
    }
  ]
}"""


class _AttemptOutcome:
    """Internal: result of one planner LLM attempt."""

    def __init__(
        self,
        graph: TaskGraph | None,
        meta: dict[str, Any],
        raw: str,
        model: str,
        error: str | None = None,
    ) -> None:
        self.graph = graph
        self.meta = meta
        self.raw = raw
        self.model = model
        self.error = error

    @property
    def ok(self) -> bool:
        return self.graph is not None


def _unwrap_code_fence(text: str) -> str:
    """Strip a markdown code fence if the model wrapped JSON in one."""
    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        if first_newline != -1:
            stripped = stripped[first_newline + 1 :]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    return stripped.strip()


def _parse_plan_meta(planner_json: dict[str, Any]) -> dict[str, Any]:
    """Extract the planner's optional self-review metadata."""
    return {
        "needs_review": bool(planner_json.get("needs_review", False)),
        "confidence": str(planner_json.get("confidence", "high")).lower(),
        "concerns": str(planner_json.get("concerns", "") or ""),
    }


class Planner:
    """Task decomposition planner with self-review and iterative refinement.

    Flow per :meth:`plan` call (bounded by ``MAX_PLANNER_ATTEMPTS`` LLM calls):

    1. Generate a plan (JSON) from the user prompt.
    2. Parse and validate structure (existing graph builder) and self-check:
       cycles, unknown deps, budget, contract coverage, task quality,
       redundancy.
    3. If issues remain, prune what heuristics can fix locally (duplicates,
       over-budget) and re-score.
    4. If the plan is still weak, re-plan with a stricter critique prompt that
       includes concrete findings. The best-scoring attempt wins.
    5. If every attempt fails to produce a usable graph, fall back to the best
       graph-like attempt; only raise when nothing usable was produced.
    """

    def __init__(self, router: ModelRouter):
        self.router = router
        self.system_prompt = load_planner_prompt()

    # ------------------------------------------------------------------
    # LLM plumbing
    # ------------------------------------------------------------------

    async def _call_planner(
        self, messages: list[dict[str, str]]
    ) -> tuple[str, str]:
        return await self.router.call(
            task_type="planning",
            messages=messages,
            temperature=0.1,
            max_tokens=4000,
            response_format={"type": "json_object"},
        )

    def _parse_response(self, response: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Parse raw planner text into (planner_json, meta). Raises GraphError."""
        cleaned = _unwrap_code_fence(response)
        try:
            planner_json = json.loads(cleaned)
        except json.JSONDecodeError as e:
            # Tolerate common trailing-comma slip in a last-ditch pass.
            try:
                planner_json = json.loads(re.sub(r",\s*([}\]])", r"\1", cleaned))
            except json.JSONDecodeError:
                raise GraphError(f"Planner returned invalid JSON: {e}") from e
        if not isinstance(planner_json, dict):
            raise GraphError("Planner output must be a JSON object")
        return planner_json, _parse_plan_meta(planner_json)

    def _build_graph(
        self, planner_json: dict[str, Any], max_tasks: int
    ) -> TaskGraph:
        """Build a graph, pruning over-budget plans instead of failing."""
        try:
            return build_graph_from_planner_output(planner_json, max_tasks=max_tasks)
        except ValueError as e:
            # Over-budget graphs get pruned to budget rather than rejected.
            if "Max task limit" in str(e):
                tasks = planner_json.get("tasks")
                if isinstance(tasks, list):
                    trimmed = dict(planner_json)
                    trimmed["tasks"] = tasks[:max_tasks]
                    return build_graph_from_planner_output(
                        trimmed, max_tasks=max_tasks
                    )
            raise

    # ------------------------------------------------------------------
    # Review loop
    # ------------------------------------------------------------------

    def _review(self, graph: TaskGraph) -> list[str]:
        """Self-check the graph. Returns concrete, actionable issues."""
        if settings.planner_review_enabled:
            return find_issues(
                graph,
                max_tasks=settings.max_tasks,
                min_contract_coverage=settings.planner_min_contract_coverage,
                min_task_score=settings.planner_min_task_score,
                max_estimated_cost=settings.planner_max_estimated_cost,
            )
        # Minimal structural-only self-check when review is disabled:
        # cycles and missing dependencies are always worth catching.
        structural: list[str] = []
        try:
            graph.validate_dependencies()
        except ValueError as e:
            structural.append(f"Missing dependency: {e}")
        try:
            graph.detect_cycles()
        except ValueError as e:
            structural.append(f"Cycle: {e}")
        return structural

    def _is_usable(self, graph: TaskGraph) -> bool:
        """A graph is usable if it has tasks and no structural defects."""
        if not graph.tasks:
            return False
        return not self._review(graph)

    def _quality_ok(self, graph: TaskGraph) -> bool:
        """Quality gate: passes when self-check reports no issues at all."""
        return not self._review(graph)

    async def plan(self, task_description: str) -> TaskGraph:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task_description},
        ]

        best: TaskGraph | None = None
        best_score = -1.0
        last_error: Exception | None = None
        response = ""
        critique_rounds = 0

        for attempt in range(1, MAX_PLANNER_ATTEMPTS + 1):
            try:
                response, model = await self._call_planner(messages)
            except ModelRoutingError as e:
                last_error = e
                logger.warning(f"Planner attempt {attempt}: model routing failed: {e}")
                break  # no provider available; retrying won't help

            try:
                planner_json, meta = self._parse_response(response)
                graph = self._build_graph(planner_json, max_tasks=settings.max_tasks)
            except GraphError as e:
                last_error = e
                logger.warning(f"Planner attempt {attempt} produced unusable output: {e}")
                # Ask the model to fix its output format.
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Previous attempt failed: {e}. "
                        "Output ONLY valid JSON matching the schema. "
                        "No markdown fences, no prose."
                    ),
                })
                continue

            logger.info(
                f"Planner attempt {attempt} via {model}: "
                f"{len(graph.tasks)} tasks, self-review "
                f"needs_review={meta['needs_review']} "
                f"confidence={meta['confidence']}"
            )

            # --- self-check ------------------------------------------------
            issues = self._review(graph)
            if issues:
                for issue in issues:
                    logger.warning(f"Planner self-check: {issue}")
            if meta["needs_review"] and meta["concerns"]:
                issues.append(f"Planner flagged own uncertainty: {meta['concerns']}")

            # --- local repair: prune duplicates / enforce budget -----------
            if issues:
                graph, dropped = prune_plan(graph, max_tasks=settings.max_tasks)
                if dropped:
                    logger.info(
                        f"Planner pruned {len(dropped)} task(s): {', '.join(dropped)}"
                    )
                issues = self._review(graph)

            # --- score and keep the best attempt ---------------------------
            review = score_plan(
                graph,
                max_tasks=settings.max_tasks,
                min_contract_coverage=settings.planner_min_contract_coverage,
                min_task_score=settings.planner_min_task_score,
                max_estimated_cost=settings.planner_max_estimated_cost,
            )
            penalty = len(issues)
            final_score = review.score - 0.1 * penalty
            if final_score > best_score:
                best_score = final_score
                best = graph

            # --- decide: accept, or re-plan with critique -------------------
            if not issues or attempt == MAX_PLANNER_ATTEMPTS:
                break  # clean plan, or out of LLM budget

            structural_only = all(
                i.startswith(("Cycle:", "Missing dependency:")) for i in issues
            )
            if structural_only and meta["confidence"] != "low" and not meta["needs_review"]:
                break  # pure structural defects; re-prompting rarely adds value

            critique = build_critique_prompt(
                task_description, graph, review, meta
            )
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": critique})
            critique_rounds += 1
            logger.info("Planner re-planning with critique feedback")
            continue

        if best is not None:
            return self._finalize(best, critique_rounds)

        max_retries = settings.planner_max_retries + 1
        raise PlannerError(
            f"Planning failed after {max_retries} attempts: {last_error}"
        )

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def _finalize(self, graph: TaskGraph, critique_rounds: int = 0) -> TaskGraph:
        """Attach review metadata and log the accepted plan."""
        review = score_plan(
            graph,
            max_tasks=settings.max_tasks,
            min_contract_coverage=settings.planner_min_contract_coverage,
            min_task_score=settings.planner_min_task_score,
            max_estimated_cost=settings.planner_max_estimated_cost,
        )
        graph.quality_score = review.score
        graph.contract_coverage = review.contract_coverage
        graph.estimated_cost = estimate_plan_cost(graph)
        graph.review_iterations = critique_rounds
        for task in graph.tasks:
            if task.id in review.task_scores:
                task.quality_score = review.task_scores[task.id]
        logger.info(
            f"Plan accepted: {len(graph.tasks)} tasks, quality={review.score:.2f}, "
            f"contract_coverage={review.contract_coverage:.0%}, "
            f"estimated_cost={graph.estimated_cost}"
        )
        return graph


class PlannerError(Exception):
    pass
