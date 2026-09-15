"""Tests for the intelligent planner: quality scoring, budget enforcement,
iterative refinement, and execution contracts."""

from __future__ import annotations

import json

import pytest

from agentcli.config import settings
from agentcli.plan_review import (
    build_critique_prompt,
    compute_contract_coverage,
    estimate_plan_cost,
    find_issues,
    find_redundant_pairs,
    prune_plan,
    score_plan,
    score_task,
)
from agentcli.planner import Planner, PlannerError
from agentcli.schemas import Complexity, Task, TaskGraph, TaskType


def _good_planner_json() -> dict:
    return {
        "needs_review": False,
        "confidence": "high",
        "concerns": "",
        "tasks": [
            {
                "id": "t1",
                "description": "Design the database schema for todos",
                "depends_on": [],
                "task_type": "planning",
                "complexity": "low",
                "expected_inputs": ["todo attributes"],
                "expected_outputs": ["schema definition"],
                "validation_criteria": ["schema includes primary key"],
            },
            {
                "id": "t2",
                "description": "Implement Todo model and session factory",
                "depends_on": ["t1"],
                "task_type": "coding",
                "complexity": "medium",
                "expected_inputs": ["schema from t1"],
                "expected_outputs": ["models.py"],
                "validation_criteria": ["model instantiates cleanly"],
            },
        ],
    }


class FakeRouter:
    """Minimal router double: returns scripted responses in order."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def call(
        self, task_type: str, messages: list[dict], **kwargs: object  # noqa: ARG002
    ) -> tuple[str, str]:
        self.calls.append({"task_type": task_type, "messages": messages})
        if not self.responses:
            pytest.fail("FakeRouter ran out of scripted responses")
        response = self.responses.pop(0)
        if isinstance(response, dict):
            response = json.dumps(response)
        return response, "fake-model"


def _graph_with_contracts(n: int = 3) -> TaskGraph:
    graph = TaskGraph(max_tasks=10)
    for i in range(n):
        graph.add_task(
            Task(
                id=f"t{i + 1}",
                description=f"Implement feature {i + 1}",
                depends_on=[f"t{i}"] if i > 0 else [],
                task_type=TaskType.CODING,
                complexity=Complexity.LOW,
                expected_outputs=[f"artifact {i + 1}"],
                validation_criteria=[f"artifact {i + 1} works"],
            )
        )
    return graph


# ---------------------------------------------------------------------------
# Task quality scoring
# ---------------------------------------------------------------------------


class TestTaskScoring:
    def test_small_actionable_task_scores_high(self) -> None:
        graph = TaskGraph(max_tasks=5)
        task = Task(
            id="t1",
            description="Implement the login endpoint",
            task_type=TaskType.CODING,
        )
        graph.add_task(task)
        assert score_task(task, graph) >= 0.6

    def test_broad_vague_task_scores_low(self) -> None:
        graph = TaskGraph(max_tasks=5)
        task = Task(
            id="t1",
            description=(
                "Handle everything for the entire system properly, "
                "design the whole architecture and build all the things"
            ),
            task_type=TaskType.GENERAL,
        )
        graph.add_task(task)
        assert score_task(task, graph) < 0.55
        assert "broad_scope" in task.quality_flags
        assert "not_action_oriented" in task.quality_flags

    def test_long_description_flagged(self) -> None:
        graph = TaskGraph(max_tasks=5)
        task = Task(
            id="t1",
            description="Implement " + "very specific detail " * 15,
            task_type=TaskType.CODING,
        )
        graph.add_task(task)
        score_task(task, graph)
        assert "long_description" in task.quality_flags

    def test_contract_boosts_testability(self) -> None:
        graph = TaskGraph(max_tasks=5)
        bare = Task(id="t1", description="Write the parser module")
        with_contract = Task(
            id="t2",
            description="Write the parser module",
            validation_criteria=["parses sample input"],
        )
        graph.add_task(bare)
        graph.add_task(with_contract)
        assert score_task(with_contract, graph) > score_task(bare, graph)


# ---------------------------------------------------------------------------
# Redundancy + budget
# ---------------------------------------------------------------------------


class TestRedundancyAndBudget:
    def test_near_duplicate_tasks_detected(self) -> None:
        graph = TaskGraph(max_tasks=10)
        graph.add_task(
            Task(id="t1", description="Implement user authentication endpoints")
        )
        graph.add_task(
            Task(id="t2", description="Implement user authentication endpoints")
        )
        graph.add_task(Task(id="t3", description="Write documentation for the API"))
        pairs = find_redundant_pairs(graph)
        assert ("t1", "t2") in pairs

    def test_prune_drops_duplicate_keeps_better(self) -> None:
        graph = TaskGraph(max_tasks=10)
        graph.add_task(
            Task(
                id="t1",
                description="Implement user authentication endpoints",
                expected_outputs=["auth routes"],
            )
        )
        graph.add_task(
            Task(id="t2", description="Implement user authentication endpoints")
        )
        graph.add_task(Task(id="t3", description="Write the API documentation"))
        pruned, dropped = prune_plan(graph, max_tasks=10)
        assert dropped == ["t2"]
        assert [t.id for t in pruned.tasks] == ["t1", "t3"]

    def test_prune_respects_budget(self) -> None:
        graph = TaskGraph(max_tasks=10)
        for i in range(6):
            graph.add_task(
                Task(
                    id=f"t{i + 1}",
                    description=f"Implement feature module {i}",
                    task_type=TaskType.CODING,
                )
            )
        pruned, dropped = prune_plan(graph, max_tasks=3)
        assert len(pruned.tasks) == 3
        assert len(dropped) == 3
        # All remaining deps must still resolve
        pruned.validate_dependencies()

    def test_estimated_cost_sums_complexity_weights(self) -> None:
        graph = TaskGraph(max_tasks=10)
        graph.add_task(Task(id="t1", description="A", complexity=Complexity.LOW))
        graph.add_task(Task(id="t2", description="B", complexity=Complexity.MEDIUM))
        graph.add_task(Task(id="t3", description="C", complexity=Complexity.HIGH))
        assert estimate_plan_cost(graph) == 7.0

    def test_cost_overrun_flagged(self) -> None:
        graph = TaskGraph(max_tasks=25)
        for i in range(12):
            graph.add_task(
                Task(
                    id=f"t{i + 1}",
                    description=f"Implement module {i}",
                    complexity=Complexity.HIGH,
                )
            )
        issues = find_issues(
            graph,
            max_tasks=20,
            min_contract_coverage=0.0,
            min_task_score=0.0,
            max_estimated_cost=40.0,
        )
        assert any("exceeds budget" in i for i in issues)


# ---------------------------------------------------------------------------
# Self-check issues
# ---------------------------------------------------------------------------


class TestSelfCheck:
    def test_cycle_flagged(self) -> None:
        graph = TaskGraph(max_tasks=10)
        graph.add_task(Task(id="t1", description="Implement A", depends_on=["t2"]))
        graph.add_task(Task(id="t2", description="Implement B", depends_on=["t1"]))
        issues = find_issues(
            graph, 10, min_contract_coverage=0.0, min_task_score=0.0,
            max_estimated_cost=100.0,
        )
        assert any("Cycle" in i for i in issues)

    def test_missing_dependency_flagged(self) -> None:
        graph = TaskGraph(max_tasks=10)
        graph.add_task(Task(id="t1", description="Implement A", depends_on=["tx"]))
        issues = find_issues(
            graph, 10, min_contract_coverage=0.0, min_task_score=0.0,
            max_estimated_cost=100.0,
        )
        assert any("unknown task" in i for i in issues)

    def test_contract_coverage_flagged(self) -> None:
        graph = TaskGraph(max_tasks=10)
        graph.add_task(Task(id="t1", description="Implement A"))
        graph.add_task(Task(id="t2", description="Implement B"))
        issues = find_issues(
            graph, 10, min_contract_coverage=0.9, min_task_score=0.0,
            max_estimated_cost=100.0,
        )
        assert any("Contract coverage" in i for i in issues)
        assert compute_contract_coverage(graph) == 0.0

    def test_clean_plan_has_no_issues(self) -> None:
        graph = _graph_with_contracts(3)
        issues = find_issues(
            graph, 10, min_contract_coverage=0.6, min_task_score=0.5,
            max_estimated_cost=100.0,
        )
        assert issues == []

    def test_score_plan_summary(self) -> None:
        review = score_plan(
            _graph_with_contracts(3), max_tasks=10,
            min_contract_coverage=0.6, min_task_score=0.5,
            max_estimated_cost=100.0,
        )
        assert 0.0 <= review.score <= 1.0
        assert review.is_acceptable
        assert review.estimated_cost > 0


# ---------------------------------------------------------------------------
# Graph building with contracts
# ---------------------------------------------------------------------------


class TestContractParsing:
    def test_contract_fields_parsed(self) -> None:
        from agentcli.graph import build_graph_from_planner_output

        graph = build_graph_from_planner_output(_good_planner_json())
        t1 = graph.get_task("t1")
        assert t1 is not None
        assert t1.expected_outputs == ["schema definition"]
        assert t1.validation_criteria == ["schema includes primary key"]

    def test_contract_fields_tolerate_strings(self) -> None:
        from agentcli.graph import build_graph_from_planner_output

        planner_json = {
            "tasks": [
                {
                    "id": "t1",
                    "description": "Implement feature",
                    "expected_inputs": "schema, spec",
                    "expected_outputs": "module file",
                }
            ]
        }
        graph = build_graph_from_planner_output(planner_json)
        t1 = graph.tasks[0]
        assert t1.expected_inputs == ["schema", "spec"]
        assert t1.expected_outputs == ["module file"]
        assert t1.validation_criteria == []

    def test_contract_render_includes_all_sections(self) -> None:
        task = Task(
            id="t1",
            description="Implement feature",
            expected_inputs=["spec"],
            expected_outputs=["module"],
            validation_criteria=["tests pass"],
        )
        rendered = task.render_contract()
        assert "Expected inputs: spec" in rendered
        assert "Required outputs: module" in rendered
        assert "Validation criteria" in rendered


# ---------------------------------------------------------------------------
# Iterative refinement loop
# ---------------------------------------------------------------------------


class TestPlannerRefinement:
    @pytest.mark.asyncio
    async def test_good_plan_accepted_without_replan(self) -> None:
        router = FakeRouter([_good_planner_json()])
        planner = Planner(router)  # type: ignore[arg-type]
        graph = await planner.plan("Build a todo API")

        assert len(graph.tasks) == 2
        assert len(router.calls) == 1
        assert graph.quality_score is not None
        assert graph.review_iterations == 0
        assert graph.estimated_cost > 0
        assert graph.contract_coverage == 1.0

    @pytest.mark.asyncio
    async def test_weak_plan_triggers_critique_replan(self) -> None:
        weak = {
            "needs_review": True,
            "confidence": "low",
            "concerns": "requirements are vague",
            "tasks": [
                {
                    "id": "t1",
                    "description": "the whole system and everything else",
                    "depends_on": [],
                    "task_type": "general",
                    "complexity": "high",
                }
            ],
        }
        better = _good_planner_json()
        router = FakeRouter([weak, better])
        planner = Planner(router)  # type: ignore[arg-type]
        graph = await planner.plan("Build something vague")

        assert len(router.calls) == 2
        assert graph.get_task("t1") is not None
        assert graph.get_task("t2") is not None
        # Critique message must reference the concrete findings
        critique_msgs = [
            m for m in router.calls[1]["messages"] if m["role"] == "user"
        ]
        assert any("REVIEW FINDINGS" in m["content"] for m in critique_msgs)
        assert graph.review_iterations == 1

    @pytest.mark.asyncio
    async def test_invalid_json_then_valid_retries(self) -> None:
        router = FakeRouter(["not json at all", _good_planner_json()])
        planner = Planner(router)  # type: ignore[arg-type]
        graph = await planner.plan("Build a todo API")
        assert len(graph.tasks) == 2
        assert len(router.calls) == 2

    @pytest.mark.asyncio
    async def test_overbudget_plan_pruned_not_rejected(self) -> None:
        big = {
            "tasks": [
                {
                    "id": f"t{i}",
                    "description": f"Implement feature {i}",
                    "depends_on": [],
                    "task_type": "coding",
                    "complexity": "low",
                    "expected_outputs": [f"artifact {i}"],
                }
                for i in range(30)
            ]
        }
        router = FakeRouter([big])
        planner = Planner(router)  # type: ignore[arg-type]
        graph = await planner.plan("Build many features")
        assert len(graph.tasks) == settings.max_tasks

    @pytest.mark.asyncio
    async def test_all_attempts_fail_raises_planner_error(self) -> None:
        router = FakeRouter(["garbage 1", "garbage 2", "garbage 3"])
        planner = Planner(router)  # type: ignore[arg-type]
        with pytest.raises(PlannerError, match="Planning failed"):
            await planner.plan("Build a todo API")

    @pytest.mark.asyncio
    async def test_router_failure_raises_planner_error(self) -> None:
        class DeadRouter:
            async def call(self, *a: object, **_kw: object) -> tuple[str, str]:  # noqa: ARG002
                from agentcli.model_router import ModelRoutingError

                raise ModelRoutingError("no providers")

        planner = Planner(DeadRouter())  # type: ignore[arg-type]
        with pytest.raises(PlannerError, match="no providers"):
            await planner.plan("Build a todo API")


# ---------------------------------------------------------------------------
# Critique prompt construction
# ---------------------------------------------------------------------------


class TestCritiquePrompt:
    def test_critique_lists_issues_and_rules(self) -> None:
        graph = _graph_with_contracts(2)
        review = score_plan(
            graph, max_tasks=10, min_contract_coverage=0.6,
            min_task_score=0.5, max_estimated_cost=100.0,
        )
        review.issues = ["Weak tasks below quality threshold 0.5: t1 (0.30: vague)"]
        prompt = build_critique_prompt(
            "Build the thing", graph, review,
            {"needs_review": True, "confidence": "low",
             "concerns": "not sure about auth scope"},
        )
        assert "REVIEW FINDINGS" in prompt
        assert "Weak tasks" in prompt
        assert "REVISION RULES" in prompt
        assert "not sure about auth scope" in prompt
        assert "FULL replacement plan" in prompt
