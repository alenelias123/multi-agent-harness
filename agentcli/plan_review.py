"""Heuristic plan review: task quality scoring, budget enforcement, and self-checks.

This module implements the local (non-LLM) half of plan refinement:

- :func:`score_task` — score how small, testable, and independent a task is.
- :func:`score_plan` / :func:`find_issues` — score the whole graph and list weaknesses.
- :func:`prune_plan` — drop redundant/broad tasks and trim the plan to budget.
- :func:`build_critique_prompt` — build the stricter re-plan prompt sent to the LLM.
- :func:`compute_contract_coverage` — how many tasks carry an execution contract.
- :func:`estimate_plan_cost` — rough expected model cost of executing the plan.

All heuristics are intentionally conservative: they flag, they don't guess.
"""

from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .schemas import Complexity, Task, TaskGraph

# Words that mark a description as actionable (starts with a verb-ish word).
_ACTION_VERBS = {
    "add", "build", "configure", "create", "define", "design", "document",
    "extract", "fix", "implement", "install", "integrate", "list", "migrate",
    "refactor", "remove", "run", "set", "setup", "split", "update", "validate",
    "verify", "write",
}

# Vague words that suggest a task is too broad or undefined.
_VAGUE_TERMS = {
    "various", "stuff", "things", "etc", "everything", "something",
    "appropriately", "properly", "handle all", "as needed",
}

_BROAD_MARKERS = (
    "entire", "whole", "all the", "complete system", "everything",
    "and everything", "end to end", "end-to-end",
)

# Multi-clause signals: long conjunction chains suggest the task does too much.
_CONJUNCTIONS = (" and then ", "; ", " plus ", " as well as ")

_TESTABLE_HINTS = (
    "test", "assert", "verify", "validate", "expect", "criterion",
    "criteria", "check", "should return", "should produce", "output",
)

COMPLEXITY_COST = {
    Complexity.LOW: 1.0,
    Complexity.MEDIUM: 2.0,
    Complexity.HIGH: 4.0,
}

QUALITY_WEIGHTS = {
    "small": 0.30,
    "actionable": 0.25,
    "independent": 0.20,
    "testable": 0.15,
    "typed": 0.10,
}


@dataclass
class PlanReview:
    """Result of reviewing a plan."""

    score: float
    issues: list[str] = field(default_factory=list)
    task_scores: dict[str, float] = field(default_factory=dict)
    task_flags: dict[str, list[str]] = field(default_factory=dict)
    contract_coverage: float = 0.0
    estimated_cost: float = 0.0

    @property
    def is_acceptable(self) -> bool:
        return not self.issues


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _has_contract(task: Task) -> bool:
    return bool(task.expected_outputs) or bool(task.validation_criteria)


def score_task(task: Task, graph: TaskGraph) -> float:
    """Score a task 0..1 on smallness, actionability, testability, independence.

    A task can be *marked* by the planner with quality flags; the returned value
    is ``max(computed_score, self_declared_score)`` so a planner that already
    flags quality is rewarded, but bad tasks can still be caught.
    """
    flags: list[str] = []
    desc = _norm(task.description)
    words = desc.split()
    score = 1.0

    # --- smallness -----------------------------------------------------
    small = 1.0
    if len(words) > 60:
        small -= 0.4
        flags.append("very_long_description")
    elif len(words) > 30:
        small -= 0.2
        flags.append("long_description")
    if any(marker in desc for marker in _BROAD_MARKERS):
        small -= 0.3
        flags.append("broad_scope")
    if any(term in desc for term in _VAGUE_TERMS):
        small -= 0.2
        flags.append("vague_wording")
    if sum(desc.count(c) for c in _CONJUNCTIONS) >= 2:
        small -= 0.2
        flags.append("multi_objective")

    # --- actionability -------------------------------------------------
    actionable = 1.0
    first_word = words[0] if words else ""
    if first_word not in _ACTION_VERBS:
        actionable -= 0.5
        flags.append("not_action_oriented")

    # --- independence --------------------------------------------------
    independent = 1.0
    if len(task.depends_on) > 3:
        independent -= 0.5
        flags.append("high_fan_in")
    # A task that depends on many tasks AND has many dependents is a hub;
    # hubs make partial failure recovery harder.
    dependents = sum(1 for t in graph.tasks if task.id in t.depends_on)
    if len(task.depends_on) + dependents > 5:
        independent -= 0.2
        flags.append("hub_task")

    # --- testability ---------------------------------------------------
    testable = 0.5  # neutral by default
    if task.validation_criteria:
        testable = 1.0
    elif any(hint in desc for hint in _TESTABLE_HINTS):
        testable = 0.8
    else:
        flags.append("no_validation_criteria")

    # --- typing --------------------------------------------------------
    typed = 0.7 if task.task_type.value != "general" else 0.4
    if typed < 0.7:
        flags.append("untyped_task")

    score = (
        QUALITY_WEIGHTS["small"] * max(small, 0.0)
        + QUALITY_WEIGHTS["actionable"] * max(actionable, 0.0)
        + QUALITY_WEIGHTS["independent"] * max(independent, 0.0)
        + QUALITY_WEIGHTS["testable"] * max(testable, 0.0)
        + QUALITY_WEIGHTS["typed"] * typed
    )
    # Compound penalty: a task exhibiting many distinct quality problems is
    # worse than the sum of its parts (each flag is an independent smell).
    score -= 0.05 * (len(flags) - 1) if len(flags) > 1 else 0.0
    task.quality_flags = list(dict.fromkeys(flags))  # dedupe, keep order
    return round(max(score, 0.0), 3)


def _similarity(a: str, b: str) -> float:
    """Similarity of two descriptions with shared template scaffolding removed.

    Raw SequenceMatcher is too generous with templated planner output (e.g.
    "Implement feature module 0" vs "...1" score ~0.96). Comparing the
    *distinctive* words instead avoids treating numbered boilerplate as
    redundancy, while still catching true near-duplicates.
    """
    a_norm, b_norm = _norm(a), _norm(b)
    ratio = SequenceMatcher(None, a_norm, b_norm).ratio()
    if ratio <= 0.5:
        return ratio
    a_words = set(a_norm.split())
    b_words = set(b_norm.split())
    stopwords = {"the", "a", "an", "and", "or", "to", "of", "for", "with"}
    a_content = a_words - stopwords
    b_content = b_words - stopwords
    if not a_content or not b_content:
        return ratio
    # The words unique to each side reveal real differences between
    # otherwise similar sentences.
    diff_ratio = 1.0 - len(a_content ^ b_content) / max(
        len(a_content | b_content), 1
    )
    return min(ratio, diff_ratio)


def find_redundant_pairs(
    graph: TaskGraph, threshold: float = 0.82
) -> list[tuple[str, str]]:
    """Return pairs of near-duplicate tasks (id_a, id_b) with similarity > threshold."""
    pairs: list[tuple[str, str]] = []
    tasks = graph.tasks
    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            if _similarity(tasks[i].description, tasks[j].description) > threshold:
                pairs.append((tasks[i].id, tasks[j].id))
    return pairs


def prune_plan(graph: TaskGraph, max_tasks: int) -> tuple[TaskGraph, list[str]]:
    """Prune the weakest/redundant tasks so the plan fits the budget.

    Strategy:
    1. Drop the lower-scored task of every near-duplicate pair.
    2. If still over budget, drop lowest-scored tasks (fewest dependencies
       first among ties, so we don't orphan large subgraphs).

    Returns ``(pruned_graph, dropped_ids)``. Pruning never re-orders the graph;
    dependencies of kept tasks always remain valid because dropped tasks are
    removed from remaining ``depends_on`` lists only when the dropped task is a
    dependency (in which case its *content* survives via its duplicate).
    """
    scored = {t.id: score_task(t, graph) for t in graph.tasks}
    dropped: list[str] = []

    # 1) Near-duplicate pairs: keep the better-scored task.
    for id_a, id_b in find_redundant_pairs(graph):
        if id_a in dropped or id_b in dropped:
            continue
        loser = id_b if scored.get(id_a, 0) >= scored.get(id_b, 0) else id_a
        dropped.append(loser)

    # 2) Budget trim: drop lowest-scoring tasks until within budget.
    if len(graph.tasks) - len(dropped) > max_tasks:
        survivors = [t for t in graph.tasks if t.id not in dropped]
        survivors.sort(key=lambda t: (scored.get(t.id, 0.0), -len(t.depends_on)))
        overflow = len(survivors) - max_tasks
        dropped.extend(t.id for t in survivors[:overflow])

    dropped = list(dict.fromkeys(dropped))
    if not dropped:
        return graph, []

    keep = {t.id for t in graph.tasks} - set(dropped)
    pruned = TaskGraph(max_tasks=max_tasks)
    for task in graph.tasks:
        if task.id in keep:
            cleaned = task.model_copy(
                update={"depends_on": [d for d in task.depends_on if d in keep]}
            )
            pruned.add_task(cleaned)

    pruned.pruned_task_ids = dropped
    return pruned, dropped


def compute_contract_coverage(graph: TaskGraph) -> float:
    """Fraction of tasks with expected_outputs or validation_criteria."""
    if not graph.tasks:
        return 0.0
    covered = sum(1 for t in graph.tasks if _has_contract(t))
    return round(covered / len(graph.tasks), 3)


def estimate_plan_cost(graph: TaskGraph) -> float:
    """Rough expected cost in abstract units: sum of complexity weights."""
    return sum(COMPLEXITY_COST.get(t.complexity, 2.0) for t in graph.tasks)


def find_issues(
    graph: TaskGraph,
    max_tasks: int,
    min_contract_coverage: float,
    min_task_score: float,
    max_estimated_cost: float,
) -> list[str]:
    """Self-check the graph: return a list of concrete, actionable issues."""
    issues: list[str] = []

    # --- structural self-checks -----------------------------------------
    try:
        graph.validate_dependencies()
    except ValueError as e:
        issues.append(f"Missing dependency: {e}")
    try:
        graph.detect_cycles()
    except ValueError as e:
        issues.append(f"Cycle: {e}")

    # --- budget checks ---------------------------------------------------
    if len(graph.tasks) == 0:
        issues.append("Plan contains no tasks")
        return issues
    if len(graph.tasks) > max_tasks:
        issues.append(
            f"Too many tasks: {len(graph.tasks)} > max {max_tasks}; "
            "merge or drop the least essential ones"
        )

    cost = estimate_plan_cost(graph)
    if cost > max_estimated_cost:
        high = [t.id for t in graph.tasks if t.complexity == Complexity.HIGH]
        issues.append(
            f"Estimated cost {cost} exceeds budget {max_estimated_cost}; "
            + (
                f"split high-complexity tasks: {', '.join(high)}"
                if high
                else "reduce scope of tasks"
            )
        )

    # --- per-task quality ------------------------------------------------
    scores: dict[str, float] = {}
    flags: dict[str, list[str]] = {}
    for task in graph.tasks:
        scores[task.id] = score_task(task, graph)
        flags[task.id] = task.quality_flags

    weak = {tid: s for tid, s in scores.items() if s < min_task_score}
    if weak:
        worst = sorted(weak.items(), key=lambda kv: kv[1])[:5]
        details = ", ".join(
            f"{tid} ({s:.2f}: {', '.join(flags[tid])})" for tid, s in worst
        )
        issues.append(
            f"Weak tasks below quality threshold {min_task_score}: {details}"
        )

    # --- contracts ---------------------------------------------------------
    coverage = compute_contract_coverage(graph)
    if coverage < min_contract_coverage:
        missing = [t.id for t in graph.tasks if not _has_contract(t)]
        issues.append(
            f"Contract coverage {coverage:.0%} below required "
            f"{min_contract_coverage:.0%}; add expected_outputs and "
            f"validation_criteria to: {', '.join(missing[:8])}"
        )

    # --- redundancy ---------------------------------------------------------
    dupes = find_redundant_pairs(graph)
    if dupes:
        pairs = ", ".join(f"{a}~{b}" for a, b in dupes[:4])
        issues.append(f"Redundant near-duplicate tasks: {pairs}")

    return issues


def score_plan(
    graph: TaskGraph,
    max_tasks: int,
    min_contract_coverage: float,
    min_task_score: float,
    max_estimated_cost: float,
) -> PlanReview:
    """Score the whole plan and collect issues. One pass, no LLM calls."""
    issues = find_issues(
        graph, max_tasks, min_contract_coverage, min_task_score, max_estimated_cost
    )
    scores = {t.id: score_task(t, graph) for t in graph.tasks}
    avg = round(sum(scores.values()) / len(scores), 3) if scores else 0.0
    return PlanReview(
        score=avg,
        issues=issues,
        task_scores=scores,
        task_flags={t.id: t.quality_flags for t in graph.tasks},
        contract_coverage=compute_contract_coverage(graph),
        estimated_cost=estimate_plan_cost(graph),
    )


def build_critique_prompt(
    task_description: str,
    graph: TaskGraph,
    review: PlanReview,
    graph_review: dict[str, str] | None = None,
) -> str:
    """Build the stricter re-plan prompt with critique feedback.

    ``graph_review`` is the planner's optional self-review dict from the
    previous attempt (needs_review / confidence / concerns), included so the
    model's own concerns are folded into the next attempt.
    """
    task_lines = "\n".join(
        f"  - {t.id} [{t.task_type.value}/{t.complexity.value}] "
        f"{t.description}"
        + (f" | flags: {', '.join(t.quality_flags)}" if t.quality_flags else "")
        for t in graph.tasks
    )
    issue_lines = "\n".join(f"  {i + 1}. {issue}" for i, issue in enumerate(review.issues))

    parts = [
        "Your previous plan for this task was reviewed and found deficient.",
        "",
        f"ORIGINAL TASK: {task_description}",
        "",
        "PREVIOUS PLAN:",
        task_lines or "  (empty)",
        "",
        f"REVIEW FINDINGS (plan score {review.score:.2f}, "
        f"contract coverage {review.contract_coverage:.0%}, "
        f"estimated cost {review.estimated_cost}):",
        issue_lines or "  (none listed)",
    ]

    if graph_review:
        concerns = graph_review.get("concerns")
        if concerns:
            parts += ["", "YOUR OWN STATED CONCERNS:", f"  {concerns}"]

    parts += [
        "",
        "REVISION RULES (stricter than the first attempt):",
        "1. Produce a FULL replacement plan, not a diff.",
        "2. Fix every review finding listed above.",
        "3. Split any task that combines two objectives into separate tasks.",
        "4. Every task MUST have expected_outputs and validation_criteria.",
        "5. Keep the task count within budget and prefer small, independently "
        "verifiable tasks.",
        "6. Output ONLY the JSON object, same schema as before.",
    ]
    return "\n".join(parts)

