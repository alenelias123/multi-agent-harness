# AgentCLI — Multi-Agent Development CLI (Phase 1)

A command-line tool that decomposes natural-language development tasks into a dependency graph of subtasks and executes them in parallel using a pool of free-tier LLMs via OpenRouter.

## Features

- **Natural language → Task graph**: Planner LLM breaks down your task into a DAG of subtasks
- **Parallel execution**: Independent tasks run concurrently (configurable parallelism, default 4)
- **Free-tier model routing**: Automatic fallback between free OpenRouter models on rate limits/errors
- **SQLite persistence**: Every run and task stored with `user_id`/`run_id` for future multi-user support
- **Rich CLI**: Progress streaming, history, and run inspection

## Installation

```bash
# Clone and install
git clone <repo-url>
cd agentcli
pip install -e ".[dev]"

# Or install directly
pip install -e .
```

## Configuration

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

Get a free API key at [openrouter.ai](https://openrouter.ai).

Optional settings (in `.env` or environment):
```env
# Planner model (default: openrouter/auto)
PLANNER_MODEL=google/gemini-flash-1.5

# Max parallel tasks (default: 4)
MAX_PARALLELISM=4

# Max tasks per run (default: 20)
MAX_TASKS=20

# Task retry attempts (default: 2)
TASK_MAX_RETRIES=2

# Log level (default: INFO)
LOG_LEVEL=DEBUG
```

## Usage

### Run a task
```bash
agentcli run "Create a FastAPI REST API for a todo app with CRUD operations, using SQLite and SQLAlchemy"
```

### View history
```bash
agentcli history
agentcli history -n 10
```

### Inspect a run
```bash
agentcli show <run_id>
```

## Architecture

```
agentcli/
├── cli.py              # Typer CLI entrypoint
├── config.py           # Settings, model chains, fallback config
├── planner.py          # Planning LLM call + JSON validation
├── graph.py            # TaskGraph, DAG validation, ready-set computation
├── executor.py         # Async DAG executor with semaphore concurrency control
├── model_router.py     # Model selection, fallback, retry/backoff
├── providers/
│   └── openrouter.py   # Async OpenAI-compatible client for OpenRouter
├── storage.py          # SQLite persistence (runs, tasks)
├── schemas.py          # Pydantic models (Task, TaskGraph, TaskResult, Run)
└── prompts/
    └── planner_system.txt  # System prompt enforcing strict JSON output
```

## Task Graph Schema

```json
{
  "tasks": [
    {
      "id": "t1",
      "description": "Design database schema",
      "depends_on": [],
      "task_type": "planning",
      "complexity": "low"
    }
  ]
}
```

- `task_type`: `planning` | `coding` | `analysis` | `writing` | `general`
- `complexity`: `low` | `medium` | `high`

## Model Routing

Configured in `config.py` → `task_type_models`:

```python
{
    "planning": ["google/gemini-flash-1.5", "meta-llama/llama-3.1-8b-instruct:free", ...],
    "coding": ["qwen/qwen-2.5-coder-32b-instruct:free", "meta-llama/llama-3.1-8b-instruct:free", ...],
    "analysis": [...],
    "writing": [...],
    "general": [...],
}
```

On 429/5xx errors, the router automatically tries the next model in the chain with exponential backoff.

## Database Schema

```sql
runs(run_id, user_id, task_description, status, created_at, completed_at, task_graph_json, final_output, error)
tasks(task_id, run_id, description, depends_on_json, status, output, model_used, error, attempts, started_at, completed_at)
```

All tables include `user_id` (defaults to `"local"`) for Phase 2 multi-user compatibility.

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=agentcli

# Run specific test file
pytest tests/test_graph.py
pytest tests/test_executor.py
```

## Example Output

```
$ agentcli run "Create a Python CLI tool that fetches weather data from an API"

Planning task: Create a Python CLI tool that fetches weather data from an API
┌───────────────────────────────────── Task Graph ─────────────────────────────────────┐
│ ID   │ Description                          │ Depends On │ Type      │ Complexity │
├──────┼──────────────────────────────────────┼────────────┼───────────┼────────────┤
│ t1   │ Design CLI command structure...      │ —          │ planning  │ low        │
│ t2   │ Implement API client with httpx      │ t1         │ coding    │ medium     │
│ t3   │ Add CLI commands using typer         │ t1         │ coding    │ medium     │
│ t4   │ Write unit tests with pytest         │ t2, t3     │ coding    │ medium     │
└──────┴──────────────────────────────────────┴────────────┴───────────┴────────────┘

[t1] running on gemini-flash-1.5... done
[t2] running on qwen-2.5-coder-32b... done
[t3] running on qwen-2.5-coder-32b... done
[t4] running on qwen-2.5-coder-32b... done

┌──────────────────────────── Final Result ────────────────────────────┐
│ # Weather CLI Tool                                                     │
│                                                                        │
│ A Python CLI tool for fetching weather data...                        │
└────────────────────────────────────────────────────────────────────────┘
```

## Phase 2 Roadmap (Not Implemented)

- Multi-user authentication
- Web API server
- Redis/Celery task queue
- Web UI dashboard
- Team workspaces

## License

MIT