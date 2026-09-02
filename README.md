# AgentCLI — Multi-Agent Development CLI

A standalone command-line tool that decomposes natural-language development tasks into a dependency graph of subtasks and executes them in parallel using a pool of free-tier LLMs.

## Installation

### From source (recommended)

```bash
git clone https://github.com/alenelias123/multi-agent-harness.git
cd multi-agent-harness
pip install .
```

Or for development:

```bash
pip install -e ".[dev]"
```

### Verify it works

```bash
agentcli --version
agentcli --help
```

## Quick Start

```bash
# 1. Set up configuration (one-time)
agentcli config init

# 2. Edit ~/.config/agentcli/.env and add your API key
#    (get a free key at https://openrouter.ai/keys)

# 3. Run a task
agentcli run "Create a FastAPI REST API for a todo app with CRUD operations"

# 4. Save output to a file
agentcli run "Write a Python fizzbuzz" -o fizzbuzz.md
```

## Usage

### Run a task

```bash
agentcli run "Create a FastAPI REST API for a todo app with CRUD operations, using SQLite and SQLAlchemy"
```

### Save output to file

```bash
agentcli run "Write a Python script that scrapes Hacker News" -o output.md
```

### Interactive chat

```bash
# Start a chat session
agentcli chat

# With a specific model
agentcli chat --model openrouter/google/gemma-4-31b-it:free
```

Chat commands:
| Command | Description |
|---------|-------------|
| `/help` | Show all commands |
| `/clear` | Clear conversation history |
| `/model` | Show active model |
| `/export` | Save conversation to a file |
| `/system` | Show the system prompt |
| `/quit` | Exit |

### View history

```bash
agentcli history
agentcli history -n 10
```

### Inspect a run

```bash
agentcli show <run_id>
```

### Configuration

```bash
# Show config/data paths
agentcli config path

# Create config directory and .env template
agentcli config init
```

## Configuration

AgentCLI uses XDG-compliant paths for configuration and data storage:

| Item | Path |
|------|------|
| Config dir | `~/.config/agentcli/` |
| Data dir | `~/.local/share/agentcli/` |
| Database | `~/.local/share/agentcli/agentcli.db` |
| Env file | `~/.config/agentcli/.env` |

### Environment Variables

Create a `.env` file in `~/.config/agentcli/` (or in your CWD):

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

Get a free API key at [openrouter.ai](https://openrouter.ai) (requires adding $10 credits for 1000 free requests/day, or use the 50 req/day free tier).

Optional settings:

```env
# Planner model (default: openrouter/free — auto-routes to best available free model)
PLANNER_MODEL=openrouter/free

# Max parallel tasks (default: 4)
MAX_PARALLELISM=4

# Max tasks per run (default: 20)
MAX_TASKS=20

# Task retry attempts (default: 2)
TASK_MAX_RETRIES=2

# Log level (default: INFO)
LOG_LEVEL=DEBUG

# Enable Freebuff provider (requires: npm install -g freebuff)
FREEBUFF_ENABLED=true
```

### CLI Options

Override the database location on any command:

```bash
agentcli run "task" --db-path /tmp/custom.db
agentcli history --db-path /tmp/custom.db
```

### Python Module

You can also run it as a Python module:

```bash
python -m agentcli --version
python -m agentcli run "your task here"
```

## Architecture

```
agentcli/
├── __main__.py           # python -m agentcli entrypoint
├── _version.py           # Single source of truth for version
├── cli.py                # Typer CLI with run, chat, history, show, config commands
├── chat.py               # Interactive chat REPL with Rich rendering
├── config.py             # Settings, XDG paths, model chains, fallback config
├── planner.py            # Planning LLM call + JSON validation
├── graph.py              # TaskGraph, DAG validation, ready-set computation
├── executor.py           # Async DAG executor with semaphore concurrency control
├── model_router.py       # Model selection, fallback, retry/backoff
├── providers/
│   ├── base.py           # Abstract provider interface
│   ├── openrouter.py     # Async OpenAI-compatible client for OpenRouter
│   └── freebuff.py       # Freebuff CLI wrapper
├── storage.py            # SQLite persistence (runs, tasks)
├── schemas.py            # Pydantic models (Task, TaskGraph, TaskResult, Run)
└── prompts/
    ├── chat_system.txt     # System prompt for interactive chat
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

Configured in `config.py` → `task_type_models` (August 2026 free models):

| Task Type | Models |
|-----------|--------|
| planning | nemotron-3-ultra-550b, gemma-4-31b-it, freebuff, openrouter/free |
| coding | gemma-4-31b-it, north-mini-code, nemotron-3.5-lightning, freebuff, openrouter/free |
| analysis | nemotron-3-ultra-550b, gemma-4-31b-it, freebuff, openrouter/free |
| writing | gemma-4-31b-it, nemotron-3-ultra-550b, minimax-m3, freebuff, openrouter/free |
| general | nemotron-3.5-lightning, gemma-4-31b-it, stealth/ox-alpha, freebuff, openrouter/free |

On 429/5xx errors, the router automatically tries the next model in the chain with exponential backoff. The `openrouter/free` meta-model always appears as the final fallback.

### Free-tier rate limits

OpenRouter's free tier has a **50 requests/day** limit (or 1000/day with $10 credits). The CLI uses up to `N × retries × models` LLM calls per run, where N is the number of tasks. Plan accordingly or add credits at [openrouter.ai/settings/credits](https://openrouter.ai/settings/credits).

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

[t1] running on nemotron-3-ultra-550b... done
[t2] running on gemma-4-31b-it... done
[t3] running on gemma-4-31b-it... done
[t4] running on gemma-4-31b-it... done

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
