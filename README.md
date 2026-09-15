# AgentCLI — Multi-Agent Development CLI

A standalone command-line tool that decomposes natural-language development tasks into a
dependency graph of subtasks and executes them in parallel using a pool of free-tier and
pay-per-use LLMs.

- **Intelligent planner** — generates a task DAG with execution contracts, then
  self-reviews and iteratively refines its own plan (quality scoring, budget
  enforcement, redundancy pruning, critique-driven re-planning).
- **Resilient executor** — runs the DAG with adaptive retries (failure classification,
  model switching, prompt adaptation), output validation, and partial failure recovery.
- **Multi-provider routing** — OpenRouter, OpenCode, and Freebuff with automatic
  fallback, health tracking, and exponential backoff.
- **Shared context** — a namespaced SQLite-backed context store lets parallel tasks
  read each other's outputs and share state.
- **Full-screen TUI** — everything the CLI offers, bundled into one interactive
  dashboard. Runs by default with no arguments.

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

# 2. Install and configure Freebuff (one command!)
agentcli setup

# 3. Launch the TUI dashboard (default when no command is given)
agentcli

# ...or run a task directly from the CLI
agentcli run "Create a FastAPI REST API for a todo app with CRUD operations"
```

## Usage

### Launch the TUI dashboard

Running `agentcli` with no command opens the full-screen dashboard, which bundles
every CLI feature into one interactive screen:

```bash
agentcli            # open the dashboard
agentcli dashboard --run <run_id>   # focus on a specific run
```

Tabs: **Plan** (generate, edit, approve & run), **Logs** (live task output),
**History** (past runs with full detail), **Chat**, **Context**, **Sessions**, and
**Config**. Key bindings: `q` quit, `r` refresh, `e` edit mode, `a` add task,
`x` delete task.

### Run a task

```bash
agentcli run "Create a FastAPI REST API for a todo app, using SQLite and SQLAlchemy"
```

Useful options:

```bash
# Review the plan before executing (y = proceed, N = cancel, e = edit tasks)
agentcli run "Build a web scraper" --review

# Save the final aggregated output to a file
agentcli run "Write a Python fizzbuzz" -o fizzbuzz.md

# Don't stream task outputs live
agentcli run "Refactor the parser" --no-stream

# Override the database location
agentcli run "task" --db-path /tmp/custom.db
```

### Interactive chat

```bash
agentcli chat
agentcli chat --model openrouter/google/gemma-4-31b-it:free
agentcli chat --system-prompt my_prompt.txt
```

Chat commands:

| Command | Description |
|---------|-------------|
| `/help` | Show all commands |
| `/clear` | Clear conversation history |
| `/model` | Show active model |
| `/history` | Show message counts |
| `/export [file]` | Save conversation to a Markdown file |
| `/system` | Show the system prompt |
| `/quit` | Exit |

### View history & inspect runs

```bash
agentcli history          # recent runs
agentcli history -n 10
agentcli show <run_id>    # task graph, per-task results, final output
```

### Parallel AI sessions (tmux)

Run multiple chat sessions in parallel, each in its own tmux window:

```bash
agentcli sessions create --name research --model openrouter/google/gemma-4-31b-it:free
agentcli sessions list
agentcli sessions attach <session_id>       # enter the session (Ctrl-B d to detach)
agentcli sessions send <session_id> -m "explain the plan"
agentcli sessions logs <session_id> -l 100  # peek at output without attaching
agentcli sessions kill <session_id>
agentcli sessions kill-all
```

Requires `tmux` (`sudo apt install tmux` / `brew install tmux`).

### Context sharing

Tasks automatically store their outputs in a namespaced context store so downstream
and parallel tasks can read them. You can manage it directly:

```bash
agentcli context stats
agentcli context ls --namespace run:abc123
agentcli context get --key task:t1:output --run-id abc123
agentcli context set --key arch --value "microservices" --run-id abc123
agentcli context search --key "FastAPI"
agentcli context clear --namespace run:abc123
```

Namespaces:

| Namespace | Scope | Example |
|-----------|-------|---------|
| `global` | All agents, all runs | Project conventions, shared knowledge |
| `run:{id}` | Single run | Task outputs, run-specific state |
| `agent:{id}` | Single agent | Agent memory, preferences (optional TTL) |

### Portable context files (`.agentref.json`)

`.agentref.json` files are agent-native context snapshots: compact JSON, designed to
be read **only by agents**. A downstream agent can load a *slice* (by namespace, key
prefix, or tags) and pay prompt tokens only for what it needs.

```bash
# Export the current context store to .agentref.json
agentcli ref create --run-id <run_id>

# Inspect it
agentcli ref show
agentcli ref stats

# Slice: only entries tagged 'architecture' or in namespace 'run:abc123'
agentcli ref slice --tags architecture --output .agentref.slice.json

# Import back into the context store
agentcli ref import --source .agentref.json
```

### Setup commands

```bash
# Freebuff: install CLI + check/configure auth
agentcli setup
agentcli setup --auth                 # interactive auth flow
agentcli setup --token YOUR_TOKEN     # set token directly

# OpenCode: configure API key and tier
agentcli opencode --api-key YOUR_KEY --tier zen   # or --tier go

# Both providers in one go
agentcli providers --opencode-key YOUR_KEY --freebuff-auth
```

### Configuration

```bash
agentcli config path   # show config/data/db/env paths
agentcli config init   # create config dir and .env template
```

Paths (XDG-compliant):

| Item | Path |
|------|------|
| Config dir | `~/.config/agentcli/` |
| Data dir | `~/.local/share/agentcli/` |
| Database | `~/.local/share/agentcli/agentcli.db` |
| Env file | `~/.config/agentcli/.env` (or `./.env`) |

## Configuration Reference

Environment variables (in `./.env` or `~/.config/agentcli/.env`):

### Providers

```env
# OpenRouter (free tier available — https://openrouter.ai/keys)
OPENROUTER_API_KEY=sk-or-...

# OpenCode (pay-per-use Zen or low-cost Go — https://opencode.ai/auth)
OPENCODE_API_KEY=
OPENCODE_TIER=zen                 # 'zen' (pay-per-use) or 'go' (subscription)
# OPENCODE_BASE_URL=              # auto-selected from tier if unset

# Freebuff (zero-config, via the freebuff CLI)
FREEBUFF_ENABLED=true
FREEBUFF_TOKEN=                   # auto-detected from freebuff config if installed
FREEBUFF_AUTO_INSTALL=true
FREEBUFF_TIMEOUT=120
```

Custom OpenAI-compatible providers can be added via the `custom_providers` JSON
setting (map of name → `{base_url, api_key, timeout, enabled, priority}`).

### Planner

```env
PLANNER_MODEL=openrouter/free         # model used for planning
PLANNER_MAX_RETRIES=2
PLANNER_REVIEW_ENABLED=true           # master switch for heuristic plan review
PLANNER_MIN_TASK_SCORE=0.55           # tasks below this score trigger re-planning
PLANNER_MIN_CONTRACT_COVERAGE=0.6     # fraction of tasks that must declare contracts
PLANNER_MAX_ESTIMATED_COST=40.0       # complexity-weight budget (low=1, med=2, high=4)
```

### Execution

```env
MAX_TASKS=20              # max tasks per plan
MAX_PARALLELISM=4         # max concurrent task executions
TASK_MAX_RETRIES=2        # max attempts per task
LOG_LEVEL=INFO
```

## How Planning Works

The planner is a bounded self-review loop (up to 3 LLM calls per plan):

1. **Generate** — the planner LLM outputs a strict JSON plan: tasks with
   `depends_on`, `task_type`, `complexity`, and an *execution contract*
   (`expected_inputs`, `expected_outputs`, `validation_criteria`).
2. **Self-check** — local heuristics grade every task (smallness, actionability,
   independence, testability, typing), check structural integrity (cycles, unknown
   dependencies), budget (estimated cost), contract coverage, and near-duplicate
   redundancy.
3. **Local repair** — duplicates and over-budget plans are pruned without another
   LLM call; dependencies of kept tasks are re-validated.
4. **Critique re-plan** — if issues remain, a stricter re-plan prompt is built that
   includes every concrete finding (plus the model's own stated concerns). The
   best-scoring attempt wins.

Each task keeps a `quality_score` and `quality_flags`, and the graph carries
`quality_score`, `contract_coverage`, `estimated_cost`, and `review_iterations`.

## How Execution Works

- **DAG scheduling** — tasks run as soon as all dependencies succeed, up to
  `MAX_PARALLELISM` concurrent tasks, prioritized by critical-path length.
- **Context injection** — every task prompt includes upstream task outputs plus
  shared state; outputs are stored back into the context store.
- **Execution contracts** — `expected_inputs` / `expected_outputs` /
  `validation_criteria` are rendered into the task prompt so the model knows
  exactly what to produce.
- **Output validation** — task-type-specific validators (code presence, structure,
  minimum length, criteria keywords) reject weak outputs before they propagate.
- **Adaptive retries** — failures are classified (rate limit, timeout, network,
  model error, invalid output, validation failed) and drive exponential-backoff
  retries with prompt adaptation on repeat attempts.
- **Partial failure recovery** — when a task fails, only its direct dependents are
  skipped; parallel branches continue. Skipped tasks are recorded in the run.

## Model Routing

Models are configured per task type in `config.py` → `task_type_models`, using
`provider/model` references:

| Task Type | Example chain (first = preferred) |
|-----------|-----------------------------------|
| planning | opencode/claude-sonnet-4-5, opencode/gemini-2.5-pro, nemotron-3-ultra, gemma-4-31b, freebuff, openrouter/free |
| coding | opencode/claude-sonnet-4-5, opencode/codex-mini, gemini-2.5-flash, gemma-4-31b, north-mini-code, freebuff, openrouter/free |
| analysis | opencode/claude-sonnet-4-5, opencode/gemini-2.5-pro, nemotron-3-ultra, gemma-4-31b, freebuff, openrouter/free |
| writing | opencode/gemini-2.5-flash, opencode/claude-haiku, gemma-4-31b, nemotron-3-ultra, minimax-m3, freebuff, openrouter/free |
| general | opencode/gemini-2.5-flash, opencode/claude-haiku, nemotron-3.5-lightning, gemma-4-31b, freebuff, openrouter/free |

On 429/5xx errors the router automatically tries the next model in the chain with
jittered exponential backoff, tracks per-model health (3 consecutive failures ⇒
60 s cooldown), and falls back to `openrouter/free` last.

### Free-tier rate limits

OpenRouter's free tier allows **50 requests/day** (or 1000/day with $10 credits).
A run can use up to `N × retries × models` LLM calls, where N is the number of
tasks. Plan accordingly, or add credits at
[openrouter.ai/settings/credits](https://openrouter.ai/settings/credits).

## Architecture

```
agentcli/
├── __main__.py           # python -m agentcli entrypoint
├── _version.py           # Single source of truth for version
├── cli.py                # Typer CLI: run, chat, history, show, config, setup,
│                         #   context, ref, sessions, opencode, providers, dashboard
├── tui.py                # Textual full-screen dashboard (default launch mode)
├── chat.py               # Interactive chat REPL with Rich rendering
├── config.py             # Settings, XDG paths, model chains, planner tuning
├── setup.py              # Freebuff/OpenCode install, auth, token management
├── context.py            # SharedContext store, ContextBridge, AgentReference
├── planner.py            # Plan generation, self-review loop, critique re-planning
├── plan_review.py        # Heuristic scoring, pruning, budgets, critique prompts
├── graph.py              # TaskGraph, DAG validation, ASCII rendering
├── executor.py           # DAG executor: validation, adaptive retries, recovery
├── model_router.py       # Provider routing, fallback, health tracking
├── sessions.py           # tmux-backed parallel chat session manager
├── storage.py            # SQLite persistence (runs, task results)
├── schemas.py            # Pydantic models (Task, TaskGraph, TaskResult, Run)
├── providers/
│   ├── base.py           # Abstract provider interface
│   ├── openrouter.py     # Async OpenAI-compatible client for OpenRouter
│   ├── opencode.py       # OpenCode Zen/Go clients
│   └── freebuff.py       # Freebuff CLI wrapper with token detection
└── prompts/
    ├── chat_system.txt     # System prompt for interactive chat
    └── planner_system.txt  # System prompt enforcing strict JSON plan output

scripts/
└── install_freebuff.sh   # Shell script for standalone Freebuff installation
```

## Task Graph Schema

```json
{
  "needs_review": false,
  "confidence": "high",
  "concerns": "",
  "tasks": [
    {
      "id": "t1",
      "description": "Design database schema",
      "depends_on": [],
      "task_type": "planning",
      "complexity": "low",
      "expected_inputs": ["list of todo attributes"],
      "expected_outputs": ["schema definition with columns and types"],
      "validation_criteria": ["schema includes a primary key"]
    }
  ]
}
```

- `task_type`: `planning` | `coding` | `analysis` | `writing` | `general`
- `complexity`: `low` | `medium` | `high`
- The contract fields are optional but strongly encouraged — they raise task
  quality scores and are rendered into the executor prompt.

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

## Development

```bash
pip install -e ".[dev]"

# Run the test suite (180 tests)
pytest

# Type checking
mypy agentcli

# Linting / auto-fix
ruff check agentcli tests
ruff check agentcli tests --fix
```

Code style: ruff with `line-length = 100`, targeting Python 3.11+. All three
gates (pytest, ruff, mypy) are expected to pass before committing.

### Programmatic usage

```python
from agentcli.context import get_shared_context, get_context_bridge

# Get the shared context store
ctx = get_shared_context()

# Write / read / search
ctx.write(key="arch", value="microservices", namespace="global", tags=["architecture"])
value = ctx.read(key="arch", namespace="global")
results = ctx.search(query="FastAPI")

# Context bridge used by the executor
bridge = get_context_bridge()
bridge.store_task_output(run_id="run-1", task_id="t1", output="done")
bridge.store_shared_state(run_id="run-1", key="project_context", value="Python + FastAPI")

# Portable agentref files
from agentcli.context import export_reference, import_reference
ref = export_reference(ctx, output_path=".agentref.json")   # save a snapshot
ref.slice(tags=["architecture"]).save(".agentref.slice.json")  # partial load
count = import_reference(".agentref.json")                  # import into the store
```

You can also run it as a Python module:

```bash
python -m agentcli --version
python -m agentcli run "your task here"
```

## Roadmap

- Multi-user authentication
- Web API server
- Redis/Celery task queue
- Team workspaces

## License

MIT
