import json
import logging
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from .config import settings
from .schemas import Run, RunStatus, TaskGraph, TaskResult, TaskStatus

logger = logging.getLogger(__name__)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    task_description TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    task_graph_json TEXT,
    final_output TEXT,
    error TEXT
);

CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    description TEXT NOT NULL,
    depends_on_json TEXT NOT NULL,
    status TEXT NOT NULL,
    output TEXT,
    model_used TEXT,
    error TEXT,
    attempts INTEGER DEFAULT 1,
    started_at TEXT,
    completed_at TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_tasks_run_id ON tasks(run_id);
CREATE INDEX IF NOT EXISTS idx_runs_user_id ON runs(user_id);
CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at);
"""


class Storage:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or settings.db_path
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def save_run(self, run: Run) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs
                (run_id, user_id, task_description, status, created_at, completed_at,
                 task_graph_json, final_output, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.user_id,
                    run.task_description,
                    run.status.value,
                    run.created_at.isoformat(),
                    run.completed_at.isoformat() if run.completed_at else None,
                    run.task_graph.model_dump_json() if run.task_graph else None,
                    run.final_output,
                    run.error,
                ),
            )

    def save_task_result(
        self,
        run_id: str,
        result: TaskResult,
        task_description: str,
        depends_on: list[str],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO tasks
                (task_id, run_id, description, depends_on_json, status, output,
                 model_used, error, attempts, started_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.task_id,
                    run_id,
                    task_description,
                    json.dumps(depends_on),
                    result.status.value,
                    result.output,
                    result.model_used,
                    result.error,
                    result.attempts,
                    result.started_at.isoformat() if result.started_at else None,
                    result.completed_at.isoformat() if result.completed_at else None,
                ),
            )

    def get_run(self, run_id: str) -> Run | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if not row:
                return None
            return self._row_to_run(row)

    def list_runs(self, user_id: str | None = None, limit: int = 50) -> list[Run]:
        with self._connect() as conn:
            if user_id:
                rows = conn.execute(
                    "SELECT * FROM runs WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                    (user_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [self._row_to_run(row) for row in rows]

    def get_task_results(self, run_id: str) -> list[TaskResult]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE run_id = ? ORDER BY task_id",
                (run_id,),
            ).fetchall()
            return [self._row_to_task_result(row) for row in rows]

    def get_task_graph(self, run_id: str) -> TaskGraph | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT task_graph_json FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if not row or not row["task_graph_json"]:
                return None
            return TaskGraph.model_validate_json(row["task_graph_json"])

    def _row_to_run(self, row: sqlite3.Row) -> Run:
        run = Run(
            run_id=row["run_id"],
            user_id=row["user_id"],
            task_description=row["task_description"],
            status=RunStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            completed_at = (
                datetime.fromisoformat(row["completed_at"])
                if row["completed_at"]
                else None
            ),
            final_output=row["final_output"],
            error=row["error"],
        )
        if row["task_graph_json"]:
            run.task_graph = TaskGraph.model_validate_json(
                row["task_graph_json"]
            )
        return run

    def _row_to_task_result(self, row: sqlite3.Row) -> TaskResult:
        started_at = (
            datetime.fromisoformat(row["started_at"])
            if row["started_at"]
            else None
        )
        completed_at = (
            datetime.fromisoformat(row["completed_at"])
            if row["completed_at"]
            else None
        )
        return TaskResult(
            task_id=row["task_id"],
            status=TaskStatus(row["status"]),
            output=row["output"],
            model_used=row["model_used"],
            error=row["error"],
            attempts=row["attempts"],
            started_at=started_at,
            completed_at=completed_at,
        )


storage = Storage()