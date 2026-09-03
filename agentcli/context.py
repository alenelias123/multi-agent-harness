"""Context sharing system for multiple agents.

Provides a shared context store that allows multiple agents running in
parallel to read/write context, enabling coordination and knowledge transfer
between tasks.

Architecture:
  - SharedContext: Thread-safe in-memory store with optional persistence
  - ContextEntry: Individual context items with metadata
  - AgentContext: Per-agent context wrapper with namespace isolation
  - ContextBridge: Bridges context between chat sessions and task executor

Storage:
  Context is stored in SQLite alongside run data, enabling cross-session
  context sharing. Each entry has:
    - namespace: isolated scope (run_id, agent_id, or "global")
    - key: identifier for the context item
    - value: the actual content
    - metadata: origin, timestamp, tags for filtering
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

logger = logging.getLogger(__name__)


# ── Data Models ──────────────────────────────────────────────────────────

@dataclass
class ContextEntry:
    """A single piece of shared context."""
    key: str
    value: str
    namespace: str = "global"
    source_agent: str | None = None
    source_task_id: str | None = None
    tags: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None  # TTL-based expiry
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "namespace": self.namespace,
            "source_agent": self.source_agent,
            "source_task_id": self.source_task_id,
            "tags": self.tags,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "metadata": self.metadata,
        }


@dataclass
class AgentContext:
    """Per-agent context wrapper with namespace isolation.

    An agent can read from shared/global context and write to its own
    namespace.
    """
    agent_id: str
    namespace: str
    store: SharedContext

    def write(self, key: str, value: str, tags: list[str] | None = None,
              ttl: float | None = None, **metadata: Any) -> None:
        """Write to this agent's namespace."""
        self.store.write(
            key=key,
            value=value,
            namespace=self.namespace,
            source_agent=self.agent_id,
            tags=tags or [],
            ttl=ttl,
            metadata=metadata,
        )

    def read(self, key: str, fallback_to_global: bool = True) -> str | None:
        """Read from this agent's namespace, optionally falling back to global."""
        val = self.store.read(key, namespace=self.namespace)
        if val is None and fallback_to_global:
            val = self.store.read(key, namespace="global")
        return val

    def read_all(self, namespace: str | None = None,
                 tags: list[str] | None = None) -> list[ContextEntry]:
        """Read all entries, optionally filtered by namespace/tags."""
        ns = namespace or self.namespace
        return self.store.read_all(namespace=ns, tags=tags)

    def share_with(self, target_agent_id: str, key: str, value: str,
                   tags: list[str] | None = None) -> None:
        """Write to another agent's namespace."""
        self.store.write(
            key=key,
            value=value,
            namespace=f"agent:{target_agent_id}",
            source_agent=self.agent_id,
            tags=tags or [],
            metadata={"shared": True, "target_agent": target_agent_id},
        )


# ── Shared Context Store ─────────────────────────────────────────────────

class SharedContext:
    """Thread-safe shared context store with SQLite persistence.

    Context is namespaced:
      - "global"     : shared across all agents/runs
      - "run:{id}"    : scoped to a specific run
      - "agent:{id}"  : scoped to a specific agent
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or Path(":memory:")
        self._memory_cache: dict[str, dict[str, ContextEntry]] = defaultdict(dict)
        self._is_memory = db_path is None
        # For in-memory databases, keep a persistent connection
        self._persistent_conn: sqlite3.Connection | None = None
        if self._is_memory:
            self._persistent_conn = sqlite3.connect(":memory:")
            self._persistent_conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS context_entries (
                    key TEXT NOT NULL,
                    namespace TEXT NOT NULL,
                    value TEXT NOT NULL,
                    source_agent TEXT,
                    source_task_id TEXT,
                    tags_json TEXT DEFAULT '[]',
                    created_at REAL NOT NULL,
                    expires_at REAL,
                    metadata_json TEXT DEFAULT '{}',
                    PRIMARY KEY (key, namespace)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_context_namespace
                ON context_entries(namespace)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_context_tags
                ON context_entries(tags_json)
            """)

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        if self._persistent_conn is not None:
            # Reuse the persistent connection for in-memory databases
            try:
                yield self._persistent_conn
                self._persistent_conn.commit()
            except Exception:
                self._persistent_conn.rollback()
                raise
        else:
            conn = sqlite3.connect(str(self._db_path))
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def write(
        self,
        key: str,
        value: str,
        namespace: str = "global",
        source_agent: str | None = None,
        source_task_id: str | None = None,
        tags: list[str] | None = None,
        ttl: float | None = None,
        **metadata: Any,
    ) -> ContextEntry:
        """Write a context entry."""
        tags = tags or []
        expires_at = time.time() + ttl if ttl else None

        entry = ContextEntry(
            key=key,
            value=value,
            namespace=namespace,
            source_agent=source_agent,
            source_task_id=source_task_id,
            tags=tags,
            created_at=time.time(),
            expires_at=expires_at,
            metadata=metadata,
        )

        # Update in-memory cache
        cache_key = f"{namespace}:{key}"
        self._memory_cache[namespace][key] = entry

        # Persist to SQLite
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO context_entries
                (key, namespace, value, source_agent, source_task_id,
                 tags_json, created_at, expires_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    namespace,
                    value,
                    source_agent,
                    source_task_id,
                    json.dumps(tags),
                    entry.created_at,
                    expires_at,
                    json.dumps(metadata),
                ),
            )

        logger.debug(f"Context write: [{namespace}] {key} = {value[:50]}...")
        return entry

    def read(self, key: str, namespace: str = "global") -> str | None:
        """Read a single context entry's value."""
        # Check memory cache first
        entry = self._memory_cache.get(namespace, {}).get(key)
        if entry and not entry.is_expired:
            return entry.value

        # Fall back to SQLite
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM context_entries WHERE key = ? AND namespace = ?",
                (key, namespace),
            ).fetchone()

        if not row:
            return None

        expires_at = row["expires_at"]
        if expires_at and time.time() > expires_at:
            # Expired — clean up
            self.delete(key, namespace)
            return None

        return row["value"]

    def read_all(
        self,
        namespace: str | None = None,
        tags: list[str] | None = None,
        limit: int = 100,
    ) -> list[ContextEntry]:
        """Read all entries, optionally filtered by namespace and tags."""
        with self._connect() as conn:
            if namespace and tags:
                rows = conn.execute(
                    "SELECT * FROM context_entries WHERE namespace = ?",
                    (namespace,),
                ).fetchall()
            elif namespace:
                rows = conn.execute(
                    "SELECT * FROM context_entries WHERE namespace = ? LIMIT ?",
                    (namespace, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM context_entries LIMIT ?",
                    (limit,),
                ).fetchall()

        entries = []
        for row in rows:
            # Filter by tags if specified
            if tags:
                row_tags = json.loads(row["tags_json"])
                if not any(t in row_tags for t in tags):
                    continue

            expires_at = row["expires_at"]
            if expires_at and time.time() > expires_at:
                continue

            entries.append(ContextEntry(
                key=row["key"],
                value=row["value"],
                namespace=row["namespace"],
                source_agent=row["source_agent"],
                source_task_id=row["source_task_id"],
                tags=json.loads(row["tags_json"]),
                created_at=row["created_at"],
                expires_at=row["expires_at"],
                metadata=json.loads(row["metadata_json"]),
            ))

        return entries

    def delete(self, key: str, namespace: str = "global") -> bool:
        """Delete a context entry."""
        self._memory_cache.get(namespace, {}).pop(key, None)
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM context_entries WHERE key = ? AND namespace = ?",
                (key, namespace),
            )
            return cursor.rowcount > 0

    def clear_namespace(self, namespace: str) -> int:
        """Delete all entries in a namespace."""
        self._memory_cache.pop(namespace, None)
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM context_entries WHERE namespace = ?",
                (namespace,),
            )
            return cursor.rowcount

    def search(self, query: str, namespace: str | None = None,
               tags: list[str] | None = None) -> list[ContextEntry]:
        """Full-text search across context values."""
        all_entries = self.read_all(namespace=namespace, tags=tags, limit=1000)
        query_lower = query.lower()
        return [
            e for e in all_entries
            if query_lower in e.value.lower() or query_lower in e.key.lower()
        ]

    def get_context_agent(self, agent_id: str) -> AgentContext:
        """Get an agent-scoped context wrapper."""
        return AgentContext(
            agent_id=agent_id,
            namespace=f"agent:{agent_id}",
            store=self,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get context store statistics."""
        with self._connect() as conn:
            stats_row = conn.execute(
                "SELECT COUNT(*) as entry_count, "
                "COUNT(DISTINCT namespace) as namespace_count "
                "FROM context_entries"
            ).fetchone()
            tag_rows = conn.execute(
                "SELECT tags_json FROM context_entries"
            ).fetchall()

        all_tags: set[str] = set()
        for tag_row in tag_rows:
            all_tags.update(json.loads(tag_row["tags_json"]))

        return {
            "total_entries": stats_row["entry_count"],
            "total_namespaces": stats_row["namespace_count"],
            "unique_tags": sorted(all_tags),
        }


# ── Context Bridge ───────────────────────────────────────────────────────

class ContextBridge:
    """Bridges context between the task executor and chat sessions.

    When tasks complete, their outputs are automatically stored in context.
    Downstream tasks can read upstream context. Chat sessions can read
    and write context.
    """

    def __init__(self, context: SharedContext) -> None:
        self.context = context

    def store_task_output(
        self,
        run_id: str,
        task_id: str,
        output: str,
        agent_id: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Store a completed task's output in shared context."""
        namespace = f"run:{run_id}"
        self.context.write(
            key=f"task:{task_id}:output",
            value=output,
            namespace=namespace,
            source_agent=agent_id,
            source_task_id=task_id,
            tags=(tags or []) + ["task_output", f"task:{task_id}"],
            task_id=task_id,
        )

    def get_task_output(self, run_id: str, task_id: str) -> str | None:
        """Retrieve a task's output from shared context."""
        return self.context.read(
            key=f"task:{task_id}:output",
            namespace=f"run:{run_id}",
        )

    def get_all_task_outputs(self, run_id: str) -> dict[str, str]:
        """Get all task outputs for a run."""
        entries = self.context.read_all(
            namespace=f"run:{run_id}",
            tags=["task_output"],
            limit=100,
        )
        return {
            e.source_task_id: e.value
            for e in entries
            if e.source_task_id
        }

    def store_shared_state(
        self,
        run_id: str,
        key: str,
        value: str,
        agent_id: str | None = None,
    ) -> None:
        """Store shared state that all agents in a run can access."""
        self.context.write(
            key=f"state:{key}",
            value=value,
            namespace=f"run:{run_id}",
            source_agent=agent_id,
            tags=["shared_state"],
        )

    def get_shared_state(self, run_id: str, key: str) -> str | None:
        """Get shared state for a run."""
        return self.context.read(
            key=f"state:{key}",
            namespace=f"run:{run_id}",
        )

    def store_agent_memory(
        self,
        agent_id: str,
        key: str,
        value: str,
        ttl: float | None = None,
    ) -> None:
        """Store per-agent memory with optional TTL."""
        self.context.write(
            key=key,
            value=value,
            namespace=f"agent:{agent_id}",
            source_agent=agent_id,
            tags=["agent_memory"],
            ttl=ttl,
        )

    def get_agent_memory(self, agent_id: str, key: str) -> str | None:
        """Get per-agent memory."""
        return self.context.read(
            key=key,
            namespace=f"agent:{agent_id}",
        )

    def store_global(self, key: str, value: str, tags: list[str] | None = None) -> None:
        """Store in global namespace, accessible to all agents/runs."""
        self.context.write(
            key=key,
            value=value,
            namespace="global",
            tags=tags or [],
        )

    def get_global(self, key: str) -> str | None:
        """Read from global namespace."""
        return self.context.read(key=key, namespace="global")


# ── Module-level singleton ───────────────────────────────────────────────

_shared_context: SharedContext | None = None
_context_bridge: ContextBridge | None = None


def get_shared_context(db_path: Path | None = None) -> SharedContext:
    """Get or create the shared context singleton."""
    global _shared_context
    if _shared_context is None:
        _shared_context = SharedContext(db_path=db_path)
    return _shared_context


def get_context_bridge(db_path: Path | None = None) -> ContextBridge:
    """Get or create the context bridge singleton."""
    global _context_bridge
    if _context_bridge is None:
        ctx = get_shared_context(db_path=db_path)
        _context_bridge = ContextBridge(ctx)
    return _context_bridge
