"""Tests for the context sharing system."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from agentcli.context import (
    AgentContext,
    ContextBridge,
    ContextEntry,
    SharedContext,
)


class TestContextEntry:
    def test_creation(self) -> None:
        entry = ContextEntry(key="k1", value="v1")
        assert entry.key == "k1"
        assert entry.value == "v1"
        assert entry.namespace == "global"
        assert entry.tags == []
        assert entry.is_expired is False

    def test_expiry(self) -> None:
        entry = ContextEntry(
            key="k1", value="v1",
            expires_at=time.time() - 1,  # already expired
        )
        assert entry.is_expired is True

    def test_expiry_future(self) -> None:
        entry = ContextEntry(
            key="k1", value="v1",
            expires_at=time.time() + 3600,
        )
        assert entry.is_expired is False

    def test_to_dict(self) -> None:
        entry = ContextEntry(
            key="k1", value="v1", namespace="ns",
            tags=["a", "b"],
        )
        d = entry.to_dict()
        assert d["key"] == "k1"
        assert d["value"] == "v1"
        assert d["namespace"] == "ns"
        assert d["tags"] == ["a", "b"]


class TestSharedContext:
    def test_write_and_read(self) -> None:
        ctx = SharedContext()
        ctx.write(key="k1", value="hello", namespace="global")
        assert ctx.read("k1", "global") == "hello"

    def test_namespaces(self) -> None:
        ctx = SharedContext()
        ctx.write(key="k1", value="global-val", namespace="global")
        ctx.write(key="k1", value="ns-val", namespace="run:abc")

        assert ctx.read("k1", "global") == "global-val"
        assert ctx.read("k1", "run:abc") == "ns-val"

    def test_read_missing(self) -> None:
        ctx = SharedContext()
        assert ctx.read("nonexistent") is None

    def test_overwrite(self) -> None:
        ctx = SharedContext()
        ctx.write(key="k1", value="first")
        ctx.write(key="k1", value="second")
        assert ctx.read("k1") == "second"

    def test_delete(self) -> None:
        ctx = SharedContext()
        ctx.write(key="k1", value="v1")
        assert ctx.delete("k1") is True
        assert ctx.read("k1") is None

    def test_delete_nonexistent(self) -> None:
        ctx = SharedContext()
        assert ctx.delete("nonexistent") is False

    def test_clear_namespace(self) -> None:
        ctx = SharedContext()
        ctx.write(key="k1", value="v1", namespace="ns1")
        ctx.write(key="k2", value="v2", namespace="ns1")
        ctx.write(key="k3", value="v3", namespace="ns2")

        cleared = ctx.clear_namespace("ns1")
        assert cleared == 2
        assert ctx.read("k1", "ns1") is None
        assert ctx.read("k3", "ns2") == "v3"

    def test_read_all(self) -> None:
        ctx = SharedContext()
        ctx.write(key="k1", value="v1", namespace="ns")
        ctx.write(key="k2", value="v2", namespace="ns")
        ctx.write(key="k3", value="v3", namespace="other")

        entries = ctx.read_all(namespace="ns")
        assert len(entries) == 2
        assert {e.key for e in entries} == {"k1", "k2"}

    def test_read_all_with_tags(self) -> None:
        ctx = SharedContext()
        ctx.write(key="k1", value="v1", namespace="ns", tags=["a"])
        ctx.write(key="k2", value="v2", namespace="ns", tags=["b"])
        ctx.write(key="k3", value="v3", namespace="ns", tags=["a"])

        entries = ctx.read_all(namespace="ns", tags=["a"])
        assert len(entries) == 2
        assert {e.key for e in entries} == {"k1", "k3"}

    def test_search(self) -> None:
        ctx = SharedContext()
        ctx.write(key="api_design", value="REST API with FastAPI")
        ctx.write(key="db_schema", value="SQLAlchemy models")
        ctx.write(key="test_plan", value="pytest-based testing")

        results = ctx.search("FastAPI")
        assert len(results) == 1
        assert results[0].key == "api_design"

    def test_search_by_key(self) -> None:
        ctx = SharedContext()
        ctx.write(key="api_design", value="some design")

        results = ctx.search("api")
        assert len(results) == 1

    def test_stats(self) -> None:
        ctx = SharedContext()
        ctx.write(key="k1", value="v1", namespace="ns1", tags=["a"])
        ctx.write(key="k2", value="v2", namespace="ns2", tags=["b"])

        stats = ctx.get_stats()
        assert stats["total_entries"] == 2
        assert stats["total_namespaces"] == 2
        assert set(stats["unique_tags"]) == {"a", "b"}

    def test_get_agent_context(self) -> None:
        ctx = SharedContext()
        agent_ctx = ctx.get_context_agent("agent-1")

        agent_ctx.write(key="memory", value="I know Python")
        assert agent_ctx.read("memory") == "I know Python"
        assert agent_ctx.read("memory", fallback_to_global=False) == "I know Python"

    def test_expiry_in_read(self) -> None:
        ctx = SharedContext()
        ctx.write(key="k1", value="v1", ttl=0.1)
        assert ctx.read("k1") == "v1"

        # Wait for expiry
        time.sleep(0.2)
        assert ctx.read("k1") is None

    def test_persistence(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        try:
            ctx1 = SharedContext(db_path=db_path)
            ctx1.write(key="k1", value="v1")
            del ctx1

            ctx2 = SharedContext(db_path=db_path)
            assert ctx2.read("k1") == "v1"
        finally:
            db_path.unlink()


class TestAgentContext:
    def test_write_and_read(self) -> None:
        ctx = SharedContext()
        agent = AgentContext(
            agent_id="agent-1",
            namespace="agent:agent-1",
            store=ctx,
        )

        agent.write(key="k1", value="v1")
        assert agent.read("k1") == "v1"

    def test_fallback_to_global(self) -> None:
        ctx = SharedContext()
        ctx.write(key="shared", value="global-val", namespace="global")

        agent = AgentContext(
            agent_id="agent-1",
            namespace="agent:agent-1",
            store=ctx,
        )
        assert agent.read("shared") == "global-val"

    def test_no_fallback(self) -> None:
        ctx = SharedContext()
        ctx.write(key="shared", value="global-val", namespace="global")

        agent = AgentContext(
            agent_id="agent-1",
            namespace="agent:agent-1",
            store=ctx,
        )
        assert agent.read("shared", fallback_to_global=False) is None

    def test_share_with(self) -> None:
        ctx = SharedContext()
        agent1 = AgentContext(
            agent_id="agent-1",
            namespace="agent:agent-1",
            store=ctx,
        )
        agent2 = AgentContext(
            agent_id="agent-2",
            namespace="agent:agent-2",
            store=ctx,
        )

        agent1.share_with(target_agent_id="agent-2", key="k1", value="shared-v")
        assert agent2.read("k1") == "shared-v"

    def test_read_all(self) -> None:
        ctx = SharedContext()
        agent = AgentContext(
            agent_id="agent-1",
            namespace="agent:agent-1",
            store=ctx,
        )

        agent.write(key="k1", value="v1", tags=["a"])
        agent.write(key="k2", value="v2", tags=["b"])

        entries = agent.read_all()
        assert len(entries) == 2


class TestContextBridge:
    def test_store_and_get_task_output(self) -> None:
        ctx = SharedContext()
        bridge = ContextBridge(ctx)

        bridge.store_task_output(
            run_id="run-1",
            task_id="t1",
            output="task output here",
        )

        output = bridge.get_task_output("run-1", "t1")
        assert output == "task output here"

    def test_get_all_task_outputs(self) -> None:
        ctx = SharedContext()
        bridge = ContextBridge(ctx)

        bridge.store_task_output("run-1", "t1", "output 1")
        bridge.store_task_output("run-1", "t2", "output 2")

        outputs = bridge.get_all_task_outputs("run-1")
        assert outputs == {"t1": "output 1", "t2": "output 2"}

    def test_shared_state(self) -> None:
        ctx = SharedContext()
        bridge = ContextBridge(ctx)

        bridge.store_shared_state("run-1", "arch", "microservices")
        assert bridge.get_shared_state("run-1", "arch") == "microservices"

    def test_agent_memory(self) -> None:
        ctx = SharedContext()
        bridge = ContextBridge(ctx)

        bridge.store_agent_memory("agent-1", "pref", "use tabs")
        assert bridge.get_agent_memory("agent-1", "pref") == "use tabs"

    def test_global_context(self) -> None:
        ctx = SharedContext()
        bridge = ContextBridge(ctx)

        bridge.store_global("conventions", "use type hints")
        assert bridge.get_global("conventions") == "use type hints"
