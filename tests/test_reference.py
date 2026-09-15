from __future__ import annotations

import json
import time
from pathlib import Path

from agentcli.context import (
    AgentReference,
    ContextEntry,
    SharedContext,
    export_reference,
    import_reference,
)

# ── helpers ────────────────────────────────────────────────────────────────

def _fill_context(ctx: SharedContext, namespace: str = "global") -> None:
    ctx.write(
        "feat: auth", "OAuth2 with JWT", namespace=namespace, tags=["feature", "auth"]
    )
    ctx.write(
        "feat: billing", "Stripe integration", namespace=namespace, tags=["feature", "billing"]
    )
    ctx.write("arch: db", "PostgreSQL 16", namespace=namespace, tags=["architecture"])
    ctx.write(
        "task:t1:output",
        "Auth module done",
        namespace=namespace,
        source_task_id="t1",
        tags=["task_output", "task:t1"],
    )


# ── AgentReference basics ──────────────────────────────────────────────────

class TestAgentReferenceBasics:
    def test_empty_reference_serialises(self, tmp_path: Path) -> None:
        ref = AgentReference()
        p = tmp_path / "empty.agentref.json"
        ref.save(p)
        loaded = AgentReference.load(p)
        assert loaded.origin_run_id == ""
        assert loaded.namespaces == {}
        assert loaded.summary()["total_entries"] == 0

    def test_round_trip_via_dict(self) -> None:
        ref = AgentReference(origin_run_id="run-123", created_at=1_762_000_000.0)
        data = ref.to_dict()
        assert data["$schema"] == AgentReference.SCHEMA_VERSION
        assert data["origin_run_id"] == "run-123"
        reconstructed = AgentReference.from_json(ref.to_json())
        assert reconstructed.origin_run_id == "run-123"

    def test_to_json_and_load(self, tmp_path: Path) -> None:  # noqa: ARG002
        ref = AgentReference(origin_run_id="run-456")
        text = ref.to_json()
        loaded = AgentReference.from_json(text)
        assert loaded.origin_run_id == "run-456"


# ── Building a reference from SharedContext ───────────────────────────────

class TestFromContext:
    def test_from_context_includes_all_entries(self, tmp_path: Path) -> None:
        ctx = SharedContext(db_path=tmp_path / "ctx.db")
        _fill_context(ctx)
        ref = AgentReference.from_context(ctx)
        assert ref.summary()["total_entries"] == 4

    def test_from_context_namespace_filter(self, tmp_path: Path) -> None:
        ctx = SharedContext(db_path=tmp_path / "ctx.db")
        _fill_context(ctx, namespace="run:abc")
        _fill_context(ctx, namespace="global")
        ref = AgentReference.from_context(ctx, namespace="run:abc")
        assert ref.summary()["total_entries"] == 4
        assert "run:abc" in ref.namespaces_list()
        assert "global" not in ref.namespaces_list()

    def test_from_context_tags_filter(self, tmp_path: Path) -> None:
        ctx = SharedContext(db_path=tmp_path / "ctx.db")
        _fill_context(ctx)
        ref = AgentReference.from_context(ctx, tags=["task_output"])
        assert ref.summary()["total_entries"] == 1
        assert ref.keys_for_namespace("global") == ["task:t1:output"]

    def test_origin_run_id_embedded(self, tmp_path: Path) -> None:
        ctx = SharedContext(db_path=tmp_path / "ctx.db")
        _fill_context(ctx)
        ref = AgentReference.from_context(ctx, origin_run_id="run-777")
        assert ref.origin_run_id == "run-777"
        assert ref.to_dict()["origin_run_id"] == "run-777"


# ── Loading a reference back into a SharedContext ─────────────────────────

class TestLoadInto:
    def test_load_into_restores_entries(self, tmp_path: Path) -> None:
        src = SharedContext(db_path=tmp_path / "src.db")
        _fill_context(src, namespace="run:x")
        ref = AgentReference.from_context(src, origin_run_id="run-x")

        dst = SharedContext(db_path=tmp_path / "dst.db")
        count = ref.load_into(dst)
        assert count == 4
        assert dst.read("feat: auth", namespace="run:x") == "OAuth2 with JWT"
        assert dst.read("arch: db", namespace="run:x") == "PostgreSQL 16"

    def test_load_into_preserves_tags(self, tmp_path: Path) -> None:
        src = SharedContext(db_path=tmp_path / "src.db")
        _fill_context(src)
        ref = AgentReference.from_context(src)
        dst = SharedContext(db_path=tmp_path / "dst.db")
        ref.load_into(dst)
        entries = dst.read_all(namespace="global", tags=["task_output"])
        assert len(entries) == 1
        assert entries[0].source_task_id == "t1"


# ── Slice (the credit-saving primitive) ───────────────────────────────────

class TestSlice:
    def test_slice_by_namespace(self, tmp_path: Path) -> None:
        ctx = SharedContext(db_path=tmp_path / "ctx.db")
        _fill_context(ctx, namespace="run:a")
        _fill_context(ctx, namespace="run:b")
        ref = AgentReference.from_context(ctx)
        sliced = ref.slice(namespaces=["run:a"])
        assert "run:a" in sliced.namespaces_list()
        assert "run:b" not in sliced.namespaces_list()
        assert sliced.summary()["total_entries"] == 4

    def test_slice_by_key_prefix(self, tmp_path: Path) -> None:
        ctx = SharedContext(db_path=tmp_path / "ctx.db")
        _fill_context(ctx)
        ref = AgentReference.from_context(ctx)
        sliced = ref.slice(key_prefix="feat:")
        assert sliced.summary()["total_entries"] == 2
        assert "feat: auth" in sliced.keys_for_namespace("global")
        assert "arch: db" not in sliced.keys_for_namespace("global")

    def test_slice_by_tags(self, tmp_path: Path) -> None:
        ctx = SharedContext(db_path=tmp_path / "ctx.db")
        _fill_context(ctx)
        ref = AgentReference.from_context(ctx)
        sliced = ref.slice(tags=["architecture"])
        assert sliced.summary()["total_entries"] == 1
        assert sliced.keys_for_namespace("global") == ["arch: db"]

    def test_slice_combined_filters(self, tmp_path: Path) -> None:
        ctx = SharedContext(db_path=tmp_path / "ctx.db")
        _fill_context(ctx, namespace="run:p")
        _fill_context(ctx, namespace="run:q")
        ref = AgentReference.from_context(ctx)
        sliced = ref.slice(namespaces=["run:p"], tags=["task_output"])
        assert sliced.summary()["total_entries"] == 1
        assert sliced.keys_for_namespace("run:p") == ["task:t1:output"]

    def test_slice_no_filter_returns_full_copy(self, tmp_path: Path) -> None:
        ctx = SharedContext(db_path=tmp_path / "ctx.db")
        _fill_context(ctx)
        ref = AgentReference.from_context(ctx)
        sliced = ref.slice()
        assert sliced.summary()["total_entries"] == ref.summary()["total_entries"]
        assert sliced is not ref


# ── Convenience functions ─────────────────────────────────────────────────

class TestConvenienceFunctions:
    def test_export_reference_saves_file(self, tmp_path: Path) -> None:
        ctx = SharedContext(db_path=tmp_path / "ctx.db")
        _fill_context(ctx)
        out = tmp_path / "out.agentref.json"
        export_reference(ctx, output_path=out)
        assert out.exists()
        loaded = AgentReference.load(out)
        assert loaded.summary()["total_entries"] == 4

    def test_export_reference_namespace_filter(self, tmp_path: Path) -> None:
        ctx = SharedContext(db_path=tmp_path / "ctx.db")
        _fill_context(ctx)
        out = tmp_path / "filtered.agentref.json"
        ref = export_reference(ctx, namespace="global", tags=["feature"], output_path=out)
        assert ref.summary()["total_entries"] == 2

    def test_import_reference_returns_count(self, tmp_path: Path) -> None:
        ctx = SharedContext(db_path=tmp_path / "src.db")
        _fill_context(ctx)
        ref = AgentReference.from_context(ctx)
        file = tmp_path / "snap.agentref.json"
        ref.save(file)

        dst = SharedContext(db_path=tmp_path / "dst.db")
        count = import_reference(file, dst)
        assert count == 4
        assert dst.read("feat: billing", namespace="global") == "Stripe integration"


# ── ContextEntry model ────────────────────────────────────────────────────

class TestContextEntry:
    def test_to_dict_round_trip(self) -> None:
        entry = ContextEntry(
            key="k1", value="v1", namespace="ns1",
            source_agent="agent-a", tags=["tag1"],
            metadata={"custom": 42},
        )
        d = entry.to_dict()
        assert d["key"] == "k1"
        assert d["value"] == "v1"
        assert d["metadata"] == {"custom": 42}

    def test_is_expired_false_when_no_ttl(self) -> None:
        entry = ContextEntry(key="k", value="v")
        assert entry.is_expired is False

    def test_is_expired_true_when_past_ttl(self) -> None:
        entry = ContextEntry(key="k", value="v", expires_at=time.time() - 1)
        assert entry.is_expired is True


# ── File schema versioning ────────────────────────────────────────────────

class TestSchemaVersion:
    def test_saved_file_has_schema_field(self, tmp_path: Path) -> None:
        ref = AgentReference(origin_run_id="r1")
        p = tmp_path / "v.agentref.json"
        ref.save(p)
        data = json.loads(p.read_text(encoding="utf-8"))
        assert data["$schema"] == AgentReference.SCHEMA_VERSION

    def test_load_ignores_unknown_fields(self, tmp_path: Path) -> None:
        p = tmp_path / "extras.agentref.json"
        p.write_text(
            json.dumps({
                "$schema": AgentReference.SCHEMA_VERSION,
                "origin_run_id": "r2",
                "created_at": 1_762_000_000.0,
                "namespaces": {},
                "extra_field": "should be ignored",
            }),
            encoding="utf-8",
        )
        ref = AgentReference.load(p)
        assert ref.origin_run_id == "r2"
