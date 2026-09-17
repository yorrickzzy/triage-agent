import os
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

# All tests use a separate in-memory database and a mocked embeddings endpoint.
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["ENABLE_PGVECTOR"] = "false"

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from app.core import database
from app.core.config import RAG_EMBEDDING_MODEL, RAG_EMBEDDING_DIMENSIONS
from app.models.rag_chunk import RagChunk
from app.services.rag_store import RagStore


class RagStoreTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        self.factory = sessionmaker(self.engine, expire_on_commit=False)
        with patch.object(database, "engine", self.engine):
            database.create_tables()
        self.client = Mock()
        self.vector = [1.0] + [0.0] * (RAG_EMBEDDING_DIMENSIONS - 1)
        self.client.embeddings.create.return_value = SimpleNamespace(
            data=[SimpleNamespace(embedding=self.vector)]
        )
        self.store = RagStore(self.factory, embedding_client=self.client)

    def tearDown(self):
        self.engine.dispose()

    def add_chunk(self, **kwargs):
        with self.factory() as session:
            chunk = RagChunk(source_type="incident", source_id=1, text="pytest failed", **kwargs)
            session.add(chunk)
            session.commit()
            return chunk.id

    def test_embedding_request_and_validation(self):
        self.assertEqual(self.store._build_embedding("dependency missing"), self.vector)
        self.client.embeddings.create.assert_called_once_with(
            model=RAG_EMBEDDING_MODEL, input="dependency missing",
            dimensions=1536, encoding_format="float",
        )
        for vector in ([1.0] * 32, [float("nan")] * 1536, [0.0] * 1536):
            self.client.embeddings.create.return_value.data[0].embedding = vector
            with self.assertRaises(ValueError):
                self.store._build_embedding("test")
        with self.assertRaises(ValueError):
            self.store._build_embedding(" ")

    def test_multilingual_input_is_preserved_and_bounded(self):
        self.store._build_embedding("依赖缺失" * 3000)
        sent = self.client.embeddings.create.call_args.kwargs["input"]
        self.assertTrue(sent.startswith("依赖缺失"))
        self.assertLessEqual(len(sent.encode("utf-8")), 6000)

    def test_backfill_preserves_hash_and_quality_and_is_resumable(self):
        chunk_id = self.add_chunk(embedding=json.dumps([0.0] * 32), quality_score=2.0)
        self.assertEqual(self.store.reindex_embeddings(), 1)
        self.assertEqual(self.store.reindex_embeddings(), 0)
        with self.factory() as session:
            chunk = session.get(RagChunk, chunk_id)
            self.assertEqual(json.loads(chunk.embedding), [0.0] * 32)
            self.assertEqual(chunk.quality_score, 2.0)
            self.assertEqual(json.loads(chunk.semantic_embedding), self.vector)
            self.assertEqual(chunk.embedding_model, RAG_EMBEDDING_MODEL)

    def test_backfill_failure_can_be_retried(self):
        self.add_chunk()
        self.add_chunk()
        response = self.client.embeddings.create.return_value
        self.client.embeddings.create.side_effect = [response, RuntimeError("offline")]
        with self.assertRaises(RuntimeError):
            self.store.reindex_embeddings()
        self.client.embeddings.create.side_effect = None
        self.assertEqual(self.store.reindex_embeddings(), 1)

    def test_embedding_outage_does_not_block_incident_indexing(self):
        self.client.embeddings.create.side_effect = RuntimeError("offline")
        incident = SimpleNamespace(id=1, workflow_name="CI", branch="main",
            failure_class="test", summary="pytest failed", suggested_next_step="fix",
            decision_reason="test", confidence=0.9, action="report")
        chunk = self.store.index_incident(incident)
        self.assertIsNone(chunk.semantic_embedding)
        self.assertEqual(len(self.store.search_playbook("pytest")), 1)

    def test_vector_query_excludes_legacy_and_other_models(self):
        session = Mock()
        session.execute.return_value = []
        with patch("app.services.rag_store.ENABLE_PGVECTOR", True), patch("app.services.rag_store.Vector", object()):
            self.assertIsNone(self.store._try_vector_search(session, "pytest", 5, "incident", "CI"))
        statement, parameters = session.execute.call_args.args
        self.assertIn("semantic_embedding IS NOT NULL", str(statement))
        self.assertIn("embedding_model = :embedding_model", str(statement))
        self.assertIn("vector(1536)", str(statement))
        self.assertEqual(parameters["limit"], 5)
        self.assertEqual(parameters["workflow_name"], "CI")
        self.assertEqual(parameters["embedding_model"], RAG_EMBEDDING_MODEL)

    def test_query_api_failure_falls_back(self):
        session = Mock()
        self.client.embeddings.create.side_effect = RuntimeError("offline")
        with patch("app.services.rag_store.ENABLE_PGVECTOR", True), patch("app.services.rag_store.Vector", object()):
            self.assertIsNone(self.store._try_vector_search(session, "pytest", 5))
        session.rollback.assert_called_once()

    def test_legacy_schema_migration_preserves_rows(self):
        old = create_engine("sqlite:///:memory:")
        with old.begin() as conn:
            conn.execute(text("CREATE TABLE rag_chunks (id INTEGER PRIMARY KEY, embedding TEXT, text TEXT)"))
            conn.execute(text("INSERT INTO rag_chunks VALUES (1, '[0,1]', 'old case')"))
        with patch.object(database, "engine", old):
            database.create_tables()
            database.create_tables()
        with old.connect() as conn:
            self.assertEqual(conn.execute(text("SELECT embedding FROM rag_chunks")).scalar(), "[0,1]")
            self.assertIn("semantic_embedding", {c["name"] for c in inspect(conn).get_columns("rag_chunks")})
        old.dispose()


if __name__ == "__main__":
    unittest.main()
