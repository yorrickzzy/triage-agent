import math
import json
import re

from openai import OpenAI

from sqlalchemy import desc, or_, select, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import (
    ENABLE_PGVECTOR, OPENAI_API_KEY, RAG_EMBEDDING_MODEL, RAG_EMBEDDING_DIMENSIONS,
)
from app.models.rag_chunk import RagChunk

try:
    from pgvector.sqlalchemy import Vector  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    Vector = None


class RagStore:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        embedding_client=None,
    ) -> None:
        self.session_factory = session_factory
        self.embedding_client = embedding_client

    def _build_embedding(self, text: str) -> list[float]:
        # Bound UTF-8 bytes conservatively below the embedding input token limit.
        text = text.strip().encode("utf-8")[:6000].decode("utf-8", errors="ignore")
        if not text:
            raise ValueError("Embedding input must not be empty")
        if self.embedding_client is None:
            self.embedding_client = OpenAI(api_key=OPENAI_API_KEY, timeout=20.0, max_retries=1)
        response = self.embedding_client.embeddings.create(
            model=RAG_EMBEDDING_MODEL, input=text,
            dimensions=RAG_EMBEDDING_DIMENSIONS, encoding_format="float",
        )
        vector = response.data[0].embedding
        if len(vector) != RAG_EMBEDDING_DIMENSIONS or not all(math.isfinite(v) for v in vector):
            raise ValueError("Invalid semantic embedding returned by API")
        if not any(vector):
            raise ValueError("Embedding must not be a zero vector")
        return vector

    def _serialize_embedding(self, embedding: list[float]) -> str | list[float]:
        if not ENABLE_PGVECTOR or Vector is None:
            return json.dumps(embedding)

        return embedding

    def prepare_incident(self, incident) -> dict:
        text = (
            f"[incident #{incident.id}] "
            f"workflow={incident.workflow_name}; "
            f"branch={incident.branch}; "
            f"class={incident.failure_class}; "
            f"summary={incident.summary}; "
            f"next_step={incident.suggested_next_step}; "
            f"decision={incident.decision_reason}; "
            f"confidence={incident.confidence:.2f}; "
            f"action={incident.action}"
        )
        embedding = None
        try:
            embedding = self._build_embedding(text)
        except Exception as exc:
            print(f"[rag] embedding unavailable ({type(exc).__name__}); saved for backfill", flush=True)

        return {
            "source_type": "incident", "source_id": incident.id,
            "workflow_name": incident.workflow_name,
            "failure_class": incident.failure_class, "text": text,
            "semantic_embedding": self._serialize_embedding(embedding) if embedding is not None else None,
            "embedding_model": RAG_EMBEDDING_MODEL if embedding is not None else None,
            "quality_score": 0.0,
        }

    def index_prepared_incident(self, session, payload) -> int:
        existing = session.scalars(select(RagChunk).where(
            RagChunk.source_type == "incident", RagChunk.source_id == payload["source_id"]
        )).first()
        if existing is not None:
            # Refresh legacy/intermediate content, retaining reviewer feedback.
            for key, value in payload.items():
                if key != "quality_score":
                    setattr(existing, key, value)
            session.flush()
            return existing.id
        chunk = RagChunk(**payload)
        session.add(chunk)
        session.flush()
        return chunk.id

    def index_incident(self, incident) -> RagChunk:
        payload = self.prepare_incident(incident)
        with self.session_factory() as session:
            chunk_id = self.index_prepared_incident(session, payload)
            session.commit()
            return session.get(RagChunk, chunk_id)

    def reindex_embeddings(self) -> int:
        """Rebuild missing/outdated vectors; each commit is a resumable checkpoint."""
        count = 0
        last_id = -1
        while True:
            with self.session_factory() as session:
                chunk = session.scalars(
                    select(RagChunk).where(
                        RagChunk.id > last_id,
                        or_(RagChunk.semantic_embedding.is_(None),
                            RagChunk.embedding_model.is_(None),
                            RagChunk.embedding_model != RAG_EMBEDDING_MODEL),
                    ).order_by(RagChunk.id).limit(1)
                ).first()
                if chunk is None:
                    return count
                vector = self._build_embedding(chunk.text)
                chunk.semantic_embedding = self._serialize_embedding(vector)
                chunk.embedding_model = RAG_EMBEDDING_MODEL
                session.commit()
                last_id = chunk.id
                count += 1
                print(f"[rag] reindexed chunk_id={last_id}", flush=True)

    def apply_feedback(
        self,
        *,
        incident_id: int,
        verdict: str,
    ) -> None:
        score_delta = {
            "correct": 1.0,
            "not_quite": -0.25,
            "wrong": -1.0,
        }.get(verdict, 0.0)

        if score_delta == 0.0:
            return

        with self.session_factory() as session:
            statement = (
                select(RagChunk)
                .where(
                    RagChunk.source_type == "incident",
                    RagChunk.source_id == incident_id,
                )
                .order_by(desc(RagChunk.created_at))
            )

            chunk = session.scalars(statement).first()

            if chunk is None:
                return

            chunk.quality_score = float(chunk.quality_score) + score_delta
            session.commit()

    def _build_query_text(self, evidence) -> str:
        keywords = []
        for job_name in evidence.failed_jobs:
            if not isinstance(job_name, str):
                continue

            token = job_name.strip().lower()
            if token and token not in keywords:
                keywords.append(token)

        parts = [evidence.workflow_name or "", *keywords]
        return " ".join(part for part in parts if part)

    def _build_query_terms(self, query: str) -> list[str]:
        terms = []

        for token in re.findall(r"[a-z0-9]+", query.lower()):
            if token not in terms:
                terms.append(token)

        return terms

    def _try_vector_search(
        self,
        session: Session,
        query: str,
        limit: int,
        source_type: str | None = None,
        workflow_name: str | None = None,
    ) -> list[dict] | None:
        if not ENABLE_PGVECTOR or Vector is None:
            return None

        if not query.strip():
            return None

        try:
            query_embedding = self._build_embedding(query)
            statement = text(
                f"""
                SELECT id, source_type, source_id, workflow_name, failure_class, text, quality_score
                FROM rag_chunks
                WHERE (:source_type IS NULL OR source_type = :source_type)
                  AND (:workflow_name IS NULL OR workflow_name = :workflow_name)
                  AND semantic_embedding IS NOT NULL
                  AND embedding_model = :embedding_model
                  AND quality_score >= 0
                ORDER BY semantic_embedding <-> CAST(:embedding AS vector({RAG_EMBEDDING_DIMENSIONS})) ASC,
                         quality_score DESC,
                         created_at DESC
                LIMIT :limit
                """
            )
            rows = session.execute(
                statement,
                {
                    "source_type": source_type,
                    "workflow_name": workflow_name,
                    "embedding": f"[{','.join(f'{value:.6f}' for value in query_embedding)}]",
                    "limit": limit,
                    "embedding_model": RAG_EMBEDDING_MODEL,
                },
            )

            results = [
                {
                    "source_type": row.source_type,
                    "source_id": row.source_id,
                    "workflow_name": row.workflow_name,
                    "failure_class": row.failure_class,
                    "quality_score": row.quality_score,
                    "text": row.text,
                }
                for row in rows
            ]
            # Unindexed legacy records remain available through SQL fallback.
            return results or None
        except Exception as exc:  # pragma: no cover - depends on DB extension availability
            print(f"[rag] vector search unavailable, falling back to SQL ({type(exc).__name__})", flush=True)
            # Reset failed transaction state before running SQL fallback queries.
            session.rollback()
            return None

    def _sql_fallback_search(
        self,
        session: Session,
        query: str,
        limit: int,
        source_type: str | None = None,
        workflow_name: str | None = None,
    ) -> list[dict]:
        keywords = self._build_query_terms(query)

        statement = select(RagChunk)
        statement = statement.where(RagChunk.quality_score >= 0)

        if source_type is not None:
            statement = statement.where(RagChunk.source_type == source_type)

        if workflow_name is not None:
            statement = statement.where(RagChunk.workflow_name == workflow_name)

        if keywords:
            statement = statement.where(
                or_(
                    *[
                        RagChunk.text.ilike(f"%{kw}%")
                        for kw in keywords[:5]
                    ]
                )
            )

        chunks = list(
            session.scalars(
                statement
                .order_by(
                    desc(RagChunk.quality_score),
                    desc(RagChunk.created_at),
                )
                .limit(limit)
            )
        )

        return [
            {
                "source_type": chunk.source_type,
                "source_id": chunk.source_id,
                "workflow_name": chunk.workflow_name,
                "failure_class": chunk.failure_class,
                "quality_score": chunk.quality_score,
                "text": chunk.text,
            }
            for chunk in chunks
        ]

    def search_playbook(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict]:
        query = query.strip()

        with self.session_factory() as session:
            vector_results = self._try_vector_search(session, query, limit)
            if vector_results is not None:
                return vector_results

            return self._sql_fallback_search(session, query, limit)

    def retrieve_for_evidence(
        self,
        evidence,
        limit: int = 5,
    ) -> list[dict]:
        query = self._build_query_text(evidence)

        with self.session_factory() as session:
            vector_results = self._try_vector_search(
                session,
                query,
                limit,
                source_type="incident",
                workflow_name=evidence.workflow_name,
            )
            if vector_results is not None:
                return vector_results

            return self._sql_fallback_search(
                session,
                query,
                limit,
                source_type="incident",
                workflow_name=evidence.workflow_name,
            )
