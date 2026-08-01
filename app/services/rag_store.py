import hashlib
import json
import re

from sqlalchemy import desc, or_, select, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import ENABLE_PGVECTOR
from app.models.rag_chunk import RagChunk

try:
    from pgvector.sqlalchemy import Vector  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    Vector = None


class RagStore:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
    ) -> None:
        self.session_factory = session_factory

    def _build_embedding(self, text: str) -> list[float]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        if not tokens:
            return [0.0] * 32

        vector = [0.0] * 32
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % 32
            vector[index] += 1.0

        norm = sum(value * value for value in vector) ** 0.5
        if norm == 0:
            return [0.0] * 32

        return [value / norm for value in vector]

    def _serialize_embedding(self, embedding: list[float]) -> str | list[float]:
        if not ENABLE_PGVECTOR or Vector is None:
            return json.dumps(embedding)

        return embedding

    def index_incident(self, incident) -> RagChunk:
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
        embedding = self._build_embedding(text)

        with self.session_factory() as session:
            chunk = RagChunk(
                source_type="incident",
                source_id=incident.id,
                workflow_name=incident.workflow_name,
                failure_class=incident.failure_class,
                text=text,
                embedding=self._serialize_embedding(embedding),
                quality_score=0.0,
            )

            session.add(chunk)
            session.commit()
            session.refresh(chunk)

            return chunk

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

    def _try_vector_retrieve(self, session: Session, evidence, limit: int) -> list[dict] | None:
        if not ENABLE_PGVECTOR or Vector is None:
            return None

        query_text = self._build_query_text(evidence)
        query_embedding = self._build_embedding(query_text)

        try:
            statement = text(
                """
                SELECT id, source_type, source_id, workflow_name, failure_class, text, quality_score
                FROM rag_chunks
                WHERE source_type = :source_type
                AND (:workflow_name = '' OR workflow_name = :workflow_name)
                ORDER BY embedding <-> CAST(:embedding AS vector(32)) DESC,
                         quality_score DESC,
                         created_at DESC
                LIMIT :limit
                """
            )
            rows = session.execute(
                statement,
                {
                    "source_type": "incident",
                    "workflow_name": evidence.workflow_name or "",
                    "embedding": f"[{','.join(f'{value:.6f}' for value in query_embedding)}]",
                    "limit": limit,
                },
            )

            return [
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
        except Exception as exc:  # pragma: no cover - depends on DB extension availability
            print(f"[rag] vector search unavailable, falling back to SQL: {exc}", flush=True)
            # Reset failed transaction state before running SQL fallback queries.
            session.rollback()
            return None

    def _sql_fallback_retrieve(self, session: Session, evidence, limit: int) -> list[dict]:
        keywords = []
        for job_name in evidence.failed_jobs:
            if not isinstance(job_name, str):
                continue

            token = job_name.strip().lower()
            if token and token not in keywords:
                keywords.append(token)

        statement = select(RagChunk).where(
            RagChunk.source_type == "incident",
        )

        if evidence.workflow_name:
            statement = statement.where(
                RagChunk.workflow_name == evidence.workflow_name,
            )

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

    def retrieve_for_evidence(
        self,
        evidence,
        limit: int = 5,
    ) -> list[dict]:
        with self.session_factory() as session:
            vector_results = self._try_vector_retrieve(session, evidence, limit)
            if vector_results is not None:
                return vector_results

            return self._sql_fallback_retrieve(session, evidence, limit)
