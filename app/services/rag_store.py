from sqlalchemy import desc, or_, select
from sqlalchemy.orm import Session, sessionmaker

from app.models.rag_chunk import RagChunk


class RagStore:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
    ) -> None:
        self.session_factory = session_factory

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

        with self.session_factory() as session:
            chunk = RagChunk(
                source_type="incident",
                source_id=incident.id,
                workflow_name=incident.workflow_name,
                failure_class=incident.failure_class,
                text=text,
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

    def retrieve_for_evidence(
        self,
        evidence,
        limit: int = 5,
    ) -> list[dict]:
        # Keep retrieval simple: same workflow + failed-job keyword match,
        # then trust human feedback score and recency for ranking.
        keywords = []
        for job_name in evidence.failed_jobs:
            if not isinstance(job_name, str):
                continue

            token = job_name.strip().lower()
            if token and token not in keywords:
                keywords.append(token)

        with self.session_factory() as session:
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