from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.models.feedback import Feedback
from app.models.incident import Incident
from app.schemas.feedback import (
    FeedbackCreateRequest,
    FeedbackVerdict,
)


class IncidentNotFoundError(Exception):
    pass


class FeedbackStore:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
    ) -> None:
        self.session_factory = session_factory

    def create_feedback(
        self,
        request: FeedbackCreateRequest,
    ) -> Feedback:
        return self.create(
            incident_id=request.incident_id,
            verdict=request.verdict,
            comment=request.comment,
            reviewer=request.reviewer,
        )

    def create(
        self,
        *,
        incident_id: int,
        verdict: FeedbackVerdict,
        comment: str | None = None,
        reviewer: str | None = None,
    ) -> Feedback:
        with self.session_factory() as session:
            incident = session.get(Incident, incident_id)

            if incident is None:
                raise IncidentNotFoundError(
                    f"Incident {incident_id} was not found."
                )

            feedback = Feedback(
                incident_id=incident_id,
                verdict=verdict.value,
                comment=comment,
                reviewer=reviewer,
            )

            session.add(feedback)
            session.commit()
            session.refresh(feedback)

            return feedback

    def get_feedback(
        self,
        feedback_id: int,
    ) -> Feedback | None:
        with self.session_factory() as session:
            return session.get(Feedback, feedback_id)

    def list_by_incident(
        self,
        incident_id: int,
    ) -> list[Feedback]:
        with self.session_factory() as session:
            statement = (
                select(Feedback)
                .where(Feedback.incident_id == incident_id)
                .order_by(Feedback.created_at.desc())
            )

            return list(session.scalars(statement))