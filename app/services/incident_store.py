from sqlalchemy import select
from sqlalchemy.orm import Session
import json

from app.core.telemetry import record_exception, set_span_attributes, tracer
from app.models.incident import Incident


class IncidentStore:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    def create_incident(
        self,
        *,
        evidence,
        rca: dict,
        action: str = "pending",
        decision_reason: str = "Delivery is pending.",
        comment_url: str | None = None,
    ) -> Incident:
        if tracer is None:
            return self._create_incident_impl(
                evidence=evidence,
                rca=rca,
                action=action,
                decision_reason=decision_reason,
                comment_url=comment_url,
            )

        with tracer.start_as_current_span("incident.persist") as span:
            try:
                set_span_attributes(
                    span,
                    run_id=evidence.run_id,
                    workflow_name=evidence.workflow_name,
                    failure_class=rca.get("failure_class", "unknown"),
                    confidence=rca.get("confidence", 0.0),
                    abstained=bool(rca.get("abstained", False)),
                )
                incident = self._create_incident_impl(
                    evidence=evidence,
                    rca=rca,
                    action=action,
                    decision_reason=decision_reason,
                    comment_url=comment_url,
                )
                set_span_attributes(span, incident_id=incident.id)
                return incident
            except Exception as exc:
                record_exception(span, exc)
                raise

    def _create_incident_impl(
        self,
        *,
        evidence,
        rca: dict,
        action: str = "pending",
        decision_reason: str = "Delivery is pending.",
        comment_url: str | None = None,
    ) -> Incident:
        trace_payload = rca.get("reasoning_trace")
        reasoning_trace = None
        if isinstance(trace_payload, (list, dict)):
            reasoning_trace = json.dumps(
                trace_payload,
                ensure_ascii=False,
            )

        incident = Incident(
            run_id=evidence.run_id,
            workflow_name=evidence.workflow_name,
            branch=evidence.branch,
            commit_sha=evidence.sha,
            pull_request_number=evidence.pull_request_number,
            failure_class=rca.get("failure_class", "unknown"),
            summary=rca.get("summary", ""),
            suggested_next_step=rca.get(
                "suggested_next_step",
                "A human should inspect this failure.",
            ),
            confidence=float(rca.get("confidence", 0.0)),
            abstained=bool(rca.get("abstained", False)),
            action=action,
            decision_reason=decision_reason,
            comment_url=comment_url,
            reasoning_trace=reasoning_trace,
        )

        with self.session_factory() as session:
            session.add(incident)
            session.commit()
            session.refresh(incident)

        return incident

    def get_by_id(self, incident_id: int) -> Incident | None:
        with self.session_factory() as session:
            return session.get(Incident, incident_id)

    def get_by_run_id(self, run_id: int) -> list[Incident]:
        with self.session_factory() as session:
            statement = (
                select(Incident)
                .where(Incident.run_id == run_id)
                .order_by(Incident.created_at.desc())
            )

            return list(session.scalars(statement))

    def update_delivery(
        self,
        *,
        incident_id: int,
        action: str,
        decision_reason: str,
        comment_url: str | None,
    ) -> Incident:
        with self.session_factory() as session:
            incident = session.get(Incident, incident_id)

            if incident is None:
                raise ValueError(
                    f"Incident {incident_id} was not found."
                )

            incident.action = action
            incident.decision_reason = decision_reason
            incident.comment_url = comment_url

            session.commit()
            session.refresh(incident)

            return incident