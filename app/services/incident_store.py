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
        session: Session | None = None,
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

        if session is not None:
            session.add(incident)
            session.flush()
        else:
            with self.session_factory() as session:
                session.add(incident)
                session.commit()
                session.refresh(incident)

        return incident

    def create_in_transaction(self, session, *, evidence, rca):
        return self._create_incident_impl(session=session, evidence=evidence, rca=rca).id

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
        rca: dict | None = None,
        fallback_trace=None,
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
            if rca is not None:
                self.apply_rca(incident, rca, fallback_trace=fallback_trace)

            session.commit()
            session.refresh(incident)

            return incident

    @staticmethod
    def apply_rca(incident, rca, fallback_trace=None):
        incident.failure_class = rca.get("failure_class", "unknown")
        incident.summary = rca.get("summary", "")
        incident.suggested_next_step = rca.get("suggested_next_step", "")
        incident.confidence = float(rca.get("confidence", 0.0))
        incident.abstained = bool(rca.get("abstained", False))
        trace = rca.get("reasoning_trace", fallback_trace)
        if trace is not None:
            incident.reasoning_trace = json.dumps(trace, ensure_ascii=False)
