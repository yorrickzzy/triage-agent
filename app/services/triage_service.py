from dataclasses import asdict, dataclass
from typing import Literal

from app.core.telemetry import record_exception, set_span_attributes, tracer
from app.services.github_client import GitHubRerunTimeoutError
from app.models.evidence import Evidence, RawLog
from app.services.recovery_errors import ManualReviewRequired
from app.services.confidence_gate import GateResult

TriageAction = Literal[
    "pending",
    "delivered",
    "abstain",
    "rerun_pending",
    "delivery_failed",
]

@dataclass
class TriageResult:
    action: TriageAction
    rca: dict
    reason: str
    comment_url: str | None = None
    incident_id: int | None = None

class TriageService:
    def __init__(
        self,
        evidence_collector,
        rca_pipeline,
        confidence_gate,
        github_client,
        report_formatter,
        github_delivery,
        slack_client,
        incident_store,
        rag_store=None,
        tools=None,
        task_queue=None,
    ):
        self.evidence_collector = evidence_collector
        self.rca_pipeline = rca_pipeline
        self.confidence_gate = confidence_gate
        self.github_client = github_client
        self.report_formatter = report_formatter
        self.github_delivery = github_delivery
        self.slack_client = slack_client
        self.incident_store = incident_store
        self.rag_store = rag_store
        self.tools = tools
        self.task_queue = task_queue

    

    def triage(self, run_id: int, execution=None) -> TriageResult:
        if execution is None and self.task_queue is not None:
            from app.models.triage_job import TriageJob
            from app.services.triage_worker import process_task
            run = self.github_client.get_workflow_run(run_id)
            key = self.task_queue.enqueue(TriageJob(
                run_id, run["name"], run["head_branch"], run["head_sha"],
                int(run.get("run_attempt", 1))))
            task = self.task_queue.inspect(key)
            if task["status"] == "completed":
                return TriageResult(**task["result"])
            execution = self.task_queue.claim(key)
            if execution is None:
                raise RuntimeError(f"Task {key} is {task['status']}; inspect it before retrying")
            return process_task(self, execution)
        if tracer is None:
            return self._triage_impl(run_id, execution=execution)

        with tracer.start_as_current_span("triage.service") as span:
            try:
                set_span_attributes(
                    span,
                    run_id=run_id,
                )
                result = self._triage_impl(run_id, execution=execution)
                set_span_attributes(
                    span,
                    action=result.action,
                    incident_id=result.incident_id,
                )
                return result
            except Exception as exc:
                record_exception(span, exc)
                raise

    def _triage_impl(self, run_id: int, execution=None) -> TriageResult:
        if execution is None:
            evidence = self.evidence_collector.collect(run_id)
        else:
            def collect():
                snapshot = self.evidence_collector.collect(run_id)
                if snapshot.run_attempt != execution.payload["run_attempt"]:
                    raise ManualReviewRequired("CI attempt changed before evidence was collected")
                return asdict(snapshot)
            snapshot = execution.step("evidence", collect)
            evidence = Evidence(**{**snapshot, "logs": [RawLog(**log) for log in snapshot["logs"]]})
        rag_context = []
        if self.rag_store is not None:
            operation = lambda: self.rag_store.retrieve_for_evidence(evidence)
            rag_context = execution.step("rag", operation) if execution else operation()

        operation = lambda: self.rca_pipeline.run(evidence, rag_context=rag_context,
            **({"execution": execution} if execution else {}))
        rca = execution.step("rca", operation) if execution else operation()
        if execution:
            def evaluate():
                result = self.confidence_gate.evaluate(rca)
                return {"decision": result.decision, "reason": result.reason}
            gate_result = GateResult(**execution.step("gate", evaluate))
        else:
            gate_result = self.confidence_gate.evaluate(rca)

        if execution:
            incident_id = execution.step("incident", lambda session:
                self.incident_store.create_in_transaction(session, evidence=evidence, rca=rca),
                transactional=True)
            incident = self.incident_store.get_by_id(incident_id)
        else:
            incident = self.incident_store.create_incident(evidence=evidence, rca=rca)

        if self.rag_store is not None:
            if execution:
                payload = execution.step("incident_embedding", lambda: self.rag_store.prepare_incident(incident))
                execution.step("index_incident", lambda session:
                    self.rag_store.index_prepared_incident(session, payload), transactional=True)
            else:
                self.rag_store.index_incident(incident)

        try:
            if gate_result.decision == "rerun":
                result = self._probe_flakiness(
                    run_id=run_id,
                    suspected_rca=rca,
                    evidence=evidence,
                    incident_id=incident.id,
                    execution=execution,
                )

            elif gate_result.decision == "abstain":
                # Create abstain RCA with the gate's reason
                abstain_rca = self._build_abstain_rca(gate_result.reason)
                # Deliver the abstain message to PR/Slack so the user knows why we gave up
                result = self._deliver_rca(
                    evidence=evidence,
                    rca=abstain_rca,
                    reason=gate_result.reason,
                    incident_id=incident.id,
                    execution=execution,
                )
                # Mark the action as abstain (not delivered)
                result.action = "abstain"

            else:
                result = self._deliver_rca(
                    evidence=evidence,
                    rca=rca,
                    reason=gate_result.reason,
                    incident_id=incident.id,
                    execution=execution,
                )

            if execution:
                def update_incident(session):
                    from app.models.incident import Incident
                    row = session.get(Incident, incident.id)
                    row.action, row.decision_reason = result.action, result.reason
                    row.comment_url = result.comment_url
                    return row.id
                execution.step("incident_delivery", update_incident, transactional=True)
            else:
                self.incident_store.update_delivery(
                    incident_id=incident.id, action=result.action,
                    decision_reason=result.reason, comment_url=result.comment_url)

            result.incident_id = incident.id
            return result

        except Exception as exc:
            if execution:
                raise  # Task status records failures and review requirements.
            self.incident_store.update_delivery(
                incident_id=incident.id,
                action="delivery_failed",
                decision_reason=str(exc),
                comment_url=None,
            )

            raise

    def _deliver_rca(
        self,
        *,
        evidence,
        rca: dict,
        reason: str,
        incident_id: int,
        execution=None,
    ) -> TriageResult:
        operation = lambda: self.report_formatter.to_markdown(
            evidence=evidence, rca=rca, incident_id=incident_id)
        markdown = execution.step("markdown", operation) if execution else operation()

        comment_url = None
        reference_url = self.github_client.build_run_url(evidence.run_id)

        if evidence.pull_request_number is not None:
            operation = lambda: self.github_delivery.deliver(
                pull_request_number=evidence.pull_request_number, markdown=markdown)
            delivery_result = (execution.step("github_comment", operation, effect=True)
                               if execution else operation())

            comment_url = (
                delivery_result.get("html_url")
                or delivery_result.get("url")
                or None
            )
            if comment_url:
                reference_url = comment_url
        else:
            operation = lambda: self.github_delivery.deliver_commit_comment(
                sha=evidence.sha, markdown=markdown)
            delivery_result = (execution.step("github_comment", operation, effect=True)
                               if execution else operation())
            comment_url = (
                delivery_result.get("html_url")
                or delivery_result.get("url")
                or None
            )
            if comment_url:
                reference_url = comment_url
            else:
                reference_url = self.github_client.build_commit_url(evidence.sha)

        operation = lambda: self.report_formatter.to_slack(
            evidence=evidence, rca=rca, incident_id=incident_id,
            comment_url=comment_url, reference_url=reference_url)
        slack_message = execution.step("slack_text", operation) if execution else operation()

        operation = lambda: self.slack_client.send_message(slack_message)
        if execution:
            execution.step("slack_message", operation, effect=True)
        else:
            operation()

        return TriageResult(
            action="delivered",
            rca=rca,
            reason=reason,
            comment_url=comment_url,
            incident_id=incident_id,
        )

    def _probe_flakiness(
        self,
        run_id: int,
        suspected_rca: dict,
        evidence,
        incident_id: int,
        execution=None,
    ) -> TriageResult:
        def operation():
            run = self.github_client.get_workflow_run(run_id)
            if execution and int(run.get("run_attempt", 1)) != evidence.run_attempt:
                raise ManualReviewRequired("CI attempt changed before flaky rerun")
            return run
        original_run = execution.step("pre_rerun", operation) if execution else operation()
        previous_attempt = int(original_run.get("run_attempt", 1))

        operation = lambda: self.github_client.rerun_failed_jobs(run_id)
        accepted = (execution.step("flaky_rerun", operation, effect=True)
                    if execution else operation())

        if not accepted:
            return TriageResult(
                action="abstain",
                rca=self._build_abstain_rca(
                    "GitHub did not accept the rerun request."
                ),
                reason="Rerun request was not accepted.",
            )

        try:
            operation = lambda: self.github_client.wait_for_new_attempt_completion(
                run_id=run_id, previous_attempt=previous_attempt,
                timeout_seconds=300, poll_interval_seconds=5)
            rerun = execution.step("rerun_result", operation) if execution else operation()
        except GitHubRerunTimeoutError:
            if execution:
                raise  # Retry the wait using the saved baseline, without another rerun.
            return TriageResult(
                action="rerun_pending",
                rca=suspected_rca,
                reason=(
                    "The rerun did not finish before the timeout. "
                    "No flaky conclusion was made."
                ),
            )

        if rerun.get("conclusion") == "success":
            confirmed_rca = dict(suspected_rca)
            confirmed_rca["failure_class"] = "flaky"
            confirmed_rca["summary"] = (
                "The original CI attempt failed, but the rerun passed "
                "without a code change. The failure is confirmed as flaky."
            )
            confirmed_rca["suggested_next_step"] = (
                "Investigate and quarantine the nondeterministic test "
                "or dependency responsible for the intermittent failure."
            )
            confirmed_rca["confidence"] = 1.0  # High confidence: rerun proved it

            # Re-verify the flaky RCA before delivering
            if not self.rca_pipeline.verifier.verify(confirmed_rca, evidence):
                # If verifier fails, create a safe flaky RCA with no claims
                # (to avoid any ungrounded statements)
                print(
                    "[triage] flaky RCA failed grounding, creating safe version",
                    flush=True,
                )
                confirmed_rca = {
                    "failure_class": "flaky",
                    "summary": (
                        "The test failed on the first attempt but passed on rerun "
                        "without code changes. Confirmed as flaky."
                    ),
                    "claims": [],  # Empty claims = just the fact, no detailed evidence
                    "suggested_next_step": (
                        "Investigate and quarantine the nondeterministic test "
                        "or dependency responsible for the intermittent failure."
                    ),
                    "confidence": 1.0,
                    "abstained": False,
                }

            return self._deliver_rca(
                evidence=evidence,
                rca=confirmed_rca,
                reason="Flakiness confirmed by a successful rerun.",
                incident_id=incident_id,
                execution=execution,
            )

        return TriageResult(
            action="abstain",
            rca=self._build_abstain_rca(
                "The rerun also failed, so flakiness was not confirmed."
            ),
            reason=(
                "The rerun did not pass. A human should inspect "
                "the latest attempt."
            ),
        )

    @staticmethod
    def _build_abstain_rca(reason: str) -> dict:
        return {
            "failure_class": "unknown",
            "summary": reason,
            "claims": [],
            "suggested_next_step": (
                "A human should inspect this CI failure manually."
            ),
            "confidence": 0.0,
            "abstained": True,
        }
