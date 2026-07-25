from dataclasses import dataclass
from typing import Literal

from app.services.github_client import GitHubRerunTimeoutError

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
        incident_store
    ):
        self.evidence_collector = evidence_collector
        self.rca_pipeline = rca_pipeline
        self.confidence_gate = confidence_gate
        self.github_client = github_client
        self.report_formatter = report_formatter
        self.github_delivery = github_delivery
        self.slack_client = slack_client
        self.incident_store = incident_store

    

    def triage(self, run_id: int) -> TriageResult:
        evidence = self.evidence_collector.collect(run_id)
        rca = self.rca_pipeline.run(evidence)
        gate_result = self.confidence_gate.evaluate(rca)

        incident = self.incident_store.create_incident(
            evidence=evidence,
            rca=rca,
        )

        try:
            if gate_result.decision == "rerun":
                result = self._probe_flakiness(
                    run_id=run_id,
                    suspected_rca=rca,
                    evidence=evidence,
                    incident_id=incident.id,
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
                )
                # Mark the action as abstain (not delivered)
                result.action = "abstain"

            else:
                result = self._deliver_rca(
                    evidence=evidence,
                    rca=rca,
                    reason=gate_result.reason,
                    incident_id=incident.id,
                )

            self.incident_store.update_delivery(
                incident_id=incident.id,
                action=result.action,
                decision_reason=result.reason,
                comment_url=result.comment_url,
            )

            result.incident_id = incident.id
            return result

        except Exception as exc:
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
    ) -> TriageResult:
        markdown = self.report_formatter.to_markdown(
            evidence=evidence,
            rca=rca,
            incident_id=incident_id,
        )

        comment_url = None

        if evidence.pull_request_number is not None:
            delivery_result = self.github_delivery.deliver(
                pull_request_number=evidence.pull_request_number,
                markdown=markdown,
            )

            comment_url = (
                delivery_result.get("html_url")
                or delivery_result.get("url")
                or None
            )

        slack_message = self.report_formatter.to_slack(
            evidence=evidence,
            rca=rca,
            incident_id=incident_id,
            comment_url=comment_url,
        )

        self.slack_client.send_message(slack_message)

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
    ) -> TriageResult:
        original_run = self.github_client.get_workflow_run(run_id)
        previous_attempt = int(original_run.get("run_attempt", 1))

        accepted = self.github_client.rerun_failed_jobs(run_id)

        if not accepted:
            return TriageResult(
                action="abstain",
                rca=self._build_abstain_rca(
                    "GitHub did not accept the rerun request."
                ),
                reason="Rerun request was not accepted.",
            )

        try:
            rerun = self.github_client.wait_for_new_attempt_completion(
                run_id=run_id,
                previous_attempt=previous_attempt,
                timeout_seconds=300,
                poll_interval_seconds=5,
            )
        except GitHubRerunTimeoutError:
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