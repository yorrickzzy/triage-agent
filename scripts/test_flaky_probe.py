from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.triage_service import TriageService


class FakeEvidenceCollector:
    def collect(self, run_id: int):
        return {"run_id": run_id}


class FakePipeline:
    def run(self, evidence):
        return {
            "failure_class": "flaky",
            "summary": "The test may be flaky.",
            "claims": [],
            "suggested_next_step": "Rerun the failed jobs.",
            "confidence": 0.95,
            "abstained": False,
        }


class FakeGateResult:
    decision = "rerun"
    reason = "Suspected flaky failure requires a rerun."


class FakeConfidenceGate:
    def evaluate(self, rca):
        return FakeGateResult()


class FakeGitHubClient:
    def __init__(self, rerun_conclusion: str):
        self.rerun_conclusion = rerun_conclusion
        self.rerun_called = False

    def get_workflow_run(self, run_id: int) -> dict:
        return {
            "id": run_id,
            "run_attempt": 1,
            "status": "completed",
            "conclusion": "failure",
        }

    def rerun_failed_jobs(self, run_id: int) -> bool:
        self.rerun_called = True
        return True

    def wait_for_new_attempt_completion(
        self,
        run_id: int,
        previous_attempt: int,
        timeout_seconds: int = 300,
        poll_interval_seconds: int = 5,
    ) -> dict:
        return {
            "id": run_id,
            "run_attempt": previous_attempt + 1,
            "status": "completed",
            "conclusion": self.rerun_conclusion,
        }


def run_case(conclusion: str) -> None:
    github = FakeGitHubClient(conclusion)

    service = TriageService(
        evidence_collector=FakeEvidenceCollector(),
        rca_pipeline=FakePipeline(),
        confidence_gate=FakeConfidenceGate(),
        github_client=github,
    )

    result = service.triage(28731123848)

    print("=" * 60)
    print("rerun conclusion:", conclusion)
    print("action:", result.action)
    print("reason:", result.reason)
    print("failure_class:", result.rca["failure_class"])
    print("abstained:", result.rca["abstained"])


run_case("success")
run_case("failure")