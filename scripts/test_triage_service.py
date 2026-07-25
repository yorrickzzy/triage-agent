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
            "suggested_next_step": "Rerun the failed job.",
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
    def __init__(self):
        self.rerun_called_with = None

    def rerun_failed_jobs(self, run_id: int) -> bool:
        self.rerun_called_with = run_id
        return True


github = FakeGitHubClient()

service = TriageService(
    evidence_collector=FakeEvidenceCollector(),
    rca_pipeline=FakePipeline(),
    confidence_gate=FakeConfidenceGate(),
    github_client=github,
)

result = service.triage(28731123848)

print("action:", result.action)
print("reason:", result.reason)
print("rerun_called_with:", github.rerun_called_with)
print("rca:", result.rca)