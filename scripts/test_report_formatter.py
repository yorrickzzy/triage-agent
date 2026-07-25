from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.github_client import GitHubClient
from app.services.evidence_collector import EvidenceCollector
from app.services.openai_reasoner import OpenAIReasoner
from app.services.failure_classifier import FailureClassifier
from app.services.grounding_verifier import GroundingVerifier
from app.services.rca_pipeline import RCAPipeline
from app.services.report_formatter import ReportFormatter


RUN_ID = 28731123848


def main() -> None:
    github_client = GitHubClient()
    evidence_collector = EvidenceCollector(github_client)

    reasoner = OpenAIReasoner()
    classifier = FailureClassifier()
    verifier = GroundingVerifier()

    pipeline = RCAPipeline(
        reasoner=reasoner,
        classifier=classifier,
        verifier=verifier,
    )

    formatter = ReportFormatter()

    evidence = evidence_collector.collect(RUN_ID)
    rca = pipeline.run(evidence)
    markdown = formatter.to_markdown(rca)

    print(markdown)


if __name__ == "__main__":
    main()