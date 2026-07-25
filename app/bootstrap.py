from app.core.database import SessionLocal, create_tables
from app.services.github_client import GitHubClient
from app.services.evidence_collector import EvidenceCollector
from app.services.redactor import Redactor
from app.services.openai_reasoner import OpenAIReasoner
from app.services.failure_classifier import FailureClassifier
from app.services.grounding_verifier import GroundingVerifier
from app.services.rca_pipeline import RCAPipeline
from app.services.confidence_gate import ConfidenceGate
from app.services.report_formatter import ReportFormatter
from app.services.github_delivery import GitHubDelivery
from app.services.triage_service import TriageService
from app.services.slack_client import SlackClient
from app.services.incident_store import IncidentStore
from app.services.rag_store import RagStore


def build_triage_service() -> TriageService:
    """
    Build the production TriageService and all of its dependencies.

    This function is shared by:
    - webhook worker
    - manual delivery test
    - future runtime entry points
    """

    github_client = GitHubClient()
    redactor = Redactor()
    evidence_collector = EvidenceCollector(github_client, redactor=redactor)

    reasoner = OpenAIReasoner()
    classifier = FailureClassifier()
    verifier = GroundingVerifier()

    rca_pipeline = RCAPipeline(
        reasoner=reasoner,
        classifier=classifier,
        verifier=verifier,
    )

    confidence_gate = ConfidenceGate()
    report_formatter = ReportFormatter()
    github_delivery = GitHubDelivery(github_client)
    slack_client = SlackClient()
    incident_store = IncidentStore(SessionLocal)
    rag_store = RagStore(SessionLocal)

    return TriageService(
        evidence_collector=evidence_collector,
        rca_pipeline=rca_pipeline,
        confidence_gate=confidence_gate,
        github_client=github_client,
        report_formatter=report_formatter,
        github_delivery=github_delivery,
        slack_client=slack_client,
        incident_store=incident_store,
        rag_store=rag_store,
    )