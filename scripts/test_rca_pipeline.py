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
from app.services.confidence_gate import ConfidenceGate

run_id = 28731123848

client = GitHubClient()
collector = EvidenceCollector(client)

reasoner = OpenAIReasoner()
classifier = FailureClassifier()
verifier = GroundingVerifier()

pipeline = RCAPipeline(
    reasoner,
    classifier,
    verifier,
)

confidence_gate = ConfidenceGate()

evidence = collector.collect(run_id)

rca = pipeline.run(evidence)

gate_result = confidence_gate.evaluate(rca)

print("========== RCA ==========")
print(rca)

print("\n========== Gate ==========")
print(gate_result.decision)
print(gate_result.reason)