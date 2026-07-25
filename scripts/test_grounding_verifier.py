from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.github_client import GitHubClient
from app.services.evidence_collector import EvidenceCollector
from app.services.openai_reasoner import OpenAIReasoner
from app.services.grounding_verifier import GroundingVerifier

run_id = 28731123848

client = GitHubClient()
collector = EvidenceCollector(client)
reasoner = OpenAIReasoner()
verifier = GroundingVerifier()

evidence = collector.collect(run_id)
rca = reasoner.generate(evidence)

raw_log = evidence.logs[0]

print("RCA:")
print(rca)

ok = verifier.verify(rca, raw_log)

print("grounding verified:", ok)

bad_rca = {
    "claims": [
        {
            "statement": "fake claim",
            "citations": [
                {
                    "line_start": 174,
                    "line_end": 174,
                    "quote": "This text does not exist in the log"
                }
            ]
        }
    ]
}

print("bad grounding verified:", verifier.verify(bad_rca, raw_log))