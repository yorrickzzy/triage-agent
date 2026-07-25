from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.github_client import GitHubClient
from app.services.evidence_collector import EvidenceCollector
from app.services.openai_reasoner import OpenAIReasoner

run_id = 28731123848

client = GitHubClient()
collector = EvidenceCollector(client)
reasoner = OpenAIReasoner()

evidence = collector.collect(run_id)
rca = reasoner.generate(evidence)

print(rca)