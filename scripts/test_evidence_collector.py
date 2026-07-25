from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.github_client import GitHubClient
from app.services.evidence_collector import EvidenceCollector

client = GitHubClient()
collector = EvidenceCollector(client)

evidence = collector.collect(29178076814)

print("run_id:", evidence.run_id)
print("workflow:", evidence.workflow_name)
print("branch:", evidence.branch)
print("sha:", evidence.sha)
print("failed_jobs:", evidence.failed_jobs)

for log in evidence.logs:
    print("job:", log.job_name)
    print("log length:", len(log.text))
    print("log tail:")
    print(log.text[-1500:])

print("=" * 60)
print("diff preview:")
print(evidence.diff[:1500])
print("last_green_run_id:", evidence.last_green_run_id)

first_log = evidence.logs[0]

print("total lines:", len(first_log.lines))
print("line 174:", first_log.text_between(174, 174))
print("line 175-176:")
print(first_log.text_between(175, 176))
print("pull_request_number:", evidence.pull_request_number)