from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.github_client import GitHubClient

client = GitHubClient()

run_id = "28731123848"

run = client.get_workflow_run(run_id)
print("run:", run["id"], run["name"], run["conclusion"])

jobs = client.get_failed_jobs(run_id)
print("failed jobs:", [job["name"] for job in jobs])

if jobs:
    logs = client.download_job_logs(jobs[0]["id"])
    print(logs[:1000])