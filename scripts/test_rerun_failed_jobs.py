from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.github_client import GitHubClient

RUN_ID = 28731123848

client = GitHubClient()

try:
    accepted = client.rerun_failed_jobs(RUN_ID)

    print("rerun accepted:", accepted)

except Exception as exc:
    print("rerun failed:", exc)