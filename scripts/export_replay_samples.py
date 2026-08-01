from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from app.core.config import GITHUB_OWNER, GITHUB_REPO, GITHUB_TOKEN
from app.services.evidence_collector import EvidenceCollector
from app.services.github_client import GitHubClient
from app.services.redactor import Redactor

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export real failed GitHub runs into replay fixture samples."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of failed runs to export.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts/replay_samples.json"),
        help="Output fixture JSON path.",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default=None,
        help="Optional branch filter.",
    )
    return parser.parse_args()


def list_failed_runs(count: int, branch: str | None = None) -> list[dict]:
    if not GITHUB_TOKEN or not GITHUB_OWNER or not GITHUB_REPO:
        raise RuntimeError("Missing GITHUB_TOKEN/GITHUB_OWNER/GITHUB_REPO in environment")

    base = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/actions/runs"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    runs: list[dict] = []
    page = 1

    while len(runs) < count and page <= 10:
        params = {
            "status": "completed",
            "per_page": 50,
            "page": page,
        }
        if branch:
            params["branch"] = branch

        response = requests.get(base, headers=headers, params=params, timeout=30)
        if not response.ok:
            raise RuntimeError(
                f"GitHub API failed: {response.status_code} {response.text[:300]}"
            )

        payload = response.json()
        batch = payload.get("workflow_runs", [])
        if not batch:
            break

        for run in batch:
            if run.get("conclusion") == "failure":
                runs.append(run)
                if len(runs) >= count:
                    break

        page += 1

    return runs[:count]


def evidence_to_dict(evidence) -> dict:
    return {
        "run_id": evidence.run_id,
        "workflow_name": evidence.workflow_name,
        "branch": evidence.branch,
        "sha": evidence.sha,
        "failed_jobs": evidence.failed_jobs,
        "diff": evidence.diff,
        "last_green_run_id": evidence.last_green_run_id,
        "pull_request_number": evidence.pull_request_number,
        "logs": [
            {
                "job_id": log.job_id,
                "job_name": log.job_name,
                "text": log.text,
                "lines": log.lines,
            }
            for log in evidence.logs
        ],
    }


def main() -> int:
    args = parse_args()

    github = GitHubClient()
    collector = EvidenceCollector(github, redactor=Redactor())

    failed_runs = list_failed_runs(args.count, args.branch)
    if not failed_runs:
        raise RuntimeError("No failed runs found for export")

    samples = []
    for run in failed_runs:
        run_id = int(run["id"])
        evidence = collector.collect(run_id)

        samples.append(
            {
                "run_id": run_id,
                "expected_failure_class": None,
                "expected_abstain": None,
                "true_cause": None,
                "ambiguous": None,
                "note": "Fill labels manually after review",
                "evidence": evidence_to_dict(evidence),
            }
        )

        print(f"exported run_id={run_id}")

    payload = {"samples": samples}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"wrote {len(samples)} samples to {args.output}")
    print("Next: fill expected_failure_class/true_cause/ambiguous manually.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())