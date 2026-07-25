import requests
import time

from app.core.config import GITHUB_OWNER, GITHUB_REPO, GITHUB_TOKEN


class GitHubAPIError(Exception):
    pass

class GitHubRerunTimeoutError(Exception):
    pass

class GitHubClient:
    def __init__(self):
        if not GITHUB_TOKEN:
            raise RuntimeError("GITHUB_TOKEN is not configured")
        if not GITHUB_OWNER or not GITHUB_REPO:
            raise RuntimeError("GITHUB_OWNER or GITHUB_REPO is not configured")

        self.owner = GITHUB_OWNER
        self.repo = GITHUB_REPO
        self.base_url = f"https://api.github.com/repos/{self.owner}/{self.repo}"

        self.headers = {
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def _get_json(self, path: str, params: dict | None = None) -> dict:
        url = f"{self.base_url}{path}"

        response = requests.get(
            url,
            headers=self.headers,
            params=params,
            timeout=20,
        )

        if not response.ok:
            raise GitHubAPIError(
                f"GitHub API failed: GET {path} "
                f"status={response.status_code} body={response.text[:500]}"
            )

        return response.json()

    def _get_text(
        self,
        path: str,
        params: dict | None = None,
        accept: str | None = None,
    ) -> str:
        url = f"{self.base_url}{path}"

        headers = dict(self.headers)
        if accept:
            headers["Accept"] = accept

        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=30,
        )

        if not response.ok:
            raise GitHubAPIError(
                f"GitHub API failed: GET {path} "
                f"status={response.status_code} body={response.text[:500]}"
            )

        return response.text

    def get_workflow_run(self, run_id: int) -> dict:
        return self._get_json(f"/actions/runs/{run_id}")

    def get_workflow_jobs(self, run_id: int) -> list[dict]:
        data = self._get_json(
            f"/actions/runs/{run_id}/jobs",
            params={"per_page": 100},
        )
        return data.get("jobs", [])

    def get_failed_jobs(self, run_id: int) -> list[dict]:
        jobs = self.get_workflow_jobs(run_id)
        return [
            job
            for job in jobs
            if job.get("conclusion") == "failure"
        ]

    def download_job_logs(self, job_id: int) -> str:
        return self._get_text(f"/actions/jobs/{job_id}/logs")

    def get_commit(self, sha: str) -> dict:
        return self._get_json(f"/commits/{sha}")

    def compare_commits(self, base_sha: str, head_sha: str) -> str:
        return self._get_text(
            f"/compare/{base_sha}...{head_sha}",
            accept="application/vnd.github.v3.diff",
        )

    def list_workflow_runs(
        self,
        workflow_id: int | str,
        branch: str,
        status: str = "success",
        per_page: int = 20,
    ) -> list[dict]:
        data = self._get_json(
            f"/actions/workflows/{workflow_id}/runs",
            params={
                "branch": branch,
                "status": status,
                "per_page": per_page,
            },
        )
        return data.get("workflow_runs", [])

    def get_last_green_run(
        self,
        workflow_id: int | str,
        branch: str,
        current_run_id: int,
    ) -> dict | None:
        runs = self.list_workflow_runs(
            workflow_id=workflow_id,
            branch=branch,
            status="success",
            per_page=20,
        )

        for run in runs:
            if run.get("id") != current_run_id:
                return run

        return None

    def _post(
        self,
        path: str,
        json_body: dict | None = None,
    ) -> requests.Response:
        url = f"{self.base_url}{path}"

        response = requests.post(
            url,
            headers=self.headers,
            json=json_body,
            timeout=20,
        )

        if not response.ok:
            raise GitHubAPIError(
                f"GitHub API failed: POST {path} "
                f"status={response.status_code} "
                f"body={response.text[:500]}"
            )

        return response
    
    def rerun_failed_jobs(self, run_id: int) -> bool:
        response = self._post(
            f"/actions/runs/{run_id}/rerun-failed-jobs"
        )

        return response.status_code == 201
    
    def wait_for_new_attempt_completion(
        self,
        run_id: int,
        previous_attempt: int,
        timeout_seconds: int = 300,
        poll_interval_seconds: int = 5,
    ) -> dict:
        """
        等待同一个 workflow run 出现新的 run_attempt，并完成运行。

        返回完成后的 workflow run JSON。
        """
        deadline = time.monotonic() + timeout_seconds

        while time.monotonic() < deadline:
            run = self.get_workflow_run(run_id)

            current_attempt = int(run.get("run_attempt", 1))
            status = run.get("status")
            conclusion = run.get("conclusion")

            print(
                "[rerun] "
                f"run_id={run_id}, "
                f"attempt={current_attempt}, "
                f"status={status}, "
                f"conclusion={conclusion}",
                flush=True,
            )

            # 必须是新的 attempt，不能误把旧的 completed run 当作结果
            if (
                current_attempt > previous_attempt
                and status == "completed"
            ):
                return run

            time.sleep(poll_interval_seconds)

        raise GitHubRerunTimeoutError(
            f"Timed out waiting for rerun completion: "
            f"run_id={run_id}, previous_attempt={previous_attempt}"
        )

    def post_issue_comment(
        self,
        issue_number: int,
        body: str,
    ) -> dict:
        if issue_number <= 0:
            raise ValueError("issue_number must be positive")

        if not body.strip():
            raise ValueError("Comment body cannot be empty")

        response = self._post(
            f"/issues/{issue_number}/comments",
            json_body={"body": body},
        )

        return response.json()