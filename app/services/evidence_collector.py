from app.models.evidence import Evidence, RawLog


class EvidenceCollector:
    def __init__(self, github_client, redactor=None):
        self.github = github_client
        self.redactor = redactor

    def collect(self, run_id: int) -> Evidence:
        run = self.github.get_workflow_run(run_id)
        failed_jobs = self.github.get_failed_jobs(run_id)

        logs = []

        for job in failed_jobs:
            text = self.github.download_job_logs(job["id"])
            text = text.lstrip("\ufeff")

            if self.redactor:
                text = self.redactor.redact_text(text)

            logs.append(
                RawLog(
                    job_id=job["id"],
                    job_name=job["name"],
                    text=text,
                    lines=text.splitlines(),
                )
            )

        diff = self._collect_diff(run["head_sha"])

        if self.redactor:
            diff = self.redactor.redact_text(diff)

        last_green = self.github.get_last_green_run(
            workflow_id=run["workflow_id"],
            branch=run["head_branch"],
            current_run_id=run["id"],
        )

        pull_requests = run.get("pull_requests", [])
        pull_request_number = (
            pull_requests[0].get("number")
            if pull_requests
            else None
        )

        latest = self.github.get_workflow_run(run_id)
        if int(latest.get("run_attempt", 1)) != int(run.get("run_attempt", 1)):
            raise RuntimeError("CI attempt changed while collecting evidence; retry required")

        return Evidence(
            run_id=run["id"],
            workflow_name=run["name"],
            branch=run["head_branch"],
            sha=run["head_sha"],
            failed_jobs=[job["name"] for job in failed_jobs],
            logs=logs,
            diff=diff,
            last_green_run_id=last_green["id"] if last_green else None,
            pull_request_number=pull_request_number,
            run_attempt=int(run.get("run_attempt", 1)),
        )

    def _collect_diff(self, sha: str) -> str:
        commit = self.github.get_commit(sha)
        parents = commit.get("parents", [])

        if not parents:
            return ""

        parent_sha = parents[0]["sha"]
        return self.github.compare_commits(parent_sha, sha)
