from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.models.evidence import RawLog


@dataclass(frozen=True)
class RunRef:
    run_id: int
    workflow_id: int | str
    workflow_name: str
    branch: str
    sha: str
    status: str | None = None
    conclusion: str | None = None


@runtime_checkable
class TriageToolbelt(Protocol):
    def get_failed_logs(self, run_id: int) -> list[RawLog]:
        ...

    def get_commit_diff(self, sha: str) -> str:
        ...

    def get_last_green_run(
        self,
        workflow_id: int | str,
        branch: str,
        current_run_id: int,
    ) -> RunRef | None:
        ...

    def rerun_job(self, run_id: int) -> bool:
        ...

    def search_playbook(self, query: str, limit: int = 5) -> list[dict]:
        ...


class DefaultTriageToolbelt:
    def __init__(self, github_client, rag_store) -> None:
        self.github = github_client
        self.rag_store = rag_store

    def get_failed_logs(self, run_id: int) -> list[RawLog]:
        failed_jobs = self.github.get_failed_jobs(run_id)
        logs: list[RawLog] = []

        for job in failed_jobs:
            text = self.github.download_job_logs(job["id"])
            text = text.lstrip("\ufeff")

            logs.append(
                RawLog(
                    job_id=job["id"],
                    job_name=job["name"],
                    text=text,
                    lines=text.splitlines(),
                )
            )

        return logs

    def get_commit_diff(self, sha: str) -> str:
        commit = self.github.get_commit(sha)
        parents = commit.get("parents", [])

        if not parents:
            return ""

        parent_sha = parents[0]["sha"]
        return self.github.compare_commits(parent_sha, sha)

    def get_last_green_run(
        self,
        workflow_id: int | str,
        branch: str,
        current_run_id: int,
    ) -> RunRef | None:
        run = self.github.get_last_green_run(
            workflow_id=workflow_id,
            branch=branch,
            current_run_id=current_run_id,
        )

        if run is None:
            return None

        return RunRef(
            run_id=run["id"],
            workflow_id=run.get("workflow_id", workflow_id),
            workflow_name=run.get("name", ""),
            branch=run.get("head_branch", branch),
            sha=run.get("head_sha", ""),
            status=run.get("status"),
            conclusion=run.get("conclusion"),
        )

    def rerun_job(self, run_id: int) -> bool:
        return self.github.rerun_failed_jobs(run_id)

    def search_playbook(self, query: str, limit: int = 5) -> list[dict]:
        return self.rag_store.search_playbook(query, limit=limit)