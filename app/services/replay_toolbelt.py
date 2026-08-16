import re

from app.models.evidence import Evidence, RawLog
from app.services.triage_toolbelt import RunRef


class ReplayToolbelt:
    """Toolbelt backed exclusively by one replay fixture."""

    def __init__(
        self,
        evidence: Evidence,
        playbook_chunks: list[dict] | None = None,
        rerun_accepted: bool = False,
    ) -> None:
        self.evidence = evidence
        self.playbook_chunks = list(playbook_chunks or [])
        self.rerun_accepted = rerun_accepted

    def get_failed_logs(self, run_id: int) -> list[RawLog]:
        if run_id != self.evidence.run_id:
            raise ValueError("Replay fixture does not contain logs for this run")

        return self.evidence.logs

    def get_commit_diff(self, sha: str) -> str:
        if sha != self.evidence.sha:
            raise ValueError("Replay fixture does not contain a diff for this SHA")

        return self.evidence.diff

    def get_last_green_run(
        self,
        workflow_id: int | str,
        branch: str,
        current_run_id: int,
    ) -> RunRef | None:
        if current_run_id != self.evidence.run_id:
            raise ValueError("Replay fixture does not contain this workflow run")

        if self.evidence.last_green_run_id is None:
            return None

        return RunRef(
            run_id=self.evidence.last_green_run_id,
            workflow_id=workflow_id,
            workflow_name=self.evidence.workflow_name,
            branch=branch,
            sha="",
            status="completed",
            conclusion="success",
        )

    def rerun_job(self, run_id: int) -> bool:
        if run_id != self.evidence.run_id:
            raise ValueError("Replay fixture does not contain this workflow run")

        return self.rerun_accepted

    def search_playbook(self, query: str, limit: int = 5) -> list[dict]:
        if limit < 1:
            return []

        query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))

        def score(chunk: dict) -> int:
            chunk_text = str(chunk.get("text", "")).lower()
            chunk_terms = set(re.findall(r"[a-z0-9]+", chunk_text))
            return len(query_terms & chunk_terms)

        ranked = sorted(
            self.playbook_chunks,
            key=lambda chunk: (score(chunk), chunk.get("quality_score", 0.0)),
            reverse=True,
        )
        return ranked[:limit]