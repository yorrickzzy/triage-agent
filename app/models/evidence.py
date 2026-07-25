from dataclasses import dataclass
from typing import List

@dataclass
class RawLog:
    job_id: int
    job_name: str
    text: str
    lines: List[str]

    def text_between(self, line_start: int, line_end: int) -> str:
        if line_start < 1 or line_end < line_start:
            return ""

        # line_start / line_end 使用 1-based 行号
        selected = self.lines[line_start - 1: line_end]
        return "\n".join(selected)


@dataclass
class Evidence:
    run_id: int
    workflow_name: str
    branch: str
    sha: str
    failed_jobs: List[str]
    logs: List[RawLog]
    diff: str
    last_green_run_id: int | None
    pull_request_number: int | None