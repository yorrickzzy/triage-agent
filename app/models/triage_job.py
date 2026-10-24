from dataclasses import dataclass


@dataclass
class TriageJob:
    run_id: int
    workflow: str
    branch: str
    sha: str
    run_attempt: int = 1
