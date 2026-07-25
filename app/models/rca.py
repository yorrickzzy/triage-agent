from dataclasses import dataclass
from typing import List


@dataclass
class Citation:
    line_start: int
    line_end: int
    quote: str


@dataclass
class Claim:
    statement: str
    citations: List[Citation]


@dataclass
class RCA:
    failure_class: str
    summary: str
    claims: List[Claim]
    suggested_next_step: str
    confidence: float
    abstained: bool