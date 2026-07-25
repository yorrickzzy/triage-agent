from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class FeedbackVerdict(str, Enum):
    correct = "correct"
    wrong = "wrong"
    not_quite = "not_quite"


class FeedbackCreateRequest(BaseModel):
    incident_id: int = Field(gt=0)
    verdict: FeedbackVerdict
    comment: str | None = Field(default=None, max_length=2000)
    reviewer: str | None = Field(default=None, max_length=100)


class FeedbackResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    incident_id: int
    verdict: FeedbackVerdict
    comment: str | None
    reviewer: str | None
    created_at: datetime