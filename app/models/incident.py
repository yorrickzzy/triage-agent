from datetime import datetime, timezone

from sqlalchemy import BigInteger, Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class Incident(Base):
    __tablename__ = "incidents"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )

    run_id: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        index=True,
    )

    workflow_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    branch: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    commit_sha: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    pull_request_number: Mapped[int | None] = mapped_column(
        nullable=True,
    )

    failure_class: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    summary: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    suggested_next_step: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    abstained: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    action: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    decision_reason: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    comment_url: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    reasoning_trace: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
