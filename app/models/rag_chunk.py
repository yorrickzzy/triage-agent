from datetime import datetime, timezone

from sqlalchemy import BigInteger, DateTime, Float, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base

try:
    from pgvector.sqlalchemy import Vector  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    Vector = None


class RagChunk(Base):
    __tablename__ = "rag_chunks"

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    source_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )

    source_id: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        index=True,
    )

    workflow_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    failure_class: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        index=True,
    )

    text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    quality_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
    )

    if Vector is not None:
        embedding: Mapped[list[float] | None] = mapped_column(
            Vector(32),
            nullable=True,
        )
    else:
        embedding: Mapped[str | None] = mapped_column(
            Text,
            nullable=True,
        )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )