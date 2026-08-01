from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import DATABASE_URL, ENABLE_PGVECTOR


if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not configured")


engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


def get_db_session() -> Generator[Session, None, None]:
    session = SessionLocal()

    try:
        yield session
    finally:
        session.close()


def create_tables() -> None:
    # 必须先导入模型，SQLAlchemy 才知道有哪些表
    from app.models.feedback import Feedback  # noqa: F401
    from app.models.incident import Incident  # noqa: F401
    from app.models.rag_chunk import RagChunk  # noqa: F401

    Base.metadata.create_all(bind=engine)

    with engine.begin() as conn:
        if ENABLE_PGVECTOR:
            # Provision pgvector mode when explicitly enabled.
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

            # Ensure the embedding column exists first.
            conn.execute(
                text(
                    "ALTER TABLE rag_chunks "
                    "ADD COLUMN IF NOT EXISTS embedding TEXT"
                )
            )

            # Convert text/json-string embeddings into vector values.
            conn.execute(
                text(
                    """
                    ALTER TABLE rag_chunks
                    ALTER COLUMN embedding TYPE vector(32)
                    USING CASE
                      WHEN embedding IS NULL OR btrim(embedding::text) = '' THEN NULL
                      ELSE embedding::vector(32)
                    END
                    """
                )
            )
        else:
            # Backward-compatible fallback without pgvector.
            conn.execute(
                text(
                    "ALTER TABLE rag_chunks "
                    "ADD COLUMN IF NOT EXISTS embedding TEXT"
                )
            )