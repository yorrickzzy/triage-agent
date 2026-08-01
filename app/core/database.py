from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import DATABASE_URL


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

    # Backward-compatible migration for existing databases:
    # keep a nullable embedding column available for RAG metadata.
    with engine.begin() as conn:
        conn.execute(
            text(
                "ALTER TABLE rag_chunks "
                "ADD COLUMN IF NOT EXISTS embedding TEXT"
            )
        )