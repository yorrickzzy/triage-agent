from collections.abc import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import DATABASE_URL, ENABLE_PGVECTOR, RAG_EMBEDDING_DIMENSIONS


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
    from app.models.triage_task import TriageTask  # noqa: F401
    # 必须先导入模型，SQLAlchemy 才知道有哪些表
    from app.models.feedback import Feedback  # noqa: F401
    from app.models.incident import Incident  # noqa: F401
    from app.models.rag_chunk import RagChunk  # noqa: F401
    from app.core.task_migration import migrate_task_tables

    with engine.begin() as connection:
        migrate_task_tables(connection)

    if ENABLE_PGVECTOR:
        if engine.dialect.name != "postgresql":
            raise RuntimeError("ENABLE_PGVECTOR requires PostgreSQL")
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(bind=engine)

    with engine.begin() as conn:
        if conn.dialect.name == "postgresql":
            conn.execute(
                text(
                    "ALTER TABLE incidents "
                    "ADD COLUMN IF NOT EXISTS reasoning_trace TEXT"
                )
            )
        elif conn.dialect.name == "sqlite":
            columns = conn.execute(text("PRAGMA table_info(incidents)"))
            existing_names = {row[1] for row in columns}
            if "reasoning_trace" not in existing_names:
                conn.execute(
                    text(
                        "ALTER TABLE incidents "
                        "ADD COLUMN reasoning_trace TEXT"
                    )
                )

        # Additive migration preserves old hashes and all incident/feedback data.
        existing = {column["name"] for column in inspect(conn).get_columns("rag_chunks")}
        for name, sql_type in (
            ("embedding", "TEXT"),
            ("semantic_embedding", "TEXT"),
            ("embedding_model", "VARCHAR(255)"),
        ):
            if name not in existing:
                conn.execute(text(f"ALTER TABLE rag_chunks ADD COLUMN {name} {sql_type}"))
        if ENABLE_PGVECTOR:
            columns = {column["name"]: column for column in inspect(conn).get_columns("rag_chunks")}
            if str(columns["semantic_embedding"]["type"]).upper() != f"VECTOR({RAG_EMBEDDING_DIMENSIONS})":
                conn.execute(text(
                    f"ALTER TABLE rag_chunks ALTER COLUMN semantic_embedding "
                    f"TYPE vector({RAG_EMBEDDING_DIMENSIONS}) "
                    f"USING NULLIF(semantic_embedding::text, '')::vector({RAG_EMBEDDING_DIMENSIONS})"
                ))
