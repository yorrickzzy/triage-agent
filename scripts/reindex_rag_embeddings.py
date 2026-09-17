"""Backfill semantic vectors using the configured OpenAI key and database."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.database import SessionLocal, create_tables
from app.services.rag_store import RagStore


if __name__ == "__main__":
    create_tables()
    count = RagStore(SessionLocal).reindex_embeddings()
    print(f"Reindexed {count} chunks. Already-current embeddings were skipped.")
