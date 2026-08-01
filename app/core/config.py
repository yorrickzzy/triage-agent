import os

from dotenv import load_dotenv

load_dotenv()


GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_OWNER = os.getenv("GITHUB_OWNER")
GITHUB_REPO = os.getenv("GITHUB_REPO")
GITHUB_WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_CHANNEL_ID = os.getenv("SLACK_CHANNEL_ID")

DATABASE_URL = os.getenv("DATABASE_URL")

APP_BASE_URL = os.getenv(
    "APP_BASE_URL",
    "http://localhost:8000",
).rstrip("/")

RCA_CONFIDENCE_THRESHOLD = float(os.getenv("RCA_CONFIDENCE_THRESHOLD", 0.8))

ENABLE_PGVECTOR = (
    os.getenv("ENABLE_PGVECTOR", "false").strip().lower()
    in {"1", "true", "yes", "on"}
)