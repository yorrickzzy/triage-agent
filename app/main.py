from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.database import create_tables
from app.core.telemetry import configure_telemetry
from app.routes.github_webhook import router as github_router
from app.services.triage_worker import start_worker
from app.routes.feedback import router as feedback_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_telemetry(service_name="triage-agent")
    create_tables()
    start_worker()

    yield


app = FastAPI(
    title="CI/CD Triage Agent",
    description=(
        "Evidence-grounded GitHub Actions failure triage, "
        "delivery, incident storage, and feedback capture."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(github_router)
app.include_router(feedback_router)

@app.get("/")
def root():
    return {
        "service": "CI/CD Triage Agent",
        "status": "running",
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}