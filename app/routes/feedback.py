from html import escape

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import HTMLResponse

from app.core.database import SessionLocal
from app.schemas.feedback import (
    FeedbackCreateRequest,
    FeedbackResponse,
    FeedbackVerdict,
)
from app.services.feedback_store import (
    FeedbackStore,
    IncidentNotFoundError,
)
from app.services.rag_store import RagStore


router = APIRouter(
    prefix="/feedback",
    tags=["feedback"],
)

feedback_store = FeedbackStore(SessionLocal)
rag_store = RagStore(SessionLocal)


@router.post(
    "",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_feedback(
    request: FeedbackCreateRequest,
) -> FeedbackResponse:
    try:
        feedback = feedback_store.create_feedback(request)
        rag_store.apply_feedback(
            incident_id=feedback.incident_id,
            verdict=feedback.verdict,
        )
        return FeedbackResponse.model_validate(feedback)

    except IncidentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.get(
    "/capture/{incident_id}/{verdict}",
    response_class=HTMLResponse,
)
def capture_feedback(
    incident_id: int,
    verdict: FeedbackVerdict,
) -> HTMLResponse:
    try:
        feedback = feedback_store.create(
            incident_id=incident_id,
            verdict=verdict,
            reviewer="report-link",
        )
        rag_store.apply_feedback(
            incident_id=feedback.incident_id,
            verdict=feedback.verdict,
        )

    except IncidentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    verdict_labels = {
        FeedbackVerdict.correct: "👍 Correct",
        FeedbackVerdict.wrong: "👎 Wrong",
        FeedbackVerdict.not_quite: "🤏 Not Quite",
    }

    selected_label = escape(verdict_labels[verdict])

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta
            name="viewport"
            content="width=device-width, initial-scale=1.0"
        >
        <title>CI/CD Triage Agent Feedback</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background: #f6f8fa;
                margin: 0;
                padding: 40px 20px;
            }}

            .card {{
                max-width: 560px;
                margin: 60px auto;
                background: white;
                border: 1px solid #d0d7de;
                border-radius: 12px;
                padding: 32px;
                text-align: center;
                box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08);
            }}

            h1 {{
                margin-top: 0;
            }}

            .verdict {{
                font-size: 20px;
                font-weight: bold;
                margin: 24px 0;
            }}

            .muted {{
                color: #57606a;
            }}
        </style>
    </head>

    <body>
        <main class="card">
            <h1>Thank you!</h1>

            <p>
                Your feedback for incident
                <strong>#{incident_id}</strong>
                has been recorded.
            </p>

            <div class="verdict">
                {selected_label}
            </div>

            <p class="muted">
                Feedback record #{feedback.id}
            </p>

            <p class="muted">
                You may now close this page.
            </p>
        </main>
    </body>
    </html>
    """

    return HTMLResponse(content=html)


@router.get(
    "/{feedback_id}",
    response_model=FeedbackResponse,
)
def get_feedback(
    feedback_id: int,
) -> FeedbackResponse:
    feedback = feedback_store.get_feedback(feedback_id)

    if feedback is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Feedback {feedback_id} was not found.",
        )

    return FeedbackResponse.model_validate(feedback)


@router.get(
    "/incident/{incident_id}",
    response_model=list[FeedbackResponse],
)
def list_incident_feedback(
    incident_id: int,
) -> list[FeedbackResponse]:
    feedback_list = feedback_store.list_by_incident(incident_id)

    return [
        FeedbackResponse.model_validate(feedback)
        for feedback in feedback_list
    ]