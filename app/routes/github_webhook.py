from fastapi import APIRouter, Request, HTTPException

from app.core.telemetry import record_exception, set_span_attributes, tracer
from app.core.security import verify_github_signature
from app.models.triage_job import TriageJob
from app.services.job_queue import enqueue_triage_job

router = APIRouter(
    prefix="/webhooks",
    tags=["GitHub Webhooks"],
)


@router.post("/github")
async def github_webhook(request: Request):
    if tracer is None:
        return await _github_webhook_impl(request)

    with tracer.start_as_current_span("webhook.received") as span:
        try:
            raw_body = await request.body()
            signature = request.headers.get("X-Hub-Signature-256")

            if not verify_github_signature(raw_body, signature):
                raise HTTPException(status_code=401, detail="Invalid GitHub signature")

            event_type = request.headers.get("X-GitHub-Event")
            payload = await request.json()
            set_span_attributes(
                span,
                event_type=event_type,
                github_delivery_id=request.headers.get("X-GitHub-Delivery"),
            )

            if event_type != "workflow_run":
                set_span_attributes(span, decision="ignored", reason="not workflow_run")
                return {
                    "status": "ignored",
                    "reason": "not workflow_run",
                }

            workflow_run = payload.get("workflow_run", {})
            run_id = workflow_run.get("id")
            conclusion = workflow_run.get("conclusion")
            workflow_name = workflow_run.get("name")
            branch = workflow_run.get("head_branch")
            sha = workflow_run.get("head_sha")

            set_span_attributes(
                span,
                run_id=run_id,
                conclusion=conclusion,
                workflow_name=workflow_name,
                branch=branch,
                sha=sha,
            )

            if conclusion != "failure":
                set_span_attributes(span, decision="ignored", reason="workflow did not fail")
                return {
                    "status": "ignored",
                    "reason": "workflow did not fail",
                    "run_id": run_id,
                    "conclusion": conclusion,
                }

            job = TriageJob(
                run_id=run_id,
                workflow=workflow_name,
                branch=branch,
                sha=sha,
                run_attempt=int(workflow_run.get("run_attempt", 1)),
            )

            task_id = enqueue_triage_job(job)

            print(f"[webhook] queued triage job: run_id={run_id}", flush=True)
            set_span_attributes(span, decision="queued")

            return {
                "status": "queued",
                "task_id": task_id,
                "run_id": run_id,
                "workflow": workflow_name,
                "branch": branch,
                "sha": sha,
            }
        except Exception as exc:
            record_exception(span, exc)
            raise


async def _github_webhook_impl(request: Request):
    raw_body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256")

    if not verify_github_signature(raw_body, signature):
        raise HTTPException(status_code=401, detail="Invalid GitHub signature")

    event_type = request.headers.get("X-GitHub-Event")
    payload = await request.json()

    if event_type != "workflow_run":
        return {
            "status": "ignored",
            "reason": "not workflow_run",
        }

    workflow_run = payload.get("workflow_run", {})

    run_id = workflow_run.get("id")
    conclusion = workflow_run.get("conclusion")
    workflow_name = workflow_run.get("name")
    branch = workflow_run.get("head_branch")
    sha = workflow_run.get("head_sha")

    if conclusion != "failure":
        return {
            "status": "ignored",
            "reason": "workflow did not fail",
            "run_id": run_id,
            "conclusion": conclusion,
        }

    job = TriageJob(
        run_id=run_id,
        workflow=workflow_name,
        branch=branch,
        sha=sha,
        run_attempt=int(workflow_run.get("run_attempt", 1)),
    )

    task_id = enqueue_triage_job(job)

    print(f"[webhook] queued triage job: run_id={run_id}", flush=True)

    return {
        "status": "queued",
        "task_id": task_id,
        "run_id": run_id,
        "workflow": workflow_name,
        "branch": branch,
        "sha": sha,
    }