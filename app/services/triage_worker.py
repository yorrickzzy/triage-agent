import threading

from app.bootstrap import build_triage_service
from app.services.job_queue import get_triage_job


def worker_loop() -> None:
    print("[worker] starting...", flush=True)
    service = build_triage_service()
    print("[worker] ready", flush=True)

    while True:
        job = get_triage_job()

        print(
            f"[worker] processing run {job.run_id}",
            flush=True,
        )

        try:
            result = service.triage(job.run_id)

            print(
                f"[worker] completed "
                f"(action={result.action}, "
                f"incident_id={result.incident_id})",
                flush=True,
            )

            if result.comment_url:
                print(
                    f"[worker] comment_url={result.comment_url}",
                    flush=True,
                )

        except Exception as e:
            print(
                f"[worker] failed for run_id={job.run_id}: {e}",
                flush=True,
            )


def start_worker() -> None:
    thread = threading.Thread(
        target=worker_loop,
        name="triage-worker",
        daemon=True,
    )
    thread.start()