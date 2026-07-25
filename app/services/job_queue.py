from queue import Queue
from app.models.triage_job import TriageJob

triage_queue: Queue[TriageJob] = Queue()


def enqueue_triage_job(job: TriageJob) -> None:
    triage_queue.put(job)


def get_triage_job() -> TriageJob:
    return triage_queue.get()