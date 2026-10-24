import threading
from dataclasses import asdict

from app.bootstrap import build_triage_service


def process_task(service, execution):
    try:
        result = service.triage(execution.payload["run_id"], execution=execution)
        execution.complete(asdict(result))
        return result
    except Exception as exc:
        execution.fail(exc)
        raise


def worker_loop(stop=None):
    stop = stop or threading.Event()
    service = build_triage_service()
    queue = service.task_queue
    queue.recover_running()
    print("[worker] ready (single worker)", flush=True)
    while not stop.is_set():
        try:
            execution = queue.claim()
            if execution is None:
                stop.wait(1)
                continue
            result = process_task(service, execution)
            print(f"[worker] completed task {execution.key}: {result.action}", flush=True)
        except Exception as exc:
            print(f"[worker] task failed: {exc}", flush=True)
            stop.wait(1)


def start_worker():
    stop = threading.Event()
    thread = threading.Thread(target=worker_loop, args=(stop,), name="triage-worker", daemon=True)
    thread.start()
    return stop, thread
