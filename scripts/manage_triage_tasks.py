"""Inspect/retry tasks and record externally verified outcomes. Sends no messages."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select
from app.core.database import SessionLocal
from app.models.triage_task import TriageTask
from app.services.job_queue import TaskQueue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list")
    listing.add_argument("--status")
    inspect = sub.add_parser("inspect")
    inspect.add_argument("task_id")
    retry = sub.add_parser("retry")
    retry.add_argument("task_id")
    resolve = sub.add_parser("resolve", help="Only after checking the remote operation")
    resolve.add_argument("task_id")
    resolve.add_argument("step", choices=["github_comment", "slack_message", "flaky_rerun", "agent_rerun", "agent_rerun_strict"])
    outcome = resolve.add_mutually_exclusive_group(required=True)
    outcome.add_argument("--result-file", type=Path, help="JSON remote response; rerun uses true/false")
    outcome.add_argument("--not-executed", action="store_true", help="Confirmed no remote effect occurred")
    args = parser.parse_args()
    queue = TaskQueue()
    if args.command == "list":
        with SessionLocal() as session:
            statement = select(TriageTask).order_by(TriageTask.id).limit(100)
            if args.status:
                statement = statement.where(TriageTask.status == args.status)
            print(json.dumps([{"id": row.id, "run_id": row.payload["run_id"],
                "run_attempt": row.payload["run_attempt"], "status": row.status,
                "error": row.error}
                for row in session.scalars(statement)], indent=2, ensure_ascii=False))
    elif args.command == "inspect":
        data = queue.inspect(args.task_id)
        print(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    elif args.command == "retry":
        queue.retry(args.task_id)
        print("Task queued for retry")
    elif args.command == "resolve":
        result = (json.loads(args.result_file.read_text(encoding="utf-8-sig"))
                  if args.result_file else None)
        if not args.not_executed:
            if args.step in {"flaky_rerun", "agent_rerun", "agent_rerun_strict"} and not isinstance(result, bool):
                parser.error("Rerun result must be a JSON boolean")
            if args.step not in {"flaky_rerun", "agent_rerun", "agent_rerun_strict"} and not isinstance(result, dict):
                parser.error("Delivery result must be a JSON response object")
        queue.resolve(args.task_id, args.step, result, not_executed=args.not_executed)
        print("Outcome recorded; use retry to continue")


if __name__ == "__main__":
    main()
