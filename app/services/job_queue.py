"""One durable task table, consumed by one worker."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.core.config import GITHUB_OWNER, GITHUB_REPO
from app.core.database import SessionLocal
from app.models.triage_task import TriageTask
from app.services.recovery_errors import ManualReviewRequired


def uncertain_effect(row):
    return any(step["status"] == "started" and step.get("effect")
               for step in (row.steps or {}).values()) or any(
        entry.get("state", {}).get("tool_started") and
        (entry.get("state", {}).get("pending") or {}).get("tool") == "rerun_job"
        for entry in (row.agent_state or {}).values())


class TaskQueue:
    def __init__(self, session_factory=SessionLocal):
        self.session_factory = session_factory

    def enqueue(self, job):
        identity = f"{GITHUB_OWNER}/{GITHUB_REPO}:{job.run_id}:{job.run_attempt}"
        key = hashlib.sha256(identity.encode()).hexdigest()
        with self.session_factory() as session:
            if session.get(TriageTask, key) is None:
                session.add(TriageTask(id=key, payload=asdict(job)))
                try:
                    session.commit()
                except IntegrityError:
                    session.rollback()
                    if session.get(TriageTask, key) is None:
                        raise
        return key

    def recover_running(self):
        """Call once at startup, after the previous single worker has stopped."""
        with self.session_factory() as session:
            for row in session.scalars(select(TriageTask).where(TriageTask.status == "running")):
                row.status = "needs_review" if uncertain_effect(row) else "pending"
                if row.status == "needs_review":
                    row.error = "Interrupted external operation requires review"
            session.commit()

    def claim(self, key=None):
        with self.session_factory() as session:
            statement = select(TriageTask).where(TriageTask.status == "pending")
            if key is not None:
                statement = statement.where(TriageTask.id == key)
            row = session.scalars(statement.order_by(TriageTask.updated_at, TriageTask.id)).first()
            if row is None:
                return None
            row.status, row.error = "running", None
            session.commit()
            return TaskExecution(self, row.id, row.payload)

    def inspect(self, key):
        with self.session_factory() as session:
            row = session.get(TriageTask, key)
            if row is None:
                raise KeyError(key)
            return {name: getattr(row, name) for name in (
                "id", "payload", "status", "steps", "agent_state", "result", "error", "updated_at")}

    def retry(self, key):
        with self.session_factory() as session:
            row = session.get(TriageTask, key)
            if row is None:
                raise KeyError(key)
            if row.status not in {"failed", "needs_review"}:
                raise ValueError("Only failed or reviewed tasks can be retried")
            if uncertain_effect(row):
                raise ManualReviewRequired("Resolve uncertain effects before retrying")
            row.status, row.error = "pending", None
            session.commit()

    def resolve(self, key, name, result=None, *, not_executed=False):
        """Record an externally verified outcome, then use retry to continue."""
        with self.session_factory() as session:
            row = session.get(TriageTask, key)
            if row is None:
                raise KeyError(key)
            if row.status != "needs_review":
                raise ValueError("Task must be in needs_review")
            if name in {"agent_rerun", "agent_rerun_strict"}:
                mode = "strict" if name.endswith("strict") else "normal"
                entry = dict((row.agent_state or {}).get(mode, {}))
                state = dict(entry.get("state", {}))
                pending = state.get("pending") or {}
                if pending.get("tool") != "rerun_job" or not state.get("tool_started"):
                    raise ValueError("No uncertain agent rerun")
                if not_executed:
                    state["tool_started"] = False
                else:
                    if not isinstance(result, bool):
                        raise ValueError("Agent rerun result must be true or false")
                    output = {"ok": True, "tool": "rerun_job", "result": {
                        "run_id": (pending.get("args") or {}).get("run_id") or row.payload["run_id"],
                        "accepted": result}}
                    state["tool_exchange"] = [*state.get("tool_exchange", []), {
                        "type": "function_call_output", "call_id": pending["call_id"],
                        "output": json.dumps(output)}]
                    state["observations"] = [*state.get("observations", []), {
                        "tool_request": pending, "tool_result": output,
                        "resolved_by_operator": True}]
                    state.update(turn=state.get("turn", 0) + 1, pending=None, tool_started=False)
                entry["state"] = state
                row.agent_state = {**(row.agent_state or {}), mode: entry}
            else:
                step = (row.steps or {}).get(name)
                if not step or step["status"] != "started" or not step.get("effect"):
                    raise ValueError("Not an uncertain external step")
                steps = dict(row.steps)
                if not_executed:
                    del steps[name]
                else:
                    steps[name] = {"status": "completed", "result": result}
                row.steps = steps
            session.commit()


class TaskExecution:
    def __init__(self, queue, key, payload):
        self.queue, self.key, self.payload = queue, key, payload

    def _running(self, session):
        row = session.get(TriageTask, self.key)
        if row is None or row.status != "running":
            raise RuntimeError("Task is not running")
        row.updated_at = datetime.now(timezone.utc)
        return row

    def step(self, name, operation, *, effect=False, transactional=False):
        with self.queue.session_factory() as session:
            row = self._running(session)
            previous = (row.steps or {}).get(name)
            if previous and previous["status"] == "completed":
                return previous["result"]
            if previous and previous["status"] == "started" and effect:
                raise ManualReviewRequired(f"Uncertain outcome for step {name}")
            if transactional:
                result = operation(session)
                row.steps = {**(row.steps or {}), name: {"status": "completed", "result": result}}
                session.commit()
                return result
            row.steps = {**(row.steps or {}), name: {"status": "started", "effect": effect}}
            session.commit()
        result = operation()
        with self.queue.session_factory() as session:
            row = self._running(session)
            row.steps = {**row.steps, name: {"status": "completed", "result": result}}
            session.commit()
        return result

    def checkpoint(self, strict=False):
        return TaskCheckpoint(self, "strict" if strict else "normal")

    def complete(self, result):
        with self.queue.session_factory() as session:
            row = self._running(session)
            row.result, row.status, row.error = result, "completed", None
            session.commit()

    def fail(self, error):
        with self.queue.session_factory() as session:
            row = self._running(session)
            row.status = "needs_review" if isinstance(error, ManualReviewRequired) or uncertain_effect(row) else "failed"
            row.error = str(error)
            session.commit()


class TaskCheckpoint:
    """Model state lives in the same task row, separated by grounding pass."""
    def __init__(self, execution, mode):
        self.execution, self.mode = execution, mode
        entry = execution.queue.inspect(execution.key)["agent_state"].get(mode, {})
        self.state, self.result = entry.get("state", {}), entry.get("result")

    def _write(self):
        # Native SDK response items must become plain JSON for database storage.
        entry = json.loads(json.dumps({"state": self.state, "result": self.result},
            default=lambda item: item.model_dump(mode="json", exclude_none=True)))
        with self.execution.queue.session_factory() as session:
            row = self.execution._running(session)
            row.agent_state = {**(row.agent_state or {}), self.mode: entry}
            session.commit()

    def save(self, state):
        self.state = state
        self._write()

    def complete(self, result):
        self.result = result
        self._write()


def enqueue_triage_job(job):
    return TaskQueue().enqueue(job)
