"""One-time compatibility migration; old tables are retained as read-only backups."""
from datetime import datetime, timezone
import hashlib
import json

from sqlalchemy import MetaData, Table, inspect, select, text

from app.core.config import GITHUB_OWNER, GITHUB_REPO, OPENAI_MODEL
from app.models.triage_task import TriageTask


def legacy_agent_key(evidence, strict):
    identity = {
        "version": 1, "repo": [GITHUB_OWNER, GITHUB_REPO],
        "run_id": evidence["run_id"], "sha": evidence["sha"],
        "run_attempt": evidence.get("run_attempt", 1),
        "workflow": evidence["workflow_name"], "branch": evidence["branch"],
        "last_green_run_id": evidence["last_green_run_id"],
        "pull_request_number": evidence["pull_request_number"],
        "jobs": evidence["failed_jobs"], "diff": evidence["diff"],
        "logs": [(log["job_id"], log["text"], log["lines"]) for log in evidence["logs"]],
        "strict": strict, "model": OPENAI_MODEL,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def migrate_task_tables(connection):
    inspector = inspect(connection)
    tables = set(inspector.get_table_names())
    old_agents = {}
    if "agent_runs" in tables:
        table = Table("agent_runs", MetaData(), autoload_with=connection)
        old_agents = {row["id"]: row for row in connection.execute(select(table)).mappings()}
    if "triage_tasks" in tables:
        columns = {column["name"] for column in inspector.get_columns("triage_tasks")}
        if "agent_state" not in columns:
            if "triage_tasks_legacy" in tables:
                raise RuntimeError("triage_tasks_legacy already exists; refusing to overwrite backup")
            table = Table("triage_tasks", MetaData(), autoload_with=connection)
            tasks = list(connection.execute(select(table)).mappings())
            indexes = inspector.get_indexes("triage_tasks")
            primary_key = inspector.get_pk_constraint("triage_tasks").get("name")
            connection.execute(text("ALTER TABLE triage_tasks RENAME TO triage_tasks_legacy"))
            quote = connection.dialect.identifier_preparer.quote
            if connection.dialect.name == "postgresql":
                names = {index["name"] for index in indexes}
                if primary_key:
                    names.add(primary_key)
                for name in names:
                    connection.execute(text(f"ALTER INDEX {quote(name)} RENAME TO {quote(name + '_legacy')}"))
            else:
                # SQLite cannot rename indexes; archived rows and PK are retained.
                for index in indexes:
                    connection.execute(text(f"DROP INDEX {quote(index['name'])}"))
            TriageTask.__table__.create(connection)
            for task in tasks:
                agent_state = {}
                evidence = (task["steps"] or {}).get("evidence", {}).get("result")
                if evidence:
                    for mode, strict in (("normal", False), ("strict", True)):
                        agent = old_agents.get(legacy_agent_key(evidence, strict))
                        if agent:
                            agent_state[mode] = {"state": agent["state"], "result": agent["result"]}
                connection.execute(TriageTask.__table__.insert().values(
                    id=task["id"], payload=task["payload"], status=task["status"],
                    steps=task["steps"], agent_state=agent_state,
                    result=task["result"], error=task["error"],
                    updated_at=datetime.now(timezone.utc)))
    if "agent_runs" in tables:
        if "agent_runs_legacy" in tables:
            raise RuntimeError("agent_runs_legacy already exists; refusing to overwrite backup")
        connection.execute(text("ALTER TABLE agent_runs RENAME TO agent_runs_legacy"))
