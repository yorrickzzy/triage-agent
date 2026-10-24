from pathlib import Path
from dataclasses import asdict
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("DATABASE_URL", "sqlite://")

from sqlalchemy import Column, Integer, JSON, MetaData, String, Table, create_engine, inspect, select
from sqlalchemy.orm import sessionmaker
from app.core import database
from app.core.task_migration import legacy_agent_key
from app.models.triage_job import TriageJob
from app.models.evidence import Evidence, RawLog
from app.models.triage_task import TriageTask
from app.services.job_queue import TaskQueue


class TaskMigrationTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://")

    def tearDown(self):
        self.engine.dispose()

    def initialize(self):
        with patch.object(database, "engine", self.engine):
            database.create_tables()

    def test_new_database_has_one_task_table_with_eight_columns(self):
        self.initialize()
        self.initialize()
        inspector = inspect(self.engine)
        self.assertNotIn("agent_runs", inspector.get_table_names())
        self.assertEqual({column["name"] for column in inspector.get_columns("triage_tasks")},
            {"id", "payload", "status", "steps", "agent_state", "result", "error", "updated_at"})

    def test_old_tasks_and_matching_model_state_migrate_without_data_loss(self):
        metadata = MetaData()
        tasks = Table("triage_tasks", metadata,
            Column("id", String, primary_key=True), Column("run_id", Integer, index=True),
            Column("run_attempt", Integer), Column("payload", JSON),
            Column("status", String, index=True), Column("steps", JSON),
            Column("result", JSON), Column("error", String))
        agents = Table("agent_runs", metadata, Column("id", String, primary_key=True),
            Column("state", JSON), Column("result", JSON))
        metadata.create_all(self.engine)
        evidence = asdict(Evidence(42, "CI", "main", "abc", ["tests"],
            [RawLog(1, "tests", "error", ["error"])], "", None, None))
        payload = asdict(TriageJob(42, "CI", "main", "abc"))
        state = {"turn": 1, "has_log_evidence": True}
        steps = {"evidence": {"status": "completed", "result": evidence}}
        with self.engine.begin() as connection:
            connection.execute(tasks.insert().values(id="task", run_id=42,
                run_attempt=1, payload=payload, status="running", steps=steps,
                error="original error"))
            connection.execute(agents.insert().values(id=legacy_agent_key(evidence, False),
                state=state, result=None))
            connection.execute(agents.insert().values(id="unmatched-archived-state",
                state={"turn": 3}, result={"summary": "historical"}))
        self.initialize()
        self.initialize()
        queue = TaskQueue(sessionmaker(self.engine, expire_on_commit=False))
        task = queue.inspect("task")
        self.assertEqual(task["payload"], payload)
        self.assertEqual(task["steps"], steps)
        self.assertEqual(task["agent_state"]["normal"]["state"], state)
        self.assertEqual(task["error"], "original error")
        tables = inspect(self.engine).get_table_names()
        self.assertNotIn("agent_runs", tables)
        self.assertIn("agent_runs_legacy", tables)
        self.assertIn("triage_tasks_legacy", tables)
        queue.recover_running()
        resumed = queue.claim()
        self.assertEqual(resumed.checkpoint().state, state)
        with self.engine.connect() as connection:
            archived = Table("agent_runs_legacy", MetaData(), autoload_with=connection)
            self.assertEqual(len(connection.execute(select(archived)).all()), 2)


if __name__ == "__main__":
    unittest.main()
