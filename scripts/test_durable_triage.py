from pathlib import Path
import os
import sys
import tempfile
import unittest
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("DATABASE_URL", "sqlite://")

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from app.models.triage_job import TriageJob
from app.models.triage_task import TriageTask
from app.models.incident import Incident
from app.models.evidence import Evidence, RawLog
from app.services.job_queue import TaskQueue
from app.services.incident_store import IncidentStore
from app.services.triage_service import TriageService
from app.services.triage_worker import process_task
from app.services.recovery_errors import ManualReviewRequired
from app.models.rag_chunk import RagChunk
from app.services.rag_store import RagStore


class Crash(BaseException):
    pass


class DurableTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.url = "sqlite:///" + str(Path(self.temp.name) / "queue.db")
        self.engine = create_engine(self.url)
        TriageTask.__table__.create(self.engine)
        Incident.__table__.create(self.engine)
        self.factory = sessionmaker(self.engine, expire_on_commit=False)
        self.queue = TaskQueue(self.factory)
        self.job = TriageJob(42, "CI", "main", "abc")
        self.evidence = Evidence(42, "CI", "main", "abc", ["test"],
            [RawLog(1, "test", "error", ["error"])], "", None, 10)
        self.service = TriageService(
            evidence_collector=Mock(), rca_pipeline=Mock(), confidence_gate=Mock(),
            github_client=Mock(), report_formatter=Mock(), github_delivery=Mock(),
            slack_client=Mock(), incident_store=IncidentStore(self.factory))
        self.service.evidence_collector.collect.return_value = self.evidence
        self.service.rca_pipeline.run.return_value = {
            "failure_class": "test", "summary": "failed", "confidence": .9,
            "claims": [], "suggested_next_step": "fix", "abstained": False}
        self.service.confidence_gate.evaluate.return_value = SimpleNamespace(decision="deliver", reason="ok")
        self.service.report_formatter.to_markdown.return_value = "report"
        self.service.report_formatter.to_slack.return_value = "slack"
        self.service.github_delivery.deliver.return_value = {"html_url": "https://example.test/comment"}
        self.service.slack_client.send_message.return_value = {"ok": True, "ts": "123"}

    def tearDown(self):
        self.engine.dispose()
        self.temp.cleanup()

    def count_incidents(self):
        with self.factory() as session:
            return session.scalar(select(func.count()).select_from(Incident))

    def test_duplicate_webhook_and_attempt_identity(self):
        key = self.queue.enqueue(self.job)
        self.assertEqual(self.queue.enqueue(self.job), key)
        self.assertNotEqual(self.queue.enqueue(TriageJob(42, "CI", "main", "abc", 2)), key)

    def test_restart_recovers_running_and_preserves_step_results(self):
        key = self.queue.enqueue(self.job)
        task = self.queue.claim()
        self.assertIsNone(self.queue.claim())
        task.step("evidence", lambda: {"logs": "snapshot"})
        self.engine.dispose()
        self.engine = create_engine(self.url)
        self.factory = sessionmaker(self.engine, expire_on_commit=False)
        self.queue = TaskQueue(self.factory)
        self.queue.recover_running()
        resumed = self.queue.claim()
        self.assertEqual(resumed.key, key)
        operation = Mock()
        self.assertEqual(resumed.step("evidence", operation), {"logs": "snapshot"})
        operation.assert_not_called()

    def test_manual_entry_point_reuses_completed_task(self):
        self.service.task_queue = self.queue
        self.service.github_client.get_workflow_run.return_value = {
            "name": "CI", "head_branch": "main", "head_sha": "abc", "run_attempt": 1}
        first = self.service.triage(42)
        second = self.service.triage(42)
        self.assertEqual(asdict(first), asdict(second))
        self.assertEqual(self.count_incidents(), 1)
        self.service.github_delivery.deliver.assert_called_once()
        self.service.slack_client.send_message.assert_called_once()

    def test_incident_rag_index_checkpoint_is_atomic_and_reused(self):
        RagChunk.__table__.create(self.engine)
        store = RagStore(self.factory)
        store._build_embedding = Mock(side_effect=RuntimeError("offline"))
        store.retrieve_for_evidence = Mock(return_value=[])
        self.service.rag_store = store
        key = self.queue.enqueue(self.job)
        task = self.queue.claim()
        original = task.step
        def step(name, operation, **options):
            if name == "github_comment":
                raise Crash()
            return original(name, operation, **options)
        task.step = step
        with self.assertRaises(Crash):
            process_task(self.service, task)
        self.queue.recover_running()
        process_task(self.service, self.queue.claim())
        with self.factory() as session:
            self.assertEqual(session.scalar(select(func.count()).select_from(RagChunk)), 1)
        store._build_embedding.assert_called_once()
        self.assertEqual(self.queue.inspect(key)["status"], "completed")

    def test_read_error_is_saved_and_requires_explicit_retry(self):
        key = self.queue.enqueue(self.job)
        self.queue.claim().fail(RuntimeError("temporary failure"))
        self.assertEqual(self.queue.inspect(key)["status"], "failed")
        self.queue.recover_running()
        self.assertIsNone(self.queue.claim())
        self.queue.retry(key)
        self.assertIsNotNone(self.queue.claim())

    def test_incident_transaction_rolls_back_with_checkpoint(self):
        self.queue.enqueue(self.job)
        task = self.queue.claim()
        def operation(session):
            self.service.incident_store.create_in_transaction(session,
                evidence=self.evidence, rca={"summary": "failure"})
            raise RuntimeError("transaction interrupted")
        with self.assertRaises(RuntimeError):
            task.step("incident", operation, transactional=True)
        self.assertEqual(self.count_incidents(), 0)

    def test_restart_after_github_success_skips_comment_and_incident(self):
        key = self.queue.enqueue(self.job)
        # Crash before Slack starts: GitHub result has already been committed.
        execution = self.queue.claim()
        original = execution.step
        def step(name, operation, **options):
            if name == "slack_message":
                raise Crash()
            return original(name, operation, **options)
        execution.step = step
        with self.assertRaises(Crash):
            process_task(self.service, execution)
        self.queue.recover_running()
        resumed = self.queue.claim()
        result = process_task(self.service, resumed)
        self.assertEqual(result.action, "delivered")
        self.assertEqual(self.count_incidents(), 1)
        self.service.evidence_collector.collect.assert_called_once()
        self.service.rca_pipeline.run.assert_called_once()
        self.service.github_delivery.deliver.assert_called_once()
        self.service.slack_client.send_message.assert_called_once()
        self.assertEqual(self.queue.inspect(key)["status"], "completed")
        self.assertIsNone(self.queue.claim())

    def test_unknown_slack_outcome_requires_review_then_operator_resolution(self):
        key = self.queue.enqueue(self.job)
        self.service.slack_client.send_message.side_effect = Crash()
        with self.assertRaises(Crash):
            process_task(self.service, self.queue.claim())
        self.queue.recover_running()
        self.assertIsNone(self.queue.claim())
        self.assertEqual(self.queue.inspect(key)["status"], "needs_review")
        self.service.slack_client.send_message.assert_called_once()
        with self.assertRaises(ManualReviewRequired):
            self.queue.retry(key)
        self.queue.resolve(key, "slack_message", {"ok": True, "ts": "123"})
        self.queue.retry(key)
        process_task(self.service, self.queue.claim())
        self.service.slack_client.send_message.assert_called_once()
        self.service.github_delivery.deliver.assert_called_once()
        self.assertEqual(self.count_incidents(), 1)

    def test_exception_during_post_is_not_automatically_retried(self):
        key = self.queue.enqueue(self.job)
        self.service.github_delivery.deliver.side_effect = TimeoutError("unknown outcome")
        with self.assertRaises(TimeoutError):
            process_task(self.service, self.queue.claim())
        self.assertEqual(self.queue.inspect(key)["status"], "needs_review")
        self.assertIsNone(self.queue.claim())

    def test_rerun_wait_retry_does_not_trigger_another_rerun(self):
        from app.services.github_client import GitHubRerunTimeoutError
        key = self.queue.enqueue(self.job)
        self.service.confidence_gate.evaluate.return_value = SimpleNamespace(decision="rerun", reason="flaky")
        self.service.github_client.get_workflow_run.return_value = {"run_attempt": 1}
        self.service.github_client.rerun_failed_jobs.return_value = True
        self.service.github_client.wait_for_new_attempt_completion.side_effect = [
            GitHubRerunTimeoutError("not done"), {"conclusion": "success"}]
        self.service.rca_pipeline.verifier.verify.return_value = True
        with self.assertRaises(GitHubRerunTimeoutError):
            process_task(self.service, self.queue.claim())
        self.assertEqual(self.queue.inspect(key)["status"], "failed")
        self.queue.retry(key)
        process_task(self.service, self.queue.claim())
        self.service.github_client.rerun_failed_jobs.assert_called_once()
        self.assertEqual(self.count_incidents(), 1)

    def test_stale_ci_attempt_is_not_analyzed_as_original(self):
        key = self.queue.enqueue(self.job)
        self.evidence.run_attempt = 2
        with self.assertRaises(ManualReviewRequired):
            process_task(self.service, self.queue.claim())
        self.assertEqual(self.queue.inspect(key)["status"], "needs_review")
        self.service.rca_pipeline.run.assert_not_called()

    def test_real_pipeline_resumes_model_state_from_same_task_row(self):
        from app.services.openai_reasoner import OpenAIReasoner
        from app.services.rca_pipeline import RCAPipeline
        reasoner = OpenAIReasoner.__new__(OpenAIReasoner)
        reasoner.client, reasoner.tools = Mock(), Mock()
        reasoner.controlled_agent_mode, reasoner.max_reasoning_turns = True, 4
        reasoner._request_action = Mock(side_effect=[({"action": "tool_call",
            "tool": "get_failed_logs", "args": {}, "call_id": "logs"}, []), Crash()])
        classifier = Mock()
        classifier.normalize.side_effect = lambda result: result
        verifier = Mock()
        verifier.verify.return_value = True
        final = self.service.rca_pipeline.run.return_value
        self.service.rca_pipeline = RCAPipeline(reasoner, classifier, verifier)
        key = self.queue.enqueue(self.job)
        with self.assertRaises(Crash):
            process_task(self.service, self.queue.claim())
        self.assertTrue(self.queue.inspect(key)["agent_state"]["normal"]["state"]["has_log_evidence"])
        self.queue.recover_running()
        reasoner.tools.search_playbook.return_value = []
        reasoner._request_action = Mock(side_effect=[({"action": "tool_call",
            "tool": "search_playbook", "args": {}, "call_id": "rag"}, []), (final, [])])
        result = process_task(self.service, self.queue.claim())
        self.assertEqual(result.action, "delivered")
        self.assertEqual(reasoner._request_action.call_count, 2)
        self.service.evidence_collector.collect.assert_called_once()
        self.assertEqual(self.count_incidents(), 1)


if __name__ == "__main__":
    unittest.main()
