from pathlib import Path
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("DATABASE_URL", "sqlite://")

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.triage_task import TriageTask
from app.models.triage_job import TriageJob
from app.models.evidence import Evidence, RawLog
from app.services.job_queue import TaskQueue
from app.services.openai_reasoner import OpenAIReasoner
from app.services.recovery_errors import ManualReviewRequired
from openai.types.responses import ResponseFunctionToolCall


class ProcessStopped(BaseException):
    pass


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.url = "sqlite:///" + str(Path(self.temp.name) / "state.db")
        self.engine = create_engine(self.url)
        TriageTask.__table__.create(self.engine)
        self.queue = TaskQueue(sessionmaker(self.engine, expire_on_commit=False))
        self.key = self.queue.enqueue(TriageJob(42, "CI", "main", "abc"))
        self.task = self.queue.claim()
        self.evidence = Evidence(42, "CI", "main", "abc", ["tests"],
            [RawLog(1, "tests", "error", ["error"])], "", None, None)

    def tearDown(self):
        self.engine.dispose()
        self.temp.cleanup()

    def reasoner(self):
        r = OpenAIReasoner.__new__(OpenAIReasoner)
        r.tools, r.client = Mock(), Mock()
        r.controlled_agent_mode, r.max_reasoning_turns = True, 4
        return r

    def generate(self, reasoner, strict=False):
        try:
            return reasoner.generate(self.evidence, strict_grounding=strict,
                                     checkpoint=self.task.checkpoint(strict))
        except Exception as exc:
            self.task.fail(exc)
            raise

    def action(self, name):
        return ({"action": "tool_call", "tool": name, "args": {}, "call_id": name},
            [ResponseFunctionToolCall(type="function_call", name=name,
                                     arguments="{}", call_id=name)])

    def test_restart_skips_completed_logs_and_reuses_result(self):
        r = self.reasoner()
        r._request_action = Mock(side_effect=[self.action("get_failed_logs"), ProcessStopped()])
        with self.assertRaises(ProcessStopped):
            self.generate(r)
        self.engine.dispose()
        self.engine = create_engine(self.url)
        self.queue = TaskQueue(sessionmaker(self.engine, expire_on_commit=False))
        self.queue.recover_running()
        self.task = self.queue.claim()
        resumed = self.reasoner()
        resumed.tools.search_playbook.return_value = []
        resumed._request_action = Mock(side_effect=[self.action("search_playbook"),
            ({"failure_class": "test"}, [])])
        result = self.generate(resumed)
        self.assertEqual(result["failure_class"], "test")
        self.assertEqual(self.generate(resumed), result)
        resumed.tools.get_failed_logs.assert_not_called()
        self.assertEqual(resumed._request_action.call_count, 2)
        entry = self.queue.inspect(self.key)["agent_state"]["normal"]
        self.assertEqual(entry["state"]["turn"], 2)
        self.assertEqual(entry["result"], result)

    def test_pending_native_call_is_restored_after_read_tool_interrupt(self):
        r = self.reasoner()
        r._request_action = Mock(side_effect=[self.action("get_failed_logs"), self.action("search_playbook")])
        r.tools.search_playbook.side_effect = ProcessStopped()
        with self.assertRaises(ProcessStopped):
            self.generate(r)
        resumed = self.reasoner()
        resumed.tools.search_playbook.return_value = []
        resumed.client.responses.create.return_value = SimpleNamespace(
            output=[], output_text='{"failure_class": "test"}')
        self.assertEqual(self.generate(resumed)["failure_class"], "test")
        replayed = resumed.client.responses.create.call_args.kwargs["input"]
        self.assertEqual(replayed[1]["type"], "function_call")
        self.assertEqual(replayed[1]["call_id"], replayed[2]["call_id"])
        resumed.tools.search_playbook.assert_called_once()

    def test_uncertain_agent_rerun_is_reviewed_and_resolved_in_task(self):
        r = self.reasoner()
        r._request_action = Mock(return_value=self.action("rerun_job"))
        r.tools.rerun_job.side_effect = ProcessStopped()
        with self.assertRaises(ProcessStopped):
            self.generate(r)
        self.queue.recover_running()
        self.assertEqual(self.queue.inspect(self.key)["status"], "needs_review")
        with self.assertRaises(ManualReviewRequired):
            self.queue.retry(self.key)
        self.queue.resolve(self.key, "agent_rerun", True)
        self.queue.retry(self.key)
        self.task = self.queue.claim()
        resumed = self.reasoner()
        resumed.tools.search_playbook.return_value = []
        resumed._request_action = Mock(side_effect=[self.action("get_failed_logs"),
            self.action("search_playbook"), ({"failure_class": "test"}, [])])
        self.assertEqual(self.generate(resumed)["failure_class"], "test")
        resumed.tools.rerun_job.assert_not_called()

    def test_model_error_preserves_state_for_explicit_retry(self):
        r = self.reasoner()
        r._request_action = Mock(side_effect=[self.action("get_failed_logs"), RuntimeError("connection lost")])
        with self.assertRaisesRegex(RuntimeError, "connection lost"):
            self.generate(r)
        self.assertEqual(self.queue.inspect(self.key)["status"], "failed")
        self.queue.retry(self.key)
        self.task = self.queue.claim()
        resumed = self.reasoner()
        resumed.tools.search_playbook.return_value = []
        resumed._request_action = Mock(side_effect=[self.action("search_playbook"),
            ({"failure_class": "test"}, [])])
        self.assertEqual(self.generate(resumed)["failure_class"], "test")
        resumed.tools.get_failed_logs.assert_not_called()

    def test_grounding_passes_have_separate_state_in_same_row(self):
        self.task.checkpoint().complete({"failure_class": "test"})
        self.assertIsNone(self.task.checkpoint(strict=True).result)
        self.task.checkpoint(strict=True).save({"turn": 1})
        state = self.queue.inspect(self.key)["agent_state"]
        self.assertEqual(state["normal"]["result"]["failure_class"], "test")
        self.assertEqual(state["strict"]["state"]["turn"], 1)


if __name__ == "__main__":
    unittest.main()
