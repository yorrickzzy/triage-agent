from pathlib import Path
import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("DATABASE_URL", "sqlite://")

from app.routes import github_webhook as webhook


class WebhookQueueTests(unittest.TestCase):
    def test_both_webhook_paths_preserve_attempt_and_return_durable_task_id(self):
        for traced in (False, True):
            with self.subTest(traced=traced):
                request = Mock()
                request.headers = {"X-GitHub-Event": "workflow_run"}
                request.body = AsyncMock(return_value=b"payload")
                request.json = AsyncMock(return_value={"workflow_run": {
                    "id": 42, "conclusion": "failure", "name": "CI",
                    "head_branch": "main", "head_sha": "abc", "run_attempt": 3}})
                tracer = Mock() if traced else None
                if traced:
                    tracer.start_as_current_span.return_value.__enter__ = Mock(return_value=Mock())
                    tracer.start_as_current_span.return_value.__exit__ = Mock(return_value=False)
                with patch.object(webhook, "tracer", tracer), \
                     patch.object(webhook, "verify_github_signature", return_value=True), \
                     patch.object(webhook, "enqueue_triage_job", return_value="durable-id") as enqueue:
                    result = asyncio.run(webhook.github_webhook(request))
                self.assertEqual(result["task_id"], "durable-id")
                self.assertEqual(enqueue.call_args.args[0].run_attempt, 3)


if __name__ == "__main__":
    unittest.main()
