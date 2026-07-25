from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openai import APITimeoutError

from app.services.openai_reasoner import OpenAIReasoner


class OpenAIReasonerTimeoutTests(unittest.TestCase):
    def test_generate_returns_fallback_when_request_times_out(self):
        reasoner = OpenAIReasoner.__new__(OpenAIReasoner)
        reasoner.client = Mock()
        reasoner.client.responses.create.side_effect = APITimeoutError(request=SimpleNamespace())

        evidence = SimpleNamespace(
            workflow_name="ci",
            branch="main",
            sha="abc123",
            failed_jobs=["test"],
            last_green_run_id=None,
            diff="",
            logs=[],
        )

        result = reasoner.generate(evidence)

        self.assertEqual(result["failure_class"], "unknown")
        self.assertTrue(result["abstained"])
        self.assertIn("OpenAI", result["summary"])


if __name__ == "__main__":
    unittest.main()
