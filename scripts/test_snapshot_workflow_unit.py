import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

os.environ.setdefault("DATABASE_URL", "sqlite://")

from app.models.evidence import Evidence, RawLog
from app.services.openai_reasoner import OpenAIReasoner
from app.services.grounding_verifier import GroundingVerifier
from app.services.rca_pipeline import RCAPipeline


class SnapshotWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.evidence = Evidence(1, "ci", "main", "sha", ["test"],
            [RawLog(7, "test", "header\nAssertionError\nend", ["header", "AssertionError", "end"])],
            "sanitized diff", None, None)
        self.reasoner = OpenAIReasoner.__new__(OpenAIReasoner)
        self.reasoner.tools = Mock()

    def test_window_uses_original_line_numbers_and_never_refetches(self):
        result = self.reasoner._execute_tool_call(self.evidence, {"tool": "get_failed_logs",
            "args": {"job_id": 7, "line_start": 2, "line_end": 2}})
        self.assertEqual(result["result"][0]["numbered_text"], "2 | AssertionError")
        self.reasoner.tools.get_failed_logs.assert_not_called()
        result = self.reasoner._execute_tool_call(self.evidence, {"tool": "get_commit_diff",
            "args": {"sha": "another-commit"}})
        self.assertFalse(result["ok"])
        self.reasoner.tools.get_commit_diff.assert_not_called()

    def test_missing_citations_and_invalid_lines_are_rejected(self):
        verifier = GroundingVerifier()
        self.assertFalse(verifier.verify({"claims": [], "abstained": False}, self.evidence))
        self.assertFalse(verifier.verify({"claims": [{"citations": []}]}, self.evidence))
        errors = verifier.validation_errors({"claims": [{"citations": [
            {"line_start": 20, "line_end": 20, "quote": "AssertionError"}]}]}, self.evidence)
        self.assertIn("20..20", errors[0])
        self.assertTrue(verifier.verify({"claims": [], "abstained": True}, self.evidence))

    def test_repair_gets_previous_result_and_specific_validation_error(self):
        first = {"failure_class": "test", "claims": [], "abstained": False}
        second = {"failure_class": "test", "abstained": False, "claims": [
            {"statement": "test failed", "citations": [
                {"line_start": 2, "line_end": 2, "quote": "AssertionError"}]}]}
        reasoner = Mock()
        reasoner.generate.side_effect = [first, second]
        classifier = SimpleNamespace(normalize=lambda result: result)
        result = RCAPipeline(reasoner, classifier, GroundingVerifier()).run(self.evidence)
        self.assertEqual(result, second)
        repair = reasoner.generate.call_args.kwargs["repair_context"]
        self.assertEqual(repair["previous_rca"], first)
        self.assertIn("at least one claim", repair["validation_errors"][0])


if __name__ == "__main__":
    unittest.main()
