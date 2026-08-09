from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.rca_pipeline import RCAPipeline


class _StubReasoner:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def generate(self, evidence, rag_context=None, strict_grounding=False):
        self.calls.append({"strict_grounding": strict_grounding})
        return self.outputs.pop(0)


class _StubClassifier:
    def normalize(self, rca):
        normalized = dict(rca)
        normalized["normalized"] = True
        return normalized


class _StubVerifier:
    def __init__(self, decisions):
        self.decisions = list(decisions)

    def verify(self, rca, evidence):
        return self.decisions.pop(0)


class RCAPipelineUnitTests(unittest.TestCase):
    def test_returns_first_pass_when_grounded(self):
        reasoner = _StubReasoner(
            [
                {
                    "failure_class": "test",
                    "summary": "ok",
                    "claims": [],
                    "suggested_next_step": "n",
                    "confidence": 0.9,
                    "abstained": False,
                }
            ]
        )
        classifier = _StubClassifier()
        verifier = _StubVerifier([True])

        pipeline = RCAPipeline(reasoner=reasoner, classifier=classifier, verifier=verifier)
        result = pipeline.run(evidence=object())

        self.assertFalse(result["abstained"])
        self.assertTrue(result["normalized"])
        self.assertEqual(reasoner.calls, [{"strict_grounding": False}])

    def test_abstain_includes_reasoning_trace_when_grounding_fails_twice(self):
        reasoner = _StubReasoner(
            [
                {
                    "failure_class": "test",
                    "summary": "first",
                    "claims": [{"statement": "x", "citations": []}],
                    "suggested_next_step": "n",
                    "confidence": 0.8,
                    "abstained": False,
                    "reasoning_trace": [{"event": "first_attempt"}],
                },
                {
                    "failure_class": "test",
                    "summary": "second",
                    "claims": [{"statement": "y", "citations": []}],
                    "suggested_next_step": "n",
                    "confidence": 0.7,
                    "abstained": False,
                    "reasoning_trace": [{"event": "second_attempt"}],
                },
            ]
        )
        classifier = _StubClassifier()
        verifier = _StubVerifier([False, False])

        pipeline = RCAPipeline(reasoner=reasoner, classifier=classifier, verifier=verifier)
        result = pipeline.run(evidence=object())

        self.assertTrue(result["abstained"])
        self.assertEqual(result["failure_class"], "unknown")
        self.assertIn("reasoning_trace", result)
        trace = result["reasoning_trace"]
        self.assertTrue(any(item.get("event") == "first_attempt" for item in trace))
        self.assertTrue(any(item.get("event") == "second_attempt" for item in trace))
        self.assertTrue(
            any(item.get("event") == "pipeline_grounding_failed" and item.get("pass") == 2 for item in trace)
        )
        self.assertEqual(
            reasoner.calls,
            [
                {"strict_grounding": False},
                {"strict_grounding": True},
            ],
        )


if __name__ == "__main__":
    unittest.main()
