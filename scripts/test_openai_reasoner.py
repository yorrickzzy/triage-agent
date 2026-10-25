from pathlib import Path
import sys
import json
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openai import APITimeoutError
from openai.types.responses import ResponseFunctionToolCall, ResponseReasoningItem

from app.services.openai_reasoner import OpenAIReasoner
from app.models.evidence import RawLog


def function_response(name, arguments, call_id="call_test"):
    return SimpleNamespace(
        output_text="",
        output=[
            ResponseReasoningItem(id="rs_test", summary=[], type="reasoning"),
            ResponseFunctionToolCall(
                type="function_call", name=name, call_id=call_id,
                arguments=json.dumps(arguments),
            ),
        ],
    )


@dataclass
class FakeRunRef:
    run_id: int
    workflow_id: int | str
    workflow_name: str
    branch: str
    sha: str


class OpenAIReasonerTimeoutTests(unittest.TestCase):
    def test_native_function_schemas_preserve_nullable_defaults(self):
        reasoner = OpenAIReasoner.__new__(OpenAIReasoner)
        reasoner.tools = Mock()
        schemas = reasoner._function_tools()
        self.assertEqual({tool["name"] for tool in schemas}, {
            "get_failed_logs", "get_commit_diff", "get_last_green_run",
            "search_playbook",
        })
        for tool in schemas:
            self.assertTrue(tool["strict"])
            parameters = tool["parameters"]
            self.assertFalse(parameters["additionalProperties"])
            self.assertEqual(set(parameters["required"]), set(parameters["properties"]))
            for name, schema in parameters["properties"].items():
                if name != "reason":
                    self.assertIn("null", schema["type"])

    def test_rejects_invalid_native_calls_before_executing_tools(self):
        reasoner = OpenAIReasoner.__new__(OpenAIReasoner)
        reasoner.client = Mock()
        reasoner.tools = Mock()
        multiple = function_response("rerun_job", {"run_id": 123})
        multiple.output.append(multiple.output[-1])
        malformed = function_response("rerun_job", {})
        malformed.output[-1].arguments = "{invalid"
        for response in [
            multiple, malformed,
            function_response("rerun_job", []),
            SimpleNamespace(output=[], output_text=json.dumps({
                "action": "tool_call", "tool": "rerun_job", "args": {},
            })),
        ]:
            with self.subTest(response=response):
                reasoner.client.responses.create.return_value = response
                with self.assertRaises(ValueError):
                    reasoner._request_action("test", [])
        reasoner.tools.rerun_job.assert_not_called()

    def test_function_output_keeps_existing_observation_budget(self):
        reasoner = OpenAIReasoner.__new__(OpenAIReasoner)
        reasoner.OBSERVATIONS_BUDGET_CHARS = 80
        output = reasoner._tool_output({"call_id": "call_1"}, {"result": "x" * 200})
        self.assertEqual(output["call_id"], "call_1")
        self.assertIn("[truncated]", output["output"])
        self.assertLess(len(output["output"]), 110)

    def test_optional_tool_failure_can_still_produce_final(self):
        reasoner = OpenAIReasoner.__new__(OpenAIReasoner)
        reasoner.client, reasoner.tools = Mock(), Mock()
        reasoner.controlled_agent_mode, reasoner.max_reasoning_turns = True, 1
        reasoner.tools.search_playbook.side_effect = RuntimeError("unavailable")
        final = {"failure_class": "test", "summary": "snapshot sufficient", "claims": [],
                 "suggested_next_step": "fix", "confidence": .9, "abstained": False}
        reasoner.client.responses.create.side_effect = [
            function_response("search_playbook", {"query": "AssertionError"}),
            SimpleNamespace(output=[], output_text=json.dumps(final))]
        evidence = SimpleNamespace(run_id=123, workflow_name="ci", branch="main", sha="abc",
            failed_jobs=["tests"], last_green_run_id=None, diff="", logs=[])
        result = reasoner.generate(evidence)
        self.assertFalse(result["abstained"])
        self.assertFalse(result["reasoning_trace"][0]["tool_result"]["ok"])
        self.assertEqual(reasoner.client.responses.create.call_args.kwargs["tool_choice"], "none")

    def test_generate_returns_fallback_when_request_times_out(self):
        reasoner = OpenAIReasoner.__new__(OpenAIReasoner)
        reasoner.client = Mock()
        reasoner.tools = None
        reasoner.controlled_agent_mode = False
        reasoner.max_reasoning_turns = 4
        reasoner.client.responses.create.side_effect = APITimeoutError(request=SimpleNamespace())

        evidence = SimpleNamespace(
            workflow_name="ci",
            branch="main",
            sha="abc123",
            failed_jobs=["test"],
            last_green_run_id=None,
            diff="",
            logs=[
                RawLog(
                    job_id=1,
                    job_name="test",
                    text="error line",
                    lines=["error line"],
                )
            ],
        )

        result = reasoner.generate(evidence)

        self.assertEqual(result["failure_class"], "unknown")
        self.assertTrue(result["abstained"])
        self.assertIn("OpenAI", result["summary"])

    def test_generate_fetches_tool_context_when_available(self):
        reasoner = OpenAIReasoner.__new__(OpenAIReasoner)
        reasoner.client = Mock()
        reasoner.tools = Mock()
        reasoner.controlled_agent_mode = False
        reasoner.max_reasoning_turns = 3
        reasoner.tools.search_playbook.return_value = [
            {
                "source_type": "incident",
                "source_id": 101,
                "workflow_name": "ci",
                "failure_class": "test",
                "quality_score": 1.0,
                "text": "historical pytest failure",
            }
        ]

        reasoner.client.responses.create.return_value = SimpleNamespace(
            output=[], output_text=(
                '{"failure_class":"test","summary":"s","claims":[],'
                '"suggested_next_step":"n","confidence":0.9,"abstained":false}'
            )
        )

        evidence = SimpleNamespace(
            workflow_name="ci",
            branch="main",
            sha="abc123",
            failed_jobs=["unit-tests"],
            last_green_run_id=12,
            diff="",
            logs=[
                RawLog(
                    job_id=2,
                    job_name="unit-tests",
                    text="AssertionError",
                    lines=["AssertionError"],
                )
            ],
        )

        result = reasoner.generate(evidence)

        self.assertEqual(result["failure_class"], "test")
        reasoner.tools.search_playbook.assert_called_once_with("ci unit-tests", limit=5)

        _, kwargs = reasoner.client.responses.create.call_args
        self.assertIn("historical pytest failure", kwargs["input"])

    def test_generate_completes_multi_turn_tool_loop(self):
        reasoner = OpenAIReasoner.__new__(OpenAIReasoner)
        reasoner.client = Mock()
        reasoner.tools = Mock()
        reasoner.controlled_agent_mode = False
        reasoner.max_reasoning_turns = 3

        # 1) initial eager context pull
        # 2) model asks for tool call
        # 3) reasoner executes tool and model returns final RCA
        reasoner.tools.search_playbook.side_effect = [
            [],
            [
                {
                    "source_type": "playbook",
                    "source_id": 7,
                    "workflow_name": "ci",
                    "failure_class": "test",
                    "quality_score": 0.9,
                    "text": "pytest failure usually maps to assertion mismatch",
                }
            ],
        ]

        reasoner.client.responses.create.side_effect = [
            function_response('search_playbook', {'query': 'pytest assertion', 'limit': 3, 'reason': 'Need similar failures'}),
            SimpleNamespace(
                output=[], output_text=(
                    '{"action":"final","rca":'
                    '{"failure_class":"test","summary":"s","claims":[],'
                    '"suggested_next_step":"n","confidence":0.8,"abstained":false}}'
                )
            ),
        ]

        evidence = SimpleNamespace(
            workflow_name="ci",
            branch="main",
            sha="abc123",
            failed_jobs=["unit-tests"],
            last_green_run_id=12,
            diff="",
            logs=[
                RawLog(
                    job_id=2,
                    job_name="unit-tests",
                    text="AssertionError",
                    lines=["AssertionError"],
                )
            ],
        )

        result = reasoner.generate(evidence)

        self.assertEqual(result["failure_class"], "test")
        self.assertEqual(reasoner.client.responses.create.call_count, 2)
        self.assertEqual(reasoner.tools.search_playbook.call_count, 2)

        _, kwargs = reasoner.client.responses.create.call_args
        self.assertIn("pytest failure usually maps to assertion mismatch", kwargs["input"][0]["content"])
        self.assertFalse(kwargs["parallel_tool_calls"])
        self.assertEqual(kwargs["input"][1].type, "reasoning")
        call = kwargs["input"][2]
        output = kwargs["input"][3]
        self.assertEqual(output["type"], "function_call_output")
        self.assertEqual(output["call_id"], call.call_id)
        self.assertEqual(json.loads(output["output"])["result"]["query"], "pytest assertion")

    def test_snapshot_can_produce_final_without_mandatory_tools(self):
        reasoner = OpenAIReasoner.__new__(OpenAIReasoner)
        reasoner.client, reasoner.tools = Mock(), Mock()
        reasoner.controlled_agent_mode, reasoner.max_reasoning_turns = True, 4
        final = {"failure_class": "test", "summary": "grounded", "claims": [],
                 "suggested_next_step": "fix", "confidence": .9, "abstained": False}
        reasoner.client.responses.create.return_value = SimpleNamespace(output=[], output_text=json.dumps(final))
        evidence = SimpleNamespace(run_id=123, workflow_name="ci", branch="main", sha="abc",
            failed_jobs=["tests"], last_green_run_id=None, diff="sanitized diff",
            logs=[RawLog(1, "tests", "AssertionError [REDACTED]", ["AssertionError [REDACTED]"])])
        self.assertEqual(reasoner.generate(evidence)["summary"], "grounded")
        reasoner.tools.get_failed_logs.assert_not_called()
        reasoner.tools.search_playbook.assert_not_called()
        prompt = reasoner.client.responses.create.call_args.kwargs["input"]
        self.assertIn("1 | AssertionError [REDACTED]", prompt)
        self.assertIn("sanitized diff", prompt)

    def test_execute_tool_call_routes_all_documented_tools(self):
        reasoner = OpenAIReasoner.__new__(OpenAIReasoner)
        reasoner.client = Mock()
        reasoner.tools = Mock()
        reasoner.controlled_agent_mode = False
        reasoner.max_reasoning_turns = 3

        reasoner.tools.get_failed_logs.return_value = [
            RawLog(
                job_id=10,
                job_name="lint",
                text="line1\nline2",
                lines=["line1", "line2"],
            )
        ]
        reasoner.tools.get_commit_diff.return_value = "diff --git a b"
        reasoner.tools.get_last_green_run.return_value = FakeRunRef(
            run_id=99,
            workflow_id="wf-1",
            workflow_name="ci",
            branch="main",
            sha="abc",
        )
        reasoner.tools.rerun_job.return_value = True
        reasoner.tools.search_playbook.return_value = [
            {"source_type": "playbook", "source_id": 1, "text": "hint"}
        ]

        evidence = SimpleNamespace(
            run_id=123,
            workflow_name="ci",
            branch="main",
            sha="abc123",
            failed_jobs=["lint"],
            last_green_run_id=12,
            diff="",
            logs=[RawLog(10, "lint", "line1\nline2", ["line1", "line2"])],
        )

        failed_logs_result = reasoner._execute_tool_call(
            evidence,
            {"tool": "get_failed_logs", "args": {}},
        )
        self.assertTrue(failed_logs_result["ok"])
        self.assertEqual(failed_logs_result["tool"], "get_failed_logs")
        self.assertEqual(failed_logs_result["result"][0]["job_id"], 10)
        reasoner.tools.get_failed_logs.assert_not_called()

        diff_result = reasoner._execute_tool_call(
            evidence,
            {"tool": "get_commit_diff", "args": {}},
        )
        self.assertTrue(diff_result["ok"])
        self.assertEqual(diff_result["result"]["sha"], "abc123")
        reasoner.tools.get_commit_diff.assert_not_called()
        self.assertEqual(diff_result["result"]["diff"], evidence.diff)

        green_result = reasoner._execute_tool_call(
            evidence,
            {"tool": "get_last_green_run", "args": {}},
        )
        self.assertTrue(green_result["ok"])
        self.assertEqual(green_result["result"]["run_id"], 99)
        reasoner.tools.get_last_green_run.assert_called_once_with(
            workflow_id="ci",
            branch="main",
            current_run_id=123,
        )

        rerun_result = reasoner._execute_tool_call(
            evidence,
            {"tool": "rerun_job", "args": {}},
        )
        self.assertFalse(rerun_result["ok"])
        reasoner.tools.rerun_job.assert_not_called()

        playbook_result = reasoner._execute_tool_call(
            evidence,
            {"tool": "search_playbook", "args": {"query": "pytest", "limit": 3}},
        )
        self.assertTrue(playbook_result["ok"])
        self.assertEqual(playbook_result["result"]["query"], "pytest")
        self.assertEqual(playbook_result["result"]["limit"], 3)
        reasoner.tools.search_playbook.assert_called_with("pytest", limit=3)

    def test_generate_marks_truncation_status_when_context_is_clipped(self):
        reasoner = OpenAIReasoner.__new__(OpenAIReasoner)
        reasoner.client = Mock()
        reasoner.tools = None
        reasoner.controlled_agent_mode = False
        reasoner.max_reasoning_turns = 1
        reasoner.LOGS_BUDGET_CHARS = 80
        reasoner.DIFF_BUDGET_CHARS = 60
        reasoner.RAG_BUDGET_CHARS = 120
        reasoner.OBSERVATIONS_BUDGET_CHARS = 120

        reasoner.client.responses.create.return_value = SimpleNamespace(
            output=[], output_text=(
                '{"failure_class":"test","summary":"s","claims":[],'
                '"suggested_next_step":"n","confidence":0.9,"abstained":false}'
            )
        )

        long_lines = [f"line {i} with AssertionError details" for i in range(1, 30)]
        evidence = SimpleNamespace(
            run_id=10,
            workflow_name="ci",
            branch="main",
            sha="abc123",
            failed_jobs=["unit-tests"],
            last_green_run_id=12,
            diff="diff\n" + ("x" * 300),
            logs=[
                RawLog(
                    job_id=2,
                    job_name="unit-tests",
                    text="\n".join(long_lines),
                    lines=long_lines,
                )
            ],
        )

        result = reasoner.generate(evidence)

        self.assertEqual(result["failure_class"], "test")

        _, kwargs = reasoner.client.responses.create.call_args
        prompt = kwargs["input"]
        self.assertIn("logs_truncated: true", prompt)
        self.assertIn("diff_truncated: true", prompt)

    def test_final_only_turn_rejects_additional_tool_calls(self):
        reasoner = OpenAIReasoner.__new__(OpenAIReasoner)
        reasoner.client, reasoner.tools = Mock(), Mock()
        reasoner.controlled_agent_mode, reasoner.max_reasoning_turns = True, 1
        reasoner.tools.search_playbook.return_value = []
        reasoner.client.responses.create.return_value = function_response("search_playbook", {"query": "error"})
        evidence = SimpleNamespace(run_id=123, workflow_name="ci", branch="main", sha="abc",
            failed_jobs=[], last_green_run_id=None, diff="", logs=[])
        self.assertTrue(reasoner.generate(evidence)["abstained"])
        self.assertEqual(reasoner.client.responses.create.call_count, 2)
        reasoner.tools.search_playbook.assert_called_once()
        self.assertEqual(reasoner.client.responses.create.call_args.kwargs["tool_choice"], "none")

    def test_controlled_mode_allows_final_when_playbook_returns_empty(self):
        reasoner = OpenAIReasoner.__new__(OpenAIReasoner)
        reasoner.client = Mock()
        reasoner.tools = Mock()
        reasoner.controlled_agent_mode = True
        reasoner.max_reasoning_turns = 4
        reasoner.OBSERVATIONS_BUDGET_CHARS = 4000

        reasoner.tools.get_failed_logs.return_value = [
            RawLog(
                job_id=10,
                job_name="unit-tests",
                text="AssertionError: expected 30 got 45",
                lines=["AssertionError: expected 30 got 45"],
            )
        ]
        reasoner.tools.search_playbook.return_value = []

        reasoner.client.responses.create.side_effect = [
            function_response('get_failed_logs', {'reason': 'need logs'}),
            function_response('search_playbook', {'query': 'ci unit-tests', 'limit': 5, 'reason': 'check historical matches'}),
            SimpleNamespace(
                output=[], output_text=(
                    '{"action":"final","rca":'
                    '{"failure_class":"test","summary":"grounded without rag hit",'
                    '"claims":[],"suggested_next_step":"fix test",'
                    '"confidence":0.9,"abstained":false}}'
                )
            ),
        ]

        evidence = SimpleNamespace(
            run_id=123,
            workflow_name="ci",
            branch="main",
            sha="abc123",
            failed_jobs=["unit-tests"],
            last_green_run_id=12,
            diff="",
            logs=[],
        )

        result = reasoner.generate(evidence)

        self.assertFalse(result["abstained"])
        self.assertEqual(result["summary"], "grounded without rag hit")
        self.assertEqual(reasoner.tools.get_failed_logs.call_count, 0)
        self.assertEqual(reasoner.tools.search_playbook.call_count, 1)

    def test_controlled_mode_rejects_other_run_snapshot(self):
        reasoner = OpenAIReasoner.__new__(OpenAIReasoner)
        reasoner.client = Mock()
        reasoner.tools = Mock()
        reasoner.controlled_agent_mode = True
        reasoner.max_reasoning_turns = 4
        reasoner.OBSERVATIONS_BUDGET_CHARS = 4000

        reasoner.tools.get_failed_logs.return_value = [
            RawLog(
                job_id=20,
                job_name="unit-tests",
                text="AssertionError",
                lines=["AssertionError"],
            )
        ]
        reasoner.tools.search_playbook.return_value = []

        reasoner.client.responses.create.side_effect = [
            function_response('get_failed_logs', {'run_id': 29178292575, 'reason': 'need logs'}),
            function_response('search_playbook', {'query': 'ci unit-tests', 'limit': 5, 'reason': 'check historical matches'}),
            SimpleNamespace(
                output=[], output_text=(
                    '{"action":"final","rca":'
                    '{"failure_class":"test","summary":"s",'
                    '"claims":[],"suggested_next_step":"n",'
                    '"confidence":0.9,"abstained":false}}'
                )
            ),
        ]

        evidence = SimpleNamespace(
            run_id=31307072970,
            workflow_name="ci",
            branch="main",
            sha="abc123",
            failed_jobs=["unit-tests"],
            last_green_run_id=12,
            diff="",
            logs=[],
        )

        result = reasoner.generate(evidence)

        self.assertFalse(result["abstained"])
        reasoner.tools.get_failed_logs.assert_not_called()
        self.assertFalse(result["reasoning_trace"][0]["tool_result"]["ok"])


if __name__ == "__main__":
    unittest.main()
