import json
from openai import APITimeoutError, OpenAI

from app.core.config import OPENAI_API_KEY, OPENAI_MODEL


class OpenAIReasoner:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def _fallback_result(self, error: Exception) -> dict:
        return {
            "failure_class": "unknown",
            "summary": f"OpenAI request failed: {error}",
            "claims": [],
            "suggested_next_step": "Check the OpenAI API key, network connectivity, and model availability.",
            "confidence": 0.0,
            "abstained": True,
        }

    def _format_log_with_line_numbers(self, raw_log) -> str:
        return "\n".join(
            f"{i + 1} | {line}"
            for i, line in enumerate(raw_log.lines)
        )

    def generate(self, evidence, strict_grounding: bool = False) -> dict:
        log_text = self._format_log_with_line_numbers(evidence.logs[0])

        strict_instruction = ""
        if strict_grounding:
            strict_instruction = """
        IMPORTANT:
        Your previous answer failed grounding verification.
        Cite only exact text copied from the numbered logs.
        Do not cite the diff.
        Do not paraphrase quotes.
        Every quote must appear verbatim in the cited log line range.
        If you cannot ground the answer, set abstained=true.
        """

        prompt = f"""
You are a CI/CD triage agent.

{strict_instruction}

Return ONLY valid JSON with this shape:
{{
  "failure_class": "lint|format|test|build|dependency|flaky",
  "summary": "...",
  "claims": [
    {{
      "statement": "...",
      "citations": [
        {{
          "line_start": 1,
          "line_end": 1,
          "quote": "exact quote from logs"
        }}
      ]
    }}
  ],
  "suggested_next_step": "...",
  "confidence": 0.0,
  "abstained": false
}}

Evidence:

Workflow: {evidence.workflow_name}
Branch: {evidence.branch}
SHA: {evidence.sha}
Failed jobs: {evidence.failed_jobs}
Last green run id: {evidence.last_green_run_id}

Diff:
{evidence.diff}

Logs:
{log_text}
"""

        try:
            response = self.client.responses.create(
                model=OPENAI_MODEL,
                input=prompt,
            )
            return json.loads(response.output_text)
        except APITimeoutError as exc:
            return self._fallback_result(exc)
        except Exception as exc:
            return self._fallback_result(exc)