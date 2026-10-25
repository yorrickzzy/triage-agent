import json
import time
from dataclasses import asdict, is_dataclass

from openai import OpenAI

from app.core.config import OPENAI_API_KEY, OPENAI_MODEL
from app.core.telemetry import record_exception, set_span_attributes, tracer
from app.services.recovery_errors import ManualReviewRequired


class OpenAIReasoner:
    MAX_REASONING_TURNS = 4
    MAX_INVESTIGATION_SECONDS = 120
    CONTROLLED_AGENT_MODE = True
    LOGS_BUDGET_CHARS = 10000
    DIFF_BUDGET_CHARS = 2500
    RAG_BUDGET_CHARS = 2500
    OBSERVATIONS_BUDGET_CHARS = 6000
    LOG_WINDOW_RADIUS = 40
    LOG_MAX_ANCHORS = 8
    LOG_FALLBACK_HEAD_LINES = 120

    def __init__(self, tools=None):
        self.client = OpenAI(api_key=OPENAI_API_KEY, timeout=30.0, max_retries=1)
        self.tools = tools
        self.max_reasoning_turns = self.MAX_REASONING_TURNS
        self.controlled_agent_mode = self.CONTROLLED_AGENT_MODE

    def _fallback_result(self, error: Exception) -> dict:
        return {
            "failure_class": "unknown",
            "summary": f"OpenAI request failed: {error}",
            "claims": [],
            "suggested_next_step": "Check the OpenAI API key, network connectivity, and model availability.",
            "confidence": 0.0,
            "abstained": True,
        }

    def _function_tools(self) -> list[dict]:
        if self.tools is None:
            return []

        definitions = [
            ("get_failed_logs", "Read the sanitized current-run snapshot. Optional job_id and original line range select up to 80 lines; null range returns error windows.",
             {"run_id": {"type": ["integer", "null"]},
              "job_id": {"type": ["integer", "null"]},
              "line_start": {"type": ["integer", "null"]},
              "line_end": {"type": ["integer", "null"]}}),
            ("get_commit_diff", "Fetch a commit diff. Null sha uses the current commit.",
             {"sha": {"type": ["string", "null"]}}),
            ("get_last_green_run", "Find the last successful run. Null values use task metadata.",
             {"workflow_id": {"type": ["integer", "string", "null"]},
              "branch": {"type": ["string", "null"]},
              "current_run_id": {"type": ["integer", "null"]}}),
            ("search_playbook", "Search historical failures. Null query uses task metadata; null limit uses 5.",
             {"query": {"type": ["string", "null"]},
              "limit": {"type": ["integer", "null"]}}),
        ]
        tools = []
        for name, description, properties in definitions:
            properties["reason"] = {
                "type": "string",
                "description": "Brief purpose of this tool call for the reasoning trace.",
            }
            tools.append({
                "type": "function",
                "name": name,
                "description": description,
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(properties),
                    "additionalProperties": False,
                },
            })
        return tools

    def _request_action(self, prompt: str, tool_exchange: list, *, final_only: bool = False) -> tuple[dict, list]:
        # Earlier observations remain in the budgeted prompt. Replay the latest
        # exchange, including reasoning items, to pair native calls and outputs.
        response = self.client.responses.create(
            model=OPENAI_MODEL,
            input=([{"role": "user", "content": prompt}] + tool_exchange
                   if tool_exchange else prompt),
            tools=self._function_tools(),
            **({"tool_choice": "none"} if final_only else {}),
            parallel_tool_calls=False,
        )
        output = list(response.output)
        calls = [item for item in output if item.type == "function_call"]
        if len(calls) > 1:
            raise ValueError("At most one tool call is allowed per turn")
        if calls:
            call = calls[0]
            args = json.loads(call.arguments)
            if not isinstance(args, dict):
                raise ValueError("Function arguments must be a JSON object")
            reason = args.pop("reason", "")
            return {
                "action": "tool_call",
                "tool": call.name,
                "args": args,
                "reason": reason,
                "call_id": call.call_id,
            }, output

        payload = json.loads(response.output_text)
        if not isinstance(payload, dict):
            raise ValueError("Final RCA must be a JSON object")
        if payload.get("action") == "tool_call":
            raise ValueError("Tool requests must use native function calling")
        return payload, []

    def _tool_output(self, payload: dict, result: dict) -> dict:
        output, _ = self._truncate_text(
            json.dumps(result, ensure_ascii=False), self.OBSERVATIONS_BUDGET_CHARS
        )
        return {
            "type": "function_call_output",
            "call_id": payload["call_id"],
            "output": output,
        }

    def _format_log_with_line_numbers(self, raw_log) -> str:
        return "\n".join(
            f"{i + 1} | {line}"
            for i, line in enumerate(raw_log.lines)
        )

    def _truncate_text(self, text: str, max_chars: int) -> tuple[str, bool]:
        if max_chars <= 0:
            return "", bool(text)

        if len(text) <= max_chars:
            return text, False

        clipped = text[:max_chars].rstrip()
        return f"{clipped}\n... [truncated]", True

    def _find_error_anchor_lines(self, lines: list[str]) -> list[int]:
        keywords = (
            "error",
            "exception",
            "traceback",
            "assert",
            "failed",
            "failure",
            "caused by",
        )

        anchors = []
        for idx, line in enumerate(lines):
            lowered = line.lower()
            if any(keyword in lowered for keyword in keywords):
                anchors.append(idx)

            if len(anchors) >= self.LOG_MAX_ANCHORS:
                break

        return anchors

    def _windowed_numbered_lines(self, raw_log) -> tuple[str, bool]:
        lines = raw_log.lines
        if not lines:
            return "", False

        anchors = self._find_error_anchor_lines(lines)
        ranges: list[tuple[int, int]] = []

        if anchors:
            for anchor in anchors:
                start = max(0, anchor - self.LOG_WINDOW_RADIUS)
                end = min(len(lines) - 1, anchor + self.LOG_WINDOW_RADIUS)
                ranges.append((start, end))
        else:
            fallback_end = min(len(lines) - 1, self.LOG_FALLBACK_HEAD_LINES - 1)
            ranges.append((0, fallback_end))

        merged: list[tuple[int, int]] = []
        for start, end in sorted(ranges):
            if not merged:
                merged.append((start, end))
                continue

            last_start, last_end = merged[-1]
            if start <= last_end + 1:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))

        output_lines = []
        selected_count = 0
        for start, end in merged:
            if output_lines:
                output_lines.append("...")

            for idx in range(start, end + 1):
                output_lines.append(f"{idx + 1} | {lines[idx]}")
                selected_count += 1

        truncated = selected_count < len(lines)
        return "\n".join(output_lines), truncated

    def _format_logs(self, logs) -> tuple[str, bool]:
        if not logs:
            return "None", False

        sections = []
        any_truncated = False
        for idx, raw_log in enumerate(logs, start=1):
            numbered_window, log_truncated = self._windowed_numbered_lines(raw_log)
            any_truncated = any_truncated or log_truncated

            sections.append(
                (
                    f"[log #{idx}] "
                    f"job_id={raw_log.job_id}; "
                    f"job_name={raw_log.job_name}; "
                    f"line_count={len(raw_log.lines)}; "
                    f"windowed={str(log_truncated).lower()}\n"
                    f"{numbered_window}"
                )
            )

        return "\n\n".join(sections), any_truncated

    def _format_rag_context(
        self,
        rag_context: list[dict] | None,
    ) -> str:
        if not rag_context:
            return "None"

        lines = []
        for idx, chunk in enumerate(rag_context, start=1):
            lines.append(
                (
                    f"{idx}. "
                    f"source={chunk.get('source_type')}#{chunk.get('source_id')}; "
                    f"workflow={chunk.get('workflow_name')}; "
                    f"class={chunk.get('failure_class')}; "
                    f"quality={chunk.get('quality_score', 0.0):.2f}\n"
                    f"{chunk.get('text', '')}"
                )
            )

        return "\n\n".join(lines)

    def _build_tool_query(self, evidence) -> str:
        parts = [evidence.workflow_name or ""]

        for job_name in evidence.failed_jobs[:5]:
            if isinstance(job_name, str) and job_name.strip():
                parts.append(job_name.strip())

        return " ".join(parts).strip()

    def _is_rca_payload(self, payload: dict) -> bool:
        return isinstance(payload, dict) and "failure_class" in payload

    def _normalize_limit(self, value) -> int:
        if not isinstance(value, int):
            return 5

        return min(max(value, 1), 10)

    def _normalize_run_id(self, value, default_value: int) -> int:
        if isinstance(value, int) and value > 0:
            return value

        return default_value

    def _serialize_raw_logs(self, logs) -> list[dict]:
        serialized = []

        for raw_log in logs:
            line_count = len(raw_log.lines)
            windowed, truncated = self._windowed_numbered_lines(raw_log)
            serialized.append(
                {
                    "job_id": raw_log.job_id,
                    "job_name": raw_log.job_name,
                    "line_count": line_count,
                    "truncated": truncated,
                    "numbered_lines": windowed,
                }
            )

        return serialized

    def _to_jsonable(self, value):
        if value is None:
            return None

        if is_dataclass(value):
            return asdict(value)

        if isinstance(value, (str, int, float, bool, list, dict)):
            return value

        if hasattr(value, "__dict__"):
            return dict(value.__dict__)

        return str(value)

    def _with_reasoning_trace(self, rca: dict, observations: list[dict]) -> dict:
        enriched = dict(rca)
        enriched["reasoning_trace"] = list(observations)
        return enriched

    def _prepend_reasoning_trace(
        self,
        rca: dict,
        observations: list[dict],
    ) -> dict:
        if not observations:
            return rca

        enriched = dict(rca)
        existing = enriched.get("reasoning_trace")
        if isinstance(existing, list):
            enriched["reasoning_trace"] = list(observations) + list(existing)
        else:
            enriched["reasoning_trace"] = list(observations)

        return enriched

    def _is_evidence_tool(self, tool_name: str | None) -> bool:
        return tool_name in {"get_failed_logs", "search_playbook"}

    def _has_required_evidence(
        self,
        *,
        has_log_evidence: bool,
        has_rag_evidence: bool,
        evidence_tool_calls: int,
    ) -> bool:
        return (
            evidence_tool_calls >= 1
            and has_log_evidence
            and has_rag_evidence
        )

    def _controlled_abstain_result(
        self,
        *,
        summary: str,
        observations: list[dict],
    ) -> dict:
        return {
            "failure_class": "unknown",
            "summary": summary,
            "claims": [],
            "suggested_next_step": "A human should inspect this CI failure manually.",
            "confidence": 0.0,
            "abstained": True,
            "reasoning_trace": list(observations),
        }

    def _build_controlled_loop_prompt(
        self,
        *,
        evidence,
        strict_instruction: str,
        observations_text: str,
        observations_truncated: bool,
        has_log_evidence: bool,
        has_rag_evidence: bool,
        evidence_tool_calls: int,
    ) -> str:
        return f"""
You are a CI/CD triage agent investigating a fixed evidence snapshot.

{strict_instruction}

You may reason in multiple turns. Request tools using native function calls.
For the final RCA, return ONLY valid JSON in this form:
{{
    "action": "final",
    "rca": {{
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
}}

Hard constraints:
- Use the supplied sanitized evidence snapshot; do not fetch the same evidence again.
- Call tools only when additional information materially reduces uncertainty.
- Historical search is optional. Build queries from concrete errors, modules and symptoms.
- Empty historical results do not require abstention. History is reference, not proof.
- Cite exact current log text with original line numbers; every substantive claim needs citations.
- If evidence is insufficient, return an abstained RCA instead of guessing.
- Suspected flakiness is a recommendation; only the outer workflow may rerun jobs.
- At most one tool call per turn.

Available tools:
- get_failed_logs(run_id, job_id, line_start, line_end): read the saved snapshot
- get_commit_diff(sha): read the saved diff
- get_last_green_run(workflow_id, branch, current_run_id)
- search_playbook(query, limit)

Task metadata:
Workflow: {evidence.workflow_name}
Branch: {evidence.branch}
SHA: {evidence.sha}
Failed jobs: {evidence.failed_jobs}
Last green run id: {evidence.last_green_run_id}

Evidence status:
- log_evidence_ready: {str(has_log_evidence).lower()}
- historical_evidence_ready: {str(has_rag_evidence).lower()}
- evidence_tool_calls: {evidence_tool_calls}
- observations_truncated: {str(observations_truncated).lower()}

Additional tool observations:
{observations_text}
"""

    def _build_prefetch_observation(
        self,
        *,
        evidence,
        input_context: list[dict] | None,
        merged_context: list[dict] | None,
    ) -> dict | None:
        query = self._build_tool_query(evidence)
        input_count = len(input_context or [])
        merged_count = len(merged_context or [])
        added_count = max(0, merged_count - input_count)

        if self.tools is None and input_count == 0:
            return None

        return {
            "event": "prefetch_rag_context",
            "toolbelt_configured": self.tools is not None,
            "query": query,
            "input_context_count": input_count,
            "output_context_count": merged_count,
            "added_context_count": added_count,
            "source": "triage_service.retrieve_for_evidence + reasoner._collect_tool_context",
        }

    def _execute_tool_call(self, evidence, payload: dict) -> dict:
        if self.tools is None:
            return {
                "ok": False,
                "error": "toolbelt is not configured",
            }

        tool_name = payload.get("tool")
        args = payload.get("args") or {}
        span_name = f"agent.tool_call.{tool_name}" if tool_name else "agent.tool_call"

        if tracer is None:
            return self._execute_tool_call_impl(evidence, payload)

        with tracer.start_as_current_span(span_name) as span:
            try:
                set_span_attributes(
                    span,
                    run_id=evidence.run_id,
                    workflow_name=evidence.workflow_name,
                    tool_name=tool_name,
                    tool_args=json.dumps(args, ensure_ascii=False)[:500],
                )
                result = self._execute_tool_call_impl(evidence, payload)
                set_span_attributes(
                    span,
                    ok=bool(result.get("ok", False)),
                    result_error=result.get("error"),
                )
                return result
            except Exception as exc:
                record_exception(span, exc)
                raise

    def _execute_tool_call_impl(self, evidence, payload: dict) -> dict:
        if self.tools is None:
            return {
                "ok": False,
                "error": "toolbelt is not configured",
            }

        tool_name = payload.get("tool")
        args = payload.get("args") or {}

        try:
            if tool_name == "get_failed_logs":
                run_id = self._normalize_run_id(args.get("run_id"), evidence.run_id)
                if run_id != evidence.run_id:
                    raise ValueError("Only the current evidence snapshot is available")
                logs = evidence.logs
                if args.get("job_id") is not None:
                    logs = [log for log in logs if log.job_id == args["job_id"]]
                    if not logs:
                        raise ValueError("Unknown job_id in evidence snapshot")
                if args.get("line_start") is not None or args.get("line_end") is not None:
                    start = args.get("line_start") if args.get("line_start") is not None else 1
                    end = args.get("line_end") if args.get("line_end") is not None else start + 79
                    if not isinstance(start, int) or not isinstance(end, int) or start < 1 or end < start:
                        raise ValueError("Invalid log line range")
                    end = min(end, start + 79)
                    return {"ok": True, "tool": tool_name, "result": [
                        {"job_id": log.job_id, "job_name": log.job_name,
                         "numbered_text": "\n".join(f"{i + 1} | {line}" for i, line in
                             enumerate(log.lines) if start <= i + 1 <= end)} for log in logs]}
                return {
                    "ok": True,
                    "tool": tool_name,
                    "result": self._serialize_raw_logs(logs),
                }

            if tool_name == "get_commit_diff":
                sha = str(args.get("sha") or evidence.sha).strip()
                if sha != evidence.sha:
                    raise ValueError("Only the current commit snapshot is available")
                diff_text = evidence.diff
                return {
                    "ok": True,
                    "tool": tool_name,
                    "result": {
                        "sha": sha,
                        "diff": diff_text,
                    },
                }

            if tool_name == "get_last_green_run":
                workflow_id = args.get("workflow_id")
                if workflow_id is None:
                    workflow_id = args.get("workflow")
                if workflow_id is None:
                    workflow_id = evidence.workflow_name

                branch = str(args.get("branch") or evidence.branch)
                current_run_id = self._normalize_run_id(
                    args.get("current_run_id"),
                    evidence.run_id,
                )

                run_ref = self.tools.get_last_green_run(
                    workflow_id=workflow_id,
                    branch=branch,
                    current_run_id=current_run_id,
                )
                return {
                    "ok": True,
                    "tool": tool_name,
                    "result": self._to_jsonable(run_ref),
                }

            if tool_name == "rerun_job":
                run_id = self._normalize_run_id(args.get("run_id"), evidence.run_id)
                raise ValueError("Reruns are executed only by the outer triage workflow")

            if tool_name == "search_playbook":
                query = str(args.get("query") or self._build_tool_query(evidence)).strip()
                limit = self._normalize_limit(args.get("limit", 5))
                results = self.tools.search_playbook(query, limit=limit)
                return {
                    "ok": True,
                    "tool": tool_name,
                    "result": {
                        "query": query,
                        "limit": limit,
                        "chunks": results,
                    },
                }

            return {
                "ok": False,
                "tool": tool_name,
                "error": f"unsupported tool: {tool_name}",
            }
        except Exception as exc:
            return {
                "ok": False,
                "tool": tool_name,
                "error": str(exc),
            }

    def _build_loop_prompt(
        self,
        *,
        evidence,
        diff_text: str,
        rag_text: str,
        log_text: str,
        strict_instruction: str,
        observations_text: str,
        truncation_status: dict,
    ) -> str:
        return f"""
You are a CI/CD triage agent.

{strict_instruction}

You may reason in multiple turns. Request tools using native function calls.
For the final RCA, return ONLY valid JSON in this form:
{{
    "action": "final",
    "rca": {{
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
}}

Rules:
- You can call at most one tool per turn.
- Use only the listed tool name.
- Prefer returning final RCA as soon as evidence is sufficient.
- Never cite diff or historical context as evidence.
- If context is truncated and evidence is insufficient, abstain.

Available tools:
- get_failed_logs(run_id)
- get_commit_diff(sha)
- get_last_green_run(workflow_id, branch, current_run_id)
- rerun_job(run_id)  # use only to validate suspected flakiness
- search_playbook(query, limit)

Evidence:

Workflow: {evidence.workflow_name}
Branch: {evidence.branch}
SHA: {evidence.sha}
Failed jobs: {evidence.failed_jobs}
Last green run id: {evidence.last_green_run_id}

Context truncation status:
- logs_truncated: {str(truncation_status.get("logs_truncated", False)).lower()}
- diff_truncated: {str(truncation_status.get("diff_truncated", False)).lower()}
- rag_truncated: {str(truncation_status.get("rag_truncated", False)).lower()}
- observations_truncated: {str(truncation_status.get("observations_truncated", False)).lower()}

Diff:
{diff_text}

Historical context (for hints only; never cite this as evidence):
{rag_text}

Logs:
{log_text}

Previous tool observations:
{observations_text}
"""

    def _merge_rag_context(
        self,
        rag_context: list[dict] | None,
        tool_chunks: list[dict],
    ) -> list[dict]:
        merged = list(rag_context or [])

        seen = {
            (chunk.get("source_type"), chunk.get("source_id"), chunk.get("text"))
            for chunk in merged
        }

        for chunk in tool_chunks:
            key = (
                chunk.get("source_type"),
                chunk.get("source_id"),
                chunk.get("text"),
            )
            if key in seen:
                continue

            merged.append(chunk)
            seen.add(key)

        return merged

    def _collect_tool_context(
        self,
        evidence,
        rag_context: list[dict] | None,
    ) -> list[dict] | None:
        if self.tools is None:
            return rag_context

        try:
            query = self._build_tool_query(evidence)
            if not query:
                return rag_context

            tool_chunks = self.tools.search_playbook(query, limit=5)
            if not tool_chunks:
                return rag_context

            return self._merge_rag_context(rag_context, tool_chunks)
        except Exception as exc:
            print(f"[reasoner] tool context unavailable: {exc}", flush=True)
            return rag_context

    def _generate_workflow_mode(
        self,
        evidence,
        rag_context: list[dict] | None,
        strict_grounding: bool,
    ) -> dict:
        merged_rag_context = self._collect_tool_context(evidence, rag_context)
        prefetch_observation = self._build_prefetch_observation(
            evidence=evidence,
            input_context=rag_context,
            merged_context=merged_rag_context,
        )
        raw_log_text, logs_windowed = self._format_logs(evidence.logs)
        log_text, logs_budget_truncated = self._truncate_text(
            raw_log_text,
            self.LOGS_BUDGET_CHARS,
        )
        diff_text, diff_truncated = self._truncate_text(
            evidence.diff,
            self.DIFF_BUDGET_CHARS,
        )

        strict_instruction = ""
        if strict_grounding:
            strict_instruction = """
        IMPORTANT:
        Your previous answer failed grounding verification.
        Cite only exact text copied from the numbered logs.
        Do not cite the diff.
        Do not cite historical context.
        Do not paraphrase quotes.
        Every quote must appear verbatim in the cited log line range.
        If you cannot ground the answer, set abstained=true.
        """

        current_context = list(merged_rag_context or [])
        observations: list[dict] = []
        if prefetch_observation is not None:
            observations.append(prefetch_observation)

        tool_exchange: list = []
        for _ in range(self.max_reasoning_turns):
            rag_raw = self._format_rag_context(current_context)
            rag_text, rag_truncated = self._truncate_text(
                rag_raw,
                self.RAG_BUDGET_CHARS,
            )

            observations_raw = "None"
            if observations:
                observations_raw = json.dumps(
                    observations,
                    ensure_ascii=False,
                    indent=2,
                )
            observations_text, observations_truncated = self._truncate_text(
                observations_raw,
                self.OBSERVATIONS_BUDGET_CHARS,
            )

            truncation_status = {
                "logs_truncated": logs_windowed or logs_budget_truncated,
                "diff_truncated": diff_truncated,
                "rag_truncated": rag_truncated,
                "observations_truncated": observations_truncated,
            }

            prompt = self._build_loop_prompt(
                evidence=evidence,
                diff_text=diff_text,
                rag_text=rag_text,
                log_text=log_text,
                strict_instruction=strict_instruction,
                observations_text=observations_text,
                truncation_status=truncation_status,
            )

            payload, tool_exchange = self._request_action(prompt, tool_exchange)

            # Backward compatibility: allow direct RCA output without action wrapper.
            if self._is_rca_payload(payload):
                return self._with_reasoning_trace(payload, observations)

            action = payload.get("action")
            if action == "final" and self._is_rca_payload(payload.get("rca")):
                return self._with_reasoning_trace(payload["rca"], observations)

            if action != "tool_call":
                return self._fallback_result(
                    ValueError(f"Invalid reasoner action: {action}")
                )

            tool_result = self._execute_tool_call(evidence, payload)
            tool_exchange.append(self._tool_output(payload, tool_result))
            observations.append(
                {
                    "tool_request": {
                        "tool": payload.get("tool"),
                        "args": payload.get("args") or {},
                        "reason": payload.get("reason", ""),
                    },
                    "tool_result": tool_result,
                }
            )

            if tool_result.get("ok") and payload.get("tool") == "search_playbook":
                current_context = self._merge_rag_context(
                    current_context,
                    (tool_result.get("result") or {}).get("chunks") or [],
                )

        return {
            "failure_class": "unknown",
            "summary": "Reasoner did not return a final RCA within the turn limit.",
            "claims": [],
            "suggested_next_step": "A human should inspect this CI failure manually.",
            "confidence": 0.0,
            "abstained": True,
            "reasoning_trace": observations,
        }

    def _generate_controlled_agent_mode(
        self, evidence, strict_grounding, rag_context, checkpoint=None,
        repair_context=None,
    ) -> dict:
        state = checkpoint.state if checkpoint else {}
        started = time.monotonic()
        elapsed = state.get("elapsed_seconds", 0.0)
        if (state.get("pending") or {}).get("tool") == "rerun_job" and state.get("tool_started"):
            raise ManualReviewRequired("Legacy rerun outcome requires manual reconciliation")
        observations = state.get("observations", [])
        tool_exchange = state.get("tool_exchange", [])
        pending = state.get("pending")
        logs, _ = self._format_logs(evidence.logs)
        logs, _ = self._truncate_text(logs, self.LOGS_BUDGET_CHARS)
        diff, _ = self._truncate_text(evidence.diff, self.DIFF_BUDGET_CHARS)
        history, _ = self._truncate_text(self._format_rag_context(rag_context), self.RAG_BUDGET_CHARS)
        snapshot = f"\nSanitized evidence snapshot (original log line numbers):\n{logs}\nCommit diff:\n{diff}\nHistorical reference:\n{history}"
        repair = ""
        if strict_grounding:
            repair = "Correct the previous RCA using the same evidence. Fix the validation errors; do not repeat investigation unnecessarily."
            if repair_context:
                details, _ = self._truncate_text(json.dumps(repair_context, ensure_ascii=False), self.OBSERVATIONS_BUDGET_CHARS)
                repair += "\n" + details

        def save(turn, pending=None):
            if checkpoint:
                checkpoint.save({"turn": turn, "observations": observations,
                    "tool_exchange": tool_exchange, "pending": pending, "tool_started": False,
                    "elapsed_seconds": elapsed + time.monotonic() - started})

        # Reserve a separate final-only turn after the investigation budget.
        for turn in range(state.get("turn", 0), self.max_reasoning_turns + 1):
            observation_text, clipped = self._truncate_text(
                json.dumps(observations, ensure_ascii=False), self.OBSERVATIONS_BUDGET_CHARS)
            prompt = self._build_controlled_loop_prompt(
                evidence=evidence, strict_instruction=repair,
                observations_text=observation_text, observations_truncated=clipped,
                has_log_evidence=bool(evidence.logs), has_rag_evidence=bool(rag_context),
                evidence_tool_calls=len(observations)) + snapshot
            final_only = (turn == self.max_reasoning_turns or
                          elapsed + time.monotonic() - started >= self.MAX_INVESTIGATION_SECONDS)
            if final_only:
                prompt += "\nInvestigation budget exhausted. Return final RCA or abstain now. No further tools."
            if pending is not None:
                payload, pending = pending, None
            else:
                payload, tool_exchange = self._request_action(prompt, tool_exchange, final_only=final_only)
                save(turn, payload)
            rca = payload if self._is_rca_payload(payload) else payload.get("rca")
            if isinstance(rca, dict) and self._is_rca_payload(rca):
                return self._with_reasoning_trace(rca, observations)
            if final_only or payload.get("action") != "tool_call":
                return self._controlled_abstain_result(
                    summary="No valid final RCA within the investigation budget.", observations=observations)
            result = self._execute_tool_call(evidence, payload)
            tool_exchange.append(self._tool_output(payload, result))
            observations.append({"tool_request": {"tool": payload.get("tool"),
                "args": payload.get("args") or {}, "reason": payload.get("reason", "")},
                "tool_result": result})
            # Optional investigation failures are observations, not automatic abstentions.
            save(turn + 1)
        return self._controlled_abstain_result(summary="Investigation budget exhausted.", observations=observations)

    def generate(
        self, evidence, rag_context: list[dict] | None = None,
        strict_grounding: bool = False, checkpoint=None, repair_context=None,
    ) -> dict:
        try:
            if getattr(self, "controlled_agent_mode", False):
                if checkpoint and checkpoint.result is not None:
                    return checkpoint.result
                result = self._generate_controlled_agent_mode(
                    evidence=evidence, strict_grounding=strict_grounding,
                    rag_context=rag_context, checkpoint=checkpoint, repair_context=repair_context)
                if checkpoint:
                    checkpoint.complete(result)
                return result
            return self._generate_workflow_mode(
                evidence=evidence, rag_context=rag_context,
                strict_grounding=strict_grounding)
        except Exception as exc:
            if checkpoint is not None:
                raise
            return self._fallback_result(exc)
