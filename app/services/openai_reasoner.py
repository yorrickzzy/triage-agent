import json
from dataclasses import asdict, is_dataclass

from openai import APITimeoutError, OpenAI

from app.core.config import OPENAI_API_KEY, OPENAI_MODEL


class OpenAIReasoner:
    MAX_REASONING_TURNS = 4
    CONTROLLED_AGENT_MODE = True
    LOGS_BUDGET_CHARS = 10000
    DIFF_BUDGET_CHARS = 2500
    RAG_BUDGET_CHARS = 2500
    OBSERVATIONS_BUDGET_CHARS = 6000
    LOG_WINDOW_RADIUS = 40
    LOG_MAX_ANCHORS = 8
    LOG_FALLBACK_HEAD_LINES = 120

    def __init__(self, tools=None):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
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
You are a CI/CD triage agent running in controlled tool-first mode.

{strict_instruction}

You may reason in multiple turns. Return ONLY valid JSON in one of these forms:

1) Request a tool call
{{
    "action": "tool_call",
    "tool": "get_failed_logs|get_commit_diff|get_last_green_run|rerun_job|search_playbook",
    "args": {{ ... }},
    "reason": "..."
}}

2) Return final RCA
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
- At most one tool call per turn.
- Before returning final RCA, you must gather BOTH:
  1) log evidence via get_failed_logs
  2) historical evidence via search_playbook
- If either evidence source is missing, request a tool call instead of final.
- Call-order guidance:
    1) If log evidence is missing, call get_failed_logs first.
    2) If log evidence exists but historical evidence is missing, call search_playbook next.
    3) Once both evidence requirements are satisfied, return final RCA immediately.
- Never cite diff or historical context as primary evidence claims.

Available tools:
- get_failed_logs(run_id)
- get_commit_diff(sha)
- get_last_green_run(workflow_id, branch, current_run_id)
- rerun_job(run_id)
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

Tool observations (the only detailed context you currently have):
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

        try:
            if tool_name == "get_failed_logs":
                run_id = self._normalize_run_id(args.get("run_id"), evidence.run_id)
                logs = self.tools.get_failed_logs(run_id)
                return {
                    "ok": True,
                    "tool": tool_name,
                    "result": self._serialize_raw_logs(logs),
                }

            if tool_name == "get_commit_diff":
                sha = str(args.get("sha") or evidence.sha).strip()
                diff_text = self.tools.get_commit_diff(sha)
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
                accepted = self.tools.rerun_job(run_id)
                return {
                    "ok": True,
                    "tool": tool_name,
                    "result": {
                        "run_id": run_id,
                        "accepted": bool(accepted),
                    },
                }

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

You may reason in multiple turns. Return ONLY valid JSON in one of these forms:

1) Request a tool call
{{
    "action": "tool_call",
    "tool": "search_playbook",
    "args": {{
        "query": "...",
        "limit": 5
    }},
    "reason": "..."
}}

2) Return final RCA
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

            response = self.client.responses.create(
                model=OPENAI_MODEL,
                input=prompt,
            )
            payload = json.loads(response.output_text)

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
        self,
        evidence,
        strict_grounding: bool,
        rag_context: list[dict] | None,
    ) -> dict:
        if self.tools is None:
            return self._controlled_abstain_result(
                summary=(
                    "Controlled mode requires a configured toolbelt, "
                    "but no tools are available."
                ),
                observations=[
                    {
                        "event": "controlled_mode_abstain",
                        "reason": "toolbelt_not_configured",
                    }
                ],
            )

        strict_instruction = ""
        if strict_grounding:
            strict_instruction = """
        IMPORTANT:
        Your previous answer failed grounding verification.
        You must gather log evidence and historical evidence through tools before final output.
        Cite only exact text copied from numbered logs.
        """

        observations: list[dict] = []
        has_log_evidence = False
        has_rag_evidence = False
        evidence_tool_calls = 0

        for _ in range(self.max_reasoning_turns):
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

            prompt = self._build_controlled_loop_prompt(
                evidence=evidence,
                strict_instruction=strict_instruction,
                observations_text=observations_text,
                observations_truncated=observations_truncated,
                has_log_evidence=has_log_evidence,
                has_rag_evidence=has_rag_evidence,
                evidence_tool_calls=evidence_tool_calls,
            )

            response = self.client.responses.create(
                model=OPENAI_MODEL,
                input=prompt,
            )
            payload = json.loads(response.output_text)

            # Backward compatibility: direct RCA payload is treated as final action.
            if self._is_rca_payload(payload):
                if self._has_required_evidence(
                    has_log_evidence=has_log_evidence,
                    has_rag_evidence=has_rag_evidence,
                    evidence_tool_calls=evidence_tool_calls,
                ):
                    return self._with_reasoning_trace(payload, observations)

                observations.append(
                    {
                        "event": "final_blocked",
                        "reason": "missing_required_evidence",
                        "required": {
                            "has_log_evidence": True,
                            "has_rag_evidence": True,
                            "evidence_tool_calls_min": 1,
                        },
                        "current": {
                            "has_log_evidence": has_log_evidence,
                            "has_rag_evidence": has_rag_evidence,
                            "evidence_tool_calls": evidence_tool_calls,
                        },
                    }
                )
                continue

            action = payload.get("action")
            if action == "final" and self._is_rca_payload(payload.get("rca")):
                if self._has_required_evidence(
                    has_log_evidence=has_log_evidence,
                    has_rag_evidence=has_rag_evidence,
                    evidence_tool_calls=evidence_tool_calls,
                ):
                    return self._with_reasoning_trace(payload["rca"], observations)

                observations.append(
                    {
                        "event": "final_blocked",
                        "reason": "missing_required_evidence",
                        "required": {
                            "has_log_evidence": True,
                            "has_rag_evidence": True,
                            "evidence_tool_calls_min": 1,
                        },
                        "current": {
                            "has_log_evidence": has_log_evidence,
                            "has_rag_evidence": has_rag_evidence,
                            "evidence_tool_calls": evidence_tool_calls,
                        },
                    }
                )
                continue

            if action != "tool_call":
                observations.append(
                    {
                        "event": "controlled_mode_abstain",
                        "reason": f"invalid_action:{action}",
                    }
                )
                return self._controlled_abstain_result(
                    summary=(
                        "Controlled mode received an invalid model action "
                        "before evidence requirements were met."
                    ),
                    observations=observations,
                )

            tool_name = payload.get("tool")
            if tool_name == "get_failed_logs":
                # In controlled mode, logs must come from the current failing run.
                # Ignore model-provided run IDs to avoid cross-run contamination.
                payload = {
                    **payload,
                    "args": {
                        **(payload.get("args") or {}),
                        "run_id": evidence.run_id,
                    },
                }
            tool_result = self._execute_tool_call(evidence, payload)
            observations.append(
                {
                    "tool_request": {
                        "tool": tool_name,
                        "args": payload.get("args") or {},
                        "reason": payload.get("reason", ""),
                    },
                    "tool_result": tool_result,
                }
            )

            if self._is_evidence_tool(tool_name):
                evidence_tool_calls += 1

            if not tool_result.get("ok"):
                observations.append(
                    {
                        "event": "controlled_mode_abstain",
                        "reason": "tool_call_failed",
                    }
                )
                return self._controlled_abstain_result(
                    summary=(
                        "Controlled mode stopped because a required tool call failed."
                    ),
                    observations=observations,
                )

            if tool_name == "get_failed_logs":
                log_rows = tool_result.get("result") or []
                if isinstance(log_rows, list):
                    # A successful log fetch attempt satisfies the evidence step,
                    # even when no failed logs are returned.
                    has_log_evidence = True

            if tool_name == "search_playbook":
                chunks = (tool_result.get("result") or {}).get("chunks") or []
                if isinstance(chunks, list):
                    # Historical retrieval can legitimately return zero matches.
                    # Treating this as "missing evidence" causes avoidable abstains.
                    has_rag_evidence = True

        observations.append(
            {
                "event": "controlled_mode_abstain",
                "reason": "max_turns_exceeded",
                "max_reasoning_turns": self.max_reasoning_turns,
            }
        )
        return self._controlled_abstain_result(
            summary=(
                "Controlled mode reached the maximum reasoning turns "
                "before satisfying evidence requirements."
            ),
            observations=observations,
        )

    def generate(
        self,
        evidence,
        rag_context: list[dict] | None = None,
        strict_grounding: bool = False,
    ) -> dict:
        try:
            if getattr(self, "controlled_agent_mode", False):
                return self._generate_controlled_agent_mode(
                    evidence=evidence,
                    strict_grounding=strict_grounding,
                    rag_context=rag_context,
                )

            return self._generate_workflow_mode(
                evidence=evidence,
                rag_context=rag_context,
                strict_grounding=strict_grounding,
            )
        except APITimeoutError as exc:
            return self._fallback_result(exc)
        except Exception as exc:
            return self._fallback_result(exc)