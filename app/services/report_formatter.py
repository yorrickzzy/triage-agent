from urllib.parse import quote

from app.core.config import APP_BASE_URL


class ReportFormatter:
    def __init__(
        self,
        app_base_url: str = APP_BASE_URL,
    ) -> None:
        self.app_base_url = app_base_url.rstrip("/")

    def _feedback_url(
        self,
        incident_id: int,
        verdict: str,
    ) -> str:
        safe_verdict = quote(verdict, safe="")

        return (
            f"{self.app_base_url}"
            f"/feedback/capture/{incident_id}/{safe_verdict}"
        )

    def _feedback_markdown(
        self,
        incident_id: int,
    ) -> str:
        correct_url = self._feedback_url(
            incident_id,
            "correct",
        )
        wrong_url = self._feedback_url(
            incident_id,
            "wrong",
        )
        not_quite_url = self._feedback_url(
            incident_id,
            "not_quite",
        )

        return f"""
---

### Feedback

Was this RCA helpful?

[👍 Correct]({correct_url}) · [👎 Wrong]({wrong_url}) · [🤏 Not Quite]({not_quite_url})

_Your feedback helps improve the CI/CD Triage Agent._
""".strip()

    def _format_claims(self, claims: list[dict]) -> str:
        if not claims:
            return "No explicit evidence claims were included in this report."

        lines = []
        for claim in claims:
            statement = claim.get("statement", "").strip()
            lines.append(f"- {statement or 'No evidence statement provided.'}")

            for citation in claim.get("citations", []):
                line_start = citation.get("line_start")
                line_end = citation.get("line_end")

                if line_start == line_end:
                    location = f"Line {line_start}"
                else:
                    location = f"Lines {line_start}-{line_end}"

                lines.append(f"  - {location}")

                quote = citation.get("quote", "")
                for q in quote.splitlines():
                    lines.append(f"    > {q}")

            lines.append("")

        return "\n".join(lines).strip()

    def to_markdown(
        self,
        *,
        evidence,
        rca: dict,
        incident_id: int,
    ) -> str:
        summary = rca.get("summary", "No summary was generated.")
        failure_class = rca.get("failure_class", "unknown")
        next_step = rca.get(
            "suggested_next_step",
            "Manual investigation is required.",
        )
        confidence = rca.get("confidence", 0.0)
        claims = rca.get("claims", [])

        report_body = f"""# 🔍 CI/CD Triage Agent Report

**Failure Class:** `{failure_class}`

**Cause:** {summary}

**Confidence:** `{confidence:.2f}`

**Next step:** {next_step}

## Evidence

{self._format_claims(claims)}"""

        feedback_section = self._feedback_markdown(incident_id)

        return f"{report_body}\n\n{feedback_section}"

    def _feedback_slack(
        self,
        incident_id: int,
    ) -> str:
        correct_url = self._feedback_url(
            incident_id,
            "correct",
        )
        wrong_url = self._feedback_url(
            incident_id,
            "wrong",
        )
        not_quite_url = self._feedback_url(
            incident_id,
            "not_quite",
        )

        return (
            "*Feedback*\n"
            "Was this RCA helpful?\n"
            f"<{correct_url}|👍 Correct>  •  "
            f"<{wrong_url}|👎 Wrong>  •  "
            f"<{not_quite_url}|🤏 Not Quite>"
        )

    def to_slack(
        self,
        *,
        evidence,
        rca: dict,
        incident_id: int,
        comment_url: str | None = None,
    ) -> str:
        failure_class = rca.get("failure_class", "unknown")
        summary = rca.get("summary", "No summary was generated.")
        next_step = rca.get(
            "suggested_next_step",
            "Manual investigation is required.",
        )
        confidence = rca.get("confidence", 0.0)

        lines = [
            "🔍 *CI/CD Triage Agent Report*",
            "",
            f"*Failure class:* `{failure_class}`",
            f"*Cause:* {summary}",
            f"*Confidence:* {confidence:.2f}",
            f"*Next step:* {next_step}",
        ]

        if comment_url:
            lines.extend([
                "",
                f"*Full cited report:* {comment_url}",
            ])

        lines.extend([
            "",
            self._feedback_slack(incident_id),
        ])

        return "\n".join(lines)