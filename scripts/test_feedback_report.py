from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from types import SimpleNamespace

from app.services.report_formatter import ReportFormatter


def main() -> None:
    evidence = SimpleNamespace(
        run_id=123,
        workflow_name="CI",
        branch="feature/test",
        sha="abc123",
        pull_request_number=2,
    )

    rca = {
        "failure_class": "test",
        "summary": "A pytest assertion failed.",
        "suggested_next_step": "Fix the expected value and rerun CI.",
        "confidence": 0.95,
        "claims": [],
    }

    formatter = ReportFormatter(
        app_base_url="http://localhost:8000",
    )

    markdown = formatter.to_markdown(
        evidence=evidence,
        rca=rca,
        incident_id=1,
    )

    slack = formatter.to_slack(
        evidence=evidence,
        rca=rca,
        incident_id=1,
    )

    print("=== GitHub Markdown ===")
    print(markdown)

    print("\n=== Slack Message ===")
    print(slack)


if __name__ == "__main__":
    main()