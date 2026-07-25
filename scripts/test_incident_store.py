from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from types import SimpleNamespace

from app.core.database import SessionLocal, create_tables
from app.services.incident_store import IncidentStore


def main() -> None:
    create_tables()

    evidence = SimpleNamespace(
        run_id=28731123848,
        workflow_name="CI",
        branch="main",
        sha="c18d64e7c009a7f4e4d2fc041faf8369a1ee6752",
        pull_request_number=2,
    )

    rca = {
        "failure_class": "test",
        "summary": (
            "Pytest failed because add(1, 2) returned 3 "
            "while the test expected 4."
        ),
        "suggested_next_step": (
            "Restore the expected value to 3 and rerun CI."
        ),
        "confidence": 0.96,
        "abstained": False,
    }

    store = IncidentStore(SessionLocal)

    incident = store.create_incident(
        evidence=evidence,
        rca=rca,
        action="delivered",
        decision_reason=(
            "RCA is grounded and confidence is sufficient."
        ),
        comment_url=(
            "https://github.com/yorrickzzy/"
            "chamber-playground/pull/2"
        ),
    )

    print("incident id:", incident.id)
    print("run id:", incident.run_id)
    print("failure class:", incident.failure_class)
    print("action:", incident.action)
    print("created at:", incident.created_at)


if __name__ == "__main__":
    main()