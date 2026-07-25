from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.bootstrap import build_triage_service


# 必须换成 PR #2 对应的失败 workflow run_id
RUN_ID = 29178076814


def main() -> None:
    service = build_triage_service()

    result = service.triage(RUN_ID)

    print("incident_id:", result.incident_id)
    print("action:", result.action)
    print("reason:", result.reason)
    print("comment_url:", result.comment_url)
    print("rca:", result.rca)


if __name__ == "__main__":
    main()