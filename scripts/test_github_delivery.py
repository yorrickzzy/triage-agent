from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.github_client import GitHubClient
from app.services.github_delivery import GitHubDelivery


# 换成 chamber-playground 中真实存在的 PR 编号
PR_NUMBER = 2


def main() -> None:
    github_client = GitHubClient()
    delivery = GitHubDelivery(github_client)

    markdown = """## 🔍 Chamber Test Report

This is a test comment posted by Chamber.

**Status:** Delivery API works.
"""

    result = delivery.deliver(
        pull_request_number=PR_NUMBER,
        markdown=markdown,
    )

    print("comment id:", result.get("id"))
    print("comment url:", result.get("html_url"))


if __name__ == "__main__":
    main()