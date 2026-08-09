from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.triage_toolbelt import DefaultTriageToolbelt, RunRef


class FakeGitHubClient:
    def __init__(self):
        self.calls = []

    def get_failed_jobs(self, run_id: int):
        self.calls.append(("get_failed_jobs", run_id))
        return [
            {"id": 11, "name": "lint"},
            {"id": 12, "name": "tests"},
        ]

    def download_job_logs(self, job_id: int) -> str:
        self.calls.append(("download_job_logs", job_id))
        return "\ufeffline 1\nline 2"

    def get_commit(self, sha: str):
        self.calls.append(("get_commit", sha))
        return {"parents": [{"sha": "parent-sha"}]}

    def compare_commits(self, base_sha: str, head_sha: str) -> str:
        self.calls.append(("compare_commits", base_sha, head_sha))
        return f"diff:{base_sha}->{head_sha}"

    def get_last_green_run(self, workflow_id, branch, current_run_id):
        self.calls.append(("get_last_green_run", workflow_id, branch, current_run_id))
        return {
            "id": 99,
            "workflow_id": workflow_id,
            "name": "CI",
            "head_branch": branch,
            "head_sha": "abc123",
            "status": "completed",
            "conclusion": "success",
        }

    def rerun_failed_jobs(self, run_id: int) -> bool:
        self.calls.append(("rerun_failed_jobs", run_id))
        return True


class FakeRagStore:
    def __init__(self):
        self.calls = []

    def search_playbook(self, query: str, limit: int = 5):
        self.calls.append(("search_playbook", query, limit))
        return [{"text": query, "quality_score": 1.0}]


class DefaultTriageToolbeltTests(unittest.TestCase):
    def setUp(self):
        self.github = FakeGitHubClient()
        self.rag_store = FakeRagStore()
        self.toolbelt = DefaultTriageToolbelt(self.github, self.rag_store)

    def test_get_failed_logs_returns_raw_logs(self):
        logs = self.toolbelt.get_failed_logs(123)

        self.assertEqual(len(logs), 2)
        self.assertEqual(logs[0].job_id, 11)
        self.assertEqual(logs[0].text, "line 1\nline 2")
        self.assertEqual(logs[0].lines, ["line 1", "line 2"])
        self.assertIn(("get_failed_jobs", 123), self.github.calls)
        self.assertIn(("download_job_logs", 11), self.github.calls)

    def test_get_commit_diff_uses_first_parent(self):
        diff = self.toolbelt.get_commit_diff("head-sha")

        self.assertEqual(diff, "diff:parent-sha->head-sha")
        self.assertIn(("get_commit", "head-sha"), self.github.calls)
        self.assertIn(("compare_commits", "parent-sha", "head-sha"), self.github.calls)

    def test_get_last_green_run_wraps_run_ref(self):
        run_ref = self.toolbelt.get_last_green_run("workflow-1", "main", 77)

        self.assertIsInstance(run_ref, RunRef)
        self.assertEqual(run_ref.run_id, 99)
        self.assertEqual(run_ref.workflow_id, "workflow-1")
        self.assertEqual(run_ref.workflow_name, "CI")
        self.assertEqual(run_ref.branch, "main")
        self.assertEqual(run_ref.sha, "abc123")

    def test_rerun_job_delegates(self):
        self.assertTrue(self.toolbelt.rerun_job(55))
        self.assertIn(("rerun_failed_jobs", 55), self.github.calls)

    def test_search_playbook_delegates(self):
        results = self.toolbelt.search_playbook("pytest error", limit=3)

        self.assertEqual(results, [{"text": "pytest error", "quality_score": 1.0}])
        self.assertIn(("search_playbook", "pytest error", 3), self.rag_store.calls)


if __name__ == "__main__":
    unittest.main()